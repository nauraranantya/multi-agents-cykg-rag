# src/graph/workflow.py
"""
Rewired around question_generation_agent.py: instead of a fixed pipeline
(guardrails -> vector -> review -> cypher -> review -> log_analysis ->
mcp_rdf -> synthesizer) with a hardcoded reflection/retry loop
(max_iterations=3) at each step, the flow is now:

    guardrails (relevance only)
        -> question_generation (several hypothesis-tagged questions)
        -> dispatch_retrieval (routes each question to cypher/vector/cskg
                                per its `source` tag, concurrently)
        -> review_evidence (hypothesis verdict per question, not a retry gate)
        -> synthesizer (reasons over the hypothesis summary as a whole)
        -> grounding_check (unchanged: one bounded retry on ungrounded citations)

Dissolved, and why:
- reflection_agent.py (vector_reflection_chain, reflection_chain): its job
  was reactive rephrase-and-retry on a single failed question.
  question_generation_agent.py now asks several well-targeted questions up
  front, so a single empty result is informative (rules out a hypothesis),
  not a failure needing recovery.
- log_analysis_agent.py: its "decide cskg_required" job is now decided
  per-question, upfront, by question_generation_agent's `source` tag rather
  than after cypher/vector results come back. Its "summarize findings" job
  is now the deterministic (non-LLM) _render_hypothesis_summary below, fed
  into the synthesizer directly -- one fewer LLM call, not just a moved one.
- routing_agent.py: already dead code (never wired into this graph even
  before this change).

review_vector_answer/review_cypher_answer collapse into one
review_evidence node (review_agent.py's review_chain now runs once per
generated question rather than once per source), since there's no longer a
per-source retry branch that needed two separate conditional edges.

Two later additions adapt this same fixed topology to the multi-turn
investigation flow (src/investigation/session.py) without changing the
graph's edges -- both are node-internal short-circuits, not new routing:
- `state['skip_guardrails']` (turn 1 of a case investigation only):
  guardrails_node skips its LLM call entirely -- the input is a fixed
  directive over a case a human already selected, not a question needing
  triage.
- `state['investigation_mode']` (guardrails_agent.py's per-turn routing
  decision, meaningful on follow-up turns): 'full_hypothesis' behaves
  exactly as described above; 'direct_question' asks
  question_generation_node for exactly one targeted question instead of
  3-5; 'answer_from_context' skips question_generation/dispatch_retrieval/
  review_evidence entirely and lets synthesizer answer straight from
  whatever question_results/log_cypher_context/log_vector_context/
  mcp_rdf_context the last turn that actually retrieved something left
  behind -- these are plain (non-reducer) state fields, so the checkpointer
  carries them forward unchanged across a turn that does no new retrieval,
  which is what makes this "cache" possible with no extra state or
  storage.
"""
import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutureTimeoutError
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, END
from src.graph.state import AgentState

from src.agents.guardrails_agent import guardrails_chain, _render_alert_context
from src.agents.question_generation_agent import generate_questions
from src.agents.review_agent import review_chain
from src.agents.synthesizer_agent import synthesis_chain, SynthesizedReport
from src.agents.vector_agent import query_vector_search
from src.agents.cypher_agent import query_cypher
from src.agents.mcp_rdf_agent import NO_DATA_MARKER as _MCP_NO_DATA_MARKER, run_mcp_agent
from src.agents.grounding_check import check_grounding

logger = logging.getLogger(__name__)  # graph assembly / module-level only

# One dedicated logger per agent/node, named so %(name)s in the log line
# (see src/utils/logging_config.py's format string) identifies which agent
# produced it at a glance -- "agent.synthesizer" vs. "agent.guardrails" --
# instead of every node sharing this module's own logger name and relying
# on an ad-hoc text tag inside the message to tell them apart. Lets you
# filter one agent's output alone, e.g. `grep 'agent.synthesizer'`.
guardrails_logger = logging.getLogger("agent.guardrails")
question_generation_logger = logging.getLogger("agent.question_generation")
dispatch_retrieval_logger = logging.getLogger("agent.dispatch_retrieval")
review_evidence_logger = logging.getLogger("agent.review_evidence")
synthesizer_logger = logging.getLogger("agent.synthesizer")
grounding_check_logger = logging.getLogger("agent.grounding_check")

# A single retrieval attempt (LLM call(s) + Neo4j query) has no timeout of
# its own by default. Every retrieval call is run under a hard wall-clock
# deadline; a timeout is treated as a normal retrieval failure (the
# except blocks below already degrade to an error string in that
# question's context) rather than blocking indefinitely -- same fix as the
# 37/11-minute MCP SPARQL hangs (socket.setdefaulttimeout in
# mcp-cskg-rdf/server.py).
NODE_CALL_TIMEOUT_SECONDS = int(os.environ.get("NODE_CALL_TIMEOUT_SECONDS", "60"))
_retrieval_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="retrieval-timeout")

# How many past turns' retrieved evidence dispatch_retrieval_node keeps
# accumulated in state['question_results'] (each entry tagged with the
# turn it came from) before dropping the oldest. Without this, a
# full_hypothesis/direct_question turn wholesale REPLACED the previous
# turn's question_results/log_cypher_context/log_vector_context/
# mcp_rdf_context (none of AgentState's fields besides `messages` have a
# LangGraph reducer, so a node's return value overwrites, never merges) --
# meaning a later 'answer_from_context' turn could only ever see the most
# recent retrieval-performing turn's evidence, not anything found earlier
# in the conversation. Bounded rather than unbounded so a long
# conversation doesn't grow this (and therefore every downstream LLM
# call's prompt size) forever.
EVIDENCE_WINDOW_TURNS = int(os.environ.get("EVIDENCE_WINDOW_TURNS", "5"))


def _window_question_results(results: list, current_turn: int) -> list:
    """Keeps only entries from the most recent EVIDENCE_WINDOW_TURNS
    distinct turns (including the current one) -- oldest turns dropped
    first, in whole-turn units (never splits a single turn's questions
    across the boundary). Entries missing a 'turn' tag (shouldn't happen
    going forward, but defensive against any pre-existing checkpointed
    state from before this field existed) are treated as belonging to the
    current turn rather than crashing."""
    turns_present = sorted({r.get('turn', current_turn) for r in results})
    keep_turns = set(turns_present[-EVIDENCE_WINDOW_TURNS:])
    return [r for r in results if r.get('turn', current_turn) in keep_turns]


def _call_with_timeout(fn, *args, timeout: int = NODE_CALL_TIMEOUT_SECONDS, **kwargs):
    future = _retrieval_executor.submit(fn, *args, **kwargs)
    try:
        return future.result(timeout=timeout)
    except _FutureTimeoutError:
        raise TimeoutError(
            f"{getattr(fn, '__name__', fn)} did not complete within {timeout}s"
        )


def _render_conversation_context(messages: list) -> str:
    """Renders every prior turn (all but the current turn's own trailing
    human message, which is already passed separately as `question`) as
    alternating Analyst/Investigator lines, for the case-investigation
    chatroom (src/investigation/session.py). 'N/A' on a fresh thread's
    first turn -- the whole point of checkpointing state['messages'] via
    src.investigation.session's checkpointed_app is so this has real
    history to render on later turns, rather than every follow-up being
    answered as if it were a brand new, context-free question."""
    if not messages or len(messages) <= 1:
        return "N/A (first turn of this investigation)"
    lines = []
    for m in messages[:-1]:
        role = "Analyst" if getattr(m, "type", None) == "human" else "Investigator"
        lines.append(f"{role}: {getattr(m, 'content', str(m))}")
    return "\n".join(lines)


def _render_prior_evidence(state: AgentState) -> str:
    """Compact summary of what's already been retrieved in this
    investigation thread (persists across checkpointed turns since
    question_results isn't a reducer -- see state.py), used by guardrails
    to judge whether a follow-up can be answered from it directly instead
    of triggering new retrieval (GuardrailsOutput.investigation_mode).
    'N/A' when nothing has been retrieved yet (turn 1, or src/run.py/
    eval/'s single-shot paths, which never populate question_results)."""
    summary = _render_hypothesis_summary(state.get('question_results') or [])
    return summary if summary != "Not applicable for this query." else "N/A (no evidence gathered yet)"


def _current_turn_number(messages: list) -> int:
    """1-indexed count of human messages so far, used to give each turn's
    answer AI message a stable id (see guardrails_node/synthesize_node) --
    `add_messages` replaces an existing message sharing an id instead of
    appending a duplicate, so a bounded grounding-check retry (synthesize_node
    running twice in one turn) doesn't leave two answers in the transcript."""
    return sum(1 for m in (messages or []) if getattr(m, "type", None) == "human") or 1


# --- Node Definition: Guardrails ---
def guardrails_node(state: AgentState):
    """Relevance check + investigation-mode routing in one call. Used to
    also decide escalation ("should this auto-triggered alert be
    investigated at all") -- removed once ingestion/investigation split
    into two processes (see guardrails_agent.py's module docstring):
    nothing auto-triggers the pipeline anymore, so there was nothing left
    to triage. An irrelevant question still ends the graph here, before
    the expensive downstream pipeline (question generation, three-way
    retrieval, review, synthesis) runs at all."""
    messages = state.get('messages') or []
    turn_no = _current_turn_number(messages)

    if state.get('skip_guardrails'):
        # Turn 1 of a case investigation: the input is a fixed, hardcoded
        # directive over a case a human already selected (see
        # src/investigation/session.py::start_investigation), not
        # analyst-authored text that could plausibly be off-topic -- and
        # there's no prior evidence yet either, so this is always a fresh
        # full-hypothesis investigation. Skips the LLM call entirely.
        guardrails_logger.info("SKIPPED (initial case investigation) -> relevant, full_hypothesis")
        return {"is_relevant": True, "investigation_mode": "full_hypothesis"}

    question = state['question']
    alert_context = _render_alert_context(state.get('alert'))
    case_context = state.get('case_context') or "N/A"
    conversation_context = _render_conversation_context(messages)
    prior_evidence_summary = _render_prior_evidence(state)
    guardrails_logger.info(f"invoked -- question={question!r}")
    result = guardrails_chain.invoke({
        "question": question,
        "alert_context": alert_context,
        "case_context": case_context,
        "conversation_context": conversation_context,
        "prior_evidence_summary": prior_evidence_summary,
    })
    guardrails_logger.info(
        f"output: decision={result.decision} investigation_mode={result.investigation_mode} "
        f"reasoning={result.reasoning!r}"
    )

    if result.decision == "irrelevant":
        answer = "Sorry, I can only answer questions related to log analysis and cybersecurity knowledge topic"
        return {
            "is_relevant": False,
            "investigation_mode": result.investigation_mode,
            "answer": answer,
            "messages": [AIMessage(content=answer, id=f"turn-{turn_no}-answer")],
        }

    return {"is_relevant": True, "investigation_mode": result.investigation_mode}


# --- Node Definition: Question Generation ---
def _guaranteed_mitigation_question(state: AgentState) -> dict | None:
    """Deterministic (non-LLM) mitigation-lookup question, appended to
    whatever question_generation produced this turn -- guarantees
    mitigation_suggestions can be grounded in real MITRE ATT&CK mitigation
    data (get_mitigations_for_technique, mcp-cskg-rdf/server.py) instead of
    being left up to chance whether the LLM's free hypothesis set happened
    to include a mitigation-shaped question (it typically hasn't --
    mitigation_suggestions has so far always been free-generated advice,
    never an actual CSKG lookup). None when no MITRE technique is known
    yet (state['known_mitre_techniques'], falling back to the single
    alert's own native tag for src/run.py/eval/) -- nothing to look up
    mitigations for."""
    techniques = state.get('known_mitre_techniques') or (state.get('alert') or {}).get('rule_mitre_id') or []
    if not techniques:
        return None
    ids = ", ".join(sorted(set(techniques)))
    return {
        "hypothesis": "Standard mitigations for the identified technique(s)",
        # Only their IDs are ever known at this point (not names) -- the
        # question says so explicitly and names the exact tool sequence,
        # since get_mitigations_for_technique matches on name text, not ID
        # (a bare ID silently returns nothing -- see mcp_rdf_agent.py's
        # system prompt for the general version of this pointer; spelling
        # it out here too, per-question, is extra reinforcement specifically
        # because this exact question caused that failure once already).
        "question": (
            f"What are the recommended mitigations for MITRE ATT&CK technique(s) {ids}? "
            f"Only the ID(s) are known, not the technique name(s) -- first resolve each ID via "
            f"get_technique_by_id to get its name, then look up mitigations by that name via "
            f"get_mitigations_for_technique (or use text_to_sparql to query by ID directly)."
        ),
        "source": "cskg",
        "rationale": "Guaranteed lookup (not LLM-chosen) so mitigation_suggestions is grounded in "
        "documented MITRE mitigation data instead of free-generated advice.",
    }


def question_generation_node(state: AgentState):
    """Generates several hypothesis-tagged investigative questions from the
    alert/question, each pre-tagged with which retrieval source answers it."""
    mode = state.get('investigation_mode') or "full_hypothesis"
    if mode == "answer_from_context":
        # Guardrails decided the evidence already gathered in a prior turn
        # answers this follow-up -- skip generating (and therefore
        # retrieving/reviewing) anything new. question_results/
        # log_cypher_context/log_vector_context/mcp_rdf_context are left
        # untouched (returning {} here, not overwriting them), so they
        # still hold whatever the last turn that actually retrieved
        # something put there -- that's the "cache" synthesize_node reads.
        question_generation_logger.info("SKIPPED (mode=answer_from_context) -- reusing prior evidence")
        return {}

    alert_summary = state['original_question']
    case_context = state.get('case_context') or "N/A"
    conversation_context = _render_conversation_context(state.get('messages') or [])
    question_generation_logger.info(f"invoked (mode={mode})")
    result = generate_questions(alert_summary, case_context=case_context, conversation_context=conversation_context, mode=mode)
    questions = [q.model_dump() for q in result.questions]
    question_generation_logger.info(
        f"output: {len(questions)} question(s) across "
        f"{len(set(q['hypothesis'] for q in questions))} hypothesis/hypotheses"
    )
    for q in questions:
        question_generation_logger.info(
            f"  - hypothesis={q['hypothesis']!r} question={q['question']!r} source={q['source']}"
        )

    mitigation_q = _guaranteed_mitigation_question(state)
    if mitigation_q:
        questions.append(mitigation_q)
        question_generation_logger.info(f"  - GUARANTEED: question={mitigation_q['question']!r} source=cskg")

    return {"generated_questions": questions}


# --- Node Definition: Dispatch Retrieval ---
async def dispatch_retrieval_node(state: AgentState):
    """Routes each generated question to cypher/vector/cskg per its `source`
    tag (a fixed rule, decided by question_generation_agent's prompt -- not
    a computed EIG/cost estimate), running all questions concurrently."""
    if state.get('investigation_mode') == "answer_from_context":
        # question_generation_node already skipped and returned {} this
        # turn, but `generated_questions` is a non-reducer field -- it
        # would still hold the *previous* turn's questions, not an empty
        # list, if we didn't check the mode explicitly here too. Skipping
        # explicitly (rather than trusting "no questions") is what
        # actually prevents re-dispatching stale questions and keeps
        # log_cypher_context/log_vector_context/mcp_rdf_context/
        # question_results as the cached evidence from whichever turn last
        # populated them.
        dispatch_retrieval_logger.info("SKIPPED (mode=answer_from_context) -- reusing cached evidence")
        return {}

    questions = state.get('generated_questions') or []
    if not questions:
        dispatch_retrieval_logger.info("no questions generated this turn -- nothing to dispatch")
        return {}
    query_timestamp = state.get('query_timestamp')
    turn_no = _current_turn_number(state.get('messages') or [])
    dispatch_retrieval_logger.info(f"invoked -- dispatching {len(questions)} question(s) concurrently")

    async def _retrieve_once(source: str, q_text: str):
        """One retrieval attempt for a single (source, question) pair.
        Raises on failure -- run_one below decides whether to retry."""
        if source == "vector":
            return await asyncio.to_thread(_call_with_timeout, query_vector_search, q_text, query_timestamp)
        elif source == "cypher":
            cypher_result = await asyncio.to_thread(_call_with_timeout, query_cypher, q_text, query_timestamp)
            return cypher_result.get("context", [])
        elif source == "cskg":
            return await asyncio.wait_for(run_mcp_agent(q_text), timeout=NODE_CALL_TIMEOUT_SECONDS)
        else:
            return f"Unknown source tag '{source}' for this question."

    _MAX_RETRIEVAL_ATTEMPTS = 2  # 1 bounded retry, same bound as grounding_check's synthesis retry elsewhere in this graph

    async def run_one(gq: dict) -> dict:
        """Retries only an actual retrieval *failure* (an exception --
        timeout, malformed query, missing index, etc.), never a
        legitimately empty/negative result: query_cypher/
        query_vector_search/run_mcp_agent returning "nothing found" is a
        normal return value, not an exception, so it flows straight
        through untouched -- an empty or negative result is itself
        evidence (rules out a hypothesis), not a failure to recover from,
        per question_generation_agent.py's module docstring. Only a
        genuine exception -- which conflates two different things a
        naive read of "no results" can't tell apart -- gets a second
        attempt, on the chance it was transient (a network blip, a
        momentary timeout) rather than a persistent bug that'll just fail
        the same way again."""
        q_text = gq["question"]
        source = gq["source"]
        context = None
        for attempt in range(1, _MAX_RETRIEVAL_ATTEMPTS + 1):
            try:
                context = await _retrieve_once(source, q_text)
                break
            except Exception as e:
                if attempt < _MAX_RETRIEVAL_ATTEMPTS:
                    dispatch_retrieval_logger.warning(
                        f"({source}) {q_text!r} failed on attempt {attempt}, retrying once: {e}"
                    )
                    continue
                dispatch_retrieval_logger.error(f"({source}) {q_text!r} failed after {attempt} attempt(s): {e}")
                context = f"Error retrieving for this question: {e}"
        dispatch_retrieval_logger.info(f"output: ({source}) {q_text!r} -> {str(context)[:300]!r}")
        return {**gq, "context": context}

    results = list(await asyncio.gather(*(run_one(gq) for gq in questions)))
    for r in results:
        r["turn"] = turn_no  # tag before accumulating -- _window_question_results/review_evidence_node key on this

    # Accumulate onto prior turns' evidence rather than replacing it (see
    # EVIDENCE_WINDOW_TURNS' docstring above for why this matters), then
    # window. log_cypher_context/log_vector_context/mcp_rdf_context/
    # generated_question_for_rdf are DERIVED fresh from the windowed list
    # every time, rather than separately accumulated in parallel -- keeps
    # question_results as the single source of truth so these can't drift
    # out of sync with it.
    accumulated = (state.get('question_results') or []) + results
    windowed = _window_question_results(accumulated, turn_no)

    cypher_ctx = [r["context"] for r in windowed if r["source"] == "cypher" and r["context"]]
    vector_ctx = "\n".join(str(r["context"]) for r in windowed if r["source"] == "vector" and r["context"])
    cskg_ctx = "\n".join(str(r["context"]) for r in windowed if r["source"] == "cskg" and r["context"])
    cskg_questions = "; ".join(r["question"] for r in windowed if r["source"] == "cskg")

    dispatch_retrieval_logger.info(
        f"accumulated evidence: {len(windowed)} question(s) across "
        f"{len(set(r['turn'] for r in windowed))} turn(s) (window={EVIDENCE_WINDOW_TURNS} turns)"
    )

    return {
        "question_results": windowed,
        "log_cypher_context": cypher_ctx or None,
        "log_vector_context": vector_ctx or None,
        "mcp_rdf_context": cskg_ctx or None,
        "generated_question_for_rdf": cskg_questions or "Not applicable for this query.",
    }


# --- Node Definition: Review Evidence ---
def review_evidence_node(state: AgentState):
    """Reviews each question's retrieved evidence against the hypothesis it
    was meant to discriminate (review_agent.py). No retry branch: results
    (including ruled-out hypotheses) flow straight to the synthesizer."""
    if state.get('investigation_mode') == "answer_from_context":
        # dispatch_retrieval_node already skipped this turn -- nothing new
        # to review. question_results still holds whichever prior turn's
        # (already-reviewed) results, reused as-is by the synthesizer.
        review_evidence_logger.info("SKIPPED (mode=answer_from_context) -- nothing new to review")
        return {}
    results = state.get('question_results') or []
    # question_results now accumulates across turns (dispatch_retrieval_node,
    # EVIDENCE_WINDOW_TURNS) -- most entries here on turn 2+ already have a
    # "review" from whenever they were first retrieved. Re-reviewing them
    # again every turn would be wasted LLM calls at best and risk a verdict
    # silently flipping between turns at worst; only entries missing a
    # review (this turn's freshly-dispatched questions) get one.
    already_reviewed = sum(1 for r in results if r.get("review"))
    review_evidence_logger.info(
        f"invoked -- {len(results) - already_reviewed} new question(s) to review, "
        f"{already_reviewed} already reviewed in a prior turn reused as-is"
    )
    reviewed = []
    for r in results:
        if r.get("review"):
            reviewed.append(r)
            continue
        review = review_chain.invoke({
            "hypothesis": r.get("hypothesis", ""),
            "question": r.get("question", ""),
            "context": str(r.get("context", "")),
        })
        review_evidence_logger.info(
            f"output: {r.get('question')!r} -> {review.hypothesis_verdict} ({review.reasoning!r})"
        )
        reviewed.append({**r, "review": review.model_dump()})
    return {"question_results": reviewed}


def _render_hypothesis_summary(question_results: list) -> str:
    """Deterministic (non-LLM) rollup of generated questions + review
    verdicts, replacing log_analysis_agent.py's old free-text summary --
    see the module docstring above for why this is one fewer LLM call, not
    a moved one. question_results can now span multiple turns
    (EVIDENCE_WINDOW_TURNS), so each line is tagged with which turn it's
    from -- without that, older and newer findings would be indistinguishable
    once accumulated together."""
    if not question_results:
        return "Not applicable for this query."
    lines = []
    for r in question_results:
        review = r.get("review", {})
        turn_tag = f"Turn {r['turn']} | " if r.get("turn") is not None else ""
        lines.append(
            f"- {turn_tag}Hypothesis: {r.get('hypothesis', '?')} | Question: {r.get('question', '?')} | "
            f"Source: {r.get('source', '?')} | Verdict: {review.get('hypothesis_verdict', '?')} "
            f"({review.get('reasoning', '')})"
        )
    return "\n".join(lines)


_NOT_APPLICABLE = "Not applicable for this query."
_NO_MCP_DATA = "No data was provided from this source."
_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}
_MAX_CONFIDENCE_BY_RICHNESS = {0: "low", 1: "medium", 2: "high", 3: "high"}

# run_mcp_agent() (src/agents/mcp_rdf_agent.py) wraps a full LLM agent, not
# a raw tool passthrough like query_cypher/query_vector_search -- its
# output is always natural-language prose, whether it found something or
# not, so an exact-string check like _NO_MCP_DATA (which only ever matches
# the *default* set when no cskg question was asked at all -- see below)
# can't tell those two cases apart. NO_DATA_MARKER (mcp_rdf_agent.py) is
# the reliable signal for that; this phrase list is a defense-in-depth
# fallback for whenever the model doesn't prefix its answer with the
# marker exactly as instructed. Deliberately conservative (only common,
# unambiguous "found nothing" phrasings) -- a false negative here just
# means richness undercounts by one, which only ever pushes confidence
# *down*, never invents false certainty; a false positive would do the
# opposite, which is the actual failure mode being fixed.
_MCP_NO_DATA_PHRASES = (
    "couldn't find", "could not find", "no relevant information",
    "no known", "does not match", "doesn't match", "unable to find",
    "i don't have access", "i currently don't have access",
    "no results found", "no data found",
    # Added after a real miss: a question asking to correlate a specific IP
    # against threat intel (something this KB structurally has no data for
    # at all -- see mcp_rdf_agent.py's scope note) got answered with a
    # capability disclaimer + an offer to help with a *different*,
    # reformulated question instead -- not caught by NO_DATA_MARKER (the
    # model didn't use it) or any phrase above.
    "unable to directly analyze", "cannot directly analyze",
    "unable to directly access", "outside the scope of this knowledge base",
    "not something this knowledge base",
)


def _mcp_context_has_data(mcp_str: str) -> bool:
    if not mcp_str or mcp_str in (_NO_MCP_DATA, _NOT_APPLICABLE):
        return False
    if mcp_str.startswith("Error") or mcp_str.startswith(_MCP_NO_DATA_MARKER):
        return False
    lowered = mcp_str.lower()
    return not any(phrase in lowered for phrase in _MCP_NO_DATA_PHRASES)


def _context_richness(log_cypher: str, log_vector: str, mcp_rdf_context) -> int:
    """Counts how many of the 3 possible evidence sources actually returned
    substantive content, vs. the LLM's own unconstrained self-assessment
    (which eval showed always says 'high' regardless of whether
    the answer was actually correct -- confidence wasn't discriminating at
    all). This is the deterministic signal `confidence` gets capped against
    below, so stated confidence tracks actual evidence availability instead
    of vibes."""
    sources_with_data = 0
    if log_cypher and log_cypher not in (_NOT_APPLICABLE, "[]", "None"):
        sources_with_data += 1
    if log_vector and log_vector != _NOT_APPLICABLE:
        sources_with_data += 1
    if _mcp_context_has_data(str(mcp_rdf_context) if mcp_rdf_context else ""):
        sources_with_data += 1
    return sources_with_data


def _cap_confidence(report, richness: int):
    """Caps report.confidence at the max level `richness` evidence sources
    can support -- 0 sources -> at most 'low', 1 -> at most 'medium', 2+ ->
    'high' allowed. Only ever lowers confidence, never raises it (the LLM's
    own judgment can still say something is less confident than the cap)."""
    allowed_max = _MAX_CONFIDENCE_BY_RICHNESS[richness]
    if _CONFIDENCE_RANK[report.confidence] > _CONFIDENCE_RANK[allowed_max]:
        synthesizer_logger.info(
            f"capping confidence {report.confidence} -> {allowed_max} "
            f"({richness} of 3 evidence sources had substantive content)"
        )
        return report.model_copy(update={"confidence": allowed_max})
    return report


# --- Node Definition: Synthesizer ---
def synthesize_node(state: AgentState):
    """Generates the final compiled report for the user (structured output,
    see src/agents/synthesizer_agent.py::SynthesizedReport)."""
    log_cypher = str(state.get('log_cypher_context')) if state.get('log_cypher_context') else "Not applicable for this query."
    log_vector = str(state.get('log_vector_context')) if state.get('log_vector_context') else "Not applicable for this query."
    generated_q = str(state.get('generated_question_for_rdf', "Not applicable for this query."))
    hypothesis_summary = _render_hypothesis_summary(state.get('question_results') or [])
    messages = state.get('messages') or []
    turn_no = _current_turn_number(messages)
    case_context = state.get('case_context') or "N/A"
    conversation_context = _render_conversation_context(messages)
    synthesizer_logger.info("invoked")

    if not state.get('mcp_rdf_context') and log_cypher == "Not applicable for this query." and log_vector == "Not applicable for this query.":
        synthesizer_logger.info("output: no evidence from any source -- returning fallback 'not found' report")
        fallback = SynthesizedReport(
            original_question=state['original_question'],
            cypher_context_summary="Not applicable for this query.",
            vector_context_summary="Not applicable for this query.",
            generated_question_for_rdf="Not applicable for this query.",
            cskg_context_summary="Not applicable for this query.",
            critical_analysis="No data was retrieved from any source.",
            contextual_linkage="Not applicable for this query.",
            final_answer="Sorry, no relevant information could be found for this alert.",
            cited_entities=[],
            mitre_techniques=[],
            confidence="low",
            recommended_priority="monitor",
            mitigation_suggestions=[],
        )
        return {
            "answer": fallback.final_answer,
            "synthesized_report": fallback.model_dump(),
            "messages": [AIMessage(content=fallback.final_answer, id=f"turn-{turn_no}-answer")],
        }

    # A previous grounding_check pass (see grounding_check_node /
    # decide_after_grounding below) may have rejected this report for citing
    # entities not present in context. Bounded to a single retry.
    is_retry = bool(state.get('grounding_result')) and not state['grounding_result'].get('grounded', True)
    retry_count = state.get('grounding_retry_count', 0) + (1 if is_retry else 0)
    retry_note = ""
    if is_retry:
        ungrounded = state['grounding_result'].get('ungrounded_entities', [])
        retry_note = (
            f"\n**IMPORTANT CORRECTION**: A previous draft cited entities not present in the "
            f"context above: {ungrounded}. Do not repeat that mistake -- only cite entities that "
            f"literally appear in the context above."
        )

    report = synthesis_chain.invoke({
        "original_question": state['original_question'],
        "hypothesis_summary": hypothesis_summary,
        "log_cypher_context": log_cypher,
        "log_vector_context": log_vector,
        "generated_question_for_rdf": generated_q,
        "mcp_rdf_context": str(state.get('mcp_rdf_context', "No data was provided from this source.")),
        "retry_note": retry_note,
        "case_context": case_context,
        "conversation_context": conversation_context,
    })

    richness = _context_richness(log_cypher, log_vector, state.get('mcp_rdf_context'))
    report = _cap_confidence(report, richness)

    synthesizer_logger.info(
        f"output: final_answer={report.final_answer!r} confidence={report.confidence} "
        f"priority={report.recommended_priority} mitre_techniques={report.mitre_techniques} "
        f"cited_entities={report.cited_entities} mitigation_suggestions={report.mitigation_suggestions}"
    )

    # Grows the set _guaranteed_mitigation_question (question_generation_node)
    # draws on for later turns -- a technique newly confirmed THIS turn
    # gets a guaranteed mitigation lookup on the NEXT turn that needs new
    # retrieval, even if it wasn't already tagged on the case/alert.
    known_techniques = sorted(set(state.get('known_mitre_techniques') or []) | set(report.mitre_techniques))

    return {
        "answer": report.final_answer,
        "synthesized_report": report.model_dump(),
        "grounding_retry_count": retry_count,
        "known_mitre_techniques": known_techniques,
        # Same id every time this node runs within one turn (including a
        # bounded grounding-check retry) -- add_messages replaces the prior
        # message sharing an id instead of appending a duplicate, so the
        # transcript only ever shows this turn's final, corrected answer.
        "messages": [AIMessage(content=report.final_answer, id=f"turn-{turn_no}-answer")],
    }


# --- Node Definition: Grounding Check ---
def grounding_check_node(state: AgentState):
    """Verifies the synthesizer's cited entities actually appear in the
    retrieved context. Non-LLM, cheap."""
    report = state.get('synthesized_report')
    if not report:
        grounding_check_logger.info("no report to check -- treating as grounded")
        return {"grounding_result": {"grounded": True, "checked_entities": [], "ungrounded_entities": []}}

    context_blobs = [
        str(state.get('log_cypher_context') or ''),
        str(state.get('log_vector_context') or ''),
        str(state.get('mcp_rdf_context') or ''),
        # case_context (src/investigation/narrative.py's static fact sheet:
        # hosts, IPs, users, MITRE tags, member-alert narratives) has been a
        # real input to synthesize_node's prompt since case-based
        # investigation was added, but was never added here -- so a
        # citation the synthesizer correctly grounded in the case's own
        # alert data (e.g. its source IP or host, stated directly in
        # case_context) was being flagged as a false-positive hallucination
        # and burning a retry for no reason. Deliberately NOT adding
        # conversation_context or hypothesis_summary here too: both contain
        # LLM-authored text (prior answers, review reasoning), and treating
        # those as grounding sources would let an entity that slipped
        # through once keep re-"grounding" itself in every later check --
        # case_context is safe specifically because it's built
        # deterministically from real ingested alert data, not from
        # another LLM's output.
        state.get('case_context') or '',
    ]
    result = check_grounding(report, context_blobs)
    if result.grounded:
        grounding_check_logger.info(f"output: grounded=True checked_entities={result.checked_entities}")
    else:
        grounding_check_logger.warning(
            f"output: grounded=False ungrounded_entities={result.ungrounded_entities} "
            f"checked_entities={result.checked_entities}"
        )
    return {"grounding_result": result.model_dump()}


# --- Perakitan Graph ---
workflow = StateGraph(AgentState)

workflow.add_node("guardrails", guardrails_node)
workflow.add_node("question_generation", question_generation_node)
workflow.add_node("dispatch_retrieval", dispatch_retrieval_node)
workflow.add_node("review_evidence", review_evidence_node)
workflow.add_node("synthesizer", synthesize_node)
workflow.add_node("grounding_check", grounding_check_node)


# 1. Decision after Guardrails
def decide_after_guardrails(state: AgentState):
    if not state.get('is_relevant', False):
        guardrails_logger.info("routing: irrelevant -> ending execution")
        return END
    return "question_generation"


# 2. Decision after Grounding Check
def decide_after_grounding(state: AgentState):
    result = state.get('grounding_result') or {}
    if result.get('grounded', True):
        grounding_check_logger.info("routing: grounded -> ending execution")
        return END
    if state.get('grounding_retry_count', 0) >= 1:
        grounding_check_logger.warning("routing: still ungrounded after retry -> surfacing as-is (bounded retry)")
        return END
    grounding_check_logger.warning("routing: ungrounded -> retrying synthesis once")
    return "synthesizer"


# --- Define Edges ---
workflow.set_entry_point("guardrails")

workflow.add_conditional_edges(
    "guardrails",
    decide_after_guardrails,
    {
        "question_generation": "question_generation",
        END: END
    }
)

workflow.add_edge("question_generation", "dispatch_retrieval")
workflow.add_edge("dispatch_retrieval", "review_evidence")
workflow.add_edge("review_evidence", "synthesizer")
workflow.add_edge("synthesizer", "grounding_check")

workflow.add_conditional_edges(
    "grounding_check",
    decide_after_grounding,
    {
        "synthesizer": "synthesizer",
        END: END
    }
)

# Compile graph
app = workflow.compile()
