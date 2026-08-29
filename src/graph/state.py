# src/graph/state.py
from typing import List, Optional
from typing_extensions import TypedDict, Annotated
from langgraph.graph import add_messages

class AgentState(TypedDict):
    question: str
    original_question: str
    is_relevant: bool

    # raw Alert fields (see src/ingestion/schema.py), set only by
    # eval/run_eval.py's per-alert scoring calls now that nothing
    # auto-triggers the live pipeline -- see guardrails_agent.py's
    # _render_alert_context. None everywhere else.
    alert: Optional[dict]

    # question_generation_agent's QuestionSet, dumped to dicts:
    # [{hypothesis, question, source, rationale}, ...]
    generated_questions: Optional[List[dict]]
    # dispatch_retrieval_node's per-question results, enriched by
    # review_evidence_node with a "review" verdict:
    # [{hypothesis, question, source, rationale, context, review}, ...]
    question_results: Optional[List[dict]]

    log_vector_context: Optional[str]
    log_cypher_context: Optional[List[dict]]

    generated_question_for_rdf: Optional[str]
    mcp_rdf_context: Optional[str]

    answer: Optional[str]
    cypher_query: Optional[str]
    error: Optional[str]
    messages: Annotated[list, add_messages]

    # structured synthesizer output + grounding check
    synthesized_report: Optional[dict]
    grounding_result: Optional[dict]
    grounding_retry_count: int

    # temporal weighting: the timestamp retrieval should be weighted against
    # (the triggering alert's timestamp for auto-triggered queries, "now"
    # for manual analyst questions)
    query_timestamp: Optional[str]

    # Static fact sheet for a clustered activity case under investigation
    # (see src/investigation/narrative.py::build_case_context) -- hosts,
    # IPs, users, time span, and member-alert narratives. Set once on the
    # first turn of an investigation thread; not a reducer, so it persists
    # unchanged across later checkpointed turns (src/investigation/
    # session.py's follow-up calls don't resend it). None outside the
    # investigation flow (src/run.py, eval/).
    case_context: Optional[str]

    # Set True by src/investigation/session.py::start_investigation() for
    # turn 1 of a case (the input is a fixed, hardcoded directive over a
    # case a human already selected, not analyst-authored text needing a
    # relevance check) -- guardrails_node skips its LLM call entirely
    # rather than spending a call confirming something already obviously
    # true by construction. Always False/absent for src/run.py, eval/, and
    # every follow-up turn.
    skip_guardrails: Optional[bool]

    # guardrails_agent.py's routing decision for how much investigation
    # this turn needs -- see GuardrailsOutput.investigation_mode.
    # "full_hypothesis": generate several hypothesis-tagged questions (the
    #   default -- always used for turn 1 and for src/run.py/eval/, which
    #   have no prior-turn evidence to reuse).
    # "direct_question": a narrow follow-up that still needs new
    #   information, but not a full multi-hypothesis investigation.
    # "answer_from_context": the evidence already gathered in a prior turn
    #   (still sitting in question_results/log_cypher_context/etc. via the
    #   checkpointer, since nothing overwrites them this turn) already
    #   answers this follow-up -- question_generation/dispatch_retrieval/
    #   review_evidence all skip themselves, and synthesizer answers
    #   directly from the cached evidence, with no new retrieval calls.
    investigation_mode: Optional[str]

    # MITRE ATT&CK technique IDs identified so far in this investigation --
    # seeded from the case's own native+sigma tags on turn 1 (src/
    # investigation/session.py::start_investigation, from Case.
    # mitre_techniques) or the single alert's native tag for src/run.py/
    # eval/ (question_generation_node falls back to state['alert']
    # ['rule_mitre_id'] when this isn't set), then grown by synthesize_node
    # with whatever SynthesizedReport.mitre_techniques each turn adds.
    # question_generation_node uses this to GUARANTEE a cskg question
    # asking get_mitigations_for_technique for these specific IDs whenever
    # there's at least one, rather than leaving whether
    # mitigation_suggestions is actually grounded in real MITRE mitigation
    # data up to chance (whether the LLM's free hypothesis set happened to
    # include a mitigation-shaped question). Persists across checkpointed
    # turns like case_context; not touched during 'answer_from_context'
    # turns, which skip question_generation entirely and reuse whatever's
    # already cached.
    known_mitre_techniques: Optional[List[str]]
