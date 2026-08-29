# src/investigation/session.py
"""
Runs the *same* multi-agent pipeline as src/run.py/eval/, but as a
checkpointed, multi-turn investigation thread instead of a single-shot
call -- the "initial investigation, then chatroom" half of the split
described in src/investigation/__init__.py's package (see PROTOTYPE.md).

Compiles the graph a *second* time with a checkpointer
(`workflow.compile(checkpointer=MemorySaver())`), importing the uncompiled
`workflow` StateGraph builder from src.graph.workflow rather than its
already-compiled `app`. This is deliberate: src/run.py and eval/run_eval.py
keep calling the original uncheckpointed `app` exactly as before (no
`thread_id` needed there), so adding checkpointing here is purely additive
and can't break either of them -- LangGraph raises if you invoke a
checkpointed graph without a thread_id, which would otherwise break every
existing call site.

State persistence is in-process only (MemorySaver) -- lost on restart, by
design for a prototype: this is a single-process FastAPI service (see
api.py), not a durable multi-instance backend. Each turn is additionally
appended to data/investigations.jsonl as a persistent audit trail, mirroring
the pattern src/ingestion/trigger.py used for auto_trigger_log.jsonl.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from src.graph.workflow import workflow
from src.ingestion.schema import Alert
from src.investigation.clustering import Case
from src.investigation.narrative import build_case_context
from src.retrieval.temporal import now_iso

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
LOG_PATH = DATA_DIR / "investigations.jsonl"

INITIAL_DIRECTIVE = (
    "Investigate the activity described in this case. Identify related MITRE ATT&CK "
    "techniques, provide an initial diagnosis of the situation, and recommend mitigation actions."
)

checkpointed_app = workflow.compile(checkpointer=MemorySaver())

# In-memory only -- see module docstring. Keyed by case_id (== LangGraph thread_id).
_transcripts: Dict[str, List["TurnRecord"]] = {}
_case_by_id: Dict[str, Case] = {}


class TurnRecord(BaseModel):
    case_id: str
    turn_index: int
    question: str
    answer: Optional[str] = None
    critical_analysis: Optional[str] = None
    mitigation_suggestions: List[str] = Field(default_factory=list)
    recommended_priority: Optional[str] = None
    confidence: Optional[str] = None
    mitre_techniques: List[str] = Field(default_factory=list)
    cited_entities: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    timestamp: str
    latency_seconds: float = 0.0


def _append_log(record: TurnRecord) -> None:
    LOG_PATH.parent.mkdir(exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(record.model_dump_json() + "\n")


async def _run_turn(case_id: str, question: str, extra_state: dict) -> TurnRecord:
    """Invokes checkpointed_app on the given thread. `grounding_retry_count`
    is always reset to 0 here even though it isn't a reducer field: without
    this, a turn that ended after a grounding-check retry (count=1) would
    leave that count sitting in the checkpoint, and the *next* turn's fresh
    synthesis would see grounding_retry_count>=1 and wrongly skip its own
    legitimate one-shot retry (see src/graph/workflow.py::decide_after_grounding)."""
    state = {
        "question": question,
        "original_question": question,
        "messages": [("human", question)],
        "grounding_retry_count": 0,
        **extra_state,
    }
    config = {"configurable": {"thread_id": case_id}, "recursion_limit": 30}
    start = time.monotonic()
    try:
        result = await checkpointed_app.ainvoke(state, config=config)
    except Exception as e:
        logger.error(f"[investigation] pipeline failed for case {case_id}: {e}", exc_info=True)
        result = {"answer": None, "error": str(e)}
    latency = time.monotonic() - start

    report = result.get("synthesized_report") or {}
    turn_index = len(_transcripts.get(case_id, [])) + 1
    record = TurnRecord(
        case_id=case_id,
        turn_index=turn_index,
        question=question,
        answer=result.get("answer"),
        critical_analysis=report.get("critical_analysis"),
        mitigation_suggestions=report.get("mitigation_suggestions") or [],
        recommended_priority=report.get("recommended_priority"),
        confidence=report.get("confidence"),
        mitre_techniques=report.get("mitre_techniques") or [],
        cited_entities=report.get("cited_entities") or [],
        error=result.get("error"),
        timestamp=now_iso(),
        latency_seconds=round(latency, 2),
    )
    logger.info(
        f"[investigation] case={case_id} turn={turn_index} latency={record.latency_seconds}s "
        f"mode={result.get('investigation_mode')} error={record.error}"
    )
    _transcripts.setdefault(case_id, []).append(record)
    _append_log(record)
    return record


async def start_investigation(case: Case, alerts: List[Alert]) -> TurnRecord:
    """Idempotent: a case that's already been opened returns its existing
    first turn instead of re-running the pipeline (and spending another
    round of LLM calls) -- use ask_followup for continued Q&A on an
    already-started case."""
    if case.case_id in _transcripts:
        return _transcripts[case.case_id][0]
    _case_by_id[case.case_id] = case
    case_context = build_case_context(case, alerts)
    return await _run_turn(case.case_id, INITIAL_DIRECTIVE, {
        "case_context": case_context,
        # Seeds question_generation_node's guaranteed mitigation-lookup
        # question (src/graph/state.py) with the case's own native+sigma
        # MITRE tags -- synthesize_node grows this set turn over turn with
        # anything newly identified.
        "known_mitre_techniques": case.mitre_techniques,
        # Anchors temporal retrieval weighting to when the case's activity
        # actually happened, not to wall-clock "now" -- same convention
        # src/ingestion/trigger.py used (alert.timestamp) for auto-triggered
        # investigations.
        "query_timestamp": case.last_seen,
        # Turn 1's input is a fixed directive over an already-human-selected
        # case, not a question needing triage -- guardrails_node skips its
        # LLM call entirely (see src/graph/workflow.py) rather than
        # spending a call to answer something already settled by
        # construction.
        "skip_guardrails": True,
    })


async def ask_followup(case_id: str, message: str) -> TurnRecord:
    if case_id not in _transcripts:
        raise ValueError(f"No investigation started for case {case_id} yet -- call start_investigation first.")
    case = _case_by_id.get(case_id)
    query_timestamp = case.last_seen if case else now_iso()
    # case_context is deliberately NOT resent: it's a non-reducer state
    # field, so LangGraph's checkpointer keeps turn 1's value automatically
    # when a later turn's partial state update omits it. skip_guardrails,
    # by contrast, MUST be explicitly reset to False here for the same
    # reason `_run_turn` always resets grounding_retry_count -- turn 1 set
    # it True, and being a non-reducer field it would otherwise stay True
    # on every later turn too, silently skipping guardrails' investigation_
    # mode routing (the whole point of this method) forever after turn 1.
    return await _run_turn(case_id, message, {"query_timestamp": query_timestamp, "skip_guardrails": False})


def get_transcript(case_id: str) -> List[TurnRecord]:
    return _transcripts.get(case_id, [])
