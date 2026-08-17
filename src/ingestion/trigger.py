from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import AsyncIterator, Optional

from src.graph.workflow import app
from src.ingestion.graph_loader import ingest_alert
from src.ingestion.schema import Alert

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
TRIGGER_SEVERITY_THRESHOLD = int(os.environ.get("TRIGGER_SEVERITY_THRESHOLD", "7"))


def build_question(alert: Alert) -> str:
    """Templates a natural-language analyst question from an alert's
    fields -- the same kind of question a human would type into
    `uv run -m src.run`, just generated instead of typed."""
    parts = [f"A security alert was raised on host '{alert.agent_name}': {alert.rule_description}"]
    if alert.data_srcip:
        parts.append(f"source IP {alert.data_srcip}")
    if alert.data_dstuser:
        parts.append(f"targeting user '{alert.data_dstuser}'")
    parts.append(
        "Investigate what happened, identify any related MITRE ATT&CK techniques, "
        "and recommend a response"
    )
    return ". ".join(parts) + "."


def _trigger_reason(alert: Alert) -> Optional[str]:
    """Two independent trigger signals, OR'd together. Verified against the
    current AIT-ADS sample (eval/analyze_trigger_threshold.py) before adding
    this: rule_level>=TRIGGER_SEVERITY_THRESHOLD alone catches almost nothing
    on real data (recall 0.039 at the level-7 default -- Wazuh's default
    ruleset scores severity by rule *type*, with no awareness of whether an
    event is part of an active attack). A native MITRE ATT&CK tag
    (rule.mitre.id, present on ~36% of AIT-ADS alerts) is a materially
    different, complementary signal -- combining rule_level>=5 OR
    has_mitre_tag raised recall from 0.609 to 0.750 and F1 from 0.712 to
    0.747 over rule_level alone in that same check, so this isn't
    redundant with severity, it's catching genuinely different alerts."""
    if alert.rule_level >= TRIGGER_SEVERITY_THRESHOLD:
        return "severity"
    if alert.rule_mitre_id:
        return "mitre_tag"
    return None


async def handle_alert(alert: Alert, log_path: Path) -> Optional[dict]:
    """Ingest the alert; if it's severe enough OR carries a native MITRE
    ATT&CK tag, also run it through the full AgCyRAG pipeline and append the
    result to `log_path`. Returns the result record, or None if the alert
    was only ingested (triggered neither condition)."""
    ingest_alert(alert)

    trigger_reason = _trigger_reason(alert)
    if trigger_reason is None:
        return None

    question = build_question(alert)
    initial_state = {
        "question": question,
        "original_question": question,
        "messages": [("human", question)],
        "cypher_iteration_count": 1,
        "vector_iteration_count": 1,
        "max_iterations": 3,
        "query_timestamp": alert.timestamp,
    }

    start = time.monotonic()
    try:
        result = await app.ainvoke(initial_state, config={"recursion_limit": 30})
    except Exception as e:
        logger.error(f"[trigger] AgCyRAG pipeline failed for alert {alert.id}: {e}", exc_info=True)
        result = {"answer": None, "error": str(e)}
    latency = time.monotonic() - start

    grounding = result.get("grounding_result")
    record = {
        "alert_id": alert.id,
        "alert_rule_level": alert.rule_level,
        "trigger_reason": trigger_reason,
        "scenario_id": alert.scenario_id,
        "question": question,
        "answer": result.get("answer"),
        "error": result.get("error"),
        "grounded": grounding.get("grounded") if isinstance(grounding, dict) else None,
        "latency_seconds": round(latency, 2),
    }
    log_path.parent.mkdir(exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return record


async def run(stream: AsyncIterator[Alert], log_path: Optional[Path] = None) -> dict:
    log_path = log_path or (DATA_DIR / "auto_trigger_log.jsonl")
    n_ingested = n_triggered = 0
    n_by_reason = {"severity": 0, "mitre_tag": 0}
    async for alert in stream:
        n_ingested += 1
        record = await handle_alert(alert, log_path)
        if record is not None:
            n_triggered += 1
            n_by_reason[record["trigger_reason"]] += 1
            print(f"[TRIGGERED] {alert.id} (level={alert.rule_level}, reason={record['trigger_reason']}) -> {log_path}")
        else:
            print(f"[ingested]  {alert.id} (level={alert.rule_level}, below threshold {TRIGGER_SEVERITY_THRESHOLD}, no MITRE tag)")
    summary = {
        "ingested": n_ingested, "triggered": n_triggered, "threshold": TRIGGER_SEVERITY_THRESHOLD,
        "triggered_by_severity": n_by_reason["severity"], "triggered_by_mitre_tag": n_by_reason["mitre_tag"],
    }
    print(f"Done: {n_ingested} alerts ingested, {n_triggered} triggered the multi-agent pipeline "
          f"(threshold={TRIGGER_SEVERITY_THRESHOLD}, {n_by_reason['severity']} by severity, "
          f"{n_by_reason['mitre_tag']} by MITRE tag alone).")
    return summary
