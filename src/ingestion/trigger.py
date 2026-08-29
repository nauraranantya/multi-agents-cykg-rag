from __future__ import annotations

import asyncio
import os
from typing import AsyncIterator, Optional

from src.ingestion.graph_loader import ensure_indices, ingest_alert
from src.ingestion.schema import Alert

TRIGGER_SEVERITY_THRESHOLD = int(os.environ.get("TRIGGER_SEVERITY_THRESHOLD", "7"))
# Ingestion no longer invokes the pipeline at all -- see src/investigation/
# for the human-invoked replacement. This module now only ingests, plus
# _trigger_reason() below, which is kept as a standalone, reusable "is this
# alert noteworthy" signal: eval/run_eval.py, eval/evaluate_alert_triage.py
# and eval/analyze_trigger_threshold.py all import it directly for
# comparison purposes, and src/investigation/clustering.py now reuses it
# too, as one input to ranking activity cases by urgency -- the same
# empirically-validated signal, just serving ranking instead of an
# auto-escalation gate now that escalation always goes through a human.


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


async def ingest_stream(stream: AsyncIterator[Alert]) -> dict:
    """Ingests the stream continuously -- and only ingests. No investigation
    is triggered here anymore (see src/investigation/ for the human-invoked
    replacement); this loop's only job is getting every alert into the
    graph as fast as the stream yields it. `_trigger_reason` is still
    computed per alert purely for an informational print (how much of this
    stream would have been auto-escalated under the old gate) -- it no
    longer causes any pipeline invocation."""
    # ingest_alert() itself never creates the vector/keyword/entities
    # indices vector_agent.py's retrieval depends on -- only
    # graph_loader.py::load_all() (the offline batch path) called
    # ensure_indices() before this fix, so every vector-source question
    # failed with "There is no such fulltext schema index: entities" for
    # anyone who ingested via the live stream instead. ensure_indices() is
    # documented idempotent, so calling it once up front here is safe.
    ensure_indices()
    counters = {"ingested": 0, "noteworthy": 0, "by_reason": {"severity": 0, "mitre_tag": 0}}

    async for alert in stream:
        counters["ingested"] += 1
        await asyncio.to_thread(ingest_alert, alert)  # never blocks the event loop, even briefly

        trigger_reason = _trigger_reason(alert)
        if trigger_reason is None:
            print(f"[ingested]  {alert.id} (level={alert.rule_level}, below threshold {TRIGGER_SEVERITY_THRESHOLD}, no MITRE tag)")
            continue

        counters["noteworthy"] += 1
        counters["by_reason"][trigger_reason] += 1
        print(f"[ingested]  {alert.id} (level={alert.rule_level}, reason={trigger_reason}, noteworthy)")

    summary = {
        "ingested": counters["ingested"], "noteworthy": counters["noteworthy"],
        "threshold": TRIGGER_SEVERITY_THRESHOLD,
        "noteworthy_by_severity": counters["by_reason"]["severity"],
        "noteworthy_by_mitre_tag": counters["by_reason"]["mitre_tag"],
    }
    print(
        f"Done: {counters['ingested']} alerts ingested, {counters['noteworthy']} noteworthy "
        f"({counters['by_reason']['severity']} by severity, {counters['by_reason']['mitre_tag']} by MITRE tag alone, "
        f"threshold={TRIGGER_SEVERITY_THRESHOLD}). Run `uv run -m src.investigation.run_api` and "
        f"GET /cases to review clustered activity for investigation."
    )
    return summary
