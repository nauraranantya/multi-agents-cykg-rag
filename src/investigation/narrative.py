# src/investigation/narrative.py
"""
Renders a Case (src/investigation/clustering.py) + its member alerts into
the static text fact sheet fed into the pipeline as `state['case_context']`
(see src/graph/state.py, src/graph/workflow.py's `_render_conversation_context`
sibling). This is the case-level analogue of the old
src/ingestion/trigger.py::build_question -- but deliberately a fact sheet,
not a question: the actual investigative ask (turn 1's directive, or a
later chat follow-up) is passed separately as `question`/`original_question`
by src/investigation/session.py, so case_context can stay constant across
every turn of one investigation thread while the question changes turn to
turn.
"""
from __future__ import annotations

from typing import List

from src.ingestion.schema import Alert
from src.investigation.attack_graph import build_attack_graph
from src.investigation.clustering import Case

MAX_ALERTS_IN_NARRATIVE = 15


def _alert_line(alert: Alert) -> str:
    bits = [f"[{alert.timestamp}] {alert.agent_name} (level {alert.rule_level}): {alert.rule_description}"]
    if alert.data_srcip:
        bits.append(f"source IP {alert.data_srcip}")
    if alert.data_dstuser:
        bits.append(f"user {alert.data_dstuser}")
    if alert.rule_mitre_id:
        bits.append("MITRE (native): " + ", ".join(alert.rule_mitre_id))
    if alert.sigma_mitre_id:
        bits.append("MITRE (sigma-matched): " + ", ".join(alert.sigma_mitre_id))
    if alert.full_log:
        bits.append(f"log: {alert.full_log[:200]}")
    return " | ".join(bits)


def build_case_context(case: Case, alerts: List[Alert]) -> str:
    """`alerts` should be the case's exact member alerts (see
    clustering.py::fetch_alerts_by_ids(case.alert_ids)). Member alerts are
    capped at MAX_ALERTS_IN_NARRATIVE and ordered by the reconstructed
    attack chain (src/investigation/attack_graph.py) when one alert clearly
    precedes the rest, falling back to severity ranking otherwise -- so a
    large cluster doesn't blow up context length, and the ordering itself
    tells a story rather than just being a severity-sorted dump."""
    lines = [
        f"Case {case.case_id}: {case.alert_count} alert(s) across host(s) {', '.join(case.hosts)} "
        f"between {case.first_seen} and {case.last_seen}.",
        f"Max severity (Wazuh rule_level, 0-15): {case.max_rule_level}. "
        f"Urgency score: {case.urgency_score}/100.",
    ]
    if case.src_ips:
        lines.append(f"Source IP(s) involved: {', '.join(case.src_ips)}")
    if case.dst_users:
        lines.append(f"Target user(s) involved: {', '.join(case.dst_users)}")
    if case.mitre_techniques:
        lines.append(f"MITRE ATT&CK technique(s) across member alerts (native + sigma-matched): "
                      f"{', '.join(case.mitre_techniques)}")
    if case.sigma_matched_rules:
        lines.append(f"Sigma detection rule(s) that matched (src/investigation/sigma_rules/): "
                      f"{', '.join(case.sigma_matched_rules)}")

    graph = build_attack_graph(case, alerts)
    by_id = {a.id: a for a in alerts}
    if graph.root_cause_alert_id and by_id.get(graph.root_cause_alert_id):
        root = by_id[graph.root_cause_alert_id]
        lines.append(
            f"Likely root-cause alert (earliest alert with the most downstream, entity-linked "
            f"successors -- see attack_graph.py): {root.id} at {root.timestamp} on {root.agent_name}: "
            f"{root.rule_description}"
        )
        chain_alerts = [by_id[aid] for aid in graph.chain_order if aid in by_id]
    else:
        chain_alerts = sorted(alerts, key=lambda a: (a.rule_level, bool(a.rule_mitre_id)), reverse=True)

    shown = chain_alerts[:MAX_ALERTS_IN_NARRATIVE]
    lines.append(
        f"\nMember alerts ({len(shown)} of {len(alerts)} shown, in reconstructed chain/chronological order):"
    )
    lines.extend(f"- {_alert_line(a)}" for a in shown)
    return "\n".join(lines)
