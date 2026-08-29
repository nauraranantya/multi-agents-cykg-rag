# src/investigation/clustering.py
"""
Turns alerts already sitting in the graph (src/ingestion/graph_loader.py)
into ranked "activity cases" for a human analyst to pick from -- the
human-invoked replacement for src/ingestion/trigger.py's old auto-escalate
gate. Computed on demand, straight from Neo4j, every time it's called: no
Case nodes are written back to the graph, so ingestion (src/ingestion/)
stays exactly what it says it is -- store alerts as a graph, nothing more
-- and "what counts as a case" is entirely the investigation side's
decision, re-derivable at any time from what's actually been ingested.

Clustering is two passes, both deterministic (no LLM):
  1. Per-host time-session clustering: alerts on the same host, chained
     together as long as consecutive alerts are within
     CASE_SESSION_GAP_MINUTES of each other. Host is the strongest
     structural anchor every alert already has in the graph
     ((Host)-[:HAS_ALERT]->(Alert), set by every alert unconditionally).
  2. Cross-host merge: two per-host session clusters merge into one case
     if they share a source IP or destination user AND their time windows
     are within CASE_MERGE_GAP_MINUTES of overlapping -- catches lateral
     movement across hosts that pass 1 alone would report as unrelated.
     Union-Find over the (small) number of per-host clusters, not alerts.

Known limitation (documented, not an oversight): case_id is a hash of its
exact member alert set at clustering time. A still-growing activity
cluster (new alerts keep landing on the same host/indicators after an
analyst already started investigating it) will hash to a *different*
case_id next time list_cases() is called, surfacing as a "new" case rather
than extending the same investigation thread. Accepted for v1 in favor of
staying stateless (no persisted Case identity to keep in sync with a live
ingestion stream running as a separate process).
"""
from __future__ import annotations

import hashlib
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from src.config.settings import graph
from src.ingestion.schema import Alert
from src.ingestion.trigger import _trigger_reason
from src.investigation.sigma_matcher import load_rules, match_alert
from src.retrieval.temporal import now_iso, temporal_weight

_SIGMA_RULES = load_rules()  # loaded once per process; rule files are static

CASE_SESSION_GAP_MINUTES = float(os.environ.get("CASE_SESSION_GAP_MINUTES", "60"))
CASE_MERGE_GAP_MINUTES = float(os.environ.get("CASE_MERGE_GAP_MINUTES", "60"))
# find_case()'s default lookback -- generous for a genuinely live feed, but
# note this is relative to wall-clock "now": a replayed *historical*
# dataset (e.g. AIT-ADS, dated ~2022) needs an explicit, much larger
# lookback_hours passed by the caller (see api.py's /cases/{id}/investigate
# lookback_hours query param) to ever be found by it.
DEFAULT_FIND_LOOKBACK_HOURS = 24.0 * 14


class Case(BaseModel):
    case_id: str
    alert_ids: List[str]
    hosts: List[str]
    src_ips: List[str] = Field(default_factory=list)
    dst_users: List[str] = Field(default_factory=list)
    mitre_techniques: List[str] = Field(default_factory=list)  # union of native + sigma-matched
    sigma_matched_rules: List[str] = Field(default_factory=list)  # rule titles, for transparency
    alert_count: int
    first_seen: str
    last_seen: str
    max_rule_level: int
    noteworthy_alert_count: int  # how many member alerts pass _trigger_reason() OR have a sigma match
    urgency_score: float = 0.0


def _parse_ts(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


_FETCH_ALERTS_CYPHER = """
MATCH (h:Host)-[:HAS_ALERT]->(a:Alert)
WHERE a.timestamp >= $start AND a.timestamp <= $end
OPTIONAL MATCH (srcip:IP)-[:ATTACK_TO]->(a)
OPTIONAL MATCH (a)-[:CONNECTS_TO]->(dstip:IP)
OPTIONAL MATCH (a)-[:TARGETS_USER]->(u:User)
OPTIONAL MATCH (a)-[:TRIGGERS]->(m:MitreTechnique)
RETURN a.id AS id, a.timestamp AS timestamp, h.id AS agent_name,
       a.rule_id AS rule_id, a.rule_level AS rule_level, a.title AS rule_description,
       a.full_log AS full_log, a.scenario_id AS scenario_id,
       collect(DISTINCT srcip.id) AS srcips, collect(DISTINCT dstip.id) AS dstips,
       collect(DISTINCT u.id) AS dstusers, collect(DISTINCT m.id) AS mitre_ids
ORDER BY a.timestamp
"""


_FETCH_ALERTS_BY_ID_CYPHER = """
MATCH (h:Host)-[:HAS_ALERT]->(a:Alert)
WHERE a.id IN $ids
OPTIONAL MATCH (srcip:IP)-[:ATTACK_TO]->(a)
OPTIONAL MATCH (a)-[:CONNECTS_TO]->(dstip:IP)
OPTIONAL MATCH (a)-[:TARGETS_USER]->(u:User)
OPTIONAL MATCH (a)-[:TRIGGERS]->(m:MitreTechnique)
RETURN a.id AS id, a.timestamp AS timestamp, h.id AS agent_name,
       a.rule_id AS rule_id, a.rule_level AS rule_level, a.title AS rule_description,
       a.full_log AS full_log, a.scenario_id AS scenario_id,
       collect(DISTINCT srcip.id) AS srcips, collect(DISTINCT dstip.id) AS dstips,
       collect(DISTINCT u.id) AS dstusers, collect(DISTINCT m.id) AS mitre_ids
ORDER BY a.timestamp
"""


def _enrich_with_sigma(alert: Alert) -> None:
    """In-place: sets sigma_mitre_id/sigma_matched_rules from
    src/investigation/sigma_matcher.py (see that module's docstring for
    why this runs here, at fetch time, rather than at ingestion)."""
    hits = match_alert(alert, _SIGMA_RULES)
    alert.sigma_mitre_id = sorted({tid for tid, _ in hits})
    alert.sigma_matched_rules = sorted({title for _, title in hits})


def _rows_to_alerts(rows: List[dict]) -> List[Alert]:
    alerts = [
        Alert(
            id=r["id"],
            timestamp=r["timestamp"],
            agent_name=r["agent_name"],
            rule_id=r["rule_id"],
            rule_level=r["rule_level"],
            rule_description=r["rule_description"] or "",
            rule_mitre_id=[m for m in (r["mitre_ids"] or []) if m],
            data_srcip=next((ip for ip in (r["srcips"] or []) if ip), None),
            data_dstip=next((ip for ip in (r["dstips"] or []) if ip), None),
            data_dstuser=next((u for u in (r["dstusers"] or []) if u), None),
            full_log=r["full_log"] or "",
            scenario_id=r["scenario_id"],
        )
        for r in rows
    ]
    for alert in alerts:
        _enrich_with_sigma(alert)
    return alerts


def fetch_alerts(start: str, end: str) -> List[Alert]:
    """One Cypher query joining every layer graph_loader.py writes, filtered
    to a timestamp range, reassembled into Alert objects so _trigger_reason
    (and the rest of Alert's fields, for narrative.py) can be reused as-is."""
    return _rows_to_alerts(graph.query(_FETCH_ALERTS_CYPHER, {"start": start, "end": end}))


def fetch_alerts_by_ids(alert_ids: List[str]) -> List[Alert]:
    """Fetches a case's exact member alerts by id -- used to render
    narrative.py's fact sheet for a case handed back by list_cases()/
    find_case(), independent of whatever lookback window found it."""
    if not alert_ids:
        return []
    return _rows_to_alerts(graph.query(_FETCH_ALERTS_BY_ID_CYPHER, {"ids": alert_ids}))


def fetch_recent_alerts(lookback_hours: float = 24.0) -> List[Alert]:
    end = now_iso()
    start = (datetime.now(timezone.utc) - timedelta(hours=lookback_hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return fetch_alerts(start, end)


class _SessionCluster:
    """One host's contiguous run of alerts (see _session_cluster_per_host).
    Not the pydantic Case -- an intermediate grouping before the cross-host
    merge in _merge_cross_host."""
    __slots__ = ("alerts",)

    def __init__(self, alerts: List[Alert]):
        self.alerts = alerts

    @property
    def host(self) -> str:
        return self.alerts[0].agent_name

    @property
    def first_seen(self) -> datetime:
        return min(_parse_ts(a.timestamp) for a in self.alerts)

    @property
    def last_seen(self) -> datetime:
        return max(_parse_ts(a.timestamp) for a in self.alerts)

    @property
    def src_ips(self) -> set:
        return {a.data_srcip for a in self.alerts if a.data_srcip}

    @property
    def dst_users(self) -> set:
        return {a.data_dstuser for a in self.alerts if a.data_dstuser}


def _session_cluster_per_host(alerts: List[Alert]) -> List[_SessionCluster]:
    by_host: Dict[str, List[Alert]] = {}
    for a in alerts:
        by_host.setdefault(a.agent_name, []).append(a)

    clusters: List[_SessionCluster] = []
    gap = timedelta(minutes=CASE_SESSION_GAP_MINUTES)
    for host_alerts in by_host.values():
        host_alerts.sort(key=lambda a: a.timestamp)
        current: List[Alert] = []
        prev_ts: Optional[datetime] = None
        for a in host_alerts:
            ts = _parse_ts(a.timestamp)
            if prev_ts is not None and (ts - prev_ts) > gap:
                clusters.append(_SessionCluster(current))
                current = []
            current.append(a)
            prev_ts = ts
        if current:
            clusters.append(_SessionCluster(current))
    return clusters


def _windows_within(a: _SessionCluster, b: _SessionCluster, gap: timedelta) -> bool:
    """True if [a.first,a.last] and [b.first,b.last] overlap, or the gap
    between them is within `gap`."""
    if a.first_seen <= b.last_seen and b.first_seen <= a.last_seen:
        return True
    delta = a.first_seen - b.last_seen if a.first_seen > b.last_seen else b.first_seen - a.last_seen
    return delta <= gap


def _merge_cross_host(clusters: List[_SessionCluster]) -> List[List[_SessionCluster]]:
    """Union-Find over per-host session clusters, merging any two that
    share a source IP or destination user within CASE_MERGE_GAP_MINUTES of
    overlapping -- O(n^2) over clusters (a handful per lookback window,
    not per alert), deliberately simple over a spatial index."""
    n = len(clusters)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    merge_gap = timedelta(minutes=CASE_MERGE_GAP_MINUTES)
    for i in range(n):
        for j in range(i + 1, n):
            ci, cj = clusters[i], clusters[j]
            if ci.host == cj.host:
                continue  # already one session cluster per host segment
            shares_indicator = bool(ci.src_ips & cj.src_ips) or bool(ci.dst_users & cj.dst_users)
            if shares_indicator and _windows_within(ci, cj, merge_gap):
                union(i, j)

    groups: Dict[int, List[_SessionCluster]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(clusters[i])
    return list(groups.values())


def _case_id(alert_ids: List[str]) -> str:
    digest = hashlib.sha1("|".join(sorted(alert_ids)).encode("utf-8")).hexdigest()
    return f"case-{digest[:12]}"


def score_urgency(case: Case) -> float:
    """Deterministic 0-100 urgency score, reusing signals already justified
    elsewhere in this codebase rather than inventing new unvalidated ones:
    _trigger_reason (the old auto-escalate gate, verified against AIT-ADS
    ground truth in eval/analyze_trigger_threshold.py) now measures how
    much of a case is independently "noteworthy"; recency reuses
    src/retrieval/temporal.py's own already-documented decay. Weights below
    are a free parameter (same caveat temporal.py makes about
    ALPHA_PER_HOUR -- not derived from any paper, just a reasonable
    starting split)."""
    noteworthy_component = min(case.noteworthy_alert_count / 3, 1.0)
    severity_component = case.max_rule_level / 15
    size_component = min(case.alert_count / 10, 1.0)
    recency_component = temporal_weight(case.last_seen, now_iso())
    score = 100 * (
        0.35 * noteworthy_component
        + 0.25 * severity_component
        + 0.15 * size_component
        + 0.25 * recency_component
    )
    return round(score, 1)


def _to_case(alerts: List[Alert]) -> Case:
    alert_ids = [a.id for a in alerts]
    timestamps = [_parse_ts(a.timestamp) for a in alerts]
    # "Noteworthy" now also counts a sigma match on its own, even absent a
    # native rule_mitre_id/high rule_level -- _trigger_reason() alone
    # predates sigma_rules and only ever looked at native fields.
    noteworthy = sum(1 for a in alerts if _trigger_reason(a) is not None or a.sigma_mitre_id)
    mitre = sorted({t for a in alerts for t in (a.rule_mitre_id + a.sigma_mitre_id)})
    sigma_rules_hit = sorted({title for a in alerts for title in a.sigma_matched_rules})
    case = Case(
        case_id=_case_id(alert_ids),
        alert_ids=alert_ids,
        hosts=sorted({a.agent_name for a in alerts}),
        src_ips=sorted({a.data_srcip for a in alerts if a.data_srcip}),
        dst_users=sorted({a.data_dstuser for a in alerts if a.data_dstuser}),
        mitre_techniques=mitre,
        sigma_matched_rules=sigma_rules_hit,
        alert_count=len(alerts),
        first_seen=min(timestamps).strftime("%Y-%m-%dT%H:%M:%SZ"),
        last_seen=max(timestamps).strftime("%Y-%m-%dT%H:%M:%SZ"),
        max_rule_level=max(a.rule_level for a in alerts),
        noteworthy_alert_count=noteworthy,
    )
    case.urgency_score = score_urgency(case)
    return case


def cluster_alerts(alerts: List[Alert]) -> List[Case]:
    if not alerts:
        return []
    session_clusters = _session_cluster_per_host(alerts)
    merged_groups = _merge_cross_host(session_clusters)
    return [_to_case([a for cluster in group for a in cluster.alerts]) for group in merged_groups]


def list_cases(lookback_hours: float = 24.0) -> List[Case]:
    alerts = fetch_recent_alerts(lookback_hours)
    cases = cluster_alerts(alerts)
    return sorted(cases, key=lambda c: c.urgency_score, reverse=True)


def case_from_range(start: str, end: str) -> Optional[Case]:
    """An analyst-given time range is treated as one case, unclustered --
    matches "investigate activity from timestamp xx-xx" literally, rather
    than silently re-splitting what the analyst explicitly scoped."""
    alerts = fetch_alerts(start, end)
    if not alerts:
        return None
    return _to_case(alerts)


def find_case(case_id: str, lookback_hours: float = DEFAULT_FIND_LOOKBACK_HOURS) -> Optional[Case]:
    """Re-clusters over a generous lookback and returns the matching case,
    or None. `case_id` is a hash of its alert set (see module docstring),
    so this only finds cases whose exact alert set still clusters
    identically."""
    for case in list_cases(lookback_hours):
        if case.case_id == case_id:
            return case
    return None
