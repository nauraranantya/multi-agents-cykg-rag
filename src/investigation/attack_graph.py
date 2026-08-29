# src/investigation/attack_graph.py
"""
Reconstructs a Case's member alerts into an attack graph: nodes for each
alert plus the hosts/IPs/users/MITRE techniques it touches, edges mirroring
the real graph's relation names (src/ingestion/graph_loader.py's schema)
plus a derived PRECEDES edge chaining alerts that share an entity, in time
order. This is the alert-granularity analogue of KRYSTAL's (Kurniawan
et al. 2022, Computers & Security) backward/forward chaining over a much
lower-level (per-syscall) provenance graph -- our graph is alert-level (we
ingest already-decoded Wazuh/Suricata/AMiner alerts, not raw process/file/
socket events), so "chaining" here means alerts connected via a shared
indicator (host/IP/user) rather than shared low-level system objects.

root_cause_alert_id is picked the same way KRYSTAL scores predecessor
alerts during backward search (its Sec.6.3): the alert with the most
downstream (transitively reachable via PRECEDES) successors, ties broken
by earliest timestamp.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional, Set

from pydantic import BaseModel, Field

from src.ingestion.schema import Alert
from src.investigation.clustering import Case


class AttackGraphNode(BaseModel):
    id: str
    type: Literal["alert", "host", "ip", "user", "mitre_technique"]
    label: str
    timestamp: Optional[str] = None
    rule_level: Optional[int] = None


class AttackGraphEdge(BaseModel):
    source: str
    target: str
    relation: Literal["HAS_ALERT", "ATTACK_TO", "CONNECTS_TO", "TARGETS_USER", "TRIGGERS", "PRECEDES"]


class CaseAttackGraph(BaseModel):
    case_id: str
    nodes: List[AttackGraphNode]
    edges: List[AttackGraphEdge]
    root_cause_alert_id: Optional[str] = None
    chain_order: List[str] = Field(default_factory=list)  # alert ids, chronological


def _alert_node_id(aid: str) -> str: return f"alert:{aid}"
def _host_node_id(host: str) -> str: return f"host:{host}"
def _ip_node_id(ip: str) -> str: return f"ip:{ip}"
def _user_node_id(user: str) -> str: return f"user:{user}"
def _mitre_node_id(tid: str) -> str: return f"mitre:{tid}"


def _entities(alert: Alert) -> Set[str]:
    """The indicators an alert 'touches', used both for node/edge
    construction and to decide whether two alerts should chain via
    PRECEDES. Host is always present; IPs/user are conditional."""
    ent = {_host_node_id(alert.agent_name)}
    if alert.data_srcip:
        ent.add(_ip_node_id(alert.data_srcip))
    if alert.data_dstip:
        ent.add(_ip_node_id(alert.data_dstip))
    if alert.data_dstuser:
        ent.add(_user_node_id(alert.data_dstuser))
    return ent


def build_attack_graph(case: Case, alerts: List[Alert]) -> CaseAttackGraph:
    nodes: Dict[str, AttackGraphNode] = {}
    edges: List[AttackGraphEdge] = []

    def add_node(node: AttackGraphNode) -> None:
        nodes.setdefault(node.id, node)

    for alert in alerts:
        aid = _alert_node_id(alert.id)
        add_node(AttackGraphNode(id=aid, type="alert", label=alert.rule_description,
                                  timestamp=alert.timestamp, rule_level=alert.rule_level))

        hid = _host_node_id(alert.agent_name)
        add_node(AttackGraphNode(id=hid, type="host", label=alert.agent_name))
        edges.append(AttackGraphEdge(source=hid, target=aid, relation="HAS_ALERT"))

        if alert.data_srcip:
            sid = _ip_node_id(alert.data_srcip)
            add_node(AttackGraphNode(id=sid, type="ip", label=alert.data_srcip))
            edges.append(AttackGraphEdge(source=sid, target=aid, relation="ATTACK_TO"))

        if alert.data_dstip:
            did = _ip_node_id(alert.data_dstip)
            add_node(AttackGraphNode(id=did, type="ip", label=alert.data_dstip))
            edges.append(AttackGraphEdge(source=aid, target=did, relation="CONNECTS_TO"))

        if alert.data_dstuser:
            uid = _user_node_id(alert.data_dstuser)
            add_node(AttackGraphNode(id=uid, type="user", label=alert.data_dstuser))
            edges.append(AttackGraphEdge(source=aid, target=uid, relation="TARGETS_USER"))

        for tid in sorted(set(alert.rule_mitre_id + alert.sigma_mitre_id)):
            mid = _mitre_node_id(tid)
            add_node(AttackGraphNode(id=mid, type="mitre_technique", label=tid))
            edges.append(AttackGraphEdge(source=aid, target=mid, relation="TRIGGERS"))

    # PRECEDES: every earlier/later pair (not just adjacent) sharing an
    # entity gets linked, so transitive chains through different shared
    # indicators (host -> different srcip -> different host) are captured.
    ordered = sorted(alerts, key=lambda a: a.timestamp)
    successors: Dict[str, Set[str]] = {a.id: set() for a in ordered}
    for i, earlier in enumerate(ordered):
        earlier_ent = _entities(earlier)
        for later in ordered[i + 1:]:
            if earlier_ent & _entities(later):
                edges.append(AttackGraphEdge(
                    source=_alert_node_id(earlier.id), target=_alert_node_id(later.id), relation="PRECEDES"
                ))
                successors[earlier.id].add(later.id)

    def _reachable_count(alert_id: str) -> int:
        seen: Set[str] = set()
        stack = list(successors.get(alert_id, ()))
        while stack:
            nxt = stack.pop()
            if nxt not in seen:
                seen.add(nxt)
                stack.extend(successors.get(nxt, ()))
        return len(seen)

    root_cause_alert_id = None
    if ordered:
        scored = sorted(
            ((i, a.id) for i, a in enumerate(ordered)),
            key=lambda pair: (-_reachable_count(pair[1]), pair[0]),
        )
        root_cause_alert_id = scored[0][1]

    return CaseAttackGraph(
        case_id=case.case_id,
        nodes=list(nodes.values()),
        edges=edges,
        root_cause_alert_id=root_cause_alert_id,
        chain_order=[a.id for a in ordered],
    )
