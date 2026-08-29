# src/investigation/sigma_matcher.py
"""
Matches src/investigation/sigma_rules/*.yml (real Sigma YAML, hand-curated
-- see that directory's rules for why: public SigmaHQ rules target raw
per-platform log fields our alert-level schema doesn't have, since we
ingest already-decoded Wazuh/Suricata/AMiner alerts, not raw system-call
audit data) against Alert objects, producing extra MITRE ATT&CK technique
tags beyond the sensor's own native rule_mitre_id -- src/ingestion/
trigger.py's _trigger_reason docstring already documents that native tags
cover only ~36% of AIT-ADS alerts.

Runs at investigation time (src/investigation/clustering.py), not
ingestion time: rule changes then apply retroactively to already-ingested
historical data without a re-ingest/backfill step, and results are never
written back to Neo4j (Alert.sigma_mitre_id is populated in-memory only,
see src/ingestion/schema.py).

Deliberately a *practical subset* of the Sigma detection/condition
language, not the full spec -- KRYSTAL (Kurniawan et al. 2022) makes the
same scoping call for its own Sigma-to-SPARQL translation. Supported:
  - Field modifiers: `|contains` (substring, case-insensitive), `|re`
    (regex), and bare (no modifier -- also substring match here, since our
    fields are free text where exact equality is rarely useful; a
    deliberate simplification from Sigma's real bare-field exact-match
    semantics).
  - A selection's value may be a single string or a list (list = OR of
    alternatives); multiple fields within one selection are ANDed.
  - `condition`: a single selection name; `sel1 and sel2`; `sel1 or sel2`;
    `sel1 and not sel2`; `N of <prefix>*` / `all of <prefix>*`.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from pydantic import BaseModel, Field

from src.ingestion.schema import Alert

RULES_DIR = Path(__file__).resolve().parent / "sigma_rules"

# attack.t1110, attack.t1204.002, ... -> T1110 / T1204.002
_TECHNIQUE_TAG_RE = re.compile(r"^attack\.t(\d{4}(?:\.\d{3})?)$", re.IGNORECASE)


class SigmaRule(BaseModel):
    id: str
    title: str
    description: str = ""
    detection: Dict[str, Any]
    level: str = "medium"
    tags: List[str] = Field(default_factory=list)

    @property
    def mitre_technique_ids(self) -> List[str]:
        ids = []
        for tag in self.tags:
            m = _TECHNIQUE_TAG_RE.match(tag.strip())
            if m:
                ids.append(f"T{m.group(1)}")
        return ids


def load_rules(rules_dir: Optional[Path] = None) -> List[SigmaRule]:
    rules_dir = rules_dir or RULES_DIR
    rules = []
    for path in sorted(rules_dir.glob("*.yml")) + sorted(rules_dir.glob("*.yaml")):
        with open(path) as f:
            raw = yaml.safe_load(f)
        rules.append(SigmaRule(
            id=raw["id"],
            title=raw["title"],
            description=raw.get("description", ""),
            detection=raw["detection"],
            level=raw.get("level", "medium"),
            tags=raw.get("tags", []),
        ))
    return rules


def _field_value(alert: Alert, field: str):
    if field == "rule_groups":
        return alert.rule_groups
    return getattr(alert, field, None)


def _match_field(actual, expected, modifier: str) -> bool:
    values = expected if isinstance(expected, list) else [expected]
    haystacks = [str(v) for v in actual if v] if isinstance(actual, list) else ([str(actual)] if actual else [])
    if not haystacks:
        return False
    for value in values:
        for hay in haystacks:
            if modifier == "re":
                if re.search(str(value), hay, re.IGNORECASE):
                    return True
            else:  # 'contains' or bare -- see module docstring
                if str(value).lower() in hay.lower():
                    return True
    return False


def _match_selection(selection: Dict[str, Any], alert: Alert) -> bool:
    for raw_field, expected in selection.items():
        field, _, modifier = raw_field.partition("|")
        if not _match_field(_field_value(alert, field), expected, modifier):
            return False
    return True


_QUANT_OF_RE = re.compile(r"^(all|\d+)\s+of\s+([A-Za-z0-9_]+)\*$", re.IGNORECASE)


def _evaluate_condition(condition: str, selection_results: Dict[str, bool]) -> bool:
    condition = condition.strip()

    m = _QUANT_OF_RE.match(condition)
    if m:
        quant, prefix = m.group(1), m.group(2)
        matched_names = [name for name in selection_results if name.startswith(prefix)]
        hit_count = sum(1 for name in matched_names if selection_results[name])
        if not matched_names:
            return False
        return hit_count == len(matched_names) if quant.lower() == "all" else hit_count >= int(quant)

    result: Optional[bool] = None
    op: Optional[str] = None
    negate_next = False
    for tok in condition.split():
        low = tok.lower()
        if low == "not":
            negate_next = True
            continue
        if low in ("and", "or"):
            op = low
            continue
        val = selection_results.get(tok, False)
        if negate_next:
            val = not val
            negate_next = False
        if result is None:
            result = val
        elif op == "and":
            result = result and val
        elif op == "or":
            result = result or val
    return bool(result)


def match_alert(alert: Alert, rules: List[SigmaRule]) -> List[Tuple[str, str]]:
    """Returns (technique_id, rule_title) for every rule that matched."""
    hits = []
    for rule in rules:
        selection_names = [name for name in rule.detection if name != "condition"]
        selection_results = {name: _match_selection(rule.detection[name], alert) for name in selection_names}
        condition = rule.detection.get("condition", selection_names[0] if len(selection_names) == 1 else "")
        if _evaluate_condition(condition, selection_results):
            for tid in rule.mitre_technique_ids:
                hits.append((tid, rule.title))
    return hits
