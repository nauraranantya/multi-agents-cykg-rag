# src/retrieval/temporal.py
"""
Temporal decay weighting on retrieval, ported from
../../defengraph_recreation/src/llm_stages.py (DefenGraph paper eq. 5:
w_t = exp(-alpha * |t_i - t_q|)). Ported verbatim rather than reimplemented,
including the same caveat: the paper never discloses how `alpha` was tuned,
so ALPHA_PER_HOUR here is a free parameter we chose, not a reproduced value.

Applied to AgCyRAG's Cypher/vector retrieval results so recent log events
outrank stale ones with otherwise-similar relevance -- AgCyRAG's retrieval
today has no notion of recency at all.
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

ALPHA_PER_HOUR = 0.05  # temporal decay rate; not specified by the DefenGraph paper
ISO_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")


def _parse_ts(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def temporal_weight(
    node_timestamp: Optional[str],
    query_timestamp: Optional[str],
    alpha_per_hour: float = ALPHA_PER_HOUR,
) -> float:
    """Returns a decay weight in (0, 1]. A missing node or query timestamp
    means "no temporal information to weight by" -- treated as full weight
    (1.0), same as DefenGraph's SKG entities (immutable, no temporal
    weighting per the paper's Sec.III-D)."""
    if not node_timestamp or not query_timestamp:
        return 1.0
    try:
        dt_hours = abs((_parse_ts(node_timestamp) - _parse_ts(query_timestamp)).total_seconds()) / 3600
    except (ValueError, TypeError):
        return 1.0
    return math.exp(-alpha_per_hour * dt_hours)


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _find_timestamp(record: Dict[str, Any], preferred_keys: Sequence[str]) -> Optional[str]:
    for key in preferred_keys:
        val = record.get(key)
        if isinstance(val, str) and ISO_TS_RE.match(val):
            return val
    for val in record.values():
        if isinstance(val, str) and ISO_TS_RE.match(val):
            return val
    return None


def weight_and_sort_records(
    records: List[Dict[str, Any]],
    query_timestamp: Optional[str],
    preferred_keys: Sequence[str] = ("timestamp", "event_timestamp"),
    prune_below: float = 0.0,
) -> List[Dict[str, Any]]:
    """Re-orders (and optionally prunes) a list of Cypher-result-style dicts
    by recency relative to `query_timestamp`. The Cypher generator controls
    the RETURN clause dynamically, so we can't assume a fixed schema --
    instead we heuristically look for an ISO-8601 timestamp under a
    preferred key first, then any string value that looks like one.
    Records with no discoverable timestamp keep weight 1.0 (unchanged
    ranking), same as DefenGraph's immutable-SKG-entity behavior. Content is
    never mutated, only order (and, if prune_below > 0, membership)."""
    if not query_timestamp or not records:
        return records
    scored = [
        (temporal_weight(_find_timestamp(r, preferred_keys), query_timestamp), r)
        for r in records if isinstance(r, dict)
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    if prune_below > 0:
        scored = [(w, r) for w, r in scored if w >= prune_below]
    return [r for _, r in scored]


def weight_and_sort_documents(
    docs: List[Any],
    query_timestamp: Optional[str],
    metadata_key: str = "timestamp",
    prune_below: float = 0.0,
) -> List[Any]:
    """Same idea as weight_and_sort_records, for langchain `Document`
    objects (e.g. from Neo4jVector.similarity_search), reading the
    timestamp out of `.metadata` (populated from the underlying Chunk
    node's properties by src/ingestion/graph_loader.py)."""
    if not query_timestamp or not docs:
        return docs
    scored = [
        (temporal_weight((getattr(d, "metadata", {}) or {}).get(metadata_key), query_timestamp), d)
        for d in docs
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    if prune_below > 0:
        scored = [(w, d) for w, d in scored if w >= prune_below]
    return [d for _, d in scored]
