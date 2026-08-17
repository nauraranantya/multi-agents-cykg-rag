# src/agents/grounding_check.py
"""
Non-LLM verification pass on the synthesizer's structured output.

defengraph_vs_agcyrag_analysis.md Sec.5.4: "a grounding check, not a full
agentic loop" -- after synthesis, confirm every entity the report claims to
cite actually appears in the retrieved evidence; reject/retry once on
failure. Deliberately cheap (regex/substring only, no LLM call) so it
doesn't inherit AgCyRAG's heavier max_steps=30 MCP-agent cost for what
should be a fast sanity check.

Limitation, by design: this can only catch entities we can *identify* as
claims (the report's own declared cited_entities/mitre_techniques, plus
MITRE-ID/IP patterns found inline in final_answer). A hallucinated claim
using no recognizable entity syntax at all won't be caught -- that's the
cost of staying non-LLM; see the analysis doc's Sec.5.4 for why that
trade-off was chosen over a heavier check.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from pydantic import BaseModel

MITRE_ID_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")
IP_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")


class GroundingResult(BaseModel):
    grounded: bool
    checked_entities: List[str]
    ungrounded_entities: List[str]


def _extract_entities(report: Dict[str, Any]) -> List[str]:
    entities = set()
    for e in report.get("cited_entities") or []:
        if e:
            entities.add(str(e).strip())
    for e in report.get("mitre_techniques") or []:
        if e:
            entities.add(str(e).strip())
    final_answer = report.get("final_answer") or ""
    entities.update(MITRE_ID_RE.findall(final_answer))
    entities.update(IP_RE.findall(final_answer))
    return sorted(entities)


def check_grounding(report: Dict[str, Any], context_blobs: List[str]) -> GroundingResult:
    """`report` is a SynthesizedReport dict (e.g. `.model_dump()`).
    `context_blobs` are the raw retrieved-context strings (cypher context,
    vector context, mcp_rdf context) the report was supposed to be grounded
    in."""
    haystack = "\n".join(b for b in context_blobs if b).lower()
    entities = _extract_entities(report)
    ungrounded = [e for e in entities if e.lower() not in haystack]
    return GroundingResult(
        grounded=len(ungrounded) == 0,
        checked_entities=entities,
        ungrounded_entities=ungrounded,
    )
