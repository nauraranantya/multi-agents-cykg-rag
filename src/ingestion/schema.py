# src/ingestion/schema.py
"""
Raw log alert schema. Wazuh-style, matching the shape already used in
../../defengraph_recreation/data/dkg_live_alerts.json (real Wazuh alert
fields + real MITRE ATT&CK technique IDs), so the same synthetic-data
conventions apply on both sides of the analysis doc's comparison.
"""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class Alert(BaseModel):
    id: str
    timestamp: str  # ISO-8601, e.g. "2026-08-04T01:02:03Z"

    agent_name: str
    agent_ip: Optional[str] = None

    rule_id: int
    rule_level: int  # Wazuh severity, roughly 0-15; higher = more severe
    rule_description: str
    rule_groups: List[str] = Field(default_factory=list)
    rule_mitre_id: List[str] = Field(default_factory=list)
    rule_mitre_tactic: List[str] = Field(default_factory=list)

    data_srcip: Optional[str] = None
    data_dstip: Optional[str] = None
    data_dstuser: Optional[str] = None

    decoder_name: Optional[str] = None
    full_log: str = ""

    scenario_id: Optional[str] = None  # which synthetic attack chain (or "noise") this belongs to

    def to_jsonl_record(self) -> dict:
        return self.model_dump(exclude_none=True)
