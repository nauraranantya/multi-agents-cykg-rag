# src/ingestion/config.py
"""
Ingestion source configuration, stored in a small JSON file
(data/ingestion_config.json) so the investigation API (writer, via
src/investigation/api.py's /ingestion/config routes) and the separate
run_live.py process (reader, at startup) can share it without any IPC.

Ingestion stays a separate, operator-started background process (see
PROTOTYPE.md) -- this only lets an operator configure *what run_live.py
replays the next time it's (re)started*, e.g. through a UI, instead of
CLI flags. It does not start, stop, or otherwise control that process, and
a running run_live.py process does not hot-reload this file -- it must be
restarted to pick up a change.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, field_validator

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "ingestion_config.json"


class IngestionConfig(BaseModel):
    alerts_path: Optional[str] = None  # None = run_live.py's own default (data/alerts.jsonl)
    speed: float = 100000.0
    max_gap_seconds: float = 0.2

    @field_validator("speed")
    @classmethod
    def _speed_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("speed must be > 0")
        return v

    @field_validator("max_gap_seconds")
    @classmethod
    def _gap_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("max_gap_seconds must be >= 0")
        return v


def load_config() -> IngestionConfig:
    if not CONFIG_PATH.exists():
        return IngestionConfig()
    return IngestionConfig(**json.loads(CONFIG_PATH.read_text()))


def save_config(config: IngestionConfig) -> IngestionConfig:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(config.model_dump_json(indent=2))
    return config
