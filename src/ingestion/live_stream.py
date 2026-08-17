# src/ingestion/live_stream.py
"""
Replays data/alerts.jsonl as an async event stream, simulating a live SIEM
feed. This decouples ingestion/triggering from any specific real SIEM
integration -- trigger.py just consumes an async iterator of Alert objects,
so a real feed could later replace this without touching trigger.py at all.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Optional

from src.ingestion.schema import Alert

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def _parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


async def stream_alerts(
    path: Optional[Path] = None,
    speed: float = 100000.0,
    max_gap_seconds: float = 0.2,
) -> AsyncIterator[Alert]:
    """Yield alerts in timestamp order, sleeping between them to simulate a
    live feed.

    `speed` compresses wall-clock time (speed=3600 means 1 simulated hour
    passes in 1 real second; the default, 100000, replays the whole ~5-day
    synthetic window in well under a minute). `max_gap_seconds` caps any
    single sleep so a multi-hour gap in the synthetic data can't stall a
    demo for real hours even at low --speed.
    """
    path = path or (DATA_DIR / "alerts.jsonl")
    alerts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                alerts.append(Alert(**json.loads(line)))
    alerts.sort(key=lambda a: a.timestamp)

    prev_ts = None
    for alert in alerts:
        ts = _parse_ts(alert.timestamp)
        if prev_ts is not None and speed > 0:
            gap = (ts - prev_ts).total_seconds() / speed
            await asyncio.sleep(min(max(gap, 0.0), max_gap_seconds))
        prev_ts = ts
        yield alert


if __name__ == "__main__":
    async def _demo():
        count = 0
        async for alert in stream_alerts():
            print(alert.timestamp, alert.agent_name, alert.rule_level, alert.rule_description)
            count += 1
        print(f"--- streamed {count} alerts ---")

    asyncio.run(_demo())
