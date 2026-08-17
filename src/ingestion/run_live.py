# src/ingestion/run_live.py
"""
CLI entrypoint: replay data/alerts.jsonl as a live stream and auto-trigger
AgCyRAG's multi-agent pipeline for severity-gated alerts.

Usage:
    uv run -m src.ingestion.run_live
    uv run -m src.ingestion.run_live --speed 3600 --max-gap 2.0
"""
from __future__ import annotations

import argparse
import asyncio
import logging

from src.ingestion.live_stream import stream_alerts
from src.ingestion.trigger import run
from src.utils.logging_config import setup_logging

logging.getLogger("mcp_use").propagate = False


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Replay synthetic alerts and auto-trigger AgCyRAG.")
    parser.add_argument("--speed", type=float, default=100000.0,
                         help="Time compression factor (simulated seconds per real second). "
                              "Default replays the ~5-day synthetic window in well under a minute.")
    parser.add_argument("--max-gap", type=float, default=0.2,
                         help="Max real seconds to sleep between alerts, regardless of --speed.")
    args = parser.parse_args()

    stream = stream_alerts(speed=args.speed, max_gap_seconds=args.max_gap)
    asyncio.run(run(stream))


if __name__ == "__main__":
    main()
