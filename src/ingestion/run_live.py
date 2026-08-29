# src/ingestion/run_live.py
"""
CLI entrypoint: replay data/alerts.jsonl as a live stream and ingest every
alert into the graph. This process only ingests -- it never invokes the
investigation pipeline (see src/ingestion/trigger.py::ingest_stream()'s
docstring). To investigate what's been ingested, run
`uv run -m src.investigation.run_api` as a separate process and use its
GET /cases / POST /cases/{case_id}/investigate endpoints.

Usage:
    uv run -m src.ingestion.run_live
    uv run -m src.ingestion.run_live --speed 3600 --max-gap 2.0
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from src.ingestion.live_stream import stream_alerts
from src.ingestion.trigger import ingest_stream
from src.utils.logging_config import setup_logging

logging.getLogger("mcp_use").propagate = False


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Replay synthetic alerts and ingest them into the graph.")
    parser.add_argument("--alerts-path", type=Path, default=None,
                         help="Path to the alerts JSONL to replay (default: data/alerts.jsonl). "
                              "Point this at a smaller file -- e.g. a truncated sample -- to run "
                              "a quick smoke test instead of the full set.")
    parser.add_argument("--speed", type=float, default=100000.0,
                         help="Time compression factor (simulated seconds per real second). "
                              "Default replays the ~5-day synthetic window in well under a minute.")
    parser.add_argument("--max-gap", type=float, default=0.2,
                         help="Max real seconds to sleep between alerts, regardless of --speed.")
    args = parser.parse_args()

    stream = stream_alerts(path=args.alerts_path, speed=args.speed, max_gap_seconds=args.max_gap)
    asyncio.run(ingest_stream(stream))


if __name__ == "__main__":
    main()
