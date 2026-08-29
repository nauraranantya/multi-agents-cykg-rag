# src/investigation/run_api.py
"""
CLI entrypoint for the investigation API (src/investigation/api.py) --
mirrors src/ingestion/run_live.py's role for ingestion. A separate process:
run this whenever an analyst wants to review/investigate what
`uv run -m src.ingestion.run_live` has already ingested.

Usage:
    uv run -m src.investigation.run_api
    uv run -m src.investigation.run_api --port 8001
"""
from __future__ import annotations

import argparse
import logging

import uvicorn

from src.utils.logging_config import setup_logging

logging.getLogger("mcp_use").propagate = False


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Run the AgCyRAG investigation API.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run("src.investigation.api:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
