# src/investigation/api.py
"""
The human-invoked investigation process (see PROTOTYPE.md's architecture
section). A separate process from ingestion (src/ingestion/run_live.py) --
this only reads from Neo4j (src/investigation/clustering.py) and invokes
the pipeline (src/investigation/session.py); it never ingests anything
itself. Run with `uv run -m src.investigation.run_api`.

Routes:
  GET  /cases                          ranked list of recent activity cases
  POST /cases/{case_id}/investigate    launch/resume a case's investigation
  GET  /cases/{case_id}/attack-graph   reconstructed attack graph for a case
  POST /investigations/time-range      launch/resume an ad-hoc time-range investigation
  POST /investigations/{case_id}/chat  ask a follow-up in an open investigation
  GET  /investigations/{case_id}/chat  read an investigation's transcript

Access control: see SECURITY_ASSESSMENT.md. Set INVESTIGATION_API_KEY to
require a matching `X-API-Key` header on every request -- unset (the local-
dev default) leaves the service open, matching this repo's existing
convention for optional/only-enforced-when-configured env vars.
"""
from __future__ import annotations

import os

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from src.investigation import attack_graph, clustering, session

INVESTIGATION_API_KEY = os.environ.get("INVESTIGATION_API_KEY")


async def require_api_key(x_api_key: str = Header(default="")) -> None:
    if INVESTIGATION_API_KEY and x_api_key != INVESTIGATION_API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key.")


app = FastAPI(title="AgCyRAG Investigation API", dependencies=[Depends(require_api_key)])


class TimeRangeRequest(BaseModel):
    start: str  # ISO-8601, e.g. "2022-01-21T00:00:00Z"
    end: str


class ChatRequest(BaseModel):
    message: str


@app.get("/cases")
async def get_cases(lookback_hours: float = 24.0) -> list[clustering.Case]:
    return clustering.list_cases(lookback_hours=lookback_hours)


@app.post("/cases/{case_id}/investigate")
async def investigate_case(case_id: str, lookback_hours: float = clustering.DEFAULT_FIND_LOOKBACK_HOURS) -> session.TurnRecord:
    """`lookback_hours` must cover whatever window you used to list this
    case via GET /cases -- case_id is a hash of an alert set (see
    clustering.py), and re-clustering over a window that doesn't reach the
    case's alerts won't find it. Pass the same (or a wider) lookback_hours
    you used there, e.g. if you listed cases with a replayed historical
    dataset far from wall-clock "now"."""
    case = clustering.find_case(case_id, lookback_hours=lookback_hours)
    if case is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Case {case_id} not found within lookback_hours={lookback_hours}. "
                f"If you listed it via GET /cases?lookback_hours=<bigger value>, pass the same "
                f"lookback_hours here, e.g. POST /cases/{case_id}/investigate?lookback_hours=<bigger value>."
            ),
        )
    alerts = clustering.fetch_alerts_by_ids(case.alert_ids)
    return await session.start_investigation(case, alerts)


@app.get("/cases/{case_id}/attack-graph")
async def get_attack_graph(
    case_id: str, lookback_hours: float = clustering.DEFAULT_FIND_LOOKBACK_HOURS
) -> attack_graph.CaseAttackGraph:
    """Nodes/edges for the reconstructed attack graph (src/investigation/
    attack_graph.py) -- same lookback_hours caveat as .../investigate."""
    case = clustering.find_case(case_id, lookback_hours=lookback_hours)
    if case is None:
        raise HTTPException(
            status_code=404,
            detail=f"Case {case_id} not found within lookback_hours={lookback_hours}.",
        )
    alerts = clustering.fetch_alerts_by_ids(case.alert_ids)
    return attack_graph.build_attack_graph(case, alerts)


@app.post("/investigations/time-range")
async def investigate_time_range(body: TimeRangeRequest) -> session.TurnRecord:
    case = clustering.case_from_range(body.start, body.end)
    if case is None:
        raise HTTPException(status_code=404, detail=f"No alerts found between {body.start} and {body.end}.")
    alerts = clustering.fetch_alerts_by_ids(case.alert_ids)
    return await session.start_investigation(case, alerts)


@app.post("/investigations/{case_id}/chat")
async def chat(case_id: str, body: ChatRequest) -> session.TurnRecord:
    try:
        return await session.ask_followup(case_id, body.message)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/investigations/{case_id}/chat")
async def get_chat(case_id: str) -> list[session.TurnRecord]:
    return session.get_transcript(case_id)
