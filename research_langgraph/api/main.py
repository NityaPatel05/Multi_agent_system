"""FastAPI deployment layer (CLAUDE.md build order step 3 / architecture.md §7).

Routes (all require the API key when `API_KEY` is set; see api/security.py):
- `POST /research`                    -> 202 {run_id, status:"queued"}; the graph runs in the
                                         background up to the human-approval interrupt.
- `GET  /research/{id}`               -> status, plan (for the approval screen), final report.
- `GET  /research/{id}/stream`        -> SSE: per-node events (node_started / node_finished
                                         with result sizes + audit entries), awaiting_approval,
                                         done, failed. Supports `Last-Event-ID` to resume.
- `POST /research/{id}/approve-plan`  -> 202; resumes the run in the background. 409 unless the
                                         run is awaiting approval and not already executing.
- `GET  /health`, `GET /ready`        -> liveness / readiness (ready pings the checkpointer).

Runs execute on a bounded worker pool (api/runs.py, MAX_CONCURRENT_RUNS); over capacity
returns 429. This service only orchestrates the LangGraph run — it never embeds the
adapters' provider calls itself.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from research_langgraph.api.runs import RunBusy, registry
from research_langgraph.api.security import rate_limit, require_api_key, warn_if_open
from research_langgraph.main import build_graph
from research_langgraph.memory.checkpointer import close_checkpointers, get_checkpointer
from research_langgraph.observability.langsmith_setup import configure_langsmith

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
load_dotenv()  # entrypoint-only (not at package import) so tests/evals stay on the offline mocks

logger = logging.getLogger(__name__)

_graph = None
_STREAM_MAX_SECONDS = 30 * 60
_STREAM_POLL_S = 0.25


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph
    configure_langsmith()
    warn_if_open()
    _graph = build_graph(checkpointer=get_checkpointer())
    try:
        yield
    finally:
        close_checkpointers()


app = FastAPI(title="Multi-Agent Research Citation Engine", lifespan=lifespan)

# Hardening via env (all optional): ALLOWED_HOSTS blocks DNS-rebinding Host headers,
# CORS_ORIGINS lets a browser frontend on another origin call the API.
if os.environ.get("ALLOWED_HOSTS"):
    app.add_middleware(TrustedHostMiddleware,
                       allowed_hosts=[h.strip() for h in os.environ["ALLOWED_HOSTS"].split(",") if h.strip()])
if os.environ.get("CORS_ORIGINS"):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in os.environ["CORS_ORIGINS"].split(",") if o.strip()],
        allow_methods=["GET", "POST"],
        allow_headers=["X-API-Key", "Authorization", "Content-Type", "Last-Event-ID"],
    )


class ResearchRequest(BaseModel):
    topic: str = Field(min_length=3, max_length=500)
    budget_mode: Literal["fast", "balanced", "thorough"] = "balanced"


class ApprovalRequest(BaseModel):
    approved: bool = True
    feedback: str = Field(default="", max_length=2000)


def _config(run_id: str) -> dict:
    return {"configurable": {"thread_id": run_id}}


def _snapshot(run_id: str):
    return _graph.get_state(_config(run_id))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    try:
        _snapshot("readiness-probe")  # round-trips the checkpointer (Postgres when configured)
    except Exception:
        logger.exception("readiness check failed")
        raise HTTPException(status_code=503, detail="checkpointer unavailable") from None
    return {"status": "ready"}


@app.post("/research", status_code=202, dependencies=[Depends(require_api_key), Depends(rate_limit)])
def start_research(req: ResearchRequest) -> dict:
    run_id = str(uuid.uuid4())
    try:
        registry.submit(_graph, run_id,
                        {"topic": req.topic, "run_id": run_id, "budget_mode": req.budget_mode})
    except RunBusy:
        raise HTTPException(status_code=429, detail="server at capacity, retry shortly",
                            headers={"Retry-After": "15"}) from None
    return {"run_id": run_id, "status": "queued"}


@app.get("/research/{run_id}", dependencies=[Depends(require_api_key)])
def get_research(run_id: str) -> dict:
    info = registry.get(run_id)
    snap = _snapshot(run_id)
    values = (snap.values if snap is not None else None) or {}
    if not values and info is None:
        raise HTTPException(status_code=404, detail="run not found")
    status = values.get("status", "unknown")
    if info is not None and info.running:
        status = "queued" if not values else "running"
    elif info is not None and info.status == "failed":
        status = "failed"
    elif info is not None and not values:
        status = info.status
    return {
        "run_id": run_id,
        "status": status,
        "final_report": values.get("final_report"),
        "evidence_map": values.get("evidence_map"),
        "awaiting_approval": bool(snap is not None and snap.next and "human_approval" in snap.next
                                  and not (info and info.running)),
        # plan, so an approval screen has something to show
        "plan": {
            "dimensions": values.get("dimensions"),
            "queries_by_dimension": values.get("queries_by_dimension"),
            "complexity": values.get("complexity"),
            "plan_rationale": values.get("plan_rationale"),
        },
    }


def _sse(event: dict) -> str:
    # No `event:` line on purpose: browser EventSource.onmessage only receives unnamed
    # events. The event kind is in the JSON `type` field.
    return f"id: {event['seq']}\ndata: {json.dumps(event, default=str)}\n\n"


@app.get("/research/{run_id}/stream", dependencies=[Depends(require_api_key)])
async def stream_research(run_id: str, request: Request):
    info = registry.get(run_id)
    if info is None:
        # Not tracked in this process (e.g. server restarted): report the checkpointed status once.
        snap = await asyncio.to_thread(_snapshot, run_id)
        values = (snap.values if snap is not None else None) or {}
        if not values:
            raise HTTPException(status_code=404, detail="run not found")

        async def once():
            yield f"data: {json.dumps({'run_id': run_id, 'type': 'status', 'status': values.get('status')})}\n\n"

        return StreamingResponse(once(), media_type="text/event-stream")

    try:
        seq = int(request.headers.get("last-event-id", "-1")) + 1
    except ValueError:
        seq = 0

    async def event_source():
        nonlocal seq
        deadline = time.monotonic() + _STREAM_MAX_SECONDS
        while time.monotonic() < deadline:
            for ev in info.events_since(seq):
                yield _sse(ev)
                seq = ev["seq"] + 1
                if ev["type"] in ("done", "failed"):
                    return
            if not info.running and not info.events_since(seq):
                # Idle (e.g. paused at the approval gate): end the stream so proxies don't
                # idle-timeout; the client reconnects with Last-Event-ID after approving.
                return
            if await request.is_disconnected():
                return
            await asyncio.sleep(_STREAM_POLL_S)

    return StreamingResponse(event_source(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/research/{run_id}/approve-plan", status_code=202,
          dependencies=[Depends(require_api_key), Depends(rate_limit)])
def approve_plan(run_id: str, req: ApprovalRequest) -> dict:
    from langgraph.types import Command

    snap = _snapshot(run_id)
    if snap is None or not snap.values:
        raise HTTPException(status_code=404, detail="run not found")
    if not (snap.next and "human_approval" in snap.next):
        # a retry after an HTTP timeout would otherwise start a 2nd concurrent execution
        raise HTTPException(status_code=409, detail="run is not awaiting plan approval")
    try:
        registry.submit(_graph, run_id,
                        Command(resume={"approved": req.approved, "feedback": req.feedback}))
    except RunBusy as e:
        raise HTTPException(status_code=409 if "already" in str(e) else 429, detail=str(e)) from None
    return {"run_id": run_id, "status": "running"}
