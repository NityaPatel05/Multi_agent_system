"""FastAPI deployment layer (CLAUDE.md build order step 3 / architecture.md §7).

Exposes `/research` (start a run, returns a run ID), `/research/{id}` (poll status),
`/research/{id}/stream` (SSE status stream), and `/research/{id}/approve-plan` (the
human-approval checkpoint). This service only orchestrates the LangGraph run — it never
embeds the adapters' provider calls itself, matching the same layering used everywhere
else in this project. Each MCP tool server is meant to be containerized independently
(architecture.md §7) so a broken search integration doesn't require redeploying this
service.

Note: requests here run the graph synchronously within the request handler for
simplicity — `/research` blocks until the human-approval interrupt, and
`/research/{id}/approve-plan` blocks until the run finishes or hits another interrupt.
For real production traffic, move the `graph.invoke(...)` calls onto a background task
queue; that's an operational change, not an architectural one.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel

from research_langgraph.main import build_graph, mark_run_failed
from research_langgraph.memory.checkpointer import get_checkpointer
from research_langgraph.observability.langsmith_setup import configure_langsmith

logger = logging.getLogger(__name__)

_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph
    configure_langsmith()
    checkpointer = get_checkpointer()
    _graph = build_graph(checkpointer=checkpointer)
    yield


app = FastAPI(title="Multi-Agent Research Citation Engine", lifespan=lifespan)


class ResearchRequest(BaseModel):
    topic: str
    budget_mode: str = "balanced"


class ApprovalRequest(BaseModel):
    approved: bool = True
    feedback: str = ""


def _config(run_id: str) -> dict:
    return {"configurable": {"thread_id": run_id}}


def _get_state_values(run_id: str) -> dict:
    state = _graph.get_state(_config(run_id))
    if state is None or not state.values:
        raise HTTPException(status_code=404, detail="run not found")
    return state.values


@app.post("/research")
def start_research(req: ResearchRequest) -> dict:
    run_id = str(uuid.uuid4())
    config = _config(run_id)
    try:
        _graph.invoke(
            {"topic": req.topic, "run_id": run_id, "budget_mode": req.budget_mode},
            config=config,
        )
    except Exception:
        logger.exception("research run %s failed before the first checkpoint", run_id)
        mark_run_failed(_graph, config)
        raise HTTPException(status_code=500, detail="failed to start research run") from None
    values = _get_state_values(run_id)
    return {"run_id": run_id, "status": values.get("status", "unknown")}


@app.get("/research/{run_id}")
def get_research(run_id: str) -> dict:
    state = _graph.get_state(_config(run_id))
    if state is None or not state.values:
        raise HTTPException(status_code=404, detail="run not found")
    return {
        "run_id": run_id,
        "status": state.values.get("status", "unknown"),
        "final_report": state.values.get("final_report"),
        "evidence_map": state.values.get("evidence_map"),
        "awaiting_approval": bool(state.next and "human_approval" in state.next),
    }


@app.get("/research/{run_id}/stream")
async def stream_research(run_id: str):
    async def event_source():
        last_status = None
        while True:
            # get_state() is a blocking, synchronous checkpointer call (a real Postgres
            # round-trip). This coroutine runs directly on the event loop (unlike the sync
            # `def` route handlers above, which Starlette auto-offloads to a thread pool),
            # so calling it inline here would stall every other request this process is
            # serving for the duration of each DB round-trip.
            state = await asyncio.to_thread(_graph.get_state, _config(run_id))
            status = state.values.get("status") if state and state.values else None
            if status != last_status:
                yield f"data: {json.dumps({'run_id': run_id, 'status': status})}\n\n"
                last_status = status
            if status in ("done", "failed"):
                break
            await asyncio.sleep(1.0)

    return StreamingResponse(event_source(), media_type="text/event-stream")


@app.post("/research/{run_id}/approve-plan")
def approve_plan(run_id: str, req: ApprovalRequest) -> dict:
    _get_state_values(run_id)  # 404s if the run doesn't exist
    config = _config(run_id)
    try:
        _graph.invoke(
            Command(resume={"approved": req.approved, "feedback": req.feedback}),
            config=config,
        )
    except Exception:
        logger.exception("research run %s failed after plan approval", run_id)
        mark_run_failed(_graph, config)
        raise HTTPException(status_code=500, detail="run failed after approval") from None
    values = _get_state_values(run_id)
    return {"run_id": run_id, "status": values.get("status", "unknown")}
