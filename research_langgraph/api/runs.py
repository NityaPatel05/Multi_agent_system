"""In-process run registry: runs the graph on a bounded worker pool and records an event
log per run for the SSE endpoint.

The checkpointer stays the source of truth for state; this registry adds (a) a `queued`/
`running` status before the first checkpoint exists, (b) a per-node event stream, and (c)
a guard so the same run can't be executed twice concurrently. It is per-process memory: after
a restart, old runs are still readable from the checkpoint, just without their event log.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from research_langgraph.main import mark_run_failed

logger = logging.getLogger(__name__)

MAX_CONCURRENT_RUNS = int(os.environ.get("MAX_CONCURRENT_RUNS", "4"))
_MAX_TRACKED_RUNS = 500
_MAX_AUDIT_PER_EVENT = 20
_DETAIL_MAX = 200


class RunBusy(Exception):
    pass


class RunInfo:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.status = "queued"
        self.running = True
        self.events: List[dict] = []
        self.created = time.time()
        self._lock = threading.Lock()

    def emit(self, type_: str, **data: Any) -> None:
        with self._lock:
            self.events.append({"seq": len(self.events), "run_id": self.run_id, "type": type_,
                                "ts": time.time(), **data})

    def events_since(self, seq: int) -> List[dict]:
        with self._lock:
            return self.events[seq:]


class RunRegistry:
    def __init__(self) -> None:
        self._runs: "OrderedDict[str, RunInfo]" = OrderedDict()
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_RUNS, thread_name_prefix="run")

    def get(self, run_id: str) -> Optional[RunInfo]:
        with self._lock:
            return self._runs.get(run_id)

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for r in self._runs.values() if r.running)

    def submit(self, graph, run_id: str, payload: Any) -> RunInfo:
        """Start (or resume) `run_id` in the background. Raises RunBusy if this run is
        already executing or the server is at its concurrent-run cap."""
        with self._lock:
            info = self._runs.get(run_id)
            if info and info.running:
                raise RunBusy("run is already executing")
            if sum(1 for r in self._runs.values() if r.running) >= MAX_CONCURRENT_RUNS:
                raise RunBusy("server at concurrent-run capacity")
            if info is None:
                info = RunInfo(run_id)
                self._runs[run_id] = info
                while len(self._runs) > _MAX_TRACKED_RUNS:  # evict oldest finished run
                    victim = next((k for k, v in self._runs.items() if not v.running), None)
                    if victim is None:
                        break
                    del self._runs[victim]
            else:
                info.running = True
            info.status = "running"
        self._pool.submit(self._execute, graph, info, payload)
        return info

    def _execute(self, graph, info: RunInfo, payload: Any) -> None:
        config = {"configurable": {"thread_id": info.run_id}}
        info.emit("run_started")
        try:
            for _mode, chunk in graph.stream(payload, config, stream_mode=["tasks"]):
                self._on_task_event(info, chunk)
            snap = graph.get_state(config)
            if snap.next and "human_approval" in snap.next:
                info.status = "awaiting_approval"
                info.emit("awaiting_approval", status="awaiting_approval")
            else:
                info.status = (snap.values or {}).get("status", "done")
                info.emit("done" if info.status == "done" else info.status, status=info.status)
        except Exception:
            logger.exception("research run %s failed", info.run_id)
            mark_run_failed(graph, config)
            info.status = "failed"
            info.emit("failed", status="failed", error="run failed; see server logs")
        finally:
            info.running = False

    @staticmethod
    def _on_task_event(info: RunInfo, chunk: dict) -> None:
        name = chunk.get("name")
        if "result" not in chunk and "error" not in chunk:
            info.emit("node_started", node=name, phase="start", task_id=chunk.get("id"))
            return
        result = chunk.get("result") or {}
        if isinstance(result, list):  # list of (channel, value) pairs in some versions
            result = {k: v for k, v in result if isinstance(k, str)}
        counts = {k: len(v) for k, v in result.items() if isinstance(v, (list, dict))}
        audit = [
            {k: (str(v)[:_DETAIL_MAX] if isinstance(v, str) else v) for k, v in e.items()
             if k in {"node", "kind", "name", "provider", "ok", "detail"}}
            for e in (result.get("audit_log") or [])[:_MAX_AUDIT_PER_EVENT]
            if isinstance(e, dict)
        ]
        info.emit("node_finished", node=name, phase="end", task_id=chunk.get("id"),
                  error=bool(chunk.get("error")), counts=counts, audit=audit)


registry = RunRegistry()
