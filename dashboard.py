#!/usr/bin/env python3
"""Orchestration dashboard — a live web view of the local agent pipelines.

Run it with `uv run dashboard.py`, then open http://127.0.0.1:8787.

Three sources feed the browser over one Server-Sent Events stream:

* **Runs started from the dashboard** — driven in-process by `quick_question`'s
  `run_pipeline`, with an approver installed so `write_file` asks in the browser.
* **Runs started from a terminal** — picked up by tailing the shared run log, so
  a `uv run quick_question.py ...` in another window animates here too.
* **Ollama itself** — polled for its model catalogue and what is resident in VRAM.

There is no authentication; it binds to loopback only and is meant for a single
developer's machine.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import events
import quick_question as qq

# Overridable so launch.sh can pick the port and stay in step with the server.
# Loopback is the default on purpose: there is no authentication here.
HOST: Final[str] = os.environ.get("DASHBOARD_HOST", "127.0.0.1")
PORT: Final[int] = int(os.environ.get("DASHBOARD_PORT", "8787"))
STATIC_DIR: Final[Path] = Path(__file__).parent / "static"

OLLAMA_POLL_SECONDS: Final[float] = 2.0
TAIL_POLL_SECONDS: Final[float] = 0.15
HEARTBEAT_SECONDS: Final[float] = 15.0
APPROVAL_TIMEOUT_SECONDS: Final[float] = 300.0
MAX_RUNS: Final[int] = 60
# A logged run still marked "running" this long after its last event was
# abandoned (a Ctrl-C'd terminal session), not left mid-flight.
STALE_RUN_SECONDS: Final[float] = 600.0
MAX_ANSWER_CHARS: Final[int] = 20_000
MAX_TRANSCRIPT_CHARS: Final[int] = 40_000
SUBSCRIBER_QUEUE_SIZE: Final[int] = 1000


@dataclass(slots=True)
class Stage:
    """One pass through a pipeline node. A retry adds a second `work` stage."""

    name: str
    title: str = ""
    model: str = ""
    state: str = "running"
    started: float = field(default_factory=time.time)
    seconds: float | None = None
    chars: int = 0
    attempt: int | None = None
    score: int | None = None
    verdict: str = ""
    tools: list[dict[str, str]] = field(default_factory=list)
    text: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "title": self.title, "model": self.model, "state": self.state,
            "started": self.started, "seconds": self.seconds, "chars": self.chars,
            "attempt": self.attempt, "score": self.score, "verdict": self.verdict,
            "tools": self.tools, "text": self.text,
        }


@dataclass(slots=True)
class Run:
    id: str
    tool: str = "quick_question"
    request: str = ""
    pipeline: list[str] = field(default_factory=list)
    started: float = field(default_factory=time.time)
    finished: float | None = None
    status: str = "running"
    source: str = ""
    seconds: float | None = None
    worker_model: str = ""
    attempts: int = 0
    score: int | None = None
    error: str = ""
    answer: str = ""
    origin: str = "cli"
    last_ts: float = field(default_factory=time.time)
    stages: list[Stage] = field(default_factory=list)
    pending_approval: dict[str, Any] | None = None

    def active_stage(self, name: str) -> Stage | None:
        for stage in reversed(self.stages):
            if stage.name == name and stage.state == "running":
                return stage
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "tool": self.tool, "request": self.request, "pipeline": self.pipeline,
            "started": self.started, "finished": self.finished, "status": self.status,
            "source": self.source, "seconds": self.seconds, "worker_model": self.worker_model,
            "attempts": self.attempts, "score": self.score, "error": self.error,
            "answer": self.answer, "origin": self.origin,
            "stages": [stage.as_dict() for stage in self.stages],
            "pending_approval": self.pending_approval,
        }


class Hub:
    """Folds events into run state and fans frames out to connected browsers."""

    def __init__(self) -> None:
        self.runs: dict[str, Run] = {}
        self.order: list[str] = []
        self.subscribers: set[asyncio.Queue[str]] = set()
        self.ollama: dict[str, Any] = {"reachable": False, "models": [], "running": [], "checked": 0.0}
        self.approvals: dict[str, asyncio.Future[bool]] = {}
        self.loop: asyncio.AbstractEventLoop | None = None

    # ---------- fan-out ----------

    def subscribe(self) -> asyncio.Queue[str]:
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=SUBSCRIBER_QUEUE_SIZE)
        self.subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[str]) -> None:
        self.subscribers.discard(queue)

    def send(self, frame_type: str, data: Any) -> None:
        """Push one frame to every live browser, dropping it for any that fell behind."""
        payload = json.dumps({"type": frame_type, "data": data}, default=str)
        for queue in list(self.subscribers):
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                self.subscribers.discard(queue)

    def on_event(self, event: events.Event) -> None:
        """Event-bus sink for runs this process drives.

        `emit()` may be called from a worker thread deep inside LangChain, so hop
        onto the event loop before touching run state or subscriber queues.
        """
        if self.loop is None:
            return
        self.loop.call_soon_threadsafe(self._apply_local, event)

    def _apply_local(self, event: events.Event) -> None:
        self._run(event.run_id).origin = "dashboard"
        self.apply(event, live=True)

    # ---------- folding ----------

    def _run(self, run_id: str) -> Run:
        run = self.runs.get(run_id)
        if run is None:
            run = Run(id=run_id)
            self.runs[run_id] = run
            self.order.append(run_id)
            while len(self.order) > MAX_RUNS:
                self.runs.pop(self.order.pop(0), None)
        return run

    def apply(self, event: events.Event, *, live: bool) -> None:
        """Update run state from one event, and (when live) forward it to browsers."""
        run = self._run(event.run_id)
        run.last_ts = event.ts
        payload = event.payload
        stage_name = str(payload.get("stage") or "")
        kind = event.kind

        if kind is events.Kind.RUN_STARTED:
            run.request = str(payload.get("request", ""))
            run.tool = str(payload.get("tool", "quick_question"))
            run.pipeline = list(payload.get("pipeline") or [])
            run.started = event.ts
            run.status = "running"
        elif kind is events.Kind.STAGE_STARTED:
            stage = run.active_stage(stage_name)
            if stage is None:
                stage = Stage(name=stage_name)
                run.stages.append(stage)
            stage.title = str(payload.get("title") or stage.title)
            stage.model = str(payload.get("model") or stage.model)
            if payload.get("attempt") is not None:
                stage.attempt = int(payload["attempt"])
                run.attempts = max(run.attempts, stage.attempt)
            if stage.model and stage_name == "work":
                run.worker_model = stage.model
        elif kind is events.Kind.TOKEN:
            stage = run.active_stage(stage_name)
            if stage is not None:
                stage.text = (stage.text + str(payload.get("text", "")))[-MAX_TRANSCRIPT_CHARS:]
        elif kind is events.Kind.TOOL_CALL:
            stage = run.active_stage(stage_name) or (run.stages[-1] if run.stages else None)
            if stage is not None:
                stage.tools.append({"tool": str(payload.get("tool", "")), "detail": str(payload.get("detail", ""))})
        elif kind is events.Kind.VERDICT:
            run.score = payload.get("score")
            stage = run.active_stage(stage_name) or (run.stages[-1] if run.stages else None)
            if stage is not None:
                stage.score = payload.get("score")
                stage.verdict = str(payload.get("verdict", ""))
        elif kind is events.Kind.STAGE_FINISHED:
            stage = run.active_stage(stage_name)
            if stage is not None:
                stage.state = "failed" if stage.verdict == "retry" else "done"
                if payload.get("seconds") is not None:
                    stage.seconds = float(payload["seconds"])
                if payload.get("chars"):
                    stage.chars = int(payload["chars"])
                if payload.get("chose"):
                    stage.model = str(payload["chose"])
        elif kind is events.Kind.APPROVAL_REQUESTED:
            run.pending_approval = {
                "run_id": run.id, "path": str(payload.get("path", "")),
                "bytes": int(payload.get("bytes") or 0), "lexer": str(payload.get("lexer", "text")),
                "content": str(payload.get("content", ""))[:MAX_ANSWER_CHARS],
            }
        elif kind is events.Kind.APPROVAL_RESOLVED:
            run.pending_approval = None
        elif kind is events.Kind.RUN_FINISHED:
            run.status = "done"
            run.finished = event.ts
            run.source = str(payload.get("source", ""))
            run.seconds = payload.get("seconds")
            run.worker_model = str(payload.get("worker_model") or run.worker_model)
            run.attempts = int(payload.get("attempts") or run.attempts)
            run.answer = str(payload.get("answer", ""))[:MAX_ANSWER_CHARS]
            run.pending_approval = None
            for stage in run.stages:
                if stage.state == "running":
                    stage.state = "done"
        elif kind is events.Kind.RUN_FAILED:
            run.status = "failed"
            run.finished = event.ts
            run.error = str(payload.get("error", ""))
            run.pending_approval = None
            for stage in run.stages:
                if stage.state == "running":
                    stage.state = "failed"

        if live:
            self.send("event", {"run_id": event.run_id, "kind": str(kind), "ts": event.ts, "payload": payload})

    def mark_stale_runs(self) -> None:
        """Retire runs the log shows as still open but that clearly never closed."""
        cutoff = time.time() - STALE_RUN_SECONDS
        for run in self.runs.values():
            if run.status == "running" and run.last_ts < cutoff:
                run.status = "stale"
                for stage in run.stages:
                    if stage.state == "running":
                        stage.state = "stale"

    def snapshot(self) -> dict[str, Any]:
        return {
            "runs": [self.runs[run_id].as_dict() for run_id in self.order if run_id in self.runs],
            "ollama": self.ollama,
            "now": time.time(),
            "config": {
                "worker_models": qq.WORKER_MODELS,
                "default_worker": qq.DEFAULT_WORKER_MODEL,
                "bump_worker": qq.BUMP_WORKER_MODEL,
                "router": qq.ROUTER_MODEL,
                "judge": qq.JUDGE_MODEL,
                "escalation": qq.ESCALATION_MODEL,
                "max_attempts": qq.MAX_LOCAL_ATTEMPTS,
                "accept_threshold": qq.JUDGE_ACCEPT_THRESHOLD,
            },
        }


hub: Final = Hub()


# ---------------------------------------------------------------- background tasks


async def tail_run_log() -> None:
    """Replay the log's tail, then follow it for runs started outside this process."""
    handle = None
    try:
        while True:
            if handle is None:
                try:
                    handle = events.RUN_LOG.open("r", encoding="utf-8", errors="replace")
                except OSError:
                    await asyncio.sleep(1.0)
                    continue
                size = events.log_size()
                if size > 4_000_000:
                    handle.seek(size - 4_000_000)
                    handle.readline()  # drop the partial line we landed inside
                for line in handle:
                    event = events.Event.from_json(line)
                    if event is not None:
                        hub.apply(event, live=False)  # cold history: fold, don't animate
                hub.mark_stale_runs()
                hub.send("snapshot", hub.snapshot())

            position = handle.tell()
            line = handle.readline()
            if not line:
                if not line.endswith("\n") and line:
                    handle.seek(position)  # a partial append; re-read it whole next tick
                try:
                    if events.log_size() < position:
                        handle.close()
                        handle = None  # log was truncated or replaced; start over
                        continue
                except OSError:
                    pass
                await asyncio.sleep(TAIL_POLL_SECONDS)
                continue
            if not line.endswith("\n"):
                handle.seek(position)
                await asyncio.sleep(TAIL_POLL_SECONDS)
                continue
            event = events.Event.from_json(line)
            # Runs this process drives already reached the browser via the sink.
            if event is not None and not events.is_local(event.run_id):
                hub.apply(event, live=True)
    except asyncio.CancelledError:
        raise
    finally:
        if handle is not None:
            handle.close()


async def poll_ollama() -> None:
    """Watch the model catalogue and VRAM residency, announcing every change."""
    async with httpx.AsyncClient(timeout=4.0) as client:
        was_reachable: bool | None = None
        resident: set[str] = set()
        while True:
            state: dict[str, Any] = {"reachable": False, "models": [], "running": [], "checked": time.time()}
            try:
                tags, running = await asyncio.gather(
                    client.get(f"{qq.OLLAMA_URL}/api/tags"),
                    client.get(f"{qq.OLLAMA_URL}/api/ps"),
                )
                tags.raise_for_status()
                running.raise_for_status()
                state["reachable"] = True
                state["models"] = tags.json().get("models") or []
                state["running"] = running.json().get("models") or []
            except (httpx.HTTPError, json.JSONDecodeError, ValueError):
                pass

            if was_reachable is not None and state["reachable"] != was_reachable:
                hub.send("notice", {
                    "level": "success" if state["reachable"] else "error",
                    "title": "Ollama reconnected" if state["reachable"] else "Ollama unreachable",
                    "body": qq.OLLAMA_URL,
                })
            was_reachable = state["reachable"]

            now_resident = {str(model.get("name", "")) for model in state["running"]}
            if state["reachable"]:
                for name in sorted(now_resident - resident):
                    hub.send("notice", {"level": "info", "title": "Model loaded into VRAM", "body": name})
                for name in sorted(resident - now_resident):
                    hub.send("notice", {"level": "muted", "title": "Model evicted from VRAM", "body": name})
                resident = now_resident

            hub.ollama = state
            hub.send("ollama", state)
            await asyncio.sleep(OLLAMA_POLL_SECONDS)


async def heartbeat() -> None:
    """Keep idle SSE connections (and any proxy between) from timing out."""
    while True:
        await asyncio.sleep(HEARTBEAT_SECONDS)
        hub.send("ping", {"now": time.time()})


# ---------------------------------------------------------------- app


@contextlib.asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    hub.loop = asyncio.get_running_loop()
    remove_sink = events.add_sink(hub.on_event)
    tasks = [asyncio.create_task(coro()) for coro in (tail_run_log, poll_ollama, heartbeat)]
    try:
        yield
    finally:
        remove_sink()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


app: Final = FastAPI(title="Agent Orchestration Dashboard", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str


class ApprovalRequest(BaseModel):
    allow: bool


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/snapshot")
async def snapshot() -> JSONResponse:
    return JSONResponse(hub.snapshot())


@app.get("/api/stream")
async def stream() -> StreamingResponse:
    async def frames() -> AsyncIterator[str]:
        queue = hub.subscribe()
        try:
            yield f"data: {json.dumps({'type': 'snapshot', 'data': hub.snapshot()}, default=str)}\n\n"
            while True:
                payload = await queue.get()
                yield f"data: {payload}\n\n"
        finally:
            hub.unsubscribe(queue)

    return StreamingResponse(
        frames(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.post("/api/ask")
async def ask(body: AskRequest) -> JSONResponse:
    question = body.question.strip()
    if not question:
        return JSONResponse({"error": "empty question"}, status_code=400)
    asyncio.create_task(_drive_run(question))
    return JSONResponse({"ok": True})


@app.post("/api/approve/{run_id}")
async def approve(run_id: str, body: ApprovalRequest) -> JSONResponse:
    future = hub.approvals.pop(run_id, None)
    if future is None or future.done():
        return JSONResponse({"error": "no pending approval"}, status_code=404)
    future.set_result(body.allow)
    return JSONResponse({"ok": True})


async def _drive_run(question: str) -> None:
    """Run the question pipeline in-process, streaming straight to the browser."""
    events.set_approver(_approve_from_browser)
    try:
        await qq.run_pipeline(question)
    except Exception as exc:  # noqa: BLE001 - surfaced in the UI, must not kill the server
        hub.send("notice", {"level": "error", "title": "Run failed", "body": str(exc)[:300]})
    finally:
        run_id = events.current_run()
        if run_id is not None:
            hub.approvals.pop(run_id, None)


async def _approve_from_browser(details: dict[str, Any]) -> bool:
    run_id = events.current_run() or ""
    loop = asyncio.get_running_loop()
    future: asyncio.Future[bool] = loop.create_future()
    hub.approvals[run_id] = future
    try:
        return await asyncio.wait_for(future, timeout=APPROVAL_TIMEOUT_SECONDS)
    except (TimeoutError, asyncio.TimeoutError):
        hub.send("notice", {"level": "warn", "title": "Write request timed out", "body": details.get("path", "")})
        return False
    finally:
        hub.approvals.pop(run_id, None)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def main() -> None:
    print(f"\n  ◆ Orchestration dashboard → http://{HOST}:{PORT}\n")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
