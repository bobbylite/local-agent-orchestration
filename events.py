#!/usr/bin/env python3
"""Event bus shared by the CLI tools and the web dashboard.

Every pipeline stage in `quick_question.py` / `quick_build.py` emits structured
events here. Two consumers pick them up:

* **In-process sinks** — the dashboard server registers one when it drives a run
  itself, so tokens reach the browser with no disk round-trip.
* **The run log** — a JSONL file every event is appended to, which the server
  tails. That is what makes a run you started in a *terminal* show up live in an
  already-open dashboard.

Both paths carry the same events; the server skips log lines for runs it owns so
nothing is delivered twice. Emitting is always safe: with no sinks registered and
the log unwritable, `emit()` is a no-op, so the CLI keeps working on its own.
"""

from __future__ import annotations

import contextvars
import json
import os
import threading
import time
import uuid
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

RUN_LOG: Final[Path] = Path(
    os.environ.get("QUICK_AGENTS_HOME", Path.home() / ".quick-agents")
) / "runs.jsonl"

# Tokens arrive far faster than any UI can paint them, so they are coalesced into
# batches before being emitted. Whichever bound is hit first wins.
TOKEN_FLUSH_SECONDS: Final[float] = 0.12
TOKEN_FLUSH_CHARS: Final[int] = 160


class Kind(StrEnum):
    RUN_STARTED = "run_started"
    STAGE_STARTED = "stage_started"
    STAGE_FINISHED = "stage_finished"
    TOKEN = "token"
    TOOL_CALL = "tool_call"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_RESOLVED = "approval_resolved"
    VERDICT = "verdict"
    RUN_FINISHED = "run_finished"
    RUN_FAILED = "run_failed"


@dataclass(slots=True)
class Event:
    run_id: str
    seq: int
    ts: float
    kind: Kind
    payload: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self) | {"kind": str(self.kind)}, default=str)

    @classmethod
    def from_json(cls, line: str) -> "Event | None":
        try:
            raw = json.loads(line)
            return cls(
                run_id=str(raw["run_id"]),
                seq=int(raw["seq"]),
                ts=float(raw["ts"]),
                kind=Kind(raw["kind"]),
                payload=dict(raw.get("payload") or {}),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None  # a torn final line from a concurrent append; skip it


Sink = Callable[[Event], None]

_sinks: Final[list[Sink]] = []
_sink_lock: Final = threading.Lock()
_write_lock: Final = threading.Lock()
_seq_lock: Final = threading.Lock()
_seq = 0

# The run a stage belongs to travels in a ContextVar rather than a global so that
# concurrent runs in the dashboard's event loop each keep their own identity.
_run_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("_run_id", default=None)
_stage: contextvars.ContextVar[str] = contextvars.ContextVar("_stage", default="")

# Runs this process is driving, newest last. The dashboard consults this to skip
# log lines for runs it already delivered in-process. Entries are deliberately
# *not* removed when a run ends: the log tailer reads behind real time, so a
# just-finished run's trailing lines still need to be recognised as local. The
# cap keeps a long-lived server from growing the registry without bound.
_LOCAL_RUN_CAP: Final[int] = 2048
_local_runs: Final[dict[str, None]] = {}


def mark_local(run_id: str) -> None:
    _local_runs[run_id] = None
    while len(_local_runs) > _LOCAL_RUN_CAP:
        _local_runs.pop(next(iter(_local_runs)))


def is_local(run_id: str) -> bool:
    return run_id in _local_runs

_token_buffer: Final[dict[str, list[str]]] = {}
_token_flushed_at: Final[dict[str, float]] = {}

# A tool that needs the user's blessing (currently only write_file) asks through
# here. The dashboard installs an approver that resolves from a browser click; on
# the CLI none is installed and the caller falls back to its terminal prompt.
Approver = Callable[[dict[str, Any]], Awaitable[bool]]
_approver: contextvars.ContextVar[Approver | None] = contextvars.ContextVar("_approver", default=None)


def set_approver(approver: Approver | None) -> None:
    _approver.set(approver)


def has_approver() -> bool:
    return _approver.get() is not None


async def request_approval(**details: Any) -> bool:
    """Ask the registered approver. Denies if none is installed."""
    approver = _approver.get()
    if approver is None:
        return False
    return await approver(details)


def add_sink(sink: Sink) -> Callable[[], None]:
    """Register an in-process consumer. Returns a callable that unregisters it."""
    with _sink_lock:
        _sinks.append(sink)

    def remove() -> None:
        with _sink_lock:
            if sink in _sinks:
                _sinks.remove(sink)

    return remove


def new_run(request: str, tool: str, pipeline: list[str]) -> str:
    """Open a run, bind it to the current context, and announce it."""
    run_id = uuid.uuid4().hex[:12]
    _run_id.set(run_id)
    mark_local(run_id)
    emit(Kind.RUN_STARTED, request=request, tool=tool, pipeline=pipeline)
    return run_id


def current_run() -> str | None:
    return _run_id.get()


def set_stage(stage: str) -> None:
    _stage.set(stage)


def emit(kind: Kind, **payload: Any) -> None:
    """Publish one event. Never raises — telemetry must not break a pipeline."""
    run_id = _run_id.get()
    if run_id is None:
        return
    if kind is not Kind.TOKEN:
        _flush_tokens(run_id)  # keep ordering: buffered text precedes what follows it
    payload.setdefault("stage", _stage.get())
    global _seq
    with _seq_lock:
        _seq += 1
        seq = _seq
    _dispatch(Event(run_id=run_id, seq=seq, ts=time.time(), kind=kind, payload=payload))


def emit_token(text: str) -> None:
    """Buffer streamed text, flushing on the size/age bounds above."""
    run_id = _run_id.get()
    if run_id is None or not text:
        return
    buffer = _token_buffer.setdefault(run_id, [])
    buffer.append(text)
    now = time.monotonic()
    started = _token_flushed_at.setdefault(run_id, now)
    if now - started >= TOKEN_FLUSH_SECONDS or sum(map(len, buffer)) >= TOKEN_FLUSH_CHARS:
        _flush_tokens(run_id)


def _flush_tokens(run_id: str) -> None:
    buffer = _token_buffer.pop(run_id, None)
    _token_flushed_at.pop(run_id, None)
    if not buffer:
        return
    global _seq
    with _seq_lock:
        _seq += 1
        seq = _seq
    _dispatch(
        Event(
            run_id=run_id,
            seq=seq,
            ts=time.time(),
            kind=Kind.TOKEN,
            payload={"stage": _stage.get(), "text": "".join(buffer)},
        )
    )


def _dispatch(event: Event) -> None:
    with _sink_lock:
        sinks = list(_sinks)
    for sink in sinks:
        try:
            sink(event)
        except Exception:  # noqa: BLE001 - a broken subscriber must not stop the run
            pass
    _append_to_log(event)


def _append_to_log(event: Event) -> None:
    try:
        with _write_lock:
            RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
            with RUN_LOG.open("a", encoding="utf-8") as handle:
                handle.write(event.to_json() + "\n")
    except OSError:
        pass  # a read-only home is not a reason to fail the run


def read_log(limit_bytes: int = 4_000_000) -> Iterator[Event]:
    """Replay the tail of the run log, oldest first, for dashboard cold starts."""
    try:
        size = RUN_LOG.stat().st_size
        with RUN_LOG.open("rb") as handle:
            if size > limit_bytes:
                handle.seek(size - limit_bytes)
                handle.readline()  # discard the partial line we landed inside
            for raw in handle:
                event = Event.from_json(raw.decode("utf-8", "replace"))
                if event is not None:
                    yield event
    except OSError:
        return


def log_size() -> int:
    try:
        return RUN_LOG.stat().st_size
    except OSError:
        return 0
