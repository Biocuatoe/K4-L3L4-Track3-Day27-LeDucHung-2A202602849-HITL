"""Service layer between the Streamlit UI (app.py) and the Exercise 4 graph.

All graph I/O lives here so it can be tested without a browser. Each call opens
the durable `AsyncSqliteSaver` on the shared SQLite file, so state survives
Streamlit reruns, browser reloads and process restarts. The graph's `thread_id`
is chosen once by `start_review` and reused by every later call.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from langgraph.types import Command

from common.checkpoint import sqlite_saver
from common.db import db_path
from common.github import parse_pr_url
from common.review import classify_outcome, pending_interrupt, validate_answers, validate_choice
from exercises.exercise_4_audit import build_graph


class StaleResumeError(RuntimeError):
    """The session is not waiting for the input we were asked to submit."""


@dataclass
class SessionView:
    thread_id: str
    # interrupted: waiting on a human | completed: graph finished
    # stalled: stopped mid-graph without an interrupt (e.g. an error) | empty: unknown thread
    status: str
    payload: dict | None = None
    values: dict[str, Any] = field(default_factory=dict)

    @property
    def outcome(self) -> str:
        return classify_outcome(self.values.get("final_action")) if self.status == "completed" else "pending"


@asynccontextmanager
async def _graph() -> AsyncIterator:
    async with sqlite_saver(db_path()) as cp:
        await cp.setup()
        yield build_graph(cp)


def _cfg(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


async def _view(app, thread_id: str) -> SessionView:
    snap = await app.aget_state(_cfg(thread_id))
    if not snap.values:
        return SessionView(thread_id, "empty")
    values = dict(snap.values)
    if not snap.next:
        return SessionView(thread_id, "completed", None, values)
    payload = pending_interrupt(snap)
    return SessionView(thread_id, "interrupted" if payload else "stalled", payload, values)


def new_thread_id() -> str:
    return str(uuid.uuid4())


async def start_review(pr_url: str, thread_id: str, reviewer: str | None = None) -> SessionView:
    """Run the graph until it finishes or pauses. Raises ValueError on a malformed PR URL."""
    parse_pr_url(pr_url)
    async with _graph() as app:
        if (await _view(app, thread_id)).status != "empty":
            raise StaleResumeError(f"Thread {thread_id} already exists; resume it instead")
        await app.ainvoke(
            {"pr_url": pr_url.strip(), "thread_id": thread_id, "reviewer_id": reviewer or None},
            _cfg(thread_id),
        )
        return await _view(app, thread_id)


def validate_resume(payload: dict, value: Any) -> Any:
    """Validate a resume value against the pending interrupt's kind."""
    kind = payload.get("kind")
    if kind == "approval_request":
        return validate_choice(value)
    if kind == "final_confirmation":
        return validate_choice(value, ("approve", "reject"))
    if kind == "escalation":
        return validate_answers(payload["questions"], value)
    raise ValueError(f"Unknown interrupt kind: {kind!r}")


async def resume_review(thread_id: str, value: Any, expected_kind: str | None = None) -> SessionView:
    """Resume the *same* thread with a validated answer.

    Refuses (StaleResumeError) when the thread is not paused on an interrupt of
    `expected_kind` — e.g. a double-click or a stale browser tab — so a decision
    can never be applied twice.
    """
    async with _graph() as app:
        view = await _view(app, thread_id)
        if view.status != "interrupted" or view.payload is None:
            raise StaleResumeError(f"Session is not waiting for input (status={view.status})")
        if expected_kind and view.payload.get("kind") != expected_kind:
            raise StaleResumeError(
                f"Session is waiting for {view.payload.get('kind')!r}, not {expected_kind!r}"
            )
        clean = validate_resume(view.payload, value)
        await app.ainvoke(Command(resume=clean), _cfg(thread_id))
        return await _view(app, thread_id)


async def continue_session(thread_id: str) -> SessionView:
    """Re-run a stalled session from its last checkpoint (e.g. after a transient LLM error)."""
    async with _graph() as app:
        view = await _view(app, thread_id)
        if view.status != "stalled":
            return view
        await app.ainvoke(None, _cfg(thread_id))
        return await _view(app, thread_id)


async def load_session(thread_id: str) -> SessionView:
    async with _graph() as app:
        return await _view(app, thread_id)
