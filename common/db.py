"""SQLite helpers: connection + audit-event writer.

The `audit_events` table lives in the same .db file as the LangGraph
SQLite checkpointer tables — different schemas, same file.

`init_schema()` is idempotent and runs on the first connection, so students
don't need a separate setup step.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Iterable

import aiosqlite

from common.schemas import AuditEntry

SCHEMA_FILE = Path(__file__).resolve().parent.parent / "audit" / "schema.sql"


def db_path() -> str:
    """Return the SQLite file path (override with HITL_DB_PATH env var)."""
    return os.environ.get("HITL_DB_PATH", "hitl_audit.db")


async def _ensure_schema(conn: aiosqlite.Connection) -> None:
    """Create / migrate `audit_events`. Safe to call on every connection."""
    async with conn.execute("PRAGMA table_info(audit_events)") as cur:
        cols = {r[1] for r in await cur.fetchall()}
    if cols and "event_key" not in cols:
        # Database created by an older version of the schema.
        await conn.execute("ALTER TABLE audit_events ADD COLUMN event_key TEXT")
    await conn.executescript(SCHEMA_FILE.read_text(encoding="utf-8"))  # all IF NOT EXISTS
    await conn.commit()


@asynccontextmanager
async def db_conn() -> AsyncIterator[aiosqlite.Connection]:
    """Open an aiosqlite connection, applying schema if first use."""
    conn = await aiosqlite.connect(db_path())
    conn.row_factory = aiosqlite.Row
    try:
        await _ensure_schema(conn)
        yield conn
    finally:
        await conn.close()


async def write_audit_event(
    *,
    thread_id: str,
    pr_url: str,
    entry: AuditEntry,
    event_key: str | None = None,
) -> bool:
    """Append one structured audit row. Returns False if it was a duplicate.

    If `event_key` is given, the row is inserted at most once per
    (thread_id, event_key) — this makes writes made before an `interrupt()`
    safe when LangGraph re-executes the node on resume.

    `thread_id` and `pr_url` are session-context columns (used for grouping
    and filtering); all other fields come from the `AuditEntry` so they map
    1-to-1 with first-class SQL columns.
    """
    async with db_conn() as conn:
        cur = await conn.execute(
            """
            INSERT OR IGNORE INTO audit_events (
                timestamp, thread_id, pr_url,
                agent_id, action, confidence, risk_level,
                reviewer_id, decision, reason, execution_time_ms, event_key
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.timestamp.isoformat(), thread_id, pr_url,
                entry.agent_id, entry.action, entry.confidence, entry.risk_level,
                entry.reviewer_id, entry.decision, entry.reason, entry.execution_time_ms, event_key,
            ),
        )
        await conn.commit()
        return cur.rowcount == 1


async def replay_events(thread_id: str) -> list[dict[str, Any]]:
    """Return every event for a thread, ordered by time. Used by audit/replay.py."""
    async with db_conn() as conn:
        async with conn.execute(
            """
            SELECT id, timestamp, agent_id, action, confidence, risk_level,
                   reviewer_id, decision, reason, execution_time_ms
              FROM audit_events
             WHERE thread_id = ?
             ORDER BY id
            """,
            (thread_id,),
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


RISK_RANK = {"low": 1, "med": 2, "high": 3}


def worst_risk(levels: Iterable[str]) -> str | None:
    """Highest severity among `levels` (high > med > low), not lexical order."""
    return max(levels, key=lambda r: RISK_RANK.get(r, 0), default=None)


async def list_sessions(limit: int = 25) -> list[dict[str, Any]]:
    """Recent review sessions, newest first, with explicitly ranked worst risk."""
    async with db_conn() as conn:
        async with conn.execute(
            """
            SELECT thread_id, pr_url,
                   MIN(timestamp) AS started,
                   MAX(timestamp) AS last_event_at,
                   COUNT(*)       AS events,
                   MAX(CASE risk_level WHEN 'high' THEN 3 WHEN 'med' THEN 2
                                       WHEN 'low' THEN 1 ELSE 0 END) AS risk_rank,
                   (SELECT e.action   FROM audit_events e
                     WHERE e.thread_id = a.thread_id ORDER BY e.id DESC LIMIT 1) AS last_action,
                   (SELECT e.decision FROM audit_events e
                     WHERE e.thread_id = a.thread_id ORDER BY e.id DESC LIMIT 1) AS last_decision
              FROM audit_events a
             GROUP BY thread_id, pr_url
             ORDER BY MAX(id) DESC
             LIMIT ?
            """,
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
    names = {v: k for k, v in RISK_RANK.items()}
    out = []
    for r in rows:
        d = dict(r)
        d["worst_risk"] = names.get(d.pop("risk_rank"), "-")
        out.append(d)
    return out
