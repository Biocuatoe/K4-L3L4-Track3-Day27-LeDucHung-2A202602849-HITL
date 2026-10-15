"""Checkpointer helpers with an explicit msgpack allow-list.

Graph state contains our Pydantic models. LangGraph warns (and will eventually
refuse) when it deserializes an unregistered type from a checkpoint, so the
project's own types are registered explicitly — no unsafe/permissive mode.
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

import aiosqlite
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

ALLOWED_TYPES = [("common.schemas", "PRAnalysis"), ("common.schemas", "ReviewComment")]


def make_serde() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_TYPES)


def memory_saver() -> InMemorySaver:
    return InMemorySaver(serde=make_serde())


@asynccontextmanager
async def sqlite_saver(path: str) -> AsyncIterator[AsyncSqliteSaver]:
    """Equivalent to AsyncSqliteSaver.from_conn_string(path) plus the allow-list."""
    async with aiosqlite.connect(path) as conn:
        yield AsyncSqliteSaver(conn, serde=make_serde())
