"""Tests verifying that runtime assert statements were replaced with ValueError raises.

Each test confirms that passing a model instance with id=None raises ValueError
instead of AssertionError (which would be silently suppressed with python -O).
"""

import sqlite3
from pathlib import Path

import pytest

from app.analysis.knowledge import initialize_knowledge, update_knowledge
from app.checker import check_topic
from app.config import LLMSettings, Settings
from app.crud import update_knowledge_state, update_topic
from app.database import get_connection, init_db
from app.models import KnowledgeState, Topic


def _make_settings() -> Settings:
    return Settings(llm=LLMSettings(model="openai/gpt-4o-mini", api_key="test-key"))


def _topic_without_id() -> Topic:
    return Topic(name="Test", description="Test topic", id=None)


def _knowledge_state_without_id() -> KnowledgeState:
    return KnowledgeState(id=None, topic_id=1, summary_text="summary", token_count=10)


@pytest.fixture
def db_conn(tmp_path: Path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        yield conn
    finally:
        conn.close()


def test_update_topic_raises_value_error_when_id_is_none(db_conn: sqlite3.Connection):
    topic = _topic_without_id()
    with pytest.raises(ValueError, match="Cannot update a topic without an ID"):
        update_topic(db_conn, topic)


def test_update_knowledge_state_raises_value_error_when_id_is_none(db_conn: sqlite3.Connection):
    state = _knowledge_state_without_id()
    with pytest.raises(ValueError, match="Cannot update a knowledge state without an ID"):
        update_knowledge_state(db_conn, state)


@pytest.mark.asyncio
async def test_check_topic_raises_value_error_when_id_is_none(db_conn: sqlite3.Connection):
    topic = _topic_without_id()
    settings = _make_settings()
    with pytest.raises(ValueError, match="Topic must have an ID"):
        await check_topic(topic, db_conn, settings)


@pytest.mark.asyncio
async def test_initialize_knowledge_raises_value_error_when_id_is_none(db_conn: sqlite3.Connection):
    topic = _topic_without_id()
    settings = _make_settings()
    with pytest.raises(ValueError, match="Topic must have an ID"):
        await initialize_knowledge(topic, [], db_conn, settings)


@pytest.mark.asyncio
async def test_update_knowledge_raises_value_error_when_id_is_none(db_conn: sqlite3.Connection):
    topic = _topic_without_id()
    settings = _make_settings()
    # NoveltyResult is needed; use a mock to avoid any LLM calls
    from app.analysis.llm import NoveltyResult

    novelty_result = NoveltyResult(has_new_info=False, summary="none", confidence=0.0)
    with pytest.raises(ValueError, match="Topic must have an ID"):
        await update_knowledge(topic, novelty_result, db_conn, settings)
