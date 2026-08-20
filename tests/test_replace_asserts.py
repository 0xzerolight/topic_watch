"""Tests verifying that runtime assert statements were replaced with ValueError raises.

Each test confirms that passing a model instance with id=None raises ValueError
instead of AssertionError (which would be silently suppressed with python -O).
"""

import sqlite3
from pathlib import Path

import pytest

from app.analysis.knowledge import KnowledgeUpdatePlan
from app.checker import (
    CheckOutcome,
    TopicSnapshot,
    _commit_check_transition,
    _commit_init_transition,
    check_topic,
)
from app.config import LLMSettings, Settings
from app.crud import update_knowledge_state, update_topic
from app.database import get_connection, init_db
from app.models import CheckResult, KnowledgeState, Topic


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
async def test_check_topic_raises_value_error_when_id_is_none(db_conn: sqlite3.Connection, db_path: Path):
    topic = _topic_without_id()
    settings = _make_settings()
    with pytest.raises(ValueError, match="Topic must have an ID"):
        await check_topic(topic, settings, db_path=db_path)


def _snapshot_without_id() -> TopicSnapshot:
    return TopicSnapshot(topic=_topic_without_id(), generation="gen", knowledge_version=0, knowledge_summary="")


def test_commit_check_transition_raises_value_error_when_id_is_none(db_conn: sqlite3.Connection):
    """The durable write refuses an id-less topic (the knowledge helpers' old guard)."""
    outcome = CheckOutcome(result=CheckResult(topic_id=1))
    with pytest.raises(ValueError, match="Topic must have an ID"):
        _commit_check_transition(db_conn, _snapshot_without_id(), outcome, settings=_make_settings())


def test_commit_init_transition_raises_value_error_when_id_is_none(db_conn: sqlite3.Connection):
    plan = KnowledgeUpdatePlan(summary_text="s", token_count=1)
    with pytest.raises(ValueError, match="Topic must have an ID"):
        _commit_init_transition(db_conn, _snapshot_without_id(), plan, [], _make_settings())
