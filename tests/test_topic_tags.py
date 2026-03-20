"""Tests for topic tags/categories feature."""

import sqlite3
from collections.abc import AsyncGenerator
from pathlib import Path

import httpx
import pytest

from app.config import LLMSettings, NotificationSettings, Settings
from app.crud import create_topic, get_topic, list_topics, update_topic
from app.database import get_connection
from app.main import app
from app.migrations.m006_topic_tags import up as m006_up
from app.models import FeedMode, Topic, TopicStatus
from app.web.dependencies import get_db_conn, get_settings

CSRF_TEST_TOKEN = "test-csrf-token-for-tests"


def _make_settings(**overrides) -> Settings:
    defaults = {
        "llm": LLMSettings(model="openai/gpt-4o-mini", api_key="test-key-12345678"),
        "notifications": NotificationSettings(urls=["json://localhost"]),
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _make_topic(
    conn: sqlite3.Connection,
    name: str,
    tags: list[str] | None = None,
    status: TopicStatus = TopicStatus.READY,
) -> Topic:
    topic = create_topic(
        conn,
        Topic(
            name=name,
            description=f"Description for {name}",
            feed_urls=["https://example.com/feed.xml"],
            feed_mode=FeedMode.MANUAL,
            status=status,
            tags=tags or [],
        ),
    )
    conn.commit()
    return topic


@pytest.fixture
async def client(
    db_conn: sqlite3.Connection,
) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Create a test client with database dependency overridden."""
    settings = _make_settings()

    def override_db():
        yield db_conn

    def override_settings():
        return settings

    app.dependency_overrides[get_db_conn] = override_db
    app.dependency_overrides[get_settings] = override_settings

    # Set db_path so background tasks (_run_init) use the test database
    db_path_str = db_conn.execute("PRAGMA database_list").fetchone()[2]
    app.state.db_path = Path(db_path_str)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        cookies={"csrf_token": CSRF_TEST_TOKEN},
        headers={"X-CSRF-Token": CSRF_TEST_TOKEN},
    ) as ac:
        yield ac

    app.dependency_overrides.clear()


# --- Model / CRUD tests ---


def test_topic_default_tags_empty(db_conn):
    topic = _make_topic(db_conn, "No Tags Topic")
    assert topic.tags == []


def test_create_topic_with_tags(db_conn):
    topic = _make_topic(db_conn, "Tagged Topic", tags=["ai", "tech"])
    assert topic.tags == ["ai", "tech"]


def test_tags_roundtrip(db_conn):
    """Tags created → fetched are identical."""
    _make_topic(db_conn, "Roundtrip Topic", tags=["security", "privacy"])
    fetched = get_topic(db_conn, 1)
    assert fetched is not None
    assert fetched.tags == ["security", "privacy"]


def test_update_topic_tags(db_conn):
    topic = _make_topic(db_conn, "Update Tags", tags=["old"])
    topic.tags = ["new", "updated"]
    update_topic(db_conn, topic)
    db_conn.commit()

    fetched = get_topic(db_conn, topic.id)
    assert fetched is not None
    assert fetched.tags == ["new", "updated"]


def test_clear_topic_tags(db_conn):
    topic = _make_topic(db_conn, "Clear Tags", tags=["remove-me"])
    topic.tags = []
    update_topic(db_conn, topic)
    db_conn.commit()

    fetched = get_topic(db_conn, topic.id)
    assert fetched is not None
    assert fetched.tags == []


def test_list_topics_tag_filter(db_conn):
    _make_topic(db_conn, "AI Topic", tags=["ai", "tech"])
    _make_topic(db_conn, "Security Topic", tags=["security"])
    _make_topic(db_conn, "Tech Topic", tags=["tech"])

    ai_topics = list_topics(db_conn, tag="ai")
    assert len(ai_topics) == 1
    assert ai_topics[0].name == "AI Topic"

    tech_topics = list_topics(db_conn, tag="tech")
    assert len(tech_topics) == 2
    names = {t.name for t in tech_topics}
    assert names == {"AI Topic", "Tech Topic"}


def test_list_topics_tag_filter_no_match(db_conn):
    _make_topic(db_conn, "Topic A", tags=["ai"])
    result = list_topics(db_conn, tag="nonexistent")
    assert result == []


def test_list_topics_no_filter_returns_all(db_conn):
    _make_topic(db_conn, "Alpha", tags=["a"])
    _make_topic(db_conn, "Beta", tags=["b"])
    _make_topic(db_conn, "Gamma")

    result = list_topics(db_conn)
    assert len(result) == 3


def test_list_topics_active_only_with_tag(db_conn):
    from app.crud import update_topic as ut

    _make_topic(db_conn, "Active Tagged", tags=["tech"])
    t2 = _make_topic(db_conn, "Inactive Tagged", tags=["tech"])
    t2.is_active = False
    ut(db_conn, t2)
    db_conn.commit()

    result = list_topics(db_conn, active_only=True, tag="tech")
    assert len(result) == 1
    assert result[0].name == "Active Tagged"


# --- Migration test ---


def test_migration_adds_tags_column(tmp_path: Path):
    """Migration adds tags column to existing topics table."""
    db_path = tmp_path / "migration_test.db"
    conn = get_connection(db_path)

    # Create table without tags column (simulate pre-migration state)
    conn.execute(
        """CREATE TABLE topics (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            description TEXT NOT NULL,
            feed_urls TEXT NOT NULL DEFAULT '[]',
            feed_mode TEXT NOT NULL DEFAULT 'auto',
            created_at TEXT NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1,
            status TEXT NOT NULL DEFAULT 'researching',
            error_message TEXT,
            check_interval_hours INTEGER
        )"""
    )
    conn.commit()

    # Verify column does not exist yet
    columns_before = {row[1] for row in conn.execute("PRAGMA table_info(topics)").fetchall()}
    assert "tags" not in columns_before

    # Run migration
    m006_up(conn)
    conn.commit()

    # Verify column now exists
    columns_after = {row[1] for row in conn.execute("PRAGMA table_info(topics)").fetchall()}
    assert "tags" in columns_after

    conn.close()


def test_migration_idempotent(tmp_path: Path):
    """Running migration twice does not raise an error."""
    db_path = tmp_path / "idempotent_test.db"
    conn = get_connection(db_path)

    conn.execute(
        """CREATE TABLE topics (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT NOT NULL,
            feed_urls TEXT NOT NULL DEFAULT '[]',
            feed_mode TEXT NOT NULL DEFAULT 'auto',
            created_at TEXT NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1,
            status TEXT NOT NULL DEFAULT 'researching',
            error_message TEXT,
            check_interval_hours INTEGER
        )"""
    )
    conn.commit()

    m006_up(conn)
    conn.commit()
    # Second run should not raise
    m006_up(conn)
    conn.commit()

    conn.close()


# --- Route tests ---


async def test_dashboard_shows_tag_filter(client, db_conn):
    _make_topic(db_conn, "AI News", tags=["ai"])
    _make_topic(db_conn, "Security Watch", tags=["security"])

    response = await client.get("/")
    assert response.status_code == 200
    assert "ai" in response.text
    assert "security" in response.text


async def test_dashboard_tag_filter_param(client, db_conn):
    _make_topic(db_conn, "AI News", tags=["ai"])
    _make_topic(db_conn, "Security Watch", tags=["security"])

    response = await client.get("/?tag=ai")
    assert response.status_code == 200
    assert "AI News" in response.text
    assert "Security Watch" not in response.text


async def test_dashboard_tag_filter_no_match(client, db_conn):
    _make_topic(db_conn, "AI News", tags=["ai"])

    response = await client.get("/?tag=nonexistent")
    assert response.status_code == 200
    assert "AI News" not in response.text


async def test_create_topic_with_tags_via_form(client, db_conn):
    response = await client.post(
        "/topics",
        data={
            "name": "Tagged New Topic",
            "description": "A topic with tags",
            "feed_mode": "auto",
            "tags": "ai, tech, ml",
            "csrf_token": CSRF_TEST_TOKEN,
        },
        follow_redirects=False,
    )
    # Should redirect (302 or 303)
    assert response.status_code in (302, 303)

    # Verify tags were saved
    topics = list_topics(db_conn)
    assert len(topics) == 1
    assert set(topics[0].tags) == {"ai", "tech", "ml"}


async def test_edit_topic_updates_tags(client, db_conn):
    topic = _make_topic(db_conn, "Edit Me", tags=["old-tag"])
    topic.status = TopicStatus.READY
    update_topic(db_conn, topic)
    db_conn.commit()

    response = await client.post(
        f"/topics/{topic.id}/edit",
        data={
            "name": "Edit Me",
            "description": "Description for Edit Me",
            "feed_mode": "manual",
            "feed_urls": "https://example.com/feed.xml",
            "tags": "new-tag, another",
            "csrf_token": CSRF_TEST_TOKEN,
        },
        follow_redirects=False,
    )
    assert response.status_code in (302, 303)

    fetched = get_topic(db_conn, topic.id)
    assert fetched is not None
    assert set(fetched.tags) == {"new-tag", "another"}


async def test_topic_detail_shows_tags(client, db_conn):
    topic = _make_topic(db_conn, "Detail Topic", tags=["visible-tag"])

    response = await client.get(f"/topics/{topic.id}")
    assert response.status_code == 200
    assert "visible-tag" in response.text
