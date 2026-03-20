"""Tests for topic search/filter functionality (CRUD + route)."""

import sqlite3
from collections.abc import AsyncGenerator

import httpx
import pytest

from app.config import LLMSettings, NotificationSettings, Settings
from app.crud import create_topic, search_dashboard_data
from app.main import app
from app.models import FeedMode, Topic, TopicStatus
from app.web.dependencies import get_db_conn, get_settings


def _make_settings(**overrides) -> Settings:
    defaults = {
        "llm": LLMSettings(model="openai/gpt-4o-mini", api_key="test-key-12345678"),
        "notifications": NotificationSettings(urls=["json://localhost"]),
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _make_topic(conn: sqlite3.Connection, name: str, status: TopicStatus = TopicStatus.READY) -> Topic:
    topic = create_topic(
        conn,
        Topic(
            name=name,
            description=f"Description for {name}",
            feed_urls=["https://example.com/feed.xml"],
            feed_mode=FeedMode.MANUAL,
            status=status,
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

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac

    app.dependency_overrides.clear()


# --- CRUD tests ---


def test_search_no_filters_returns_all(db_conn):
    _make_topic(db_conn, "Alpha")
    _make_topic(db_conn, "Beta")
    _make_topic(db_conn, "Gamma")

    result = search_dashboard_data(db_conn)
    assert len(result) == 3


def test_search_query_filter_returns_matching(db_conn):
    _make_topic(db_conn, "Alpha News")
    _make_topic(db_conn, "Beta Updates")
    _make_topic(db_conn, "Alpha Tech")

    result = search_dashboard_data(db_conn, query="Alpha")
    names = [r["topic"].name for r in result]
    assert len(result) == 2
    assert "Alpha News" in names
    assert "Alpha Tech" in names
    assert "Beta Updates" not in names


def test_search_query_case_insensitive(db_conn):
    _make_topic(db_conn, "Python Programming")
    _make_topic(db_conn, "JavaScript Frameworks")

    result = search_dashboard_data(db_conn, query="python")
    names = [r["topic"].name for r in result]
    assert len(result) == 1
    assert "Python Programming" in names


def test_search_status_filter(db_conn):
    _make_topic(db_conn, "Ready Topic", status=TopicStatus.READY)
    _make_topic(db_conn, "Error Topic", status=TopicStatus.ERROR)
    _make_topic(db_conn, "Another Ready", status=TopicStatus.READY)

    result = search_dashboard_data(db_conn, status="ready")
    assert len(result) == 2
    for r in result:
        assert r["topic"].status == TopicStatus.READY


def test_search_error_status_filter(db_conn):
    _make_topic(db_conn, "Ready Topic", status=TopicStatus.READY)
    _make_topic(db_conn, "Error Topic", status=TopicStatus.ERROR)

    result = search_dashboard_data(db_conn, status="error")
    assert len(result) == 1
    assert result[0]["topic"].name == "Error Topic"


def test_search_combined_query_and_status(db_conn):
    _make_topic(db_conn, "Tech Ready", status=TopicStatus.READY)
    _make_topic(db_conn, "Tech Error", status=TopicStatus.ERROR)
    _make_topic(db_conn, "News Ready", status=TopicStatus.READY)

    result = search_dashboard_data(db_conn, query="Tech", status="ready")
    assert len(result) == 1
    assert result[0]["topic"].name == "Tech Ready"


def test_search_no_match_returns_empty(db_conn):
    _make_topic(db_conn, "Alpha")
    _make_topic(db_conn, "Beta")

    result = search_dashboard_data(db_conn, query="zzz_no_match")
    assert result == []


def test_search_result_has_expected_keys(db_conn):
    _make_topic(db_conn, "Test Topic")

    result = search_dashboard_data(db_conn)
    assert len(result) == 1
    item = result[0]
    assert "topic" in item
    assert "last_check" in item
    assert "article_count" in item
    assert item["last_check"] is None
    assert item["article_count"] == 0


# --- Route tests ---


async def test_search_route_returns_html(client, db_conn):
    _make_topic(db_conn, "Foo Topic")
    _make_topic(db_conn, "Bar Topic")

    response = await client.get("/topics/search")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


async def test_search_route_filters_by_query(client, db_conn):
    _make_topic(db_conn, "Python News")
    _make_topic(db_conn, "JavaScript News")

    response = await client.get("/topics/search?q=Python")
    assert response.status_code == 200
    assert "Python News" in response.text
    assert "JavaScript News" not in response.text


async def test_search_route_filters_by_status(client, db_conn):
    _make_topic(db_conn, "Ready One", status=TopicStatus.READY)
    _make_topic(db_conn, "Error One", status=TopicStatus.ERROR)

    response = await client.get("/topics/search?status=error")
    assert response.status_code == 200
    assert "Error One" in response.text
    assert "Ready One" not in response.text


async def test_search_route_no_match_shows_empty_message(client, db_conn):
    _make_topic(db_conn, "Existing Topic")

    response = await client.get("/topics/search?q=zzz_no_match")
    assert response.status_code == 200
    assert "No topics match your search" in response.text


async def test_search_route_all_status_returns_all(client, db_conn):
    _make_topic(db_conn, "Ready One", status=TopicStatus.READY)
    _make_topic(db_conn, "Error One", status=TopicStatus.ERROR)

    response = await client.get("/topics/search?status=all")
    assert response.status_code == 200
    assert "Ready One" in response.text
    assert "Error One" in response.text


async def test_search_route_empty_query_returns_all(client, db_conn):
    _make_topic(db_conn, "Topic A")
    _make_topic(db_conn, "Topic B")

    response = await client.get("/topics/search?q=")
    assert response.status_code == 200
    assert "Topic A" in response.text
    assert "Topic B" in response.text
