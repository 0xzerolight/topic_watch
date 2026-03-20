"""Tests for POST /topics/{topic_id}/checks/{check_id}/notify (force notify)."""

import sqlite3
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.analysis.llm import NoveltyResult
from app.config import LLMSettings, NotificationSettings, Settings
from app.crud import create_check_result, create_topic, get_check_result
from app.main import app
from app.models import CheckResult, FeedMode, Topic, TopicStatus
from app.web.dependencies import get_db_conn, get_settings

CSRF_TEST_TOKEN = "test-csrf-token-for-force-notify-tests"


def _make_settings(**overrides) -> Settings:
    defaults = {
        "llm": LLMSettings(model="openai/gpt-4o-mini", api_key="test-key-12345678"),
        "notifications": NotificationSettings(urls=["json://localhost"]),
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _make_topic(conn: sqlite3.Connection, name: str = "Test Topic") -> Topic:
    """Create and persist a topic, return with id."""
    topic = Topic(
        name=name,
        description="A test topic",
        feed_urls=[],
        feed_mode=FeedMode.AUTO,
        status=TopicStatus.READY,
        status_changed_at=datetime.now(UTC),
    )
    return create_topic(conn, topic)


def _make_check_result(
    conn: sqlite3.Connection,
    topic_id: int,
    has_new_info: bool = True,
    llm_response: str | None = None,
) -> CheckResult:
    """Create and persist a check result, return with id."""
    if llm_response is None and has_new_info:
        novelty = NoveltyResult(
            has_new_info=True,
            summary="Something new happened",
            key_facts=["Fact one", "Fact two"],
            source_urls=["https://example.com/article1"],
            confidence=0.9,
        )
        llm_response = novelty.model_dump_json()

    result = CheckResult(
        topic_id=topic_id,
        checked_at=datetime.now(UTC),
        articles_found=5,
        articles_new=2,
        has_new_info=has_new_info,
        llm_response=llm_response,
        notification_sent=False,
        notification_error=None,
    )
    return create_check_result(conn, result)


@pytest.fixture
async def client(
    db_conn: sqlite3.Connection,
) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Test client with db and settings overrides, CSRF token pre-configured."""
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
        cookies={"csrf_token": CSRF_TEST_TOKEN},
        headers={"X-CSRF-Token": CSRF_TEST_TOKEN},
    ) as ac:
        yield ac

    app.dependency_overrides.clear()


# --- get_check_result CRUD ---


def test_get_check_result_returns_result(db_conn: sqlite3.Connection) -> None:
    """get_check_result returns the correct CheckResult by id."""
    topic = _make_topic(db_conn)
    assert topic.id is not None
    check = _make_check_result(db_conn, topic.id)
    assert check.id is not None

    fetched = get_check_result(db_conn, check.id)
    assert fetched is not None
    assert fetched.id == check.id
    assert fetched.topic_id == topic.id
    assert fetched.has_new_info is True


def test_get_check_result_returns_none_for_missing(db_conn: sqlite3.Connection) -> None:
    """get_check_result returns None for a nonexistent id."""
    result = get_check_result(db_conn, 999999)
    assert result is None


# --- Force notify: success ---


async def test_force_notify_success(
    client: httpx.AsyncClient,
    db_conn: sqlite3.Connection,
) -> None:
    """Force notify returns 'Sent!' when send_notification returns True."""
    topic = _make_topic(db_conn)
    assert topic.id is not None
    check = _make_check_result(db_conn, topic.id, has_new_info=True)
    assert check.id is not None

    with patch("app.web.routes.send_notification", new_callable=AsyncMock) as mock_send:
        mock_send.return_value = True
        response = await client.post(f"/topics/{topic.id}/checks/{check.id}/notify")

    assert response.status_code == 200
    assert "Sent!" in response.text
    assert "var(--pico-ins-color, green)" in response.text


async def test_force_notify_calls_send_with_correct_args(
    client: httpx.AsyncClient,
    db_conn: sqlite3.Connection,
) -> None:
    """Force notify calls send_notification with a title derived from topic name."""
    topic = _make_topic(db_conn, name="Climate News")
    assert topic.id is not None
    check = _make_check_result(db_conn, topic.id, has_new_info=True)
    assert check.id is not None

    with patch("app.web.routes.send_notification", new_callable=AsyncMock) as mock_send:
        mock_send.return_value = True
        await client.post(f"/topics/{topic.id}/checks/{check.id}/notify")

    mock_send.assert_called_once()
    title, body = mock_send.call_args.args[0], mock_send.call_args.args[1]
    assert "Climate News" in title
    assert "Something new happened" in body


# --- Force notify: delivery failure ---


async def test_force_notify_delivery_failure(
    client: httpx.AsyncClient,
    db_conn: sqlite3.Connection,
) -> None:
    """Force notify returns 'Delivery failed' when send_notification returns False."""
    topic = _make_topic(db_conn)
    assert topic.id is not None
    check = _make_check_result(db_conn, topic.id, has_new_info=True)
    assert check.id is not None

    with patch("app.web.routes.send_notification", new_callable=AsyncMock) as mock_send:
        mock_send.return_value = False
        response = await client.post(f"/topics/{topic.id}/checks/{check.id}/notify")

    assert response.status_code == 200
    assert "Delivery failed" in response.text
    assert "var(--pico-del-color, red)" in response.text


# --- Force notify: no new info ---


async def test_force_notify_no_new_info_returns_400(
    client: httpx.AsyncClient,
    db_conn: sqlite3.Connection,
) -> None:
    """Force notify returns 400 for a check result with has_new_info=False."""
    topic = _make_topic(db_conn)
    assert topic.id is not None
    check = _make_check_result(db_conn, topic.id, has_new_info=False, llm_response=None)
    assert check.id is not None

    with patch("app.web.routes.send_notification", new_callable=AsyncMock) as mock_send:
        response = await client.post(f"/topics/{topic.id}/checks/{check.id}/notify")

    assert response.status_code == 400
    assert "No new info" in response.text
    mock_send.assert_not_called()


# --- Force notify: not found cases ---


async def test_force_notify_nonexistent_check_returns_404(
    client: httpx.AsyncClient,
    db_conn: sqlite3.Connection,
) -> None:
    """Force notify returns 404 when the check result does not exist."""
    topic = _make_topic(db_conn)
    assert topic.id is not None

    response = await client.post(f"/topics/{topic.id}/checks/999999/notify")

    assert response.status_code == 404
    assert "not found" in response.text.lower()


async def test_force_notify_nonexistent_topic_returns_404(
    client: httpx.AsyncClient,
    db_conn: sqlite3.Connection,
) -> None:
    """Force notify returns 404 when the topic does not exist."""
    topic = _make_topic(db_conn)
    assert topic.id is not None
    check = _make_check_result(db_conn, topic.id, has_new_info=True)
    assert check.id is not None

    response = await client.post(f"/topics/999999/checks/{check.id}/notify")

    assert response.status_code == 404


async def test_force_notify_check_from_different_topic_returns_404(
    client: httpx.AsyncClient,
    db_conn: sqlite3.Connection,
) -> None:
    """Force notify returns 404 when check result belongs to a different topic."""
    topic_a = _make_topic(db_conn, name="Topic A")
    topic_b = _make_topic(db_conn, name="Topic B")
    assert topic_a.id is not None
    assert topic_b.id is not None

    # Check belongs to topic_b
    check = _make_check_result(db_conn, topic_b.id, has_new_info=True)
    assert check.id is not None

    # Request uses topic_a's id but topic_b's check id
    response = await client.post(f"/topics/{topic_a.id}/checks/{check.id}/notify")

    assert response.status_code == 404


# --- Force notify: exception handling ---


async def test_force_notify_exception_returns_error_message(
    client: httpx.AsyncClient,
    db_conn: sqlite3.Connection,
) -> None:
    """Force notify returns an error message when send_notification raises."""
    topic = _make_topic(db_conn)
    assert topic.id is not None
    check = _make_check_result(db_conn, topic.id, has_new_info=True)
    assert check.id is not None

    with patch("app.web.routes.send_notification", new_callable=AsyncMock) as mock_send:
        mock_send.side_effect = RuntimeError("SMTP connection refused")
        response = await client.post(f"/topics/{topic.id}/checks/{check.id}/notify")

    assert response.status_code == 200
    assert "Error" in response.text
    assert "SMTP connection refused" in response.text
    assert "var(--pico-del-color, red)" in response.text


# --- CSRF protection ---


async def test_force_notify_requires_csrf(db_conn: sqlite3.Connection) -> None:
    """POST /topics/{topic_id}/checks/{check_id}/notify returns 403 without CSRF."""
    settings = _make_settings()

    def override_db():
        yield db_conn

    def override_settings():
        return settings

    app.dependency_overrides[get_db_conn] = override_db
    app.dependency_overrides[get_settings] = override_settings

    topic = _make_topic(db_conn)
    assert topic.id is not None
    check = _make_check_result(db_conn, topic.id, has_new_info=True)
    assert check.id is not None

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as ac:
            response = await ac.post(f"/topics/{topic.id}/checks/{check.id}/notify")

        assert response.status_code == 403
    finally:
        app.dependency_overrides.clear()
