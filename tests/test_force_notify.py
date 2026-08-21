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
from app.models import CheckResult, FeedMode, NotificationDelivery, Topic, TopicStatus
from app.web.dependencies import get_db_conn, get_settings
from app.webhooks import WebhookOutcome

CSRF_TEST_TOKEN = "test-csrf-token-for-force-notify-tests"


async def _ok_send(title: str, body: str, url: str, timeout_s: float) -> NotificationDelivery:
    """Stub for app.checker.send_single_notification: every target delivers."""
    return NotificationDelivery(url=url, ok=True)


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
    """Force notify reports success when every configured channel delivered."""
    topic = _make_topic(db_conn)
    assert topic.id is not None
    check = _make_check_result(db_conn, topic.id, has_new_info=True)
    assert check.id is not None

    with patch("app.checker.send_single_notification", side_effect=_ok_send):
        response = await client.post(f"/topics/{topic.id}/checks/{check.id}/notify")

    assert response.status_code == 200
    assert "Sent!" in response.text
    assert "var(--pico-ins-color, green)" in response.text
    # The resend went through a durable intent, now recorded as delivered.
    row = db_conn.execute("SELECT status, check_result_id FROM pending_notifications").fetchone()
    assert row["status"] == "sent"
    assert row["check_result_id"] == check.id


async def test_force_notify_calls_send_with_correct_args(
    client: httpx.AsyncClient,
    db_conn: sqlite3.Connection,
) -> None:
    """The resent message carries a title derived from the topic name."""
    topic = _make_topic(db_conn, name="Climate News")
    assert topic.id is not None
    check = _make_check_result(db_conn, topic.id, has_new_info=True)
    assert check.id is not None

    seen: list[tuple[str, str]] = []

    async def record(title, body, url, timeout_s):  # noqa: ANN001
        seen.append((title, body))
        return NotificationDelivery(url=url, ok=True)

    with patch("app.checker.send_single_notification", side_effect=record):
        await client.post(f"/topics/{topic.id}/checks/{check.id}/notify")

    assert len(seen) == 1
    title, body = seen[0]
    assert "Climate News" in title
    assert "Something new happened" in body


# --- Force notify: delivery failure ---


async def test_force_notify_delivery_failure(
    client: httpx.AsyncClient,
    db_conn: sqlite3.Connection,
) -> None:
    """A failed resend says so and leaves the intent queued for the drain."""
    topic = _make_topic(db_conn)
    assert topic.id is not None
    check = _make_check_result(db_conn, topic.id, has_new_info=True)
    assert check.id is not None

    async def fail(title, body, url, timeout_s):  # noqa: ANN001
        return NotificationDelivery(url=url, ok=False, error="down")

    with patch("app.checker.send_single_notification", side_effect=fail):
        response = await client.post(f"/topics/{topic.id}/checks/{check.id}/notify")

    assert response.status_code == 200
    assert "Delivery failed" in response.text
    assert "queued for retry" in response.text
    assert "var(--pico-del-color, red)" in response.text
    row = db_conn.execute("SELECT status, retry_count FROM pending_notifications").fetchone()
    assert row["status"] == "pending"
    assert row["retry_count"] == 1


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

    with patch("app.checker.send_single_notification", side_effect=_ok_send) as mock_send:
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
    """Force notify returns an error message when the delivery layer raises."""
    topic = _make_topic(db_conn)
    assert topic.id is not None
    check = _make_check_result(db_conn, topic.id, has_new_info=True)
    assert check.id is not None

    with patch("app.web.routers.topics.deliver_notification_intents", new_callable=AsyncMock) as mock_send:
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


# --- channel parity (AUG-109) ---


async def test_force_notify_sends_webhooks_for_a_webhook_only_setup(
    db_conn: sqlite3.Connection,
) -> None:
    """A webhook-only configuration must be resent through, not reported failed."""
    settings = _make_settings(notifications=NotificationSettings(urls=[], webhook_urls=["https://hooks.example.com/x"]))

    def override_db():
        yield db_conn

    app.dependency_overrides[get_db_conn] = override_db
    app.dependency_overrides[get_settings] = lambda: settings
    try:
        topic = _make_topic(db_conn, name="Webhook Only")
        assert topic.id is not None
        check = _make_check_result(db_conn, topic.id, has_new_info=True)
        assert check.id is not None

        posted: list[str] = []

        async def record(url, payload, timeout=10.0):  # noqa: ANN001
            posted.append(url)
            return WebhookOutcome(ok=True, status=200)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            cookies={"csrf_token": CSRF_TEST_TOKEN},
            headers={"X-CSRF-Token": CSRF_TEST_TOKEN},
        ) as ac:
            with patch("app.webhooks.send_webhook", side_effect=record):
                response = await ac.post(f"/topics/{topic.id}/checks/{check.id}/notify")

        assert response.status_code == 200
        assert "Sent!" in response.text
        assert "webhooks 1/1" in response.text
        assert posted == ["https://hooks.example.com/x"]
        row = db_conn.execute("SELECT status, check_result_id FROM pending_webhooks").fetchone()
        assert row["status"] == "sent"
        assert row["check_result_id"] == check.id
    finally:
        app.dependency_overrides.clear()


async def test_force_notify_without_any_target_says_so(db_conn: sqlite3.Connection) -> None:
    settings = _make_settings(notifications=NotificationSettings(urls=[], webhook_urls=[]))

    def override_db():
        yield db_conn

    app.dependency_overrides[get_db_conn] = override_db
    app.dependency_overrides[get_settings] = lambda: settings
    try:
        topic = _make_topic(db_conn, name="No Targets")
        assert topic.id is not None
        check = _make_check_result(db_conn, topic.id, has_new_info=True)
        assert check.id is not None

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            cookies={"csrf_token": CSRF_TEST_TOKEN},
            headers={"X-CSRF-Token": CSRF_TEST_TOKEN},
        ) as ac:
            response = await ac.post(f"/topics/{topic.id}/checks/{check.id}/notify")

        assert response.status_code == 400
        assert "No delivery target configured" in response.text
    finally:
        app.dependency_overrides.clear()
