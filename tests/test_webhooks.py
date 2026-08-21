"""Tests for the webhook delivery module."""

import sqlite3
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.analysis.llm import NoveltyResult
from app.config import LLMSettings, NotificationSettings, Settings
from app.crud import (
    apply_webhook_outcome,
    create_topic,
    create_webhook_intents,
    list_pending_webhooks,
    release_stale_webhook_claims,
)
from app.models import Topic, TopicStatus, to_db_utc
from app.webhooks import (
    WebhookOutcome,
    _build_webhook_payload,
    build_webhook_intents,
    deliver_webhook_intents,
    retry_pending_webhooks,
    send_webhook,
)
from tests.helpers import conn_db_path


def _ok(status: int = 200) -> WebhookOutcome:
    return WebhookOutcome(ok=True, status=status)


def _fail(**kwargs) -> WebhookOutcome:
    return WebhookOutcome(ok=False, **kwargs)


async def _deliver(
    conn: sqlite3.Connection,
    topic_id: int,
    settings: Settings,
    *,
    topic_name: str = "Hooked",
    novelty: NoveltyResult | None = None,
) -> int:
    """The live path in one step: build intents, commit them, then deliver."""
    intents = build_webhook_intents(topic_name, novelty or _make_novelty(), settings, topic_id)
    create_webhook_intents(conn, intents)
    conn.commit()
    return await deliver_webhook_intents(intents, settings, conn_db_path(conn), conn)


def _make_settings(**overrides) -> Settings:
    defaults = {
        "llm": LLMSettings(model="openai/gpt-4o-mini", api_key="test-key"),
        "notifications": NotificationSettings(urls=[]),
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _make_novelty(**overrides) -> NoveltyResult:
    defaults = {
        "has_new_info": True,
        "summary": "New milestone reached",
        "key_facts": ["Fact one", "Fact two"],
        "source_urls": ["https://example.com/article"],
        "confidence": 0.85,
    }
    defaults.update(overrides)
    return NoveltyResult(**defaults)


# --- _build_webhook_payload ---


class TestBuildWebhookPayload:
    """Tests for payload construction."""

    def test_payload_contains_topic_name(self) -> None:
        novelty = _make_novelty()
        payload = _build_webhook_payload("My Topic", novelty)
        assert payload["topic"] == "My Topic"

    def test_payload_contains_summary(self) -> None:
        novelty = _make_novelty(summary="Something happened")
        payload = _build_webhook_payload("T", novelty)
        assert payload["summary"] == "Something happened"

    def test_payload_summary_defaults_to_empty_string_when_none(self) -> None:
        novelty = _make_novelty(summary=None)
        payload = _build_webhook_payload("T", novelty)
        assert payload["summary"] == ""

    def test_payload_contains_key_facts(self) -> None:
        novelty = _make_novelty(key_facts=["Fact A", "Fact B"])
        payload = _build_webhook_payload("T", novelty)
        assert payload["key_facts"] == ["Fact A", "Fact B"]

    def test_payload_contains_source_urls(self) -> None:
        novelty = _make_novelty(source_urls=["https://a.com", "https://b.com"])
        payload = _build_webhook_payload("T", novelty)
        assert payload["source_urls"] == ["https://a.com", "https://b.com"]

    def test_payload_contains_confidence(self) -> None:
        novelty = _make_novelty(confidence=0.72)
        payload = _build_webhook_payload("T", novelty)
        assert payload["confidence"] == pytest.approx(0.72)

    def test_payload_contains_relevance(self) -> None:
        novelty = _make_novelty(relevance=0.61)
        payload = _build_webhook_payload("T", novelty)
        assert payload["relevance"] == pytest.approx(0.61)

    def test_relevance_is_float_type(self) -> None:
        novelty = _make_novelty(relevance=1.0)
        payload = _build_webhook_payload("T", novelty)
        assert isinstance(payload["relevance"], float)

    def test_payload_contains_timestamp(self) -> None:
        novelty = _make_novelty()
        payload = _build_webhook_payload("T", novelty)
        assert "timestamp" in payload
        assert isinstance(payload["timestamp"], str)
        # Should be an ISO 8601 string with timezone info
        assert "T" in payload["timestamp"]
        assert "+00:00" in payload["timestamp"] or "Z" in payload["timestamp"]

    def test_payload_has_all_expected_fields(self) -> None:
        novelty = _make_novelty()
        payload = _build_webhook_payload("T", novelty)
        expected_keys = {
            "topic",
            "reasoning",
            "summary",
            "key_facts",
            "source_urls",
            "confidence",
            "relevance",
            "importance",
            "timestamp",
        }
        assert set(payload.keys()) == expected_keys

    def test_payload_includes_importance(self) -> None:
        novelty = _make_novelty(importance=5)
        payload = _build_webhook_payload("T", novelty)
        assert payload["importance"] == 5

    def test_key_facts_is_list_type(self) -> None:
        novelty = _make_novelty(key_facts=[])
        payload = _build_webhook_payload("T", novelty)
        assert isinstance(payload["key_facts"], list)

    def test_source_urls_is_list_type(self) -> None:
        novelty = _make_novelty(source_urls=[])
        payload = _build_webhook_payload("T", novelty)
        assert isinstance(payload["source_urls"], list)

    def test_payload_contains_reasoning(self) -> None:
        novelty = _make_novelty(reasoning="Article [1] mentions a new date.")
        payload = _build_webhook_payload("T", novelty)
        assert payload["reasoning"] == "Article [1] mentions a new date."

    def test_payload_reasoning_defaults_to_empty(self) -> None:
        novelty = _make_novelty()
        payload = _build_webhook_payload("T", novelty)
        assert payload["reasoning"] == ""

    def test_confidence_is_float_type(self) -> None:
        novelty = _make_novelty(confidence=1.0)
        payload = _build_webhook_payload("T", novelty)
        assert isinstance(payload["confidence"], float)


# --- send_webhook ---


class TestSendWebhook:
    """Tests for the individual send_webhook function."""

    async def test_returns_true_on_success(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("app.webhooks.httpx.AsyncClient", return_value=mock_client):
            result = await send_webhook("https://example.com/hook", {"key": "value"})

        assert result.ok is True

    async def test_returns_false_on_timeout(self) -> None:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("app.webhooks.httpx.AsyncClient", return_value=mock_client):
            result = await send_webhook("https://example.com/hook", {})

        assert result.ok is False

    async def test_returns_false_on_http_error(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "Server Error",
                request=MagicMock(),
                response=mock_response,
            )
        )

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("app.webhooks.httpx.AsyncClient", return_value=mock_client):
            result = await send_webhook("https://example.com/hook", {})

        assert result.ok is False

    async def test_returns_false_on_connection_error(self) -> None:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("Connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("app.webhooks.httpx.AsyncClient", return_value=mock_client):
            result = await send_webhook("https://example.com/hook", {})

        assert result.ok is False

    async def test_blocks_private_url(self) -> None:
        result = await send_webhook("http://localhost:9200/hook", {"key": "value"})
        assert result.ok is False

    async def test_blocks_internal_ip(self) -> None:
        result = await send_webhook("http://169.254.169.254/metadata", {})
        assert result.ok is False

    async def test_blocks_loopback_ip(self) -> None:
        result = await send_webhook("http://127.0.0.1:8080/hook", {})
        assert result.ok is False

    async def test_blocks_file_scheme_before_post(self) -> None:
        """file:// is rejected by the scheme allowlist before any POST (OVH-141).

        The httpx client must never be constructed for a non-http(s) scheme, so
        a transport/config change can't expose the first hop.
        """
        with patch("app.webhooks.httpx.AsyncClient") as mock_cls:
            result = await send_webhook("file:///etc/passwd", {"key": "value"})
        assert result.ok is False
        mock_cls.assert_not_called()

    async def test_blocks_gopher_scheme_before_post(self) -> None:
        """gopher:// is rejected by the scheme allowlist before any POST (OVH-141)."""
        with patch("app.webhooks.httpx.AsyncClient") as mock_cls:
            result = await send_webhook("gopher://example.com/7", {})
        assert result.ok is False
        mock_cls.assert_not_called()

    async def test_blocks_ftp_scheme_before_post(self) -> None:
        """ftp:// is rejected by the scheme allowlist before any POST (OVH-141)."""
        with patch("app.webhooks.httpx.AsyncClient") as mock_cls:
            result = await send_webhook("ftp://example.com/file", {})
        assert result.ok is False
        mock_cls.assert_not_called()

    async def test_malformed_ipv6_url_returns_false_not_raises(self) -> None:
        """OVH-131: a malformed IPv6 literal makes urlparse raise ValueError, but
        send_webhook honors its 'Never raises' contract — returns False, no POST.
        """
        # urlparse("http://[::1") raises ValueError("Invalid IPv6 URL").
        with patch("app.webhooks.httpx.AsyncClient") as mock_cls:
            result = await send_webhook("http://[::1", {"key": "value"})
        assert result.ok is False
        mock_cls.assert_not_called()

    async def test_posts_to_correct_url(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        url = "https://hooks.example.com/trigger"
        payload = {"topic": "test"}

        with patch("app.webhooks.httpx.AsyncClient", return_value=mock_client):
            await send_webhook(url, payload)

        mock_client.post.assert_called_once_with(url, json=payload)


# --- send_webhooks ---


class TestWebhookOutcomeClassification:
    """AUG-324: the outcome has to say what happened, not just whether it worked."""

    @staticmethod
    def _client_returning(status: int, headers: dict | None = None) -> AsyncMock:
        response = MagicMock()
        response.status_code = status
        response.headers = headers or {}
        response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError("err", request=MagicMock(), response=response)
        )
        client = AsyncMock()
        client.post = AsyncMock(return_value=response)
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        return client

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 410, 413, 422])
    async def test_permanent_statuses_are_not_retryable(self, status: int) -> None:
        with patch("app.webhooks.httpx.AsyncClient", return_value=self._client_returning(status)):
            outcome = await send_webhook("https://example.com/hook", {})
        assert outcome.ok is False
        assert outcome.status == status
        assert outcome.retryable is False
        assert outcome.error == f"HTTP {status}"

    @pytest.mark.parametrize("status", [408, 425, 429, 500, 503])
    async def test_transient_statuses_stay_retryable(self, status: int) -> None:
        with patch("app.webhooks.httpx.AsyncClient", return_value=self._client_returning(status)):
            outcome = await send_webhook("https://example.com/hook", {})
        assert outcome.retryable is True

    async def test_retry_after_delta_seconds_is_parsed(self) -> None:
        client = self._client_returning(429, {"Retry-After": "120"})
        with patch("app.webhooks.httpx.AsyncClient", return_value=client):
            outcome = await send_webhook("https://example.com/hook", {})
        assert outcome.retry_after_s == pytest.approx(120.0)

    async def test_retry_after_http_date_is_parsed(self) -> None:
        when = datetime.now(UTC) + timedelta(seconds=300)
        client = self._client_returning(503, {"Retry-After": format_datetime(when, usegmt=True)})
        with patch("app.webhooks.httpx.AsyncClient", return_value=client):
            outcome = await send_webhook("https://example.com/hook", {})
        assert outcome.retry_after_s is not None
        assert 250 < outcome.retry_after_s <= 300

    async def test_retry_after_is_clamped_and_never_negative(self) -> None:
        client = self._client_returning(429, {"Retry-After": "999999"})
        with patch("app.webhooks.httpx.AsyncClient", return_value=client):
            outcome = await send_webhook("https://example.com/hook", {})
        assert outcome.retry_after_s == pytest.approx(3600.0)

        past = datetime.now(UTC) - timedelta(hours=1)
        client = self._client_returning(429, {"Retry-After": format_datetime(past, usegmt=True)})
        with patch("app.webhooks.httpx.AsyncClient", return_value=client):
            outcome = await send_webhook("https://example.com/hook", {})
        assert outcome.retry_after_s == pytest.approx(0.0)

    async def test_unparseable_retry_after_is_ignored(self) -> None:
        client = self._client_returning(429, {"Retry-After": "next tuesday"})
        with patch("app.webhooks.httpx.AsyncClient", return_value=client):
            outcome = await send_webhook("https://example.com/hook", {})
        assert outcome.retry_after_s is None
        assert outcome.retryable is True

    async def test_permanent_status_ignores_retry_after(self) -> None:
        client = self._client_returning(422, {"Retry-After": "60"})
        with patch("app.webhooks.httpx.AsyncClient", return_value=client):
            outcome = await send_webhook("https://example.com/hook", {})
        assert outcome.retryable is False
        assert outcome.retry_after_s is None

    async def test_blocked_targets_are_terminal_not_transient(self) -> None:
        """A private address or a bad scheme will not become valid on a retry."""
        for url in ("http://127.0.0.1:8080/hook", "file:///etc/passwd", "http://[::1"):
            outcome = await send_webhook(url, {})
            assert outcome.ok is False, url
            assert outcome.retryable is False, url


class TestWebhookIntents:
    """Building and delivering per-target webhook intents."""

    def test_no_webhook_urls_builds_no_intents(self) -> None:
        settings = _make_settings(notifications=NotificationSettings(urls=[], webhook_urls=[]))
        assert build_webhook_intents("My Topic", _make_novelty(), settings, 1) == []

    def test_one_intent_per_configured_url(self) -> None:
        urls = ["https://a.com/hook", "https://b.com/hook"]
        settings = _make_settings(notifications=NotificationSettings(urls=[], webhook_urls=urls))
        intents = build_webhook_intents("My Topic", _make_novelty(), settings, 7, 42)
        assert [i.url for i in intents] == urls
        assert {i.topic_id for i in intents} == {7}
        assert {i.check_result_id for i in intents} == {42}

    def test_intent_payload_carries_the_novelty_fields(self) -> None:
        settings = _make_settings(
            notifications=NotificationSettings(urls=[], webhook_urls=["https://hook.example.com"])
        )
        novelty = _make_novelty(
            summary="Big news", key_facts=["fact1"], source_urls=["https://src.com"], confidence=0.9
        )
        payload = build_webhook_intents("My Topic", novelty, settings, 1)[0].payload
        assert payload["topic"] == "My Topic"
        assert payload["summary"] == "Big news"
        assert payload["key_facts"] == ["fact1"]
        assert payload["source_urls"] == ["https://src.com"]
        assert payload["confidence"] == pytest.approx(0.9)
        assert "timestamp" in payload

    async def test_returns_count_of_successful_deliveries(self, db_conn: sqlite3.Connection) -> None:
        topic = _make_topic(db_conn)
        settings = _make_settings(
            notifications=NotificationSettings(
                urls=[],
                webhook_urls=["https://a.com/hook", "https://b.com/hook", "https://c.com/hook"],
            )
        )

        async def fake_send(url: str, payload: dict, timeout: float = 10.0) -> WebhookOutcome:
            return _ok() if url != "https://c.com/hook" else _fail(status=500, error="HTTP 500")

        with patch("app.webhooks.send_webhook", side_effect=fake_send):
            delivered = await _deliver(db_conn, topic.id, settings)

        assert delivered == 2

    async def test_sends_to_all_configured_urls(self, db_conn: sqlite3.Connection) -> None:
        topic = _make_topic(db_conn)
        urls = ["https://a.com/hook", "https://b.com/hook"]
        settings = _make_settings(notifications=NotificationSettings(urls=[], webhook_urls=urls))
        called: list[str] = []

        async def capture(url: str, payload: dict, timeout: float = 10.0) -> WebhookOutcome:
            called.append(url)
            return _ok()

        with patch("app.webhooks.send_webhook", side_effect=capture):
            await _deliver(db_conn, topic.id, settings)

        assert sorted(called) == sorted(urls)


# --- pending_webhooks retry queue ---


def _make_topic(conn: sqlite3.Connection) -> Topic:
    topic = create_topic(conn, Topic(name="Hooked", description="d", status=TopicStatus.READY))
    conn.commit()
    return topic


class TestWebhookRetryQueue:
    """Tests for the persistent webhook delivery queue."""

    async def test_failed_webhook_stays_queued(self, db_conn: sqlite3.Connection) -> None:
        """A failed delivery keeps its intent, which is what the retry drain finds."""
        topic = _make_topic(db_conn)
        settings = _make_settings(notifications=NotificationSettings(urls=[], webhook_urls=["https://a.com/hook"]))

        with patch("app.webhooks.send_webhook", return_value=_fail(status=500, error="HTTP 500")):
            count = await _deliver(db_conn, topic.id, settings)

        assert count == 0
        pending = list_pending_webhooks(db_conn)
        assert len(pending) == 1
        assert pending[0].url == "https://a.com/hook"
        assert pending[0].topic_id == topic.id
        assert pending[0].payload["topic"] == "Hooked"
        assert pending[0].last_error == "HTTP 500"

    async def test_successful_webhook_leaves_the_queue(self, db_conn: sqlite3.Connection) -> None:
        topic = _make_topic(db_conn)
        settings = _make_settings(notifications=NotificationSettings(urls=[], webhook_urls=["https://a.com/hook"]))

        with patch("app.webhooks.send_webhook", return_value=_ok()):
            await _deliver(db_conn, topic.id, settings)

        assert list_pending_webhooks(db_conn) == []
        row = db_conn.execute("SELECT status, delivered_at FROM pending_webhooks").fetchone()
        assert row["status"] == "sent"
        assert row["delivered_at"] is not None

    async def test_intent_exists_before_any_post(self, db_conn: sqlite3.Connection) -> None:
        """TW-AUD-004: the row is durable before the first POST, not after a failure."""
        topic = _make_topic(db_conn)
        settings = _make_settings(notifications=NotificationSettings(urls=[], webhook_urls=["https://a.com/hook"]))
        seen: list[int] = []

        async def observe(url: str, payload: dict, timeout: float = 10.0) -> WebhookOutcome:
            seen.append(db_conn.execute("SELECT COUNT(*) FROM pending_webhooks").fetchone()[0])
            return _ok()

        with patch("app.webhooks.send_webhook", side_effect=observe):
            await _deliver(db_conn, topic.id, settings)

        assert seen == [1]

    async def test_retry_resends_and_records_on_success(self, db_conn: sqlite3.Connection) -> None:
        topic = _make_topic(db_conn)
        settings = _make_settings(notifications=NotificationSettings(urls=[], webhook_urls=["https://a.com/hook"]))

        with patch("app.webhooks.send_webhook", return_value=_fail(status=500, error="HTTP 500")):
            await _deliver(db_conn, topic.id, settings)
        assert len(list_pending_webhooks(db_conn)) == 1
        # A retry is scheduled, so clear the backoff to exercise the drain now.
        db_conn.execute("UPDATE pending_webhooks SET next_attempt_at = NULL")
        db_conn.commit()

        with patch("app.webhooks.send_webhook", new_callable=AsyncMock, return_value=_ok()) as mock_send:
            await retry_pending_webhooks(db_conn, settings)

        mock_send.assert_awaited_once()
        sent_url, sent_payload = mock_send.await_args.args[0], mock_send.await_args.args[1]
        assert sent_url == "https://a.com/hook"
        assert sent_payload["topic"] == "Hooked"
        assert list_pending_webhooks(db_conn) == []

    async def test_retry_failure_increments_and_keeps(self, db_conn: sqlite3.Connection) -> None:
        topic = _make_topic(db_conn)
        settings = _make_settings(notifications=NotificationSettings(urls=[], webhook_urls=["https://a.com/hook"]))

        with patch("app.webhooks.send_webhook", return_value=_fail(status=500, error="HTTP 500")):
            await _deliver(db_conn, topic.id, settings)
        db_conn.execute("UPDATE pending_webhooks SET next_attempt_at = NULL")
        db_conn.commit()

        with patch("app.webhooks.send_webhook", new_callable=AsyncMock, return_value=_fail(status=503)):
            await retry_pending_webhooks(db_conn, settings)

        row = db_conn.execute("SELECT retry_count FROM pending_webhooks").fetchone()
        assert row["retry_count"] == 2
        # Still pending (default max_retries=3).
        assert len(list_pending_webhooks(db_conn)) == 1

    async def test_exhausted_retries_are_abandoned(self, db_conn: sqlite3.Connection) -> None:
        topic = _make_topic(db_conn)
        settings = _make_settings(notifications=NotificationSettings(urls=[], webhook_urls=["https://a.com/hook"]))

        with patch("app.webhooks.send_webhook", return_value=_fail(status=500, error="HTTP 500")):
            await _deliver(db_conn, topic.id, settings)

        for _ in range(4):
            db_conn.execute("UPDATE pending_webhooks SET next_attempt_at = NULL")
            db_conn.commit()
            with patch("app.webhooks.send_webhook", new_callable=AsyncMock, return_value=_fail(status=503)):
                await retry_pending_webhooks(db_conn, settings)

        assert list_pending_webhooks(db_conn) == []
        row = db_conn.execute("SELECT status, retry_count FROM pending_webhooks").fetchone()
        assert row["status"] == "abandoned"
        assert row["retry_count"] == 3

    async def test_terminal_status_abandons_without_retry(self, db_conn: sqlite3.Connection) -> None:
        """AUG-324: a 422 will still be a 422; do not burn the retry budget on it."""
        topic = _make_topic(db_conn)
        settings = _make_settings(notifications=NotificationSettings(urls=[], webhook_urls=["https://a.com/hook"]))

        with patch("app.webhooks.send_webhook", return_value=_fail(status=422, retryable=False, error="HTTP 422")):
            await _deliver(db_conn, topic.id, settings)

        row = db_conn.execute("SELECT status, retry_count, next_attempt_at FROM pending_webhooks").fetchone()
        assert row["status"] == "abandoned"
        assert row["retry_count"] == 1
        assert row["next_attempt_at"] is None
        assert list_pending_webhooks(db_conn) == []

    async def test_retry_after_sets_the_due_time_and_the_drain_waits(self, db_conn: sqlite3.Connection) -> None:
        """AUG-324: a 429 is scheduled at the receiver's stated recovery time."""
        topic = _make_topic(db_conn)
        settings = _make_settings(notifications=NotificationSettings(urls=[], webhook_urls=["https://a.com/hook"]))

        with patch(
            "app.webhooks.send_webhook",
            return_value=_fail(status=429, retry_after_s=1800.0, error="HTTP 429"),
        ):
            await _deliver(db_conn, topic.id, settings)

        row = db_conn.execute("SELECT status, next_attempt_at FROM pending_webhooks").fetchone()
        assert row["status"] == "pending"
        due = datetime.fromisoformat(row["next_attempt_at"])
        assert timedelta(minutes=25) < due - datetime.now(UTC) < timedelta(minutes=35)

        # Not due yet: the drain must skip it entirely.
        with patch("app.webhooks.send_webhook", new_callable=AsyncMock, return_value=_ok()) as mock_send:
            await retry_pending_webhooks(db_conn, settings)
        mock_send.assert_not_awaited()

        # Once due, it goes out.
        db_conn.execute("UPDATE pending_webhooks SET next_attempt_at = NULL")
        db_conn.commit()
        with patch("app.webhooks.send_webhook", new_callable=AsyncMock, return_value=_ok()) as mock_send:
            await retry_pending_webhooks(db_conn, settings)
        mock_send.assert_awaited_once()

    async def test_an_escaping_error_still_records_a_retryable_failure(self, db_conn: sqlite3.Connection) -> None:
        """An exception must land an outcome, or the row is stuck 'sending' for good.

        Nothing else frees it: retry_count never moves so it is never abandoned,
        retention prunes only terminal rows, and the queue view hides 'sending'.
        """
        topic = _make_topic(db_conn)
        settings = _make_settings(notifications=NotificationSettings(urls=[], webhook_urls=["https://a.com/hook"]))

        async def boom(url: str, payload: dict, timeout: float = 10.0) -> WebhookOutcome:
            raise RuntimeError("unexpected")

        with patch("app.webhooks.send_webhook", side_effect=boom):
            delivered = await _deliver(db_conn, topic.id, settings)

        assert delivered == 0
        row = db_conn.execute("SELECT * FROM pending_webhooks").fetchone()
        assert row["status"] == "pending"
        assert row["retry_count"] == 1
        assert row["last_error"] == "RuntimeError"
        assert row["next_attempt_at"] is not None

    async def test_a_failed_apply_after_a_delivered_post_is_not_re_sent(self, db_conn: sqlite3.Connection) -> None:
        """Exactly-once holds when the apply write fails, not just when the POST does.

        A locked database after a POST the endpoint accepted must not re-arm the
        row: the receiver would get the payload twice.
        """
        topic = _make_topic(db_conn)
        settings = _make_settings(notifications=NotificationSettings(urls=[], webhook_urls=["https://a.com/hook"]))
        posts: list[str] = []

        async def record(url: str, payload: dict, timeout: float = 10.0) -> WebhookOutcome:
            posts.append(url)
            return _ok()

        applies = {"n": 0}

        def flaky_apply(*args, **kwargs):  # noqa: ANN002, ANN003
            applies["n"] += 1
            if applies["n"] == 1:
                raise sqlite3.OperationalError("database is locked")
            return apply_webhook_outcome(*args, **kwargs)

        with (
            patch("app.webhooks.send_webhook", side_effect=record),
            patch("app.webhooks.apply_webhook_outcome", side_effect=flaky_apply),
        ):
            await _deliver(db_conn, topic.id, settings)
            # The stale-claim window elapses; a re-armed row would POST again.
            release_stale_webhook_claims(db_conn, to_db_utc(datetime.now(UTC) + timedelta(hours=1)))
            db_conn.commit()
            await retry_pending_webhooks(db_conn, settings)

        assert posts == ["https://a.com/hook"]
        row = db_conn.execute("SELECT status FROM pending_webhooks").fetchone()
        assert row["status"] == "sent"


class TestAbandonedWebhookLogging:
    """A permanently-dropped webhook must be observable (OVH-040)."""

    async def test_abandoned_webhook_warns_with_ids_and_redacted_url(self, db_conn: sqlite3.Connection, caplog) -> None:  # noqa: ANN001
        """Retiring an exhausted delivery emits a WARNING naming topic/check ids.

        The secret-bearing full URL must NOT appear in the log; only the
        redacted destination.
        """
        from app.crud import create_pending_webhook

        topic = _make_topic(db_conn)
        secret_url = "https://hooks.slack.com/services/T0/B0/SECRETWEBHOOKTOKEN123"
        wid = create_pending_webhook(
            db_conn,
            topic_id=topic.id,
            url=secret_url,
            payload={"topic": "Hooked"},
            check_result_id=4242,
        )
        # Drive it straight to exhaustion.
        db_conn.execute("UPDATE pending_webhooks SET retry_count = max_retries WHERE id = ?", (wid,))
        db_conn.commit()
        settings = _make_settings(notifications=NotificationSettings(urls=[], webhook_urls=[]))

        import logging

        with caplog.at_level(logging.WARNING, logger="app.webhooks"):
            await retry_pending_webhooks(db_conn, settings)

        abandon_logs = [r.getMessage() for r in caplog.records if "Abandoning webhook" in r.getMessage()]
        assert len(abandon_logs) == 1
        msg = abandon_logs[0]
        assert f"topic_id={topic.id}" in msg
        assert "check_result_id=4242" in msg
        # Redacted host present, secret token absent.
        assert "hooks.slack.com" in msg
        assert "SECRETWEBHOOKTOKEN123" not in msg
        # The row is retired, not deleted: it is the delivery ledger (AUG-153).
        row = db_conn.execute("SELECT status FROM pending_webhooks WHERE id = ?", (wid,)).fetchone()
        assert row["status"] == "abandoned"


class TestWebhookRetryCrashSafety:
    """Per-item commits must survive a mid-loop crash (no rollback of work)."""

    async def test_crash_midloop_preserves_already_applied_results(self, db_conn: sqlite3.Connection) -> None:
        """If applying item 2 crashes, item 1's apply must already be committed.

        Old code committed once after the whole loop, so a crash rolled back
        every apply from that pass — letting a failing URL retry unbounded.
        Per-item commits must keep already-applied work durable, and the drain
        must keep ownership of every sibling until it settles (AUG-263).
        """
        topic = _make_topic(db_conn)
        settings = _make_settings(
            notifications=NotificationSettings(
                urls=[],
                webhook_urls=["https://first.com/hook", "https://second.com/hook"],
            )
        )

        # Enqueue two intents.
        with patch("app.webhooks.send_webhook", return_value=_fail(status=500, error="HTTP 500")):
            await _deliver(db_conn, topic.id, settings)
        db_conn.execute("UPDATE pending_webhooks SET next_attempt_at = NULL")
        db_conn.commit()
        pending = list_pending_webhooks(db_conn)
        assert len(pending) == 2
        first_id, second_id = pending[0].id, pending[1].id

        # Retry: both sends "succeed", but every apply for the SECOND crashes —
        # the recovery apply included, as a genuinely unwritable database would.
        from app.crud import apply_webhook_outcome as real_apply

        def crashing_apply(conn, intent_id, claim_token, **kwargs):  # noqa: ANN001
            if intent_id == second_id:
                raise RuntimeError("simulated crash applying item 2")
            return real_apply(conn, intent_id, claim_token, **kwargs)

        with (
            patch("app.webhooks.send_webhook", new_callable=AsyncMock, return_value=_ok()),
            patch("app.webhooks.apply_webhook_outcome", side_effect=crashing_apply),
        ):
            await retry_pending_webhooks(db_conn, settings)

        # Item 1's apply was committed before item 2 crashed.
        statuses = {r["id"]: r["status"] for r in db_conn.execute("SELECT id, status FROM pending_webhooks").fetchall()}
        assert statuses[first_id] == "sent"
        assert sorted(statuses.values()) == ["sending", "sent"]

    async def test_no_connection_held_across_send(self, db_conn: sqlite3.Connection) -> None:
        """The network send must run with no open transaction on the snapshot conn."""
        topic = _make_topic(db_conn)
        settings = _make_settings(notifications=NotificationSettings(urls=[], webhook_urls=["https://a.com/hook"]))

        with patch("app.webhooks.send_webhook", return_value=_fail(status=500, error="HTTP 500")):
            await _deliver(db_conn, topic.id, settings)
        db_conn.execute("UPDATE pending_webhooks SET next_attempt_at = NULL")
        db_conn.commit()

        in_transaction_during_send: list[bool] = []

        async def observe(url: str, payload: dict, timeout: float = 10.0) -> WebhookOutcome:
            # When a connection holds an uncommitted write, in_transaction is
            # True. The claim must have been committed before the send.
            in_transaction_during_send.append(db_conn.in_transaction)
            return _ok()

        with patch("app.webhooks.send_webhook", side_effect=observe):
            await retry_pending_webhooks(db_conn, settings)

        assert in_transaction_during_send == [False]

    async def test_db_path_mode_no_conn(self, tmp_path) -> None:  # noqa: ANN001
        """Scheduler-style call (db_path, no conn) retries and clears on success."""
        from app.database import get_connection, init_db

        db_path = tmp_path / "test.db"
        init_db(db_path)
        conn = get_connection(db_path)
        topic = _make_topic(conn)
        settings = _make_settings(notifications=NotificationSettings(urls=[], webhook_urls=["https://a.com/hook"]))

        with patch("app.webhooks.send_webhook", return_value=_fail(status=500, error="HTTP 500")):
            await _deliver(conn, topic.id, settings)
        conn.execute("UPDATE pending_webhooks SET next_attempt_at = NULL")
        conn.commit()
        conn.close()

        with patch("app.webhooks.send_webhook", new_callable=AsyncMock, return_value=_ok()):
            await retry_pending_webhooks(settings=settings, db_path=db_path)

        verify = get_connection(db_path)
        try:
            assert list_pending_webhooks(verify) == []
        finally:
            verify.close()


class TestWebhookDrainSingleFlight:
    """Overlapping drains must deliver each queued webhook exactly once (OVH-017)."""

    async def _enqueue(self, db_path, urls: list[str]) -> None:  # noqa: ANN001
        from app.database import get_connection

        conn = get_connection(db_path)
        try:
            topic = _make_topic(conn)
            settings = _make_settings(notifications=NotificationSettings(urls=[], webhook_urls=urls))
            create_webhook_intents(conn, build_webhook_intents("Hooked", _make_novelty(), settings, topic.id))
            conn.commit()
        finally:
            conn.close()

    async def test_two_concurrent_drains_deliver_each_item_once(self, tmp_path) -> None:  # noqa: ANN001
        """Two drains launched together send each pending webhook exactly once."""
        import asyncio
        import collections

        from app.database import get_connection, init_db

        db_path = tmp_path / "test.db"
        init_db(db_path)
        urls = ["https://a.com/hook", "https://b.com/hook", "https://c.com/hook"]
        await self._enqueue(db_path, urls)
        settings = _make_settings(notifications=NotificationSettings(urls=[], webhook_urls=urls))

        sent_counts: collections.Counter[str] = collections.Counter()
        first_send_started = asyncio.Event()
        release = asyncio.Event()

        async def slow_send(url: str, payload: dict, timeout: float = 10.0) -> WebhookOutcome:
            # Block the first drain mid-send so the second drain overlaps it.
            first_send_started.set()
            await release.wait()
            sent_counts[url] += 1
            return _ok()

        with patch("app.webhooks.send_webhook", side_effect=slow_send):
            drain1 = asyncio.create_task(retry_pending_webhooks(settings=settings, db_path=db_path))
            await first_send_started.wait()
            # Second drain starts while the first holds the single-flight lock.
            drain2 = asyncio.create_task(retry_pending_webhooks(settings=settings, db_path=db_path))
            await asyncio.sleep(0)  # let drain2 observe the locked guard
            release.set()
            await asyncio.gather(drain1, drain2)

        # Each URL delivered exactly once despite the overlapping drain.
        assert dict(sent_counts) == dict.fromkeys(urls, 1)

        verify = get_connection(db_path)
        try:
            assert list_pending_webhooks(verify) == []
        finally:
            verify.close()

    async def test_claimed_row_skipped_by_second_drainer(self, tmp_path) -> None:  # noqa: ANN001
        """A row another process already claimed is skipped, not re-sent."""
        from app.crud import claim_webhook_intent
        from app.database import get_connection, init_db

        db_path = tmp_path / "test.db"
        init_db(db_path)
        await self._enqueue(db_path, ["https://a.com/hook"])
        settings = _make_settings(notifications=NotificationSettings(urls=[], webhook_urls=["https://a.com/hook"]))

        # Simulate another process having claimed the only pending row.
        claimer = get_connection(db_path)
        try:
            row = claimer.execute("SELECT id FROM pending_webhooks").fetchone()
            webhook_id = row["id"]
            assert claim_webhook_intent(claimer, webhook_id, "other-owner", "2999-01-01T00:00:00+00:00") is True
            claimer.commit()
        finally:
            claimer.close()

        send_calls: list[str] = []

        async def record_send(url: str, payload: dict, timeout: float = 10.0) -> WebhookOutcome:
            send_calls.append(url)
            return _ok()

        with patch("app.webhooks.send_webhook", side_effect=record_send):
            await retry_pending_webhooks(settings=settings, db_path=db_path)

        # The claimed row was neither listed nor re-sent by this drainer.
        assert send_calls == []
        verify = get_connection(db_path)
        try:
            # Row still present and still claimed (untouched by this drain).
            row = verify.execute("SELECT claimed_at FROM pending_webhooks").fetchone()
            assert row is not None
            assert row["claimed_at"] == "2999-01-01T00:00:00+00:00"
        finally:
            verify.close()


class TestWebhookDrainBoundedConcurrency:
    """OVH-139: a single drain processes its queue with bounded concurrency.

    A backlog of K failed deliveries must NOT serialize to K x timeout. The
    drain mirrors the live path (bounded asyncio.gather), while still claiming
    each row exactly once (the 1.6 single-flight/claim invariant must hold).
    """

    async def _enqueue(self, db_path, urls: list[str]) -> None:  # noqa: ANN001
        from app.database import get_connection

        conn = get_connection(db_path)
        try:
            topic = _make_topic(conn)
            settings = _make_settings(notifications=NotificationSettings(urls=[], webhook_urls=urls))
            create_webhook_intents(conn, build_webhook_intents("Hooked", _make_novelty(), settings, topic.id))
            conn.commit()
        finally:
            conn.close()

    async def test_sends_overlap_within_one_drain(self, tmp_path) -> None:  # noqa: ANN001
        """Multiple pending sends run concurrently, not strictly one-at-a-time."""
        import asyncio

        from app.database import get_connection, init_db

        db_path = tmp_path / "test.db"
        init_db(db_path)
        urls = ["https://a.com/hook", "https://b.com/hook", "https://c.com/hook"]
        await self._enqueue(db_path, urls)
        settings = _make_settings(notifications=NotificationSettings(urls=[], webhook_urls=urls))

        concurrent = 0
        max_concurrent = 0
        gate = asyncio.Event()
        started = 0

        async def overlapping_send(url: str, payload: dict, timeout: float = 10.0) -> WebhookOutcome:
            nonlocal concurrent, max_concurrent, started
            concurrent += 1
            started += 1
            max_concurrent = max(max_concurrent, concurrent)
            # Release the gate once all three sends are in flight; if the drain
            # were strictly sequential this would deadlock (only one ever starts).
            if started >= len(urls):
                gate.set()
            try:
                await asyncio.wait_for(gate.wait(), timeout=5.0)
            finally:
                concurrent -= 1
            return _ok()

        with patch("app.webhooks.send_webhook", side_effect=overlapping_send):
            await retry_pending_webhooks(settings=settings, db_path=db_path)

        # Sequential code would peak at 1; bounded-concurrent overlaps them.
        assert max_concurrent >= 2

        verify = get_connection(db_path)
        try:
            assert list_pending_webhooks(verify) == []
        finally:
            verify.close()

    async def test_each_item_claimed_exactly_once_no_double_send(self, tmp_path) -> None:  # noqa: ANN001
        """Bounded concurrency must not break the per-item claim: one send each."""
        import collections

        from app.crud import claim_webhook_intent
        from app.database import get_connection, init_db

        db_path = tmp_path / "test.db"
        init_db(db_path)
        urls = [f"https://h{i}.com/hook" for i in range(5)]
        await self._enqueue(db_path, urls)
        settings = _make_settings(notifications=NotificationSettings(urls=[], webhook_urls=urls))

        sent_counts: collections.Counter[str] = collections.Counter()
        claim_calls = {"n": 0}

        def counting_claim(conn, intent_id, claim_token, now_iso):  # noqa: ANN001
            claim_calls["n"] += 1
            return claim_webhook_intent(conn, intent_id, claim_token, now_iso)

        async def record_send(url: str, payload: dict, timeout: float = 10.0) -> WebhookOutcome:
            sent_counts[url] += 1
            return _ok()

        with (
            patch("app.webhooks.send_webhook", side_effect=record_send),
            patch("app.webhooks.claim_webhook_intent", side_effect=counting_claim),
        ):
            await retry_pending_webhooks(settings=settings, db_path=db_path)

        # Exactly one claim attempt per row and exactly one send per URL.
        assert claim_calls["n"] == len(urls)
        assert dict(sent_counts) == dict.fromkeys(urls, 1)

        verify = get_connection(db_path)
        try:
            assert list_pending_webhooks(verify) == []
        finally:
            verify.close()


class TestWebhookTotalDeadline:
    """The configured timeout bounds total wall time, not just each phase (AUG-247)."""

    async def test_trickling_response_hits_the_total_deadline(self) -> None:
        import asyncio

        async def _slow_body():
            for _ in range(200):
                await asyncio.sleep(0.02)
                yield b"x" * 16

        def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=_slow_body())

        transport = httpx.MockTransport(_handler)
        real_client = httpx.AsyncClient

        def _client_with_transport(*_args, **kwargs):
            kwargs.pop("transport", None)
            return real_client(transport=transport, **kwargs)

        start = asyncio.get_running_loop().time()
        with patch("app.webhooks.httpx.AsyncClient", side_effect=_client_with_transport):
            result = await send_webhook("https://hook.example.com/x", {"key": "value"}, timeout=0.3)
        elapsed = asyncio.get_running_loop().time() - start

        assert result.ok is False
        assert result.error == "deadline exceeded"
        # A slow endpoint is transient, so the intent is rescheduled rather than
        # abandoned.
        assert result.retryable is True
        # Each chunk lands inside httpx's per-operation read timeout, so only a
        # total deadline can stop this: 200 x 20ms would otherwise take 4s.
        assert elapsed < 2.0
