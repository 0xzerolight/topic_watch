"""Tests for the webhook delivery module."""

import sqlite3
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.analysis.llm import NoveltyResult
from app.config import LLMSettings, NotificationSettings, Settings
from app.crud import create_topic, list_pending_webhooks
from app.models import Topic, TopicStatus
from app.webhooks import (
    _build_webhook_payload,
    retry_pending_webhooks,
    send_webhook,
    send_webhooks,
)


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
            "timestamp",
        }
        assert set(payload.keys()) == expected_keys

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

        assert result is True

    async def test_returns_false_on_timeout(self) -> None:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("app.webhooks.httpx.AsyncClient", return_value=mock_client):
            result = await send_webhook("https://example.com/hook", {})

        assert result is False

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

        assert result is False

    async def test_returns_false_on_connection_error(self) -> None:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("Connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("app.webhooks.httpx.AsyncClient", return_value=mock_client):
            result = await send_webhook("https://example.com/hook", {})

        assert result is False

    async def test_blocks_private_url(self) -> None:
        result = await send_webhook("http://localhost:9200/hook", {"key": "value"})
        assert result is False

    async def test_blocks_internal_ip(self) -> None:
        result = await send_webhook("http://169.254.169.254/metadata", {})
        assert result is False

    async def test_blocks_loopback_ip(self) -> None:
        result = await send_webhook("http://127.0.0.1:8080/hook", {})
        assert result is False

    async def test_blocks_file_scheme_before_post(self) -> None:
        """file:// is rejected by the scheme allowlist before any POST (OVH-141).

        The httpx client must never be constructed for a non-http(s) scheme, so
        a transport/config change can't expose the first hop.
        """
        with patch("app.webhooks.httpx.AsyncClient") as mock_cls:
            result = await send_webhook("file:///etc/passwd", {"key": "value"})
        assert result is False
        mock_cls.assert_not_called()

    async def test_blocks_gopher_scheme_before_post(self) -> None:
        """gopher:// is rejected by the scheme allowlist before any POST (OVH-141)."""
        with patch("app.webhooks.httpx.AsyncClient") as mock_cls:
            result = await send_webhook("gopher://example.com/7", {})
        assert result is False
        mock_cls.assert_not_called()

    async def test_blocks_ftp_scheme_before_post(self) -> None:
        """ftp:// is rejected by the scheme allowlist before any POST (OVH-141)."""
        with patch("app.webhooks.httpx.AsyncClient") as mock_cls:
            result = await send_webhook("ftp://example.com/file", {})
        assert result is False
        mock_cls.assert_not_called()

    async def test_malformed_ipv6_url_returns_false_not_raises(self) -> None:
        """OVH-131: a malformed IPv6 literal makes urlparse raise ValueError, but
        send_webhook honors its 'Never raises' contract — returns False, no POST.
        """
        # urlparse("http://[::1") raises ValueError("Invalid IPv6 URL").
        with patch("app.webhooks.httpx.AsyncClient") as mock_cls:
            result = await send_webhook("http://[::1", {"key": "value"})
        assert result is False
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


class TestSendWebhooks:
    """Tests for the send_webhooks orchestrator."""

    async def test_returns_zero_with_no_webhook_urls(self) -> None:
        settings = _make_settings(notifications=NotificationSettings(urls=[], webhook_urls=[]))
        novelty = _make_novelty()

        result = await send_webhooks("My Topic", novelty, settings)

        assert result == 0

    async def test_returns_count_of_successful_deliveries(self) -> None:
        settings = _make_settings(
            notifications=NotificationSettings(
                urls=[],
                webhook_urls=["https://a.com/hook", "https://b.com/hook", "https://c.com/hook"],
            )
        )
        novelty = _make_novelty()

        # First two succeed, third fails
        async def fake_send_webhook(url: str, payload: dict, timeout: float = 10.0) -> bool:
            return url != "https://c.com/hook"

        with patch("app.webhooks.send_webhook", side_effect=fake_send_webhook):
            result = await send_webhooks("My Topic", novelty, settings)

        assert result == 2

    async def test_returns_full_count_when_all_succeed(self) -> None:
        settings = _make_settings(
            notifications=NotificationSettings(
                urls=[],
                webhook_urls=["https://a.com/hook", "https://b.com/hook"],
            )
        )
        novelty = _make_novelty()

        with patch("app.webhooks.send_webhook", return_value=True):
            result = await send_webhooks("My Topic", novelty, settings)

        assert result == 2

    async def test_returns_zero_when_all_fail(self) -> None:
        settings = _make_settings(
            notifications=NotificationSettings(
                urls=[],
                webhook_urls=["https://a.com/hook", "https://b.com/hook"],
            )
        )
        novelty = _make_novelty()

        with patch("app.webhooks.send_webhook", return_value=False):
            result = await send_webhooks("My Topic", novelty, settings)

        assert result == 0

    async def test_sends_to_all_configured_urls(self) -> None:
        urls = ["https://a.com/hook", "https://b.com/hook"]
        settings = _make_settings(notifications=NotificationSettings(urls=[], webhook_urls=urls))
        novelty = _make_novelty()
        called_urls: list[str] = []

        async def capture_url(url: str, payload: dict, timeout: float = 10.0) -> bool:
            called_urls.append(url)
            return True

        with patch("app.webhooks.send_webhook", side_effect=capture_url):
            await send_webhooks("My Topic", novelty, settings)

        assert sorted(called_urls) == sorted(urls)

    async def test_passes_correct_payload_fields(self) -> None:
        settings = _make_settings(
            notifications=NotificationSettings(
                urls=[],
                webhook_urls=["https://hook.example.com"],
            )
        )
        novelty = _make_novelty(
            summary="Big news",
            key_facts=["fact1"],
            source_urls=["https://src.com"],
            confidence=0.9,
        )
        captured_payloads: list[dict] = []

        async def capture_payload(url: str, payload: dict, timeout: float = 10.0) -> bool:
            captured_payloads.append(payload)
            return True

        with patch("app.webhooks.send_webhook", side_effect=capture_payload):
            await send_webhooks("My Topic", novelty, settings)

        assert len(captured_payloads) == 1
        payload = captured_payloads[0]
        assert payload["topic"] == "My Topic"
        assert payload["summary"] == "Big news"
        assert payload["key_facts"] == ["fact1"]
        assert payload["source_urls"] == ["https://src.com"]
        assert payload["confidence"] == pytest.approx(0.9)
        assert "timestamp" in payload


# --- pending_webhooks retry queue ---


def _make_topic(conn: sqlite3.Connection) -> Topic:
    topic = create_topic(conn, Topic(name="Hooked", description="d", status=TopicStatus.READY))
    conn.commit()
    return topic


class TestWebhookRetryQueue:
    """Tests for the persistent webhook retry queue."""

    async def test_failed_webhook_is_enqueued(self, db_conn: sqlite3.Connection) -> None:
        """A failed delivery is persisted to pending_webhooks instead of dropped."""
        topic = _make_topic(db_conn)
        settings = _make_settings(notifications=NotificationSettings(urls=[], webhook_urls=["https://a.com/hook"]))
        novelty = _make_novelty()

        with patch("app.webhooks.send_webhook", return_value=False):
            count = await send_webhooks("Hooked", novelty, settings, conn=db_conn, topic_id=topic.id)

        assert count == 0
        pending = list_pending_webhooks(db_conn)
        assert len(pending) == 1
        assert pending[0]["url"] == "https://a.com/hook"
        assert pending[0]["topic_id"] == topic.id
        assert pending[0]["payload"]["topic"] == "Hooked"

    async def test_successful_webhook_not_enqueued(self, db_conn: sqlite3.Connection) -> None:
        topic = _make_topic(db_conn)
        settings = _make_settings(notifications=NotificationSettings(urls=[], webhook_urls=["https://a.com/hook"]))
        novelty = _make_novelty()

        with patch("app.webhooks.send_webhook", return_value=True):
            await send_webhooks("Hooked", novelty, settings, conn=db_conn, topic_id=topic.id)

        assert list_pending_webhooks(db_conn) == []

    async def test_no_enqueue_without_conn(self, db_conn: sqlite3.Connection) -> None:
        """Without a conn/topic_id, failures are not enqueued (legacy behaviour)."""
        settings = _make_settings(notifications=NotificationSettings(urls=[], webhook_urls=["https://a.com/hook"]))
        novelty = _make_novelty()

        with patch("app.webhooks.send_webhook", return_value=False):
            await send_webhooks("Hooked", novelty, settings)

        assert list_pending_webhooks(db_conn) == []

    async def test_retry_resends_and_clears_on_success(self, db_conn: sqlite3.Connection) -> None:
        topic = _make_topic(db_conn)
        settings = _make_settings(notifications=NotificationSettings(urls=[], webhook_urls=["https://a.com/hook"]))
        novelty = _make_novelty()

        # First delivery fails → enqueued.
        with patch("app.webhooks.send_webhook", return_value=False):
            await send_webhooks("Hooked", novelty, settings, conn=db_conn, topic_id=topic.id)
        assert len(list_pending_webhooks(db_conn)) == 1

        # Retry succeeds → row cleared.
        with patch("app.webhooks.send_webhook", new_callable=AsyncMock, return_value=True) as mock_send:
            await retry_pending_webhooks(db_conn, settings)

        mock_send.assert_awaited_once()
        sent_url, sent_payload = mock_send.await_args.args[0], mock_send.await_args.args[1]
        assert sent_url == "https://a.com/hook"
        assert sent_payload["topic"] == "Hooked"
        assert list_pending_webhooks(db_conn) == []

    async def test_retry_failure_increments_and_keeps(self, db_conn: sqlite3.Connection) -> None:
        topic = _make_topic(db_conn)
        settings = _make_settings(notifications=NotificationSettings(urls=[], webhook_urls=["https://a.com/hook"]))
        novelty = _make_novelty()

        with patch("app.webhooks.send_webhook", return_value=False):
            await send_webhooks("Hooked", novelty, settings, conn=db_conn, topic_id=topic.id)

        with patch("app.webhooks.send_webhook", new_callable=AsyncMock, return_value=False):
            await retry_pending_webhooks(db_conn, settings)

        row = db_conn.execute("SELECT retry_count FROM pending_webhooks").fetchone()
        assert row["retry_count"] == 1
        # Still pending (default max_retries=3).
        assert len(list_pending_webhooks(db_conn)) == 1

    async def test_exhausted_retries_are_dropped(self, db_conn: sqlite3.Connection) -> None:
        topic = _make_topic(db_conn)
        settings = _make_settings(notifications=NotificationSettings(urls=[], webhook_urls=["https://a.com/hook"]))
        novelty = _make_novelty()

        with patch("app.webhooks.send_webhook", return_value=False):
            await send_webhooks("Hooked", novelty, settings, conn=db_conn, topic_id=topic.id)

        # Fail the retry up to max_retries; the row hits retry_count == max_retries
        # and is purged by delete_expired_webhooks on the following pass.
        for _ in range(4):
            with patch("app.webhooks.send_webhook", new_callable=AsyncMock, return_value=False):
                await retry_pending_webhooks(db_conn, settings)

        assert list_pending_webhooks(db_conn) == []
        remaining = db_conn.execute("SELECT COUNT(*) FROM pending_webhooks").fetchone()[0]
        assert remaining == 0


class TestAbandonedWebhookLogging:
    """A permanently-dropped webhook must be observable (OVH-040)."""

    async def test_abandoned_webhook_warns_with_ids_and_redacted_url(self, db_conn: sqlite3.Connection, caplog) -> None:  # noqa: ANN001
        """Pruning an exhausted delivery emits a WARNING naming topic/check ids.

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
        # The row is gone.
        remaining = db_conn.execute("SELECT COUNT(*) FROM pending_webhooks").fetchone()[0]
        assert remaining == 0


class TestWebhookRetryCrashSafety:
    """Per-item commits must survive a mid-loop crash (no rollback of work)."""

    async def test_crash_midloop_preserves_already_applied_results(self, db_conn: sqlite3.Connection) -> None:
        """If applying item 2 crashes, item 1's delete must already be committed.

        Old code committed once after the whole loop, so a crash rolled back
        every delete/increment from that pass — letting a failing URL retry
        unbounded. Per-item commits must keep already-applied work durable.
        """
        topic = _make_topic(db_conn)
        settings = _make_settings(
            notifications=NotificationSettings(
                urls=[],
                webhook_urls=["https://first.com/hook", "https://second.com/hook"],
            )
        )
        novelty = _make_novelty()

        # Enqueue two failed webhooks.
        with patch("app.webhooks.send_webhook", return_value=False):
            await send_webhooks("Hooked", novelty, settings, conn=db_conn, topic_id=topic.id)
        pending = list_pending_webhooks(db_conn)
        assert len(pending) == 2
        first_id = pending[0]["id"]

        # Retry: both sends "succeed", but applying the SECOND result crashes.
        from app.crud import delete_pending_webhook as real_delete_pending_webhook

        call_count = {"n": 0}

        def crashing_delete(conn, webhook_id):  # noqa: ANN001
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise RuntimeError("simulated crash applying item 2")
            real_delete_pending_webhook(conn, webhook_id)

        with (
            patch("app.webhooks.send_webhook", new_callable=AsyncMock, return_value=True),
            patch("app.webhooks.delete_pending_webhook", side_effect=crashing_delete),
            pytest.raises(RuntimeError, match="simulated crash"),
        ):
            await retry_pending_webhooks(db_conn, settings)

        # Item 1's delete was committed before item 2 crashed: it is gone,
        # item 2 remains. The pass did NOT roll back already-applied work.
        remaining = db_conn.execute("SELECT id FROM pending_webhooks").fetchall()
        remaining_ids = {r["id"] for r in remaining}
        assert first_id not in remaining_ids
        assert len(remaining_ids) == 1

    async def test_no_connection_held_across_send(self, db_conn: sqlite3.Connection) -> None:
        """The network send must run with no open transaction on the snapshot conn."""
        topic = _make_topic(db_conn)
        settings = _make_settings(notifications=NotificationSettings(urls=[], webhook_urls=["https://a.com/hook"]))
        novelty = _make_novelty()

        with patch("app.webhooks.send_webhook", return_value=False):
            await send_webhooks("Hooked", novelty, settings, conn=db_conn, topic_id=topic.id)

        in_transaction_during_send: list[bool] = []

        async def observe(url: str, payload: dict, timeout: float = 10.0) -> bool:
            # When a connection holds an uncommitted write, in_transaction is
            # True. The snapshot must have been committed before the send.
            in_transaction_during_send.append(db_conn.in_transaction)
            return True

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
        novelty = _make_novelty()

        with patch("app.webhooks.send_webhook", return_value=False):
            await send_webhooks("Hooked", novelty, settings, conn=conn, topic_id=topic.id)
        conn.close()

        with patch("app.webhooks.send_webhook", new_callable=AsyncMock, return_value=True):
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
            novelty = _make_novelty()
            with patch("app.webhooks.send_webhook", return_value=False):
                await send_webhooks("Hooked", novelty, settings, conn=conn, topic_id=topic.id)
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

        async def slow_send(url: str, payload: dict, timeout: float = 10.0) -> bool:
            # Block the first drain mid-send so the second drain overlaps it.
            first_send_started.set()
            await release.wait()
            sent_counts[url] += 1
            return True

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
        from app.crud import claim_pending_webhook
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
            assert claim_pending_webhook(claimer, webhook_id, "2999-01-01T00:00:00+00:00") is True
            claimer.commit()
        finally:
            claimer.close()

        send_calls: list[str] = []

        async def record_send(url: str, payload: dict, timeout: float = 10.0) -> bool:
            send_calls.append(url)
            return True

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
