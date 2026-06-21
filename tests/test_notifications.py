"""Tests for the notification module: Apprise wrapper and formatting.

Also covers the persistent notification retry-drain single-flight/claim
behaviour (OVH-017), which lives in ``app.checker.retry_pending_notifications``.
"""

import asyncio
import collections
import threading
from unittest.mock import MagicMock, patch

from app.analysis.llm import NoveltyResult
from app.checker import retry_pending_notifications
from app.config import LLMSettings, NotificationSettings, Settings
from app.crud import create_pending_notification, create_topic, list_pending_notifications
from app.database import get_connection, init_db
from app.models import PendingNotification, Topic, TopicStatus
from app.notifications import (
    format_notification,
    redact_url,
    send_notification,
    send_notification_per_url,
)


def _make_settings(**overrides) -> Settings:
    defaults = {
        "llm": LLMSettings(model="openai/gpt-4o-mini", api_key="test-key"),
        "notifications": NotificationSettings(urls=["json://localhost"]),
    }
    defaults.update(overrides)
    return Settings(**defaults)


# --- format_notification ---


class TestFormatNotification:
    """Tests for notification formatting."""

    def test_title_includes_topic_name(self) -> None:
        novelty = NoveltyResult(
            has_new_info=True,
            summary="New release date announced",
            confidence=0.9,
        )
        title, body = format_notification("Elden Ring DLC", novelty)
        assert title == "Topic Watch: Elden Ring DLC"

    def test_body_includes_summary(self) -> None:
        novelty = NoveltyResult(
            has_new_info=True,
            summary="Price announced at $39.99",
            confidence=0.9,
        )
        _, body = format_notification("Test", novelty)
        assert "Price announced at $39.99" in body

    def test_body_includes_key_facts(self) -> None:
        novelty = NoveltyResult(
            has_new_info=True,
            summary="Update",
            key_facts=["Fact one", "Fact two"],
            confidence=0.9,
        )
        _, body = format_notification("Test", novelty)
        assert "Fact one" in body
        assert "Fact two" in body
        assert "Key facts:" in body

    def test_body_includes_source_urls(self) -> None:
        novelty = NoveltyResult(
            has_new_info=True,
            summary="Update",
            source_urls=["https://example.com/article"],
            confidence=0.9,
        )
        _, body = format_notification("Test", novelty)
        assert "https://example.com/article" in body
        assert "Sources:" in body

    def test_handles_no_summary(self) -> None:
        novelty = NoveltyResult(
            has_new_info=True,
            key_facts=["A fact"],
            confidence=0.9,
        )
        title, body = format_notification("Test", novelty)
        assert title == "Topic Watch: Test"
        assert "A fact" in body

    def test_handles_empty_novelty(self) -> None:
        novelty = NoveltyResult(has_new_info=True, confidence=0.8)
        title, body = format_notification("Test", novelty)
        assert title == "Topic Watch: Test"
        assert isinstance(body, str)

    def test_body_includes_confidence_percentage(self) -> None:
        novelty = NoveltyResult(
            has_new_info=True,
            summary="Update",
            confidence=0.9,
        )
        _, body = format_notification("Test", novelty)
        assert "Confidence: 90%" in body

    def test_body_confidence_truncates_to_int(self) -> None:
        novelty = NoveltyResult(
            has_new_info=True,
            summary="Update",
            confidence=0.856,
        )
        _, body = format_notification("Test", novelty)
        assert "Confidence: 85%" in body

    def test_body_includes_relevance_percentage(self) -> None:
        novelty = NoveltyResult(
            has_new_info=True,
            summary="Update",
            confidence=0.9,
            relevance=0.8,
        )
        _, body = format_notification("Test", novelty)
        assert "Relevance: 80%" in body

    def test_body_relevance_truncates_to_int(self) -> None:
        novelty = NoveltyResult(
            has_new_info=True,
            summary="Update",
            confidence=0.9,
            relevance=0.736,
        )
        _, body = format_notification("Test", novelty)
        assert "Relevance: 73%" in body


# --- send_notification ---


class TestSendNotification:
    """Tests for the Apprise send wrapper (async)."""

    @patch("app.notifications.apprise.Apprise")
    async def test_sends_successfully(self, mock_apprise: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_instance.notify.return_value = True
        mock_apprise.return_value = mock_instance
        settings = _make_settings()

        result = await send_notification("Title", "Body", settings)

        assert result is True
        mock_instance.add.assert_called_once_with("json://localhost")
        mock_instance.notify.assert_called_once_with(title="Title", body="Body")

    @patch("app.notifications.apprise.Apprise")
    async def test_adds_multiple_urls(self, mock_apprise: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_instance.notify.return_value = True
        mock_apprise.return_value = mock_instance
        settings = _make_settings(
            notifications=NotificationSettings(urls=["json://localhost", "slack://token/channel"])
        )

        await send_notification("T", "B", settings)

        assert mock_instance.add.call_count == 2

    async def test_returns_false_when_no_urls(self) -> None:
        settings = _make_settings(notifications=NotificationSettings(urls=[]))
        result = await send_notification("Title", "Body", settings)
        assert result is False

    @patch("app.notifications.apprise.Apprise")
    async def test_returns_false_on_delivery_failure(self, mock_apprise: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_instance.notify.return_value = False
        mock_apprise.return_value = mock_instance
        settings = _make_settings()

        result = await send_notification("Title", "Body", settings)

        assert result is False

    @patch("app.notifications.apprise.Apprise")
    async def test_returns_false_on_timeout(self, mock_apprise: MagicMock) -> None:
        """A hung notify is abandoned after apprise_timeout_seconds, returning False."""
        mock_instance = MagicMock()

        _blocker = threading.Event()

        def slow_notify(**_kwargs):
            # Block for 2s — longer than the 1s timeout so wait_for fires,
            # but short enough that the abandoned thread exits quickly.
            _blocker.wait(2)
            return True

        mock_instance.notify.side_effect = slow_notify
        mock_apprise.return_value = mock_instance
        settings = _make_settings(apprise_timeout_seconds=1)

        result = await send_notification("Title", "Body", settings)

        assert result is False

    @patch("app.notifications.apprise.Apprise")
    async def test_returns_false_on_exception(self, mock_apprise: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_instance.notify.side_effect = Exception("Connection refused")
        mock_apprise.return_value = mock_instance
        settings = _make_settings()

        result = await send_notification("Title", "Body", settings)

        assert result is False


# --- pending notification drain single-flight / claim (OVH-017) ---


def _enqueue_notifications(db_path, count: int) -> list[int]:  # noqa: ANN001
    """Insert ``count`` pending notifications, returning their ids."""
    conn = get_connection(db_path)
    try:
        topic = create_topic(conn, Topic(name="Notif", description="d", status=TopicStatus.READY))
        conn.commit()
        ids: list[int] = []
        for i in range(count):
            n = create_pending_notification(
                conn,
                PendingNotification(topic_id=topic.id, title=f"T{i}", body=f"B{i}"),
            )
            ids.append(n.id)
        conn.commit()
        return ids
    finally:
        conn.close()


class TestNotificationDrainSingleFlight:
    """Overlapping drains deliver each pending notification exactly once."""

    async def test_two_concurrent_drains_deliver_each_item_once(self, tmp_path) -> None:  # noqa: ANN001
        """Two drains launched together send each pending notification once."""
        db_path = tmp_path / "test.db"
        init_db(db_path)
        _enqueue_notifications(db_path, 3)
        settings = _make_settings()

        sent_counts: collections.Counter[str] = collections.Counter()
        first_send_started = asyncio.Event()
        release = asyncio.Event()

        async def slow_send(title: str, body: str, _settings, *, url=None) -> bool:  # noqa: ANN001
            first_send_started.set()
            await release.wait()
            sent_counts[title] += 1
            return True

        with patch("app.checker.send_notification", side_effect=slow_send):
            drain1 = asyncio.create_task(retry_pending_notifications(settings=settings, db_path=db_path))
            await first_send_started.wait()
            drain2 = asyncio.create_task(retry_pending_notifications(settings=settings, db_path=db_path))
            await asyncio.sleep(0)  # let drain2 observe the single-flight guard
            release.set()
            await asyncio.gather(drain1, drain2)

        assert dict(sent_counts) == {"T0": 1, "T1": 1, "T2": 1}

        verify = get_connection(db_path)
        try:
            assert list_pending_notifications(verify) == []
        finally:
            verify.close()

    async def test_claimed_row_skipped_by_second_drainer(self, tmp_path) -> None:  # noqa: ANN001
        """A row another process already claimed is skipped, not re-sent."""
        from app.crud import claim_pending_notification

        db_path = tmp_path / "test.db"
        init_db(db_path)
        ids = _enqueue_notifications(db_path, 1)

        claimer = get_connection(db_path)
        try:
            assert claim_pending_notification(claimer, ids[0], "2999-01-01T00:00:00+00:00") is True
            claimer.commit()
        finally:
            claimer.close()

        send_calls: list[str] = []

        async def record_send(title: str, body: str, _settings, *, url=None) -> bool:  # noqa: ANN001
            send_calls.append(title)
            return True

        with patch("app.checker.send_notification", side_effect=record_send):
            await retry_pending_notifications(settings=_make_settings(), db_path=db_path)

        assert send_calls == []
        verify = get_connection(db_path)
        try:
            row = verify.execute("SELECT claimed_at FROM pending_notifications").fetchone()
            assert row is not None
            assert row["claimed_at"] == "2999-01-01T00:00:00+00:00"
        finally:
            verify.close()


# --- URL redaction (OVH-027) ---


class TestRedactUrl:
    """redact_url never leaks userinfo/token/query, keeps scheme+host.

    Fold-in: app.notifications.redact_url is now the canonical
    app.log_redaction.redact_url, which keeps a short non-secret leading path
    segment for context while still dropping userinfo/query/long secret segments.
    """

    def test_keeps_scheme_and_host(self) -> None:
        red = redact_url("slack://host.example.com/path")
        assert red.startswith("slack://host.example.com")

    def test_strips_userinfo_and_token(self) -> None:
        # tgram://<bot-token>@... — the token must not survive redaction.
        red = redact_url("tgram://123456:ABCDEF-secret-token@chat")
        assert "secret-token" not in red
        assert "ABCDEF" not in red

    def test_strips_query_string(self) -> None:
        red = redact_url("json://host/?password=hunter2")
        assert "hunter2" not in red
        assert red.startswith("json://host")

    def test_handles_garbage(self) -> None:
        # Never raises; returns a non-empty masked placeholder.
        assert redact_url("") != ""
        assert isinstance(redact_url("::not a url::"), str)


# --- per-URL delivery (OVH-027 / OVH-039) ---


class TestSendNotificationPerUrl:
    """Per-URL delivery so partial failures can be re-queued individually."""

    @patch("app.notifications.apprise.Apprise")
    async def test_returns_one_result_per_url(self, mock_apprise: MagicMock) -> None:
        instances = [MagicMock(), MagicMock()]
        instances[0].add.return_value = True
        instances[0].notify.return_value = True
        instances[1].add.return_value = True
        instances[1].notify.return_value = False
        mock_apprise.side_effect = instances
        settings = _make_settings(notifications=NotificationSettings(urls=["json://a", "json://b"]))

        results = await send_notification_per_url("T", "B", settings)

        assert [(r.url, r.ok) for r in results] == [("json://a", True), ("json://b", False)]

    @patch("app.notifications.apprise.Apprise")
    async def test_invalid_url_marked_failed_not_dropped(self, mock_apprise: MagicMock) -> None:
        """A URL apprise can't add() is reported as failed (OVH-027), not silently dropped."""
        good, bad = MagicMock(), MagicMock()
        good.add.return_value = True
        good.notify.return_value = True
        bad.add.return_value = False  # invalid URL
        mock_apprise.side_effect = [good, bad]
        settings = _make_settings(notifications=NotificationSettings(urls=["json://good", "::invalid::"]))

        results = await send_notification_per_url("T", "B", settings)

        by_url = {r.url: r for r in results}
        assert by_url["json://good"].ok is True
        assert by_url["::invalid::"].ok is False
        # The valid URL still delivered.
        good.notify.assert_called_once()
        # The invalid URL was never notify()'d (add failed first).
        bad.notify.assert_not_called()

    @patch("app.notifications.apprise.Apprise")
    async def test_only_named_url_sent_when_url_given(self, mock_apprise: MagicMock) -> None:
        """Passing url= sends to only that URL (the retry-drain per-row path)."""
        inst = MagicMock()
        inst.add.return_value = True
        inst.notify.return_value = True
        mock_apprise.return_value = inst
        settings = _make_settings(notifications=NotificationSettings(urls=["json://a", "json://b", "json://c"]))

        results = await send_notification_per_url("T", "B", settings, url="json://b")

        assert [r.url for r in results] == ["json://b"]
        inst.add.assert_called_once_with("json://b")

    @patch("app.notifications.apprise.Apprise")
    async def test_invalid_url_logged_redacted(self, mock_apprise: MagicMock, caplog) -> None:  # noqa: ANN001
        """The invalid URL warning is emitted with the token redacted."""
        import logging as _logging

        inst = MagicMock()
        inst.add.return_value = False
        mock_apprise.return_value = inst
        settings = _make_settings(notifications=NotificationSettings(urls=["tgram://123:SECRETTOKEN@chat"]))

        with caplog.at_level(_logging.WARNING, logger="app.notifications"):
            await send_notification_per_url("T", "B", settings, url="tgram://123:SECRETTOKEN@chat")

        joined = " ".join(r.getMessage() for r in caplog.records)
        assert "SECRETTOKEN" not in joined
        assert "invalid notification url" in joined.lower()


class TestSendNotificationStillNonRaising:
    """send_notification keeps its boolean contract and never raises (OVH-039)."""

    @patch("app.notifications.apprise.Apprise")
    async def test_returns_false_when_one_of_many_fails(self, mock_apprise: MagicMock) -> None:
        first, second = MagicMock(), MagicMock()
        first.add.return_value = True
        first.notify.return_value = True
        second.add.return_value = True
        second.notify.return_value = False
        mock_apprise.side_effect = [first, second]
        settings = _make_settings(notifications=NotificationSettings(urls=["json://a", "json://b"]))

        result = await send_notification("T", "B", settings)
        assert result is False

    @patch("app.notifications.apprise.Apprise")
    async def test_does_not_raise_on_per_url_exception(self, mock_apprise: MagicMock) -> None:
        inst = MagicMock()
        inst.add.return_value = True
        inst.notify.side_effect = Exception("boom")
        mock_apprise.return_value = inst
        settings = _make_settings()

        result = await send_notification("T", "B", settings)
        assert result is False


# --- retry does not re-send already-succeeded URLs (OVH-039) ---


class TestRetryOnlyResendsFailedUrls:
    """A partial failure queues only the failed URL; retry never re-hits the rest."""

    async def test_partial_failure_requeues_only_failed_url(self, tmp_path) -> None:  # noqa: ANN001
        """check_topic-style queueing: a 3-URL send with one failure queues 1 row for that URL."""
        from app.checker import _queue_failed_notifications
        from app.models import NotificationDelivery

        db_path = tmp_path / "test.db"
        init_db(db_path)
        conn = get_connection(db_path)
        try:
            topic = create_topic(conn, Topic(name="Notif", description="d", status=TopicStatus.READY))
            conn.commit()
            deliveries = [
                NotificationDelivery(url="json://a", ok=True),
                NotificationDelivery(url="json://b", ok=False, error="HTTP 500"),
                NotificationDelivery(url="json://c", ok=True),
            ]
            _queue_failed_notifications(conn, topic.id, "T", "B", deliveries)
            conn.commit()
            pending = list_pending_notifications(conn)
        finally:
            conn.close()

        assert len(pending) == 1
        assert pending[0].url == "json://b"
        assert pending[0].last_error == "HTTP 500"

    async def test_retry_drain_sends_only_to_queued_url(self, tmp_path) -> None:  # noqa: ANN001
        """A pending row carrying url=json://b retries only json://b, not the whole batch."""
        db_path = tmp_path / "test.db"
        init_db(db_path)
        conn = get_connection(db_path)
        try:
            topic = create_topic(conn, Topic(name="Notif", description="d", status=TopicStatus.READY))
            conn.commit()
            create_pending_notification(
                conn,
                PendingNotification(topic_id=topic.id, title="T", body="B", url="json://b"),
            )
            conn.commit()
        finally:
            conn.close()

        # The retry drain must call send_notification scoped to the queued URL only.
        sent_urls: list[str | None] = []

        async def record_send(title: str, body: str, _settings, *, url=None) -> bool:  # noqa: ANN001
            sent_urls.append(url)
            return True

        with patch("app.checker.send_notification", side_effect=record_send):
            await retry_pending_notifications(settings=_make_settings(), db_path=db_path)

        assert sent_urls == ["json://b"]
