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
from app.notifications import format_notification, send_notification


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

        async def slow_send(title: str, body: str, _settings) -> bool:  # noqa: ANN001
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

        async def record_send(title: str, body: str, _settings) -> bool:  # noqa: ANN001
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
