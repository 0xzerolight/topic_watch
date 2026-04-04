"""Tests for the core check loop: check_topic, check_all_topics, retry logic."""

import json
import sqlite3
from unittest.mock import AsyncMock, patch

from app.analysis.llm import NoveltyResult
from app.checker import check_all_topics, check_topic, retry_pending_notifications
from app.config import LLMSettings, NotificationSettings, Settings
from app.crud import (
    create_knowledge_state,
    create_pending_notification,
    create_topic,
    list_pending_notifications,
)
from app.models import (
    Article,
    KnowledgeState,
    PendingNotification,
    Topic,
    TopicStatus,
)
from app.scraping import FetchResult


def _make_settings(**overrides) -> Settings:
    defaults = {
        "llm": LLMSettings(model="openai/gpt-4o-mini", api_key="test-key"),
        "notifications": NotificationSettings(urls=["json://localhost"]),
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _make_topic(conn: sqlite3.Connection, **overrides) -> Topic:
    defaults = {
        "name": "Test Topic",
        "description": "A test topic",
        "feed_urls": ["https://example.com/feed.xml"],
        "status": TopicStatus.READY,
    }
    defaults.update(overrides)
    topic = create_topic(conn, Topic(**defaults))
    conn.commit()
    return topic


def _make_article(**overrides) -> Article:
    defaults = {
        "id": 1,
        "topic_id": 1,
        "title": "Test Article",
        "url": "https://example.com/article-1",
        "content_hash": "abc123",
        "raw_content": "Article content here.",
        "source_feed": "https://example.com/feed.xml",
    }
    defaults.update(overrides)
    return Article(**defaults)


# --- check_topic ---


class TestCheckTopic:
    """Tests for the single-topic check pipeline."""

    async def test_happy_path_new_info_sends_notification(self, db_conn: sqlite3.Connection) -> None:
        """New articles + new info → knowledge updated, notification sent."""
        topic = _make_topic(db_conn)
        create_knowledge_state(
            db_conn,
            KnowledgeState(topic_id=topic.id, summary_text="Old summary.", token_count=20),
        )
        db_conn.commit()
        settings = _make_settings()

        articles = [_make_article(topic_id=topic.id)]
        novelty = NoveltyResult(
            has_new_info=True,
            summary="New release date",
            key_facts=["June 2025"],
            source_urls=["https://example.com/article-1"],
            confidence=0.9,
            relevance=0.9,
        )

        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=articles, total_feed_entries=len(articles)),
            ),
            patch(
                "app.checker.analyze_articles",
                new_callable=AsyncMock,
                return_value=novelty,
            ),
            patch(
                "app.checker.update_knowledge",
                new_callable=AsyncMock,
            ) as mock_update,
            patch(
                "app.checker.send_notification",
                return_value=True,
            ) as mock_send,
        ):
            result = await check_topic(topic, db_conn, settings)

        assert result.has_new_info is True
        assert result.notification_sent is True
        assert result.articles_found == 1
        assert result.id is not None
        mock_update.assert_called_once()
        mock_send.assert_called_once()

    async def test_no_new_info_no_notification(self, db_conn: sqlite3.Connection) -> None:
        """Articles found but LLM says nothing new."""
        topic = _make_topic(db_conn)
        create_knowledge_state(
            db_conn,
            KnowledgeState(topic_id=topic.id, summary_text="Known facts.", token_count=20),
        )
        db_conn.commit()
        settings = _make_settings()

        articles = [_make_article(topic_id=topic.id)]
        novelty = NoveltyResult(has_new_info=False, confidence=0.9)

        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=articles, total_feed_entries=len(articles)),
            ),
            patch(
                "app.checker.analyze_articles",
                new_callable=AsyncMock,
                return_value=novelty,
            ),
            patch("app.checker.send_notification") as mock_send,
        ):
            result = await check_topic(topic, db_conn, settings)

        assert result.has_new_info is False
        assert result.notification_sent is False
        mock_send.assert_not_called()

    async def test_no_new_articles_early_return(self, db_conn: sqlite3.Connection) -> None:
        """No new articles → early return without LLM call."""
        topic = _make_topic(db_conn)
        settings = _make_settings()

        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=[], total_feed_entries=0),
            ),
            patch(
                "app.checker.analyze_articles",
                new_callable=AsyncMock,
            ) as mock_analyze,
        ):
            result = await check_topic(topic, db_conn, settings)

        assert result.articles_found == 0
        assert result.has_new_info is False
        mock_analyze.assert_not_called()

    async def test_scraping_failure_records_result(self, db_conn: sqlite3.Connection) -> None:
        """Scraping error should not crash, should record a result."""
        topic = _make_topic(db_conn)
        settings = _make_settings()

        with patch(
            "app.checker.fetch_new_articles_for_topic",
            new_callable=AsyncMock,
            side_effect=Exception("Network error"),
        ):
            result = await check_topic(topic, db_conn, settings)

        assert result.articles_found == 0
        assert result.id is not None

    async def test_skips_non_ready_topic(self, db_conn: sqlite3.Connection) -> None:
        """Topics not in READY status should be skipped."""
        topic = _make_topic(db_conn, name="Researching", status=TopicStatus.RESEARCHING)
        settings = _make_settings()

        result = await check_topic(topic, db_conn, settings)

        assert result.articles_found == 0
        assert result.id is not None

    async def test_notification_failure_captured_and_queued(self, db_conn: sqlite3.Connection) -> None:
        """Notification failure should be recorded and queued for retry."""
        topic = _make_topic(db_conn, name="NotifFail")
        create_knowledge_state(
            db_conn,
            KnowledgeState(topic_id=topic.id, summary_text="Summary.", token_count=10),
        )
        db_conn.commit()
        settings = _make_settings()

        articles = [_make_article(topic_id=topic.id)]
        novelty = NoveltyResult(has_new_info=True, summary="Update", confidence=0.9, relevance=0.9)

        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=articles, total_feed_entries=len(articles)),
            ),
            patch(
                "app.checker.analyze_articles",
                new_callable=AsyncMock,
                return_value=novelty,
            ),
            patch("app.checker.update_knowledge", new_callable=AsyncMock),
            patch(
                "app.checker.send_notification",
                side_effect=Exception("SMTP error"),
            ),
        ):
            result = await check_topic(topic, db_conn, settings)

        assert result.has_new_info is True
        assert result.notification_sent is False
        assert result.notification_error is not None

        # Verify a pending notification was actually queued in the DB
        pending = list_pending_notifications(db_conn)
        assert len(pending) == 1
        assert pending[0].topic_id == topic.id
        assert "Topic Watch:" in pending[0].title

    async def test_notification_delivery_failure_queued(self, db_conn: sqlite3.Connection) -> None:
        """When send_notification returns False, notification is queued for retry."""
        topic = _make_topic(db_conn, name="DeliveryFail")
        create_knowledge_state(
            db_conn,
            KnowledgeState(topic_id=topic.id, summary_text="Summary.", token_count=10),
        )
        db_conn.commit()
        settings = _make_settings()

        articles = [_make_article(topic_id=topic.id)]
        novelty = NoveltyResult(has_new_info=True, summary="Update", confidence=0.9, relevance=0.9)

        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=articles, total_feed_entries=len(articles)),
            ),
            patch(
                "app.checker.analyze_articles",
                new_callable=AsyncMock,
                return_value=novelty,
            ),
            patch("app.checker.update_knowledge", new_callable=AsyncMock),
            patch(
                "app.checker.send_notification",
                return_value=False,
            ),
        ):
            result = await check_topic(topic, db_conn, settings)

        assert result.notification_sent is False
        assert result.notification_error == "Delivery failed"

        # Verify queued for retry
        pending = list_pending_notifications(db_conn)
        assert len(pending) == 1

    async def test_llm_response_stored_as_json(self, db_conn: sqlite3.Connection) -> None:
        """The NoveltyResult should be serialized to llm_response."""
        topic = _make_topic(db_conn, name="JsonStore")
        create_knowledge_state(
            db_conn,
            KnowledgeState(topic_id=topic.id, summary_text="S.", token_count=5),
        )
        db_conn.commit()
        settings = _make_settings()

        articles = [_make_article(topic_id=topic.id)]
        novelty = NoveltyResult(
            has_new_info=True,
            summary="New thing",
            confidence=0.85,
            relevance=0.9,
        )

        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=articles, total_feed_entries=len(articles)),
            ),
            patch(
                "app.checker.analyze_articles",
                new_callable=AsyncMock,
                return_value=novelty,
            ),
            patch("app.checker.update_knowledge", new_callable=AsyncMock),
            patch("app.checker.send_notification", return_value=True),
        ):
            result = await check_topic(topic, db_conn, settings)

        parsed = json.loads(result.llm_response)
        assert parsed["has_new_info"] is True
        assert parsed["summary"] == "New thing"

    async def test_knowledge_summary_passed_to_analyze(self, db_conn: sqlite3.Connection) -> None:
        """The current knowledge summary must be retrieved and passed to analyze_articles."""
        topic = _make_topic(db_conn, name="KnowledgePass")
        create_knowledge_state(
            db_conn,
            KnowledgeState(
                topic_id=topic.id,
                summary_text="Specific knowledge summary XYZ.",
                token_count=20,
            ),
        )
        db_conn.commit()
        settings = _make_settings()

        articles = [_make_article(topic_id=topic.id)]
        novelty = NoveltyResult(has_new_info=False, confidence=0.5)

        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=articles, total_feed_entries=len(articles)),
            ),
            patch(
                "app.checker.analyze_articles",
                new_callable=AsyncMock,
                return_value=novelty,
            ) as mock_analyze,
        ):
            await check_topic(topic, db_conn, settings)

        # Verify the knowledge summary was actually passed
        call_args = mock_analyze.call_args
        knowledge_summary_arg = call_args[0][1]  # second positional arg
        assert knowledge_summary_arg == "Specific knowledge summary XYZ."

    async def test_knowledge_update_failure_still_notifies(self, db_conn: sqlite3.Connection) -> None:
        """If update_knowledge fails, notification should still be sent."""
        topic = _make_topic(db_conn, name="KUFail")
        create_knowledge_state(
            db_conn,
            KnowledgeState(topic_id=topic.id, summary_text="Old.", token_count=5),
        )
        db_conn.commit()
        settings = _make_settings()

        articles = [_make_article(topic_id=topic.id)]
        novelty = NoveltyResult(has_new_info=True, summary="New info", confidence=0.9, relevance=0.9)

        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=articles, total_feed_entries=len(articles)),
            ),
            patch(
                "app.checker.analyze_articles",
                new_callable=AsyncMock,
                return_value=novelty,
            ),
            patch(
                "app.checker.update_knowledge",
                new_callable=AsyncMock,
                side_effect=Exception("Knowledge update crashed"),
            ),
            patch(
                "app.checker.send_notification",
                return_value=True,
            ) as mock_send,
        ):
            result = await check_topic(topic, db_conn, settings)

        # Notification should still be sent despite knowledge update failure
        mock_send.assert_called_once()
        assert result.notification_sent is True
        assert result.has_new_info is True

    async def test_low_confidence_skips_notification_and_leaves_articles_unprocessed(
        self, db_conn: sqlite3.Connection
    ) -> None:
        """New info with confidence below threshold → no notification, no knowledge update,
        and articles are NOT marked as processed (so they get re-examined next cycle)."""
        topic = _make_topic(db_conn)
        create_knowledge_state(
            db_conn,
            KnowledgeState(topic_id=topic.id, summary_text="Known facts.", token_count=20),
        )
        db_conn.commit()
        settings = _make_settings(min_confidence_threshold=0.6)

        articles = [_make_article(topic_id=topic.id)]
        novelty = NoveltyResult(
            has_new_info=True,
            summary="Possibly new info",
            confidence=0.3,
        )

        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=articles, total_feed_entries=len(articles)),
            ),
            patch(
                "app.checker.analyze_articles",
                new_callable=AsyncMock,
                return_value=novelty,
            ),
            patch("app.checker.update_knowledge", new_callable=AsyncMock) as mock_update,
            patch("app.checker.send_notification") as mock_send,
            patch("app.checker.mark_articles_processed") as mock_mark,
        ):
            result = await check_topic(topic, db_conn, settings)

        # has_new_info is True (LLM detected it) but notification not sent
        assert result.has_new_info is True
        assert result.notification_sent is False
        mock_update.assert_not_called()
        mock_send.assert_not_called()
        mock_mark.assert_not_called()

    async def test_high_confidence_sends_notification(self, db_conn: sqlite3.Connection) -> None:
        """New info with confidence above threshold → normal flow."""
        topic = _make_topic(db_conn)
        create_knowledge_state(
            db_conn,
            KnowledgeState(topic_id=topic.id, summary_text="Known facts.", token_count=20),
        )
        db_conn.commit()
        settings = _make_settings(min_confidence_threshold=0.6)

        articles = [_make_article(topic_id=topic.id)]
        novelty = NoveltyResult(
            has_new_info=True,
            summary="Confirmed new release date",
            confidence=0.9,
            relevance=0.9,
        )

        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=articles, total_feed_entries=len(articles)),
            ),
            patch(
                "app.checker.analyze_articles",
                new_callable=AsyncMock,
                return_value=novelty,
            ),
            patch("app.checker.update_knowledge", new_callable=AsyncMock) as mock_update,
            patch("app.checker.send_notification", return_value=True) as mock_send,
        ):
            result = await check_topic(topic, db_conn, settings)

        assert result.has_new_info is True
        assert result.notification_sent is True
        mock_update.assert_called_once()
        mock_send.assert_called_once()

    async def test_low_relevance_skips_notification(self, db_conn: sqlite3.Connection) -> None:
        """New info with high confidence but low relevance → no notification, articles not processed."""
        topic = _make_topic(db_conn)
        create_knowledge_state(
            db_conn,
            KnowledgeState(topic_id=topic.id, summary_text="Known facts.", token_count=20),
        )
        db_conn.commit()
        settings = _make_settings(min_relevance_threshold=0.5)

        articles = [_make_article(topic_id=topic.id)]
        novelty = NoveltyResult(
            has_new_info=True,
            summary="Tangentially related info",
            confidence=0.9,
            relevance=0.2,
        )

        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=articles, total_feed_entries=len(articles)),
            ),
            patch(
                "app.checker.analyze_articles",
                new_callable=AsyncMock,
                return_value=novelty,
            ),
            patch("app.checker.update_knowledge", new_callable=AsyncMock) as mock_update,
            patch("app.checker.send_notification") as mock_send,
            patch("app.checker.mark_articles_processed") as mock_mark,
        ):
            result = await check_topic(topic, db_conn, settings)

        assert result.has_new_info is True
        assert result.notification_sent is False
        mock_update.assert_not_called()
        mock_send.assert_not_called()
        mock_mark.assert_not_called()


# --- check_all_topics ---


class TestCheckAllTopics:
    """Tests for the multi-topic check loop."""

    async def test_checks_all_ready_topics(self, db_conn: sqlite3.Connection) -> None:
        _make_topic(db_conn, name="Topic A")
        _make_topic(db_conn, name="Topic B")
        settings = _make_settings()

        with patch(
            "app.checker.fetch_new_articles_for_topic",
            new_callable=AsyncMock,
            return_value=FetchResult(articles=[], total_feed_entries=0),
        ):
            results = await check_all_topics(db_conn, settings)

        assert len(results) == 2

    async def test_skips_researching_topics(self, db_conn: sqlite3.Connection) -> None:
        _make_topic(db_conn, name="Ready", status=TopicStatus.READY)
        _make_topic(db_conn, name="Research", status=TopicStatus.RESEARCHING)
        settings = _make_settings()

        with patch(
            "app.checker.fetch_new_articles_for_topic",
            new_callable=AsyncMock,
            return_value=FetchResult(articles=[], total_feed_entries=0),
        ):
            results = await check_all_topics(db_conn, settings)

        assert len(results) == 1

    async def test_error_isolation(self, db_conn: sqlite3.Connection) -> None:
        """One topic failing should not prevent others from being checked."""
        _make_topic(db_conn, name="Good Topic")
        _make_topic(db_conn, name="Bad Topic")
        settings = _make_settings()

        async def mock_fetch(topic, conn, max_articles=10, **kwargs):
            if topic.name == "Bad Topic":
                raise Exception("Unexpected error")
            return FetchResult(articles=[], total_feed_entries=0)

        with patch(
            "app.checker.fetch_new_articles_for_topic",
            side_effect=mock_fetch,
        ):
            results = await check_all_topics(db_conn, settings)

        # Bad Topic's scraping error is caught inside check_topic,
        # so both topics produce a CheckResult.
        assert len(results) == 2

    async def test_returns_empty_when_no_topics(self, db_conn: sqlite3.Connection) -> None:
        settings = _make_settings()
        results = await check_all_topics(db_conn, settings)
        assert results == []

    async def test_skips_inactive_topics(self, db_conn: sqlite3.Connection) -> None:
        _make_topic(db_conn, name="Active", status=TopicStatus.READY, is_active=True)
        _make_topic(db_conn, name="Inactive", status=TopicStatus.READY, is_active=False)
        settings = _make_settings()

        with patch(
            "app.checker.fetch_new_articles_for_topic",
            new_callable=AsyncMock,
            return_value=FetchResult(articles=[], total_feed_entries=0),
        ):
            results = await check_all_topics(db_conn, settings)

        assert len(results) == 1

    async def test_outer_error_boundary_isolates_check_topic_crash(self, db_conn: sqlite3.Connection) -> None:
        """When check_topic itself raises, other topics still get checked."""
        _make_topic(db_conn, name="Good Topic")
        _make_topic(db_conn, name="Crash Topic")
        settings = _make_settings()

        original_check_topic = check_topic

        async def mock_check(topic, conn, settings):
            if topic.name == "Crash Topic":
                raise RuntimeError("Unexpected crash in check_topic")
            return await original_check_topic(topic, conn, settings)

        with (
            patch("app.checker.check_topic", side_effect=mock_check),
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=[], total_feed_entries=0),
            ),
        ):
            results = await check_all_topics(db_conn, settings)

        # Only the good topic produces a result; crash topic is excluded
        assert len(results) == 1


# --- retry_pending_notifications ---


class TestRetryPendingNotifications:
    """Tests for the notification retry system."""

    async def test_successful_retry_deletes_notification(self, db_conn: sqlite3.Connection) -> None:
        """When retry succeeds, the pending notification is removed."""
        topic = _make_topic(db_conn)
        create_pending_notification(
            db_conn,
            PendingNotification(topic_id=topic.id, title="Retry Title", body="Retry Body"),
        )
        db_conn.commit()

        settings = _make_settings()

        with patch(
            "app.checker.send_notification",
            new_callable=AsyncMock,
            return_value=True,
        ):
            await retry_pending_notifications(db_conn, settings)

        assert list_pending_notifications(db_conn) == []

    async def test_failed_retry_increments_count(self, db_conn: sqlite3.Connection) -> None:
        """When retry fails, the retry count is incremented."""
        topic = _make_topic(db_conn)
        create_pending_notification(
            db_conn,
            PendingNotification(topic_id=topic.id, title="T", body="B", retry_count=0),
        )
        db_conn.commit()

        settings = _make_settings()

        with patch(
            "app.checker.send_notification",
            new_callable=AsyncMock,
            return_value=False,
        ):
            await retry_pending_notifications(db_conn, settings)

        pending = list_pending_notifications(db_conn)
        assert len(pending) == 1
        assert pending[0].retry_count == 1

    async def test_exception_during_retry_increments_count(self, db_conn: sqlite3.Connection) -> None:
        """When retry raises an exception, the retry count is incremented."""
        topic = _make_topic(db_conn)
        create_pending_notification(
            db_conn,
            PendingNotification(topic_id=topic.id, title="T", body="B", retry_count=0),
        )
        db_conn.commit()

        settings = _make_settings()

        with patch(
            "app.checker.send_notification",
            new_callable=AsyncMock,
            side_effect=Exception("SMTP error"),
        ):
            await retry_pending_notifications(db_conn, settings)

        pending = list_pending_notifications(db_conn)
        assert len(pending) == 1
        assert pending[0].retry_count == 1

    async def test_expired_notifications_deleted(self, db_conn: sqlite3.Connection) -> None:
        """Notifications that have exhausted retries are cleaned up."""
        topic = _make_topic(db_conn)
        create_pending_notification(
            db_conn,
            PendingNotification(
                topic_id=topic.id,
                title="Expired",
                body="B",
                retry_count=3,
                max_retries=3,
            ),
        )
        db_conn.commit()

        settings = _make_settings()

        with patch(
            "app.checker.send_notification",
            new_callable=AsyncMock,
        ):
            await retry_pending_notifications(db_conn, settings)

        # Expired notification should be gone (deleted before retry loop)
        # and no longer retryable (retry_count >= max_retries)
        row = db_conn.execute("SELECT COUNT(*) FROM pending_notifications").fetchone()
        assert row[0] == 0

    async def test_empty_pending_is_noop(self, db_conn: sqlite3.Connection) -> None:
        """No pending notifications means no send attempts."""
        settings = _make_settings()

        with patch(
            "app.checker.send_notification",
            new_callable=AsyncMock,
        ) as mock_send:
            await retry_pending_notifications(db_conn, settings)

        mock_send.assert_not_called()
