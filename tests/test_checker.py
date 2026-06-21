"""Tests for the core check loop: check_topic, check_all_topics, retry logic."""

import json
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from app.analysis.knowledge import KnowledgeWriteResult
from app.analysis.llm import NoveltyResult, TokenUsage
from app.checker import check_all_topics, check_topic, retry_pending_notifications
from app.config import LLMSettings, NotificationSettings, Settings
from app.crud import (
    create_article,
    create_knowledge_state,
    create_pending_notification,
    create_topic,
    get_topic,
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


def _make_write_result(
    *, prompt_tokens: int = 0, completion_tokens: int = 0, sufficient_data: bool = True
) -> KnowledgeWriteResult:
    """Build a KnowledgeWriteResult for mocking initialize/update_knowledge returns."""
    return KnowledgeWriteResult(
        state=KnowledgeState(topic_id=1, summary_text="state", token_count=0),
        usage=TokenUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
        sufficient_data=sufficient_data,
    )


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
                return_value=_make_write_result(),
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
            patch("app.checker.update_knowledge", new_callable=AsyncMock, return_value=_make_write_result()),
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
            patch("app.checker.update_knowledge", new_callable=AsyncMock, return_value=_make_write_result()),
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
            patch("app.checker.update_knowledge", new_callable=AsyncMock, return_value=_make_write_result()),
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
        """If update_knowledge fails: notification still fires, but the row is now
        distinguishable (stage_error set), the new-info article is NOT marked
        processed (so the next cycle re-attempts), and the result is recorded.

        Also pins token accounting on this branch (OVH-170): the swallowed
        knowledge-update raise contributes no tokens; only analysis tokens count.
        """
        topic = _make_topic(db_conn, name="KUFail")
        create_knowledge_state(
            db_conn,
            KnowledgeState(topic_id=topic.id, summary_text="Old.", token_count=5),
        )
        # Persist a real article so we can assert its processed flag from the DB.
        article = create_article(db_conn, _make_article(id=None, topic_id=topic.id))
        db_conn.commit()
        settings = _make_settings()

        novelty = NoveltyResult(
            has_new_info=True,
            summary="New info",
            confidence=0.9,
            relevance=0.9,
            prompt_tokens=80,
            completion_tokens=20,
        )

        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=[article], total_feed_entries=1),
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

        # The failure is now recorded distinctly (OVH-009/037).
        assert result.id is not None
        assert result.stage_error is not None
        assert result.stage_error.startswith("knowledge_update_failed")
        # The recorded row carries the stage_error too.
        row = db_conn.execute("SELECT stage_error FROM check_results WHERE id = ?", (result.id,)).fetchone()
        assert row["stage_error"] is not None
        assert row["stage_error"].startswith("knowledge_update_failed")

        # The new-info-bearing article must NOT be marked processed so the next
        # cycle re-attempts the knowledge update (no silent drift).
        assert article.id is not None
        proc = db_conn.execute("SELECT processed FROM articles WHERE id = ?", (article.id,)).fetchone()
        assert proc["processed"] == 0

        # Token accounting on this branch: only analysis tokens (knowledge
        # update raised before returning usage).
        assert result.prompt_tokens == 80
        assert result.completion_tokens == 20

    async def test_scrape_failure_sets_stage_error(self, db_conn: sqlite3.Connection) -> None:
        """A scrape failure records stage_error='scrape_failed' + summary (OVH-037)."""
        topic = _make_topic(db_conn, name="ScrapeFail")
        settings = _make_settings()

        with patch(
            "app.checker.fetch_new_articles_for_topic",
            new_callable=AsyncMock,
            side_effect=Exception("Network error"),
        ):
            result = await check_topic(topic, db_conn, settings)

        assert result.id is not None
        assert result.stage_error is not None
        assert result.stage_error.startswith("scrape_failed")
        row = db_conn.execute("SELECT stage_error FROM check_results WHERE id = ?", (result.id,)).fetchone()
        assert row["stage_error"].startswith("scrape_failed")

    async def test_analysis_failure_sets_stage_error(self, db_conn: sqlite3.Connection) -> None:
        """An LLM analysis failure (safe-default) records stage_error='analysis_failed'.

        analyze_articles stays fail-safe (returns has_new_info=False, does NOT
        raise); the failure is surfaced via NoveltyResult.error and recorded on
        the CheckResult so it is distinguishable from a clean 'nothing new' run.
        """
        topic = _make_topic(db_conn, name="AnalysisFail")
        create_knowledge_state(
            db_conn,
            KnowledgeState(topic_id=topic.id, summary_text="Known.", token_count=10),
        )
        article = create_article(db_conn, _make_article(id=None, topic_id=topic.id))
        db_conn.commit()
        settings = _make_settings()

        # Mirror the analyze_articles safe-default error path.
        failed = NoveltyResult(has_new_info=False, confidence=0.0, error="LLM analysis failed")

        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=[article], total_feed_entries=1),
            ),
            patch("app.checker.analyze_articles", new_callable=AsyncMock, return_value=failed),
            patch("app.checker.send_notification") as mock_send,
        ):
            result = await check_topic(topic, db_conn, settings)

        mock_send.assert_not_called()
        assert result.has_new_info is False
        assert result.id is not None
        assert result.stage_error is not None
        assert result.stage_error.startswith("analysis_failed")
        row = db_conn.execute("SELECT stage_error FROM check_results WHERE id = ?", (result.id,)).fetchone()
        assert row["stage_error"].startswith("analysis_failed")

    async def test_clean_no_new_info_has_no_stage_error(self, db_conn: sqlite3.Connection) -> None:
        """A clean 'nothing new' run leaves stage_error NULL (distinguishable from failures)."""
        topic = _make_topic(db_conn, name="Quiet")
        create_knowledge_state(
            db_conn,
            KnowledgeState(topic_id=topic.id, summary_text="Known.", token_count=10),
        )
        article = create_article(db_conn, _make_article(id=None, topic_id=topic.id))
        db_conn.commit()
        settings = _make_settings()

        novelty = NoveltyResult(has_new_info=False, confidence=0.9)

        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=[article], total_feed_entries=1),
            ),
            patch("app.checker.analyze_articles", new_callable=AsyncMock, return_value=novelty),
            patch("app.checker.send_notification") as mock_send,
        ):
            result = await check_topic(topic, db_conn, settings)

        mock_send.assert_not_called()
        assert result.stage_error is None
        # And the article IS marked processed (we evaluated it, no failure).
        assert article.id is not None
        proc = db_conn.execute("SELECT processed FROM articles WHERE id = ?", (article.id,)).fetchone()
        assert proc["processed"] == 1

    async def test_low_confidence_skips_notification_but_marks_processed(self, db_conn: sqlite3.Connection) -> None:
        """New info with confidence below threshold → no notification, no knowledge update,
        but articles ARE marked processed (we evaluated them) so they aren't re-analyzed."""
        topic = _make_topic(db_conn)
        create_knowledge_state(
            db_conn,
            KnowledgeState(topic_id=topic.id, summary_text="Known facts.", token_count=20),
        )
        # Persist a real article so we can assert its processed flag from the DB.
        article = create_article(db_conn, _make_article(id=None, topic_id=topic.id))
        db_conn.commit()
        settings = _make_settings(min_confidence_threshold=0.6)

        novelty = NoveltyResult(
            has_new_info=True,
            summary="Possibly new info",
            confidence=0.3,
        )

        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=[article], total_feed_entries=1),
            ),
            patch(
                "app.checker.analyze_articles",
                new_callable=AsyncMock,
                return_value=novelty,
            ) as mock_analyze,
            patch(
                "app.checker.update_knowledge", new_callable=AsyncMock, return_value=_make_write_result()
            ) as mock_update,
            patch("app.checker.send_notification") as mock_send,
        ):
            result = await check_topic(topic, db_conn, settings)

        # has_new_info is True (LLM detected it) but notification not sent
        assert result.has_new_info is True
        assert result.notification_sent is False
        mock_update.assert_not_called()
        mock_send.assert_not_called()

        # Below-threshold article is still marked processed.
        assert article.id is not None
        row = db_conn.execute("SELECT processed FROM articles WHERE id = ?", (article.id,)).fetchone()
        assert row["processed"] == 1

        # Next cycle: only unprocessed articles are fetched, so analyze is not
        # called again — proving no re-analysis loop.
        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=[], total_feed_entries=1),
            ),
            patch("app.checker.analyze_articles", new_callable=AsyncMock) as mock_analyze2,
        ):
            await check_topic(topic, db_conn, settings)
        mock_analyze2.assert_not_called()
        mock_analyze.assert_called_once()

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
            patch(
                "app.checker.update_knowledge", new_callable=AsyncMock, return_value=_make_write_result()
            ) as mock_update,
            patch("app.checker.send_notification", return_value=True) as mock_send,
        ):
            result = await check_topic(topic, db_conn, settings)

        assert result.has_new_info is True
        assert result.notification_sent is True
        mock_update.assert_called_once()
        mock_send.assert_called_once()

    async def test_low_relevance_skips_notification_but_marks_processed(self, db_conn: sqlite3.Connection) -> None:
        """New info with high confidence but low relevance → no notification, but still processed."""
        topic = _make_topic(db_conn)
        create_knowledge_state(
            db_conn,
            KnowledgeState(topic_id=topic.id, summary_text="Known facts.", token_count=20),
        )
        article = create_article(db_conn, _make_article(id=None, topic_id=topic.id))
        db_conn.commit()
        settings = _make_settings(min_relevance_threshold=0.5)

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
                return_value=FetchResult(articles=[article], total_feed_entries=1),
            ),
            patch(
                "app.checker.analyze_articles",
                new_callable=AsyncMock,
                return_value=novelty,
            ),
            patch(
                "app.checker.update_knowledge", new_callable=AsyncMock, return_value=_make_write_result()
            ) as mock_update,
            patch("app.checker.send_notification") as mock_send,
        ):
            result = await check_topic(topic, db_conn, settings)

        assert result.has_new_info is True
        assert result.notification_sent is False
        mock_update.assert_not_called()
        mock_send.assert_not_called()

        assert article.id is not None
        row = db_conn.execute("SELECT processed FROM articles WHERE id = ?", (article.id,)).fetchone()
        assert row["processed"] == 1


# --- initialize_new_topic ---


class TestInitializeNewTopicStatusChangedAt:
    """status_changed_at must be refreshed on every status transition."""

    async def test_ready_transition_sets_status_changed_at(self, db_conn: sqlite3.Connection) -> None:
        topic = _make_topic(db_conn, status=TopicStatus.NEW, status_changed_at=None)
        settings = _make_settings()
        from app.checker import initialize_new_topic

        articles = [_make_article(id=None, topic_id=topic.id)]
        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=articles, total_feed_entries=1),
            ),
            patch(
                "app.checker.initialize_knowledge",
                new_callable=AsyncMock,
                return_value=_make_write_result(),
            ),
        ):
            await initialize_new_topic(topic, db_conn, settings)

        from app.crud import get_topic

        updated = get_topic(db_conn, topic.id)
        assert updated.status == TopicStatus.READY
        assert updated.status_changed_at is not None

    async def test_no_articles_error_transition_sets_status_changed_at(self, db_conn: sqlite3.Connection) -> None:
        topic = _make_topic(db_conn, status=TopicStatus.NEW, status_changed_at=None)
        settings = _make_settings()
        from app.checker import initialize_new_topic

        with patch(
            "app.checker.fetch_new_articles_for_topic",
            new_callable=AsyncMock,
            return_value=FetchResult(articles=[], total_feed_entries=0),
        ):
            await initialize_new_topic(topic, db_conn, settings)

        from app.crud import get_topic

        updated = get_topic(db_conn, topic.id)
        assert updated.status == TopicStatus.ERROR
        assert updated.status_changed_at is not None

    async def test_exception_error_transition_sets_status_changed_at(self, db_conn: sqlite3.Connection) -> None:
        topic = _make_topic(db_conn, status=TopicStatus.NEW, status_changed_at=None)
        settings = _make_settings()
        from app.checker import initialize_new_topic

        articles = [_make_article(id=None, topic_id=topic.id)]
        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=articles, total_feed_entries=1),
            ),
            patch(
                "app.checker.initialize_knowledge",
                new_callable=AsyncMock,
                side_effect=Exception("LLM down"),
            ),
        ):
            await initialize_new_topic(topic, db_conn, settings)

        from app.crud import get_topic

        updated = get_topic(db_conn, topic.id)
        assert updated.status == TopicStatus.ERROR
        assert updated.status_changed_at is not None


class TestPerTopicThresholds:
    """Per-topic confidence/relevance overrides gate notifications."""

    async def _run(self, db_conn, topic, novelty, settings):
        article = create_article(db_conn, _make_article(id=None, topic_id=topic.id))
        db_conn.commit()
        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=[article], total_feed_entries=1),
            ),
            patch("app.checker.analyze_articles", new_callable=AsyncMock, return_value=novelty),
            patch("app.checker.update_knowledge", new_callable=AsyncMock, return_value=_make_write_result()),
            patch("app.checker.send_notification", return_value=True) as mock_send,
        ):
            result = await check_topic(topic, db_conn, settings)
        return result, mock_send

    async def test_high_per_topic_confidence_suppresses_notification(self, db_conn: sqlite3.Connection) -> None:
        """A 0.9 per-topic confidence threshold suppresses a 0.8-confidence notification."""
        topic = _make_topic(db_conn, confidence_threshold=0.9)
        settings = _make_settings(min_confidence_threshold=0.7)
        novelty = NoveltyResult(has_new_info=True, summary="x", confidence=0.8, relevance=0.9)

        result, mock_send = await self._run(db_conn, topic, novelty, settings)

        assert result.has_new_info is True
        assert result.notification_sent is False
        mock_send.assert_not_called()

    async def test_blank_threshold_inherits_global(self, db_conn: sqlite3.Connection) -> None:
        """No per-topic override → global 0.7 lets a 0.8-confidence notification through."""
        topic = _make_topic(db_conn, confidence_threshold=None, relevance_threshold=None)
        settings = _make_settings(min_confidence_threshold=0.7, min_relevance_threshold=0.5)
        novelty = NoveltyResult(has_new_info=True, summary="x", confidence=0.8, relevance=0.9)

        result, mock_send = await self._run(db_conn, topic, novelty, settings)

        assert result.notification_sent is True
        mock_send.assert_called_once()

    async def test_per_topic_relevance_threshold_suppresses(self, db_conn: sqlite3.Connection) -> None:
        topic = _make_topic(db_conn, relevance_threshold=0.9)
        settings = _make_settings(min_relevance_threshold=0.3)
        novelty = NoveltyResult(has_new_info=True, summary="x", confidence=0.95, relevance=0.5)

        result, mock_send = await self._run(db_conn, topic, novelty, settings)

        assert result.notification_sent is False
        mock_send.assert_not_called()


class TestCheckResultTokens:
    """check_results record the summed analysis + knowledge tokens."""

    async def test_tokens_summed_from_analysis_and_knowledge(self, db_conn: sqlite3.Connection) -> None:
        topic = _make_topic(db_conn)
        article = create_article(db_conn, _make_article(id=None, topic_id=topic.id))
        db_conn.commit()
        settings = _make_settings(min_confidence_threshold=0.5, min_relevance_threshold=0.5)

        novelty = NoveltyResult(
            has_new_info=True, summary="x", confidence=0.9, relevance=0.9, prompt_tokens=100, completion_tokens=40
        )

        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=[article], total_feed_entries=1),
            ),
            patch("app.checker.analyze_articles", new_callable=AsyncMock, return_value=novelty),
            patch(
                "app.checker.update_knowledge",
                new_callable=AsyncMock,
                return_value=_make_write_result(prompt_tokens=30, completion_tokens=10),
            ),
            patch("app.checker.send_notification", return_value=True),
        ):
            result = await check_topic(topic, db_conn, settings)

        assert result.prompt_tokens == 130
        assert result.completion_tokens == 50
        row = db_conn.execute(
            "SELECT prompt_tokens, completion_tokens FROM check_results WHERE id = ?", (result.id,)
        ).fetchone()
        assert row["prompt_tokens"] == 130
        assert row["completion_tokens"] == 50

    async def test_early_return_records_zero_tokens(self, db_conn: sqlite3.Connection) -> None:
        topic = _make_topic(db_conn)
        settings = _make_settings()

        with patch(
            "app.checker.fetch_new_articles_for_topic",
            new_callable=AsyncMock,
            return_value=FetchResult(articles=[], total_feed_entries=0),
        ):
            result = await check_topic(topic, db_conn, settings)

        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0
        row = db_conn.execute(
            "SELECT prompt_tokens, completion_tokens FROM check_results WHERE id = ?", (result.id,)
        ).fetchone()
        assert row["prompt_tokens"] == 0
        assert row["completion_tokens"] == 0

    async def test_tokens_only_analysis_when_below_threshold(self, db_conn: sqlite3.Connection) -> None:
        """Below-threshold check still records analysis tokens (no knowledge update runs)."""
        topic = _make_topic(db_conn, confidence_threshold=0.99)
        article = create_article(db_conn, _make_article(id=None, topic_id=topic.id))
        db_conn.commit()
        settings = _make_settings()

        novelty = NoveltyResult(
            has_new_info=True, summary="x", confidence=0.5, relevance=0.9, prompt_tokens=70, completion_tokens=20
        )

        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=[article], total_feed_entries=1),
            ),
            patch("app.checker.analyze_articles", new_callable=AsyncMock, return_value=novelty),
            patch("app.checker.update_knowledge", new_callable=AsyncMock) as mock_update,
        ):
            result = await check_topic(topic, db_conn, settings)

        mock_update.assert_not_called()
        assert result.prompt_tokens == 70
        assert result.completion_tokens == 20


class TestMultiRoundInitialization:
    """Insufficient init retries across cycles until MAX, then forces READY."""

    async def _init(self, db_conn, topic, settings, *, sufficient: bool):
        from app.checker import initialize_new_topic

        articles = [_make_article(id=None, topic_id=topic.id)]
        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=articles, total_feed_entries=1),
            ),
            patch(
                "app.checker.initialize_knowledge",
                new_callable=AsyncMock,
                return_value=_make_write_result(sufficient_data=sufficient),
            ),
        ):
            await initialize_new_topic(topic, db_conn, settings)
        return get_topic(db_conn, topic.id)

    async def test_insufficient_returns_to_new_and_increments(self, db_conn: sqlite3.Connection) -> None:
        topic = _make_topic(db_conn, status=TopicStatus.NEW, status_changed_at=None)
        settings = _make_settings()

        updated = await self._init(db_conn, topic, settings, sufficient=False)
        assert updated.status == TopicStatus.NEW
        assert updated.init_attempts == 1
        assert updated.status_changed_at is not None

    async def test_exhausted_attempts_force_ready(self, db_conn: sqlite3.Connection) -> None:
        topic = _make_topic(db_conn, status=TopicStatus.NEW, status_changed_at=None, init_attempts=3)
        settings = _make_settings()

        updated = await self._init(db_conn, topic, settings, sufficient=False)
        assert updated.status == TopicStatus.READY
        # attempts reset on READY transition
        assert updated.init_attempts == 0

    async def test_sufficient_goes_ready_and_resets(self, db_conn: sqlite3.Connection) -> None:
        topic = _make_topic(db_conn, status=TopicStatus.NEW, status_changed_at=None, init_attempts=2)
        settings = _make_settings()

        updated = await self._init(db_conn, topic, settings, sufficient=True)
        assert updated.status == TopicStatus.READY
        assert updated.init_attempts == 0

    async def test_attempts_two_insufficient_stays_new_and_increments(self, db_conn: sqlite3.Connection) -> None:
        """OVH-076 boundary: init_attempts=2 + insufficient → stays NEW, becomes 3 (last retry)."""
        topic = _make_topic(db_conn, status=TopicStatus.NEW, status_changed_at=None, init_attempts=2)
        settings = _make_settings()

        updated = await self._init(db_conn, topic, settings, sufficient=False)
        assert updated.status == TopicStatus.NEW
        assert updated.init_attempts == 3

    async def _init_empty_fetch(self, db_conn, topic, settings):
        """Drive init where the fetch returns no articles (e.g. all already stored)."""
        from app.checker import initialize_new_topic

        with patch(
            "app.checker.fetch_new_articles_for_topic",
            new_callable=AsyncMock,
            return_value=FetchResult(articles=[], total_feed_entries=0),
        ):
            await initialize_new_topic(topic, db_conn, settings)
        return get_topic(db_conn, topic.id)

    async def test_empty_fetch_first_attempt_errors(self, db_conn: sqlite3.Connection) -> None:
        """OVH-001: first attempt (init_attempts=0) with no articles → ERROR."""
        topic = _make_topic(db_conn, status=TopicStatus.NEW, status_changed_at=None, init_attempts=0)
        settings = _make_settings()

        updated = await self._init_empty_fetch(db_conn, topic, settings)
        assert updated.status == TopicStatus.ERROR
        assert updated.error_message == "No articles found during initialization"

    async def test_empty_fetch_during_reinit_stays_new(self, db_conn: sqlite3.Connection) -> None:
        """OVH-001: empty fetch on a NEW-topic re-init (init_attempts>0) keeps waiting in NEW."""
        topic = _make_topic(db_conn, status=TopicStatus.NEW, status_changed_at=None, init_attempts=1)
        settings = _make_settings()

        updated = await self._init_empty_fetch(db_conn, topic, settings)
        assert updated.status == TopicStatus.NEW
        # init_attempts unchanged: nothing was analyzed this pass.
        assert updated.init_attempts == 1
        assert updated.error_message is None

    async def test_real_second_pass_empty_fetch_stays_new(self, db_conn: sqlite3.Connection) -> None:
        """OVH-001 real path: pass 1 stores+marks articles (insufficient → NEW, attempts=1);
        pass 2 fetch returns [] because every hash is already stored → stays NEW, not ERROR."""
        from app.checker import initialize_new_topic

        topic = _make_topic(db_conn, status=TopicStatus.NEW, status_changed_at=None, init_attempts=0)
        settings = _make_settings()
        stored_hashes: set[str] = set()

        async def fake_fetch(t, conn, **kwargs):
            # Mimic real dedup: only return articles whose hash isn't already stored.
            article = _make_article(id=None, topic_id=t.id, content_hash="hash-1")
            if article.content_hash in stored_hashes:
                return FetchResult(articles=[], total_feed_entries=1)
            created = create_article(conn, article)
            conn.commit()
            stored_hashes.add(article.content_hash)
            return FetchResult(articles=[created], total_feed_entries=1)

        with (
            patch("app.checker.fetch_new_articles_for_topic", side_effect=fake_fetch),
            patch(
                "app.checker.initialize_knowledge",
                new_callable=AsyncMock,
                return_value=_make_write_result(sufficient_data=False),
            ),
        ):
            # Pass 1: stores the article, insufficient → back to NEW, attempts=1.
            await initialize_new_topic(topic, db_conn, settings)
            after_pass1 = get_topic(db_conn, topic.id)
            assert after_pass1.status == TopicStatus.NEW
            assert after_pass1.init_attempts == 1

            # Pass 2: fetch finds nothing new (already stored) → must NOT error.
            await initialize_new_topic(after_pass1, db_conn, settings)

        after_pass2 = get_topic(db_conn, topic.id)
        assert after_pass2.status == TopicStatus.NEW
        assert after_pass2.error_message is None


class TestInitNoOverwriteConcurrentEdits:
    """OVH-100: init's terminal status write must not clobber concurrent UI edits."""

    async def _drive_terminal_write(self, db_conn, topic, settings, *, sufficient: bool):
        """Run init through to its terminal status write, simulating a concurrent edit
        to feeds/thresholds that lands while the LLM await is in flight."""
        from app.checker import initialize_new_topic

        article = create_article(db_conn, _make_article(id=None, topic_id=topic.id))
        db_conn.commit()

        async def edit_during_llm(*args, **kwargs):
            # Simulate the UI editing this topic's feeds/thresholds mid-init.
            db_conn.execute(
                "UPDATE topics SET feed_urls=?, confidence_threshold=? WHERE id=?",
                ('["https://edited.example.com/feed.xml"]', 0.42, topic.id),
            )
            db_conn.commit()
            return _make_write_result(sufficient_data=sufficient)

        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=[article], total_feed_entries=1),
            ),
            patch("app.checker.initialize_knowledge", side_effect=edit_during_llm),
        ):
            await initialize_new_topic(topic, db_conn, settings)
        return get_topic(db_conn, topic.id)

    async def test_ready_write_preserves_concurrent_feed_edit(self, db_conn: sqlite3.Connection) -> None:
        topic = _make_topic(db_conn, status=TopicStatus.NEW, status_changed_at=None)
        settings = _make_settings()

        updated = await self._drive_terminal_write(db_conn, topic, settings, sufficient=True)
        assert updated.status == TopicStatus.READY
        # The concurrent edit must survive the terminal status write.
        assert updated.feed_urls == ["https://edited.example.com/feed.xml"]
        assert updated.confidence_threshold == 0.42

    async def test_insufficient_write_preserves_concurrent_feed_edit(self, db_conn: sqlite3.Connection) -> None:
        topic = _make_topic(db_conn, status=TopicStatus.NEW, status_changed_at=None)
        settings = _make_settings()

        updated = await self._drive_terminal_write(db_conn, topic, settings, sufficient=False)
        assert updated.status == TopicStatus.NEW
        assert updated.init_attempts == 1
        assert updated.feed_urls == ["https://edited.example.com/feed.xml"]
        assert updated.confidence_threshold == 0.42


# --- check_all_topics ---


class TestCheckAllTopics:
    """Tests for the multi-topic check loop."""

    async def test_checks_all_ready_topics(self, db_conn: sqlite3.Connection, tmp_path: Path) -> None:
        _make_topic(db_conn, name="Topic A")
        _make_topic(db_conn, name="Topic B")
        settings = _make_settings()

        with patch(
            "app.checker.fetch_new_articles_for_topic",
            new_callable=AsyncMock,
            return_value=FetchResult(articles=[], total_feed_entries=0),
        ):
            results = await check_all_topics(settings, db_path=tmp_path / "test.db")

        assert len(results) == 2

    async def test_skips_researching_topics(self, db_conn: sqlite3.Connection, tmp_path: Path) -> None:
        _make_topic(db_conn, name="Ready", status=TopicStatus.READY)
        _make_topic(db_conn, name="Research", status=TopicStatus.RESEARCHING)
        settings = _make_settings()

        with patch(
            "app.checker.fetch_new_articles_for_topic",
            new_callable=AsyncMock,
            return_value=FetchResult(articles=[], total_feed_entries=0),
        ):
            results = await check_all_topics(settings, db_path=tmp_path / "test.db")

        assert len(results) == 1

    async def test_error_isolation(self, db_conn: sqlite3.Connection, tmp_path: Path) -> None:
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
            results = await check_all_topics(settings, db_path=tmp_path / "test.db")

        # Bad Topic's scraping error is caught inside check_topic,
        # so both topics produce a CheckResult.
        assert len(results) == 2

    async def test_returns_empty_when_no_topics(self, db_conn: sqlite3.Connection, tmp_path: Path) -> None:
        settings = _make_settings()
        results = await check_all_topics(settings, db_path=tmp_path / "test.db")
        assert results == []

    async def test_skips_inactive_topics(self, db_conn: sqlite3.Connection, tmp_path: Path) -> None:
        _make_topic(db_conn, name="Active", status=TopicStatus.READY, is_active=True)
        _make_topic(db_conn, name="Inactive", status=TopicStatus.READY, is_active=False)
        settings = _make_settings()

        with patch(
            "app.checker.fetch_new_articles_for_topic",
            new_callable=AsyncMock,
            return_value=FetchResult(articles=[], total_feed_entries=0),
        ):
            results = await check_all_topics(settings, db_path=tmp_path / "test.db")

        assert len(results) == 1

    async def test_outer_error_boundary_isolates_check_topic_crash(
        self, db_conn: sqlite3.Connection, tmp_path: Path
    ) -> None:
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
            results = await check_all_topics(settings, db_path=tmp_path / "test.db")

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

    async def test_abandoned_notification_warns_with_ids(self, db_conn: sqlite3.Connection, caplog) -> None:  # noqa: ANN001
        """Pruning an exhausted notification emits a WARNING naming topic/check ids (OVH-040)."""
        import logging

        topic = _make_topic(db_conn)
        create_pending_notification(
            db_conn,
            PendingNotification(
                topic_id=topic.id,
                check_result_id=777,
                title="Expired",
                body="B",
                retry_count=3,
                max_retries=3,
            ),
        )
        db_conn.commit()
        settings = _make_settings()

        with (
            caplog.at_level(logging.WARNING, logger="app.checker"),
            patch("app.checker.send_notification", new_callable=AsyncMock),
        ):
            await retry_pending_notifications(db_conn, settings)

        abandon_logs = [r.getMessage() for r in caplog.records if "Abandoning notification" in r.getMessage()]
        assert len(abandon_logs) == 1
        msg = abandon_logs[0]
        assert f"topic_id={topic.id}" in msg
        assert "check_result_id=777" in msg

    async def test_empty_pending_is_noop(self, db_conn: sqlite3.Connection) -> None:
        """No pending notifications means no send attempts."""
        settings = _make_settings()

        with patch(
            "app.checker.send_notification",
            new_callable=AsyncMock,
        ) as mock_send:
            await retry_pending_notifications(db_conn, settings)

        mock_send.assert_not_called()

    async def test_no_connection_held_across_send(self, db_conn: sqlite3.Connection) -> None:
        """The send must run with the snapshot connection already committed."""
        topic = _make_topic(db_conn)
        create_pending_notification(
            db_conn,
            PendingNotification(topic_id=topic.id, title="T", body="B"),
        )
        db_conn.commit()
        settings = _make_settings()

        in_transaction: list[bool] = []

        async def observe(title, body, s):  # noqa: ANN001
            in_transaction.append(db_conn.in_transaction)
            return True

        with patch("app.checker.send_notification", side_effect=observe):
            await retry_pending_notifications(db_conn, settings)

        assert in_transaction == [False]

    async def test_crash_midloop_preserves_applied_results(self, db_conn: sqlite3.Connection) -> None:
        """A crash applying item 2 must not roll back item 1's committed delete."""
        topic = _make_topic(db_conn)
        for i in range(2):
            create_pending_notification(
                db_conn,
                PendingNotification(topic_id=topic.id, title=f"T{i}", body="B"),
            )
        db_conn.commit()
        settings = _make_settings()

        pending = list_pending_notifications(db_conn)
        assert len(pending) == 2
        first_id = pending[0].id

        from app.crud import delete_pending_notification as real_delete

        call_count = {"n": 0}

        def crashing_delete(conn, notification_id):  # noqa: ANN001
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise RuntimeError("simulated crash applying item 2")
            real_delete(conn, notification_id)

        with (
            patch("app.checker.send_notification", new_callable=AsyncMock, return_value=True),
            patch("app.checker.delete_pending_notification", side_effect=crashing_delete),
            pytest.raises(RuntimeError, match="simulated crash"),
        ):
            await retry_pending_notifications(db_conn, settings)

        remaining = db_conn.execute("SELECT id FROM pending_notifications").fetchall()
        remaining_ids = {r["id"] for r in remaining}
        assert first_id not in remaining_ids
        assert len(remaining_ids) == 1
