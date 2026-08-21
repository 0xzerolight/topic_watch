"""Tests for the core check loop: check_topic, check_all_topics, retry logic."""

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.analysis.knowledge import KnowledgeUpdatePlan
from app.analysis.llm import NoveltyResponse, NoveltyResult, TokenUsage
from app.checker import check_all_topics, check_topic, retry_pending_notifications
from app.config import LLMSettings, NotificationSettings, Settings
from app.crud import (
    MAX_ANALYSIS_ATTEMPTS,
    apply_notification_outcome,
    claim_notification_intent,
    create_article,
    create_knowledge_state,
    create_pending_notification,
    create_topic,
    get_topic,
    list_articles_for_topic,
    list_due_notification_intents,
    list_pending_notifications,
    release_stale_notification_claims,
    update_topic,
)
from app.models import (
    Article,
    FeedMode,
    KnowledgeState,
    NotificationDelivery,
    NotifyDisposition,
    PendingNotification,
    Topic,
    TopicStatus,
    to_db_utc,
)
from app.scraping import FetchResult
from tests.helpers import conn_db_path


def _make_settings(**overrides) -> Settings:
    defaults = {
        "llm": LLMSettings(model="openai/gpt-4o-mini", api_key="test-key"),
        "notifications": NotificationSettings(urls=["json://localhost"]),
        # Off by default so an unrelated test that happens to record several
        # source-failing checks never attempts a real Apprise send.
        "silence_heartbeat_checks": 0,
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _per_url_mock(*, ok: bool, error: str | None = None, url: str = "json://localhost") -> AsyncMock:
    """AsyncMock for app.checker.deliver_notification_intents returning one outcome.

    Delivery is per-target (OVH-039); this mirrors the single-URL default settings
    used across these tests. Patching at the deliver-intents seam leaves the intent
    rows themselves untouched, which is what the pipeline tests care about.
    """
    return AsyncMock(return_value=[NotificationDelivery(url=url, ok=ok, error=error)])


async def _ok_send(title: str, body: str, url: str, timeout_s: float) -> NotificationDelivery:
    """Stub for app.checker.send_single_notification: every target delivers."""
    return NotificationDelivery(url=url, ok=True)


def _fail_send(error: str):
    """Build a send_single_notification stub whose every target fails with ``error``."""

    async def _send(title: str, body: str, url: str, timeout_s: float) -> NotificationDelivery:
        return NotificationDelivery(url=url, ok=False, error=error)

    return _send


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
) -> KnowledgeUpdatePlan:
    """Build a KnowledgeUpdatePlan for mocking the prepare_* knowledge returns."""
    return KnowledgeUpdatePlan(
        summary_text="state",
        token_count=0,
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

    async def test_happy_path_new_info_sends_notification(self, db_conn: sqlite3.Connection, db_path: Path) -> None:
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
                "app.checker.prepare_knowledge_update",
                new_callable=AsyncMock,
                return_value=_make_write_result(),
            ) as mock_update,
            patch("app.checker.deliver_notification_intents", _per_url_mock(ok=True)) as mock_send,
        ):
            result = await check_topic(topic, settings, db_path=db_path)

        assert result.has_new_info is True
        assert result.notification_sent is True
        assert result.articles_found == 1
        assert result.id is not None
        mock_update.assert_called_once()
        mock_send.assert_called_once()

    async def test_no_new_info_no_notification(self, db_conn: sqlite3.Connection, db_path: Path) -> None:
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
            patch("app.checker.deliver_notification_intents") as mock_send,
        ):
            result = await check_topic(topic, settings, db_path=db_path)

        assert result.has_new_info is False
        assert result.notification_sent is False
        mock_send.assert_not_called()

    async def test_no_new_articles_early_return(self, db_conn: sqlite3.Connection, db_path: Path) -> None:
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
            result = await check_topic(topic, settings, db_path=db_path)

        assert result.articles_found == 0
        assert result.has_new_info is False
        mock_analyze.assert_not_called()

    async def test_scraping_failure_records_result(self, db_conn: sqlite3.Connection, db_path: Path) -> None:
        """Scraping error should not crash, should record a result."""
        topic = _make_topic(db_conn)
        settings = _make_settings()

        with patch(
            "app.checker.fetch_new_articles_for_topic",
            new_callable=AsyncMock,
            side_effect=Exception("Network error"),
        ):
            result = await check_topic(topic, settings, db_path=db_path)

        assert result.articles_found == 0
        assert result.id is not None

    async def test_skips_non_ready_topic(self, db_conn: sqlite3.Connection, db_path: Path) -> None:
        """A non-READY topic is skipped, and the skip is not filed as a check.

        Nothing was fetched or analyzed, so persisting a clean zero-valued row
        claimed monitoring that never happened: it broke source-failure streaks
        and its fresh ``checked_at`` pushed the first real check further out
        (AUG-134).
        """
        topic = _make_topic(db_conn, name="Researching", status=TopicStatus.RESEARCHING)
        settings = _make_settings()

        result = await check_topic(topic, settings, db_path=db_path)

        assert result.articles_found == 0
        assert result.id is None
        assert result.stage_error is not None
        assert result.stage_error.startswith("skipped: topic not ready")
        assert db_conn.execute("SELECT COUNT(*) FROM check_results").fetchone()[0] == 0

    async def test_notification_failure_captured_and_queued(self, db_conn: sqlite3.Connection, db_path: Path) -> None:
        """Notification failure should be recorded and queued for retry.

        OVH-085: the source article is persisted and asserted ``processed==1``
        after the failed send — failed-but-queued articles are still marked
        processed (the queued notification is the only recovery path; a
        reordering that only marks processed on success would fail here).
        """
        topic = _make_topic(db_conn, name="NotifFail")
        create_knowledge_state(
            db_conn,
            KnowledgeState(topic_id=topic.id, summary_text="Summary.", token_count=10),
        )
        # Persist a real article so its processed flag can be asserted from the DB.
        article = create_article(db_conn, _make_article(id=None, topic_id=topic.id))
        db_conn.commit()
        settings = _make_settings()

        novelty = NoveltyResult(has_new_info=True, summary="Update", confidence=0.9, relevance=0.9)

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
            patch("app.checker.prepare_knowledge_update", new_callable=AsyncMock, return_value=_make_write_result()),
            patch(
                "app.checker.deliver_notification_intents",
                _per_url_mock(ok=False, error="SMTP error"),
            ),
        ):
            result = await check_topic(topic, settings, db_path=db_path)

        assert result.has_new_info is True
        assert result.notification_sent is False
        assert result.notification_error is not None

        # Verify a pending notification was actually queued in the DB
        pending = list_pending_notifications(db_conn)
        assert len(pending) == 1
        assert pending[0].topic_id == topic.id
        assert "Topic Watch:" in pending[0].title
        # The intent was created inside the durable transition, so it is queued
        # whatever the send did — the outcome is applied to it, not the reason it
        # exists (TW-AUD-004).
        assert pending[0].status == "pending"
        assert pending[0].check_result_id == result.id

        # OVH-085: the article is marked processed even though the send failed.
        assert article.id is not None
        proc = db_conn.execute("SELECT processed FROM articles WHERE id = ?", (article.id,)).fetchone()
        assert proc["processed"] == 1

    async def test_notification_delivery_failure_queued(self, db_conn: sqlite3.Connection, db_path: Path) -> None:
        """When send_notification returns False, notification is queued for retry.

        OVH-085: also pins that the source article is marked ``processed==1`` on
        the delivery-returned-False failure path.
        """
        topic = _make_topic(db_conn, name="DeliveryFail")
        create_knowledge_state(
            db_conn,
            KnowledgeState(topic_id=topic.id, summary_text="Summary.", token_count=10),
        )
        article = create_article(db_conn, _make_article(id=None, topic_id=topic.id))
        db_conn.commit()
        settings = _make_settings()

        novelty = NoveltyResult(has_new_info=True, summary="Update", confidence=0.9, relevance=0.9)

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
            patch("app.checker.prepare_knowledge_update", new_callable=AsyncMock, return_value=_make_write_result()),
            patch(
                "app.checker.deliver_notification_intents",
                _per_url_mock(ok=False, error="delivery failed"),
            ),
        ):
            result = await check_topic(topic, settings, db_path=db_path)

        assert result.notification_sent is False
        # Per-URL failures are summarized redacted (scheme://host: reason) (OVH-039).
        assert result.notification_error == "json://localhost: delivery failed"

        # Verify the failed URL was queued for retry, scoped to that URL.
        pending = list_pending_notifications(db_conn)
        assert len(pending) == 1
        assert pending[0].url == "json://localhost"
        # OVH-040 traceability: the queued row is correlated to its check result
        # (previously NULL for notifications; the webhook path already did this).
        assert result.id is not None
        assert pending[0].check_result_id == result.id

        # OVH-085: the article is marked processed even though delivery failed.
        assert article.id is not None
        proc = db_conn.execute("SELECT processed FROM articles WHERE id = ?", (article.id,)).fetchone()
        assert proc["processed"] == 1

    async def test_llm_response_stored_as_json(self, db_conn: sqlite3.Connection, db_path: Path) -> None:
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
            patch("app.checker.prepare_knowledge_update", new_callable=AsyncMock, return_value=_make_write_result()),
            patch("app.checker.deliver_notification_intents", _per_url_mock(ok=True)),
        ):
            result = await check_topic(topic, settings, db_path=db_path)

        parsed = json.loads(result.llm_response)
        assert parsed["has_new_info"] is True
        assert parsed["summary"] == "New thing"

    async def test_knowledge_summary_passed_to_analyze(self, db_conn: sqlite3.Connection, db_path: Path) -> None:
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
            await check_topic(topic, settings, db_path=db_path)

        # Verify the knowledge summary was actually passed
        call_args = mock_analyze.call_args
        knowledge_summary_arg = call_args[0][1]  # second positional arg
        assert knowledge_summary_arg == "Specific knowledge summary XYZ."

    async def test_knowledge_update_failure_still_notifies(self, db_conn: sqlite3.Connection, db_path: Path) -> None:
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
                "app.checker.prepare_knowledge_update",
                new_callable=AsyncMock,
                side_effect=Exception("Knowledge update crashed"),
            ),
            patch("app.checker.deliver_notification_intents", _per_url_mock(ok=True)) as mock_send,
        ):
            result = await check_topic(topic, settings, db_path=db_path)

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

    async def test_scrape_failure_sets_stage_error(self, db_conn: sqlite3.Connection, db_path: Path) -> None:
        """A raising fetch records stage_error='pipeline_failed' + summary (OVH-037/AUG-133).

        Per-feed failures are caught inside the fetch and counted, so an exception
        escaping it is our own storage/dedup/extraction breaking — an internal
        failure, not a source outage.
        """
        topic = _make_topic(db_conn, name="ScrapeFail")
        settings = _make_settings()

        with patch(
            "app.checker.fetch_new_articles_for_topic",
            new_callable=AsyncMock,
            side_effect=Exception("Network error"),
        ):
            result = await check_topic(topic, settings, db_path=db_path)

        assert result.id is not None
        assert result.stage_error is not None
        assert result.stage_error.startswith("pipeline_failed")
        row = db_conn.execute("SELECT stage_error FROM check_results WHERE id = ?", (result.id,)).fetchone()
        assert row["stage_error"].startswith("pipeline_failed")

    async def test_analysis_failure_sets_stage_error(self, db_conn: sqlite3.Connection, db_path: Path) -> None:
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
            patch("app.checker.deliver_notification_intents") as mock_send,
        ):
            result = await check_topic(topic, settings, db_path=db_path)

        mock_send.assert_not_called()
        assert result.has_new_info is False
        assert result.id is not None
        assert result.stage_error is not None
        assert result.stage_error.startswith("analysis_failed")
        row = db_conn.execute("SELECT stage_error FROM check_results WHERE id = ?", (result.id,)).fetchone()
        assert row["stage_error"].startswith("analysis_failed")

    async def test_clean_no_new_info_has_no_stage_error(self, db_conn: sqlite3.Connection, db_path: Path) -> None:
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
            patch("app.checker.deliver_notification_intents") as mock_send,
        ):
            result = await check_topic(topic, settings, db_path=db_path)

        mock_send.assert_not_called()
        assert result.stage_error is None
        # And the article IS marked processed (we evaluated it, no failure).
        assert article.id is not None
        proc = db_conn.execute("SELECT processed FROM articles WHERE id = ?", (article.id,)).fetchone()
        assert proc["processed"] == 1

    async def test_low_confidence_skips_notification_but_marks_processed(
        self, db_conn: sqlite3.Connection, db_path: Path
    ) -> None:
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
                "app.checker.prepare_knowledge_update", new_callable=AsyncMock, return_value=_make_write_result()
            ) as mock_update,
            patch("app.checker.deliver_notification_intents") as mock_send,
        ):
            result = await check_topic(topic, settings, db_path=db_path)

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
            await check_topic(topic, settings, db_path=db_path)
        mock_analyze2.assert_not_called()
        mock_analyze.assert_called_once()

    async def test_high_confidence_sends_notification(self, db_conn: sqlite3.Connection, db_path: Path) -> None:
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
                "app.checker.prepare_knowledge_update", new_callable=AsyncMock, return_value=_make_write_result()
            ) as mock_update,
            patch("app.checker.deliver_notification_intents", _per_url_mock(ok=True)) as mock_send,
        ):
            result = await check_topic(topic, settings, db_path=db_path)

        assert result.has_new_info is True
        assert result.notification_sent is True
        mock_update.assert_called_once()
        mock_send.assert_called_once()

    async def test_low_relevance_skips_notification_but_marks_processed(
        self, db_conn: sqlite3.Connection, db_path: Path
    ) -> None:
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
                "app.checker.prepare_knowledge_update", new_callable=AsyncMock, return_value=_make_write_result()
            ) as mock_update,
            patch("app.checker.deliver_notification_intents") as mock_send,
        ):
            result = await check_topic(topic, settings, db_path=db_path)

        assert result.has_new_info is True
        assert result.notification_sent is False
        mock_update.assert_not_called()
        mock_send.assert_not_called()

        assert article.id is not None
        row = db_conn.execute("SELECT processed FROM articles WHERE id = ?", (article.id,)).fetchone()
        assert row["processed"] == 1

    async def test_confidence_equal_to_threshold_sends_notification(
        self, db_conn: sqlite3.Connection, db_path: Path
    ) -> None:
        """OVH-075: confidence EXACTLY at the threshold notifies (locks ``<`` not ``<=``).

        The gate is ``novelty.confidence < threshold``: an equal value must pass.
        A regression flipping the operator to ``<=`` would suppress at the
        boundary and this test would fail.
        """
        topic = _make_topic(db_conn, name="ConfBoundary")
        create_knowledge_state(
            db_conn,
            KnowledgeState(topic_id=topic.id, summary_text="Known facts.", token_count=20),
        )
        db_conn.commit()
        settings = _make_settings(min_confidence_threshold=0.7)

        articles = [_make_article(topic_id=topic.id)]
        novelty = NoveltyResult(
            has_new_info=True,
            summary="Boundary confidence update",
            confidence=0.7,  # exactly equal to the threshold
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
                "app.checker.prepare_knowledge_update", new_callable=AsyncMock, return_value=_make_write_result()
            ) as mock_update,
            patch("app.checker.deliver_notification_intents", _per_url_mock(ok=True)) as mock_send,
        ):
            result = await check_topic(topic, settings, db_path=db_path)

        assert result.has_new_info is True
        assert result.notification_sent is True
        mock_update.assert_called_once()
        mock_send.assert_called_once()

    async def test_relevance_equal_to_threshold_sends_notification(
        self, db_conn: sqlite3.Connection, db_path: Path
    ) -> None:
        """OVH-075: relevance EXACTLY at the threshold notifies (locks ``<`` not ``<=``)."""
        topic = _make_topic(db_conn, name="RelBoundary")
        create_knowledge_state(
            db_conn,
            KnowledgeState(topic_id=topic.id, summary_text="Known facts.", token_count=20),
        )
        db_conn.commit()
        settings = _make_settings(min_relevance_threshold=0.5)

        articles = [_make_article(topic_id=topic.id)]
        novelty = NoveltyResult(
            has_new_info=True,
            summary="Boundary relevance update",
            confidence=0.9,
            relevance=0.5,  # exactly equal to the threshold
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
                "app.checker.prepare_knowledge_update", new_callable=AsyncMock, return_value=_make_write_result()
            ) as mock_update,
            patch("app.checker.deliver_notification_intents", _per_url_mock(ok=True)) as mock_send,
        ):
            result = await check_topic(topic, settings, db_path=db_path)

        assert result.has_new_info is True
        assert result.notification_sent is True
        mock_update.assert_called_once()
        mock_send.assert_called_once()


# --- check_id_var lifecycle (OVH-088, OVH-103) ---


class TestCheckTopicResetsCheckIdVar:
    """check_topic uses the leak-safe token idiom (OVH-103): it RESTORES whatever
    correlation id the caller had set rather than clobbering it to None, so a
    nested check_topic inside an outer flow that owns its own check_id leaves that
    outer id intact. With no outer id, the var returns to its default (None)."""

    async def test_outer_check_id_restored_after_nested_check(self, db_conn: sqlite3.Connection, db_path: Path) -> None:
        """An outer caller's check_id survives a nested check_topic (OVH-103)."""
        from app.check_context import check_id_var

        topic = _make_topic(db_conn, name="ResetSuccess")
        settings = _make_settings()

        token = check_id_var.set("outer-sentinel")
        try:
            with patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=[], total_feed_entries=0),
            ):
                await check_topic(topic, settings, db_path=db_path)
            # OVH-103: the inner finally restores the prior token, not None.
            assert check_id_var.get() == "outer-sentinel"
        finally:
            check_id_var.reset(token)

    async def test_var_returns_to_default_when_no_outer_id(self, db_conn: sqlite3.Connection, db_path: Path) -> None:
        """With no outer id set, check_topic leaves the var at its default."""
        from app.check_context import check_id_var

        topic = _make_topic(db_conn, name="ResetDefault")
        settings = _make_settings()

        assert check_id_var.get() is None
        with patch(
            "app.checker.fetch_new_articles_for_topic",
            new_callable=AsyncMock,
            return_value=FetchResult(articles=[], total_feed_entries=0),
        ):
            await check_topic(topic, settings, db_path=db_path)
        assert check_id_var.get() is None

    async def test_outer_check_id_restored_after_inner_pipeline_raises(
        self, db_conn: sqlite3.Connection, db_path: Path
    ) -> None:
        """Even when the inner pipeline raises, the finally restores the outer id."""
        from app.check_context import check_id_var

        topic = _make_topic(db_conn, name="ResetRaise")
        settings = _make_settings()

        token = check_id_var.set("outer-sentinel")
        try:
            with (
                patch(
                    "app.checker._check_topic_inner",
                    new_callable=AsyncMock,
                    side_effect=RuntimeError("pipeline boom"),
                ),
                pytest.raises(RuntimeError, match="pipeline boom"),
            ):
                await check_topic(topic, settings, db_path=db_path)
            assert check_id_var.get() == "outer-sentinel"
        finally:
            check_id_var.reset(token)

    async def test_inner_check_uses_its_own_check_id_during_run(
        self, db_conn: sqlite3.Connection, db_path: Path
    ) -> None:
        """Inside the pipeline the var holds the freshly generated id (not the
        outer one), then the outer id is restored afterwards."""
        from app.check_context import check_id_var

        topic = _make_topic(db_conn, name="InnerId")
        settings = _make_settings()

        seen: dict[str, str | None] = {}

        async def _capture(*_args, **_kwargs):
            seen["inner"] = check_id_var.get()
            return FetchResult(articles=[], total_feed_entries=0)

        token = check_id_var.set("outer-sentinel")
        try:
            with patch("app.checker.fetch_new_articles_for_topic", side_effect=_capture):
                await check_topic(topic, settings, db_path=db_path)
            assert seen["inner"] is not None
            assert seen["inner"] != "outer-sentinel"
            assert check_id_var.get() == "outer-sentinel"
        finally:
            check_id_var.reset(token)


# --- init / retry-drain correlation id (OVH-102) ---


class TestInitAndRetryCarryCheckId:
    """OVH-102: the multi-round init flow and the notification retry-drain must
    run under a generated check_id so a single topic's init / a single drain is
    traceable across interleaved scheduler ticks (no '-' correlation placeholder).
    The token idiom is used so any outer caller's id is restored afterwards."""

    async def test_initialize_new_topic_sets_check_id_during_run(
        self, db_conn: sqlite3.Connection, db_path: Path
    ) -> None:
        from app.check_context import check_id_var
        from app.checker import initialize_new_topic

        topic = _make_topic(db_conn, name="InitCid", status=TopicStatus.NEW, status_changed_at=None)
        settings = _make_settings()

        seen: dict[str, str | None] = {}

        async def _capture(*_args, **_kwargs):
            seen["fetch"] = check_id_var.get()
            return FetchResult(articles=[], total_feed_entries=0)

        assert check_id_var.get() is None
        with patch("app.checker.fetch_new_articles_for_topic", side_effect=_capture):
            await initialize_new_topic(topic, settings, db_path=db_path)

        # The init flow ran under a real correlation id, not the '-' placeholder.
        assert seen["fetch"] is not None
        assert seen["fetch"] != "-"
        # And it restored the prior (default) afterwards.
        assert check_id_var.get() is None

    async def test_initialize_new_topic_restores_outer_check_id(
        self, db_conn: sqlite3.Connection, db_path: Path
    ) -> None:
        """A caller that owns an outer check_id keeps it after init returns."""
        from app.check_context import check_id_var
        from app.checker import initialize_new_topic

        topic = _make_topic(db_conn, name="InitOuter", status=TopicStatus.NEW, status_changed_at=None)
        settings = _make_settings()

        token = check_id_var.set("outer-sentinel")
        try:
            with patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=[], total_feed_entries=0),
            ):
                await initialize_new_topic(topic, settings, db_path=db_path)
            assert check_id_var.get() == "outer-sentinel"
        finally:
            check_id_var.reset(token)

    async def test_initialize_new_topic_logs_carry_check_id(
        self, db_conn: sqlite3.Connection, caplog, db_path: Path
    ) -> None:  # noqa: ANN001
        """Init log lines carry a non-'-' check_id (the whole point of OVH-102)."""
        import logging

        from app.check_context import CheckIdFilter, check_id_var
        from app.checker import initialize_new_topic

        topic = _make_topic(db_conn, name="InitLog", status=TopicStatus.NEW, status_changed_at=None)
        settings = _make_settings()

        check_id_filter = CheckIdFilter()
        caplog.handler.addFilter(check_id_filter)
        try:
            assert check_id_var.get() is None
            with (
                caplog.at_level(logging.INFO, logger="app.checker"),
                patch(
                    "app.checker.fetch_new_articles_for_topic",
                    new_callable=AsyncMock,
                    return_value=FetchResult(articles=[], total_feed_entries=0),
                ),
            ):
                await initialize_new_topic(topic, settings, db_path=db_path)
        finally:
            caplog.handler.removeFilter(check_id_filter)

        init_records = [r for r in caplog.records if r.name == "app.checker"]
        assert init_records, "expected init log lines"
        # Every init log line carried a real correlation id, never the placeholder.
        assert all(getattr(r, "check_id", "-") not in (None, "-") for r in init_records)

    async def test_retry_drain_sets_check_id_during_run(self, db_conn: sqlite3.Connection) -> None:
        """The notification retry-drain runs under a generated check_id (OVH-102)."""
        from app.check_context import check_id_var

        topic = _make_topic(db_conn)
        create_pending_notification(
            db_conn,
            PendingNotification(topic_id=topic.id, title="T", body="B", url="json://localhost"),
        )
        db_conn.commit()
        settings = _make_settings()

        seen: dict[str, str | None] = {}

        async def _capture(title, body, url, timeout_s):  # noqa: ANN001
            seen["send"] = check_id_var.get()
            return NotificationDelivery(url=url, ok=True)

        assert check_id_var.get() is None
        with patch("app.checker.send_single_notification", side_effect=_capture):
            await retry_pending_notifications(db_conn, settings)

        assert seen["send"] is not None
        assert seen["send"] != "-"
        # The drain restored the prior (default) contextvar afterwards.
        assert check_id_var.get() is None

    async def test_retry_drain_restores_outer_check_id(self, db_conn: sqlite3.Connection) -> None:
        """An outer caller's check_id survives the retry-drain (token idiom)."""
        from app.check_context import check_id_var

        topic = _make_topic(db_conn)
        create_pending_notification(
            db_conn,
            PendingNotification(topic_id=topic.id, title="T", body="B", url="json://localhost"),
        )
        db_conn.commit()
        settings = _make_settings()

        token = check_id_var.set("outer-sentinel")
        try:
            with patch("app.checker.send_single_notification", side_effect=_ok_send):
                await retry_pending_notifications(db_conn, settings)
            assert check_id_var.get() == "outer-sentinel"
        finally:
            check_id_var.reset(token)


# --- initialize_new_topic ---


class TestInitializeNewTopicStatusChangedAt:
    """status_changed_at must be refreshed on every status transition."""

    async def test_ready_transition_sets_status_changed_at(self, db_conn: sqlite3.Connection, db_path: Path) -> None:
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
                "app.checker.prepare_initial_knowledge",
                new_callable=AsyncMock,
                return_value=_make_write_result(),
            ),
        ):
            await initialize_new_topic(topic, settings, db_path=db_path)

        from app.crud import get_topic

        updated = get_topic(db_conn, topic.id)
        assert updated.status == TopicStatus.READY
        assert updated.status_changed_at is not None

    async def test_no_articles_error_transition_sets_status_changed_at(
        self, db_conn: sqlite3.Connection, db_path: Path
    ) -> None:
        topic = _make_topic(db_conn, status=TopicStatus.NEW, status_changed_at=None)
        settings = _make_settings()
        from app.checker import initialize_new_topic

        with patch(
            "app.checker.fetch_new_articles_for_topic",
            new_callable=AsyncMock,
            return_value=FetchResult(articles=[], total_feed_entries=0),
        ):
            await initialize_new_topic(topic, settings, db_path=db_path)

        from app.crud import get_topic

        updated = get_topic(db_conn, topic.id)
        assert updated.status == TopicStatus.ERROR
        assert updated.status_changed_at is not None

    async def test_exception_error_transition_sets_status_changed_at(
        self, db_conn: sqlite3.Connection, db_path: Path
    ) -> None:
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
                "app.checker.prepare_initial_knowledge",
                new_callable=AsyncMock,
                side_effect=Exception("LLM down"),
            ),
        ):
            await initialize_new_topic(topic, settings, db_path=db_path)

        from app.crud import get_topic

        updated = get_topic(db_conn, topic.id)
        assert updated.status == TopicStatus.ERROR
        assert updated.status_changed_at is not None


class TestPerTopicThresholds:
    """Per-topic confidence/relevance overrides gate notifications."""

    async def _run(self, db_conn, topic, novelty, settings):
        db_path = conn_db_path(db_conn)
        article = create_article(db_conn, _make_article(id=None, topic_id=topic.id))
        db_conn.commit()
        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=[article], total_feed_entries=1),
            ),
            patch("app.checker.analyze_articles", new_callable=AsyncMock, return_value=novelty),
            patch("app.checker.prepare_knowledge_update", new_callable=AsyncMock, return_value=_make_write_result()),
            patch("app.checker.deliver_notification_intents", _per_url_mock(ok=True)) as mock_send,
        ):
            result = await check_topic(topic, settings, db_path=db_path)
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

    async def test_per_topic_relevance_threshold_suppresses(self, db_conn: sqlite3.Connection, db_path: Path) -> None:
        topic = _make_topic(db_conn, relevance_threshold=0.9)
        settings = _make_settings(min_relevance_threshold=0.3)
        novelty = NoveltyResult(has_new_info=True, summary="x", confidence=0.95, relevance=0.5)

        result, mock_send = await self._run(db_conn, topic, novelty, settings)

        assert result.notification_sent is False
        mock_send.assert_not_called()


class TestImportanceThreshold:
    """Per-topic importance threshold suppresses sends but NOT the knowledge update.

    Unlike the confidence/relevance gates (unreliable result -> skip everything),
    below-importance info is genuinely new and on-topic: the knowledge state must
    still absorb it, or the same trivial fact re-flags as "new" every cycle.
    """

    async def _run(self, db_conn, topic, novelty, settings):
        db_path = conn_db_path(db_conn)
        article = create_article(db_conn, _make_article(id=None, topic_id=topic.id))
        db_conn.commit()
        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=[article], total_feed_entries=1),
            ),
            patch("app.checker.analyze_articles", new_callable=AsyncMock, return_value=novelty),
            patch(
                "app.checker.prepare_knowledge_update", new_callable=AsyncMock, return_value=_make_write_result()
            ) as mock_update,
            patch("app.checker.deliver_notification_intents", _per_url_mock(ok=True)) as mock_send,
            patch("app.checker.deliver_webhook_intents", new_callable=AsyncMock) as mock_webhooks,
        ):
            result = await check_topic(topic, settings, db_path=db_path)
        return result, mock_send, mock_update, mock_webhooks, article

    async def test_below_threshold_suppresses_sends_but_updates_knowledge(self, db_conn: sqlite3.Connection) -> None:
        """Importance 2 < threshold 4 → no notification/webhook, but knowledge updates
        and articles are marked processed."""
        topic = _make_topic(db_conn, importance_threshold=4)
        settings = _make_settings(min_confidence_threshold=0.5, min_relevance_threshold=0.3)
        novelty = NoveltyResult(has_new_info=True, summary="x", confidence=0.9, relevance=0.9, importance=2)

        result, mock_send, mock_update, mock_webhooks, article = await self._run(db_conn, topic, novelty, settings)

        assert result.has_new_info is True
        assert result.notification_sent is False
        mock_send.assert_not_called()
        mock_webhooks.assert_not_called()
        mock_update.assert_awaited_once()
        articles = list_articles_for_topic(db_conn, topic.id)
        assert all(a.processed for a in articles)

    async def test_at_threshold_notifies(self, db_conn: sqlite3.Connection) -> None:
        """Importance exactly at the threshold notifies (locks >= not >)."""
        topic = _make_topic(db_conn, importance_threshold=4)
        settings = _make_settings(min_confidence_threshold=0.5, min_relevance_threshold=0.3)
        novelty = NoveltyResult(has_new_info=True, summary="x", confidence=0.9, relevance=0.9, importance=4)

        result, mock_send, mock_update, _, _ = await self._run(db_conn, topic, novelty, settings)

        assert result.notification_sent is True
        mock_send.assert_called_once()
        mock_update.assert_awaited_once()

    async def test_null_threshold_notifies_at_importance_one(self, db_conn: sqlite3.Connection) -> None:
        """NULL importance_threshold = no suppression: importance 1 still notifies."""
        topic = _make_topic(db_conn, importance_threshold=None)
        settings = _make_settings(min_confidence_threshold=0.5, min_relevance_threshold=0.3)
        novelty = NoveltyResult(has_new_info=True, summary="x", confidence=0.9, relevance=0.9, importance=1)

        result, mock_send, _, _, _ = await self._run(db_conn, topic, novelty, settings)

        assert result.notification_sent is True
        mock_send.assert_called_once()


class TestCheckResultTokens:
    """check_results record the summed analysis + knowledge tokens."""

    async def test_tokens_summed_from_analysis_and_knowledge(self, db_conn: sqlite3.Connection, db_path: Path) -> None:
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
                "app.checker.prepare_knowledge_update",
                new_callable=AsyncMock,
                return_value=_make_write_result(prompt_tokens=30, completion_tokens=10),
            ),
            patch("app.checker.deliver_notification_intents", _per_url_mock(ok=True)),
        ):
            result = await check_topic(topic, settings, db_path=db_path)

        assert result.prompt_tokens == 130
        assert result.completion_tokens == 50
        row = db_conn.execute(
            "SELECT prompt_tokens, completion_tokens FROM check_results WHERE id = ?", (result.id,)
        ).fetchone()
        assert row["prompt_tokens"] == 130
        assert row["completion_tokens"] == 50

    async def test_early_return_records_zero_tokens(self, db_conn: sqlite3.Connection, db_path: Path) -> None:
        topic = _make_topic(db_conn)
        settings = _make_settings()

        with patch(
            "app.checker.fetch_new_articles_for_topic",
            new_callable=AsyncMock,
            return_value=FetchResult(articles=[], total_feed_entries=0),
        ):
            result = await check_topic(topic, settings, db_path=db_path)

        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0
        row = db_conn.execute(
            "SELECT prompt_tokens, completion_tokens FROM check_results WHERE id = ?", (result.id,)
        ).fetchone()
        assert row["prompt_tokens"] == 0
        assert row["completion_tokens"] == 0

    async def test_tokens_only_analysis_when_below_threshold(self, db_conn: sqlite3.Connection, db_path: Path) -> None:
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
            patch("app.checker.prepare_knowledge_update", new_callable=AsyncMock) as mock_update,
        ):
            result = await check_topic(topic, settings, db_path=db_path)

        mock_update.assert_not_called()
        assert result.prompt_tokens == 70
        assert result.completion_tokens == 20


class TestMultiRoundInitialization:
    """Insufficient init goes READY immediately (no bounce); sufficient init also goes READY."""

    async def _init(self, db_conn, topic, settings, *, sufficient: bool):
        db_path = conn_db_path(db_conn)
        from app.checker import initialize_new_topic

        articles = [_make_article(id=None, topic_id=topic.id)]
        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=articles, total_feed_entries=1),
            ),
            patch(
                "app.checker.prepare_initial_knowledge",
                new_callable=AsyncMock,
                return_value=_make_write_result(sufficient_data=sufficient),
            ),
        ):
            await initialize_new_topic(topic, settings, db_path=db_path)
        return get_topic(db_conn, topic.id)

    async def test_insufficient_goes_ready_immediately(self, db_conn: sqlite3.Connection) -> None:
        """Insufficient init no longer bounces to NEW — topic goes READY so baseline isn't discarded."""
        topic = _make_topic(db_conn, status=TopicStatus.NEW, status_changed_at=None)
        settings = _make_settings()

        updated = await self._init(db_conn, topic, settings, sufficient=False)
        assert updated.status == TopicStatus.READY
        assert updated.init_attempts == 0
        assert updated.status_changed_at is not None

    async def test_insufficient_goes_ready_and_mark_articles_processed_called(
        self,
        db_conn: sqlite3.Connection,
        db_path: Path,
    ) -> None:
        """mark_articles_processed fires before the READY transition — articles not discarded."""
        from app.checker import initialize_new_topic

        topic = _make_topic(db_conn, status=TopicStatus.NEW, status_changed_at=None)
        article = create_article(db_conn, _make_article(id=None, topic_id=topic.id))
        db_conn.commit()
        settings = _make_settings()

        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=[article], total_feed_entries=1),
            ),
            patch(
                "app.checker.prepare_initial_knowledge",
                new_callable=AsyncMock,
                return_value=_make_write_result(sufficient_data=False),
            ),
            patch("app.checker.mark_articles_processed") as mock_mark,
        ):
            await initialize_new_topic(topic, settings, db_path=db_path)

        updated = get_topic(db_conn, topic.id)
        assert updated.status == TopicStatus.READY
        mock_mark.assert_called_once()
        assert mock_mark.call_args.args[1] == [article.id]

    async def test_sufficient_goes_ready_and_resets(self, db_conn: sqlite3.Connection, db_path: Path) -> None:
        topic = _make_topic(db_conn, status=TopicStatus.NEW, status_changed_at=None, init_attempts=2)
        settings = _make_settings()

        updated = await self._init(db_conn, topic, settings, sufficient=True)
        assert updated.status == TopicStatus.READY
        assert updated.init_attempts == 0

    async def _init_empty_fetch(self, db_conn, topic, settings):
        """Drive init where the fetch returns no articles (e.g. all already stored)."""
        from app.checker import initialize_new_topic

        db_path = conn_db_path(db_conn)

        with patch(
            "app.checker.fetch_new_articles_for_topic",
            new_callable=AsyncMock,
            # A source did run (feeds_total=1) and had nothing new to give — the
            # only shape that still earns the generic message (AUG-135).
            return_value=FetchResult(articles=[], total_feed_entries=0, feeds_total=1),
        ):
            await initialize_new_topic(topic, settings, db_path=db_path)
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

    async def test_insufficient_init_goes_ready_immediately_articles_marked(
        self, db_conn: sqlite3.Connection, db_path: Path
    ) -> None:
        """OVH-001 real path: pass 1 stores+marks articles and goes READY (no bounce to NEW).
        Articles are marked processed before the READY transition, so they are not discarded."""
        from app.checker import initialize_new_topic

        topic = _make_topic(db_conn, status=TopicStatus.NEW, status_changed_at=None, init_attempts=0)
        settings = _make_settings()
        created_article = create_article(db_conn, _make_article(id=None, topic_id=topic.id, content_hash="hash-1"))
        db_conn.commit()

        async def fake_fetch(t, **kwargs):
            return FetchResult(articles=[created_article], total_feed_entries=1)

        with (
            patch("app.checker.fetch_new_articles_for_topic", side_effect=fake_fetch),
            patch(
                "app.checker.prepare_initial_knowledge",
                new_callable=AsyncMock,
                return_value=_make_write_result(sufficient_data=False),
            ),
        ):
            await initialize_new_topic(topic, settings, db_path=db_path)

        after_pass1 = get_topic(db_conn, topic.id)
        assert after_pass1.status == TopicStatus.READY
        assert after_pass1.init_attempts == 0
        assert after_pass1.error_message is None


class TestInitNoOverwriteConcurrentEdits:
    """OVH-100: init's terminal status write must not clobber concurrent UI edits."""

    async def _drive_terminal_write(self, db_conn, topic, settings, *, sufficient: bool):
        """Run init through to its terminal status write, simulating a concurrent edit
        to feeds/thresholds that lands while the LLM await is in flight."""
        from app.checker import initialize_new_topic

        db_path = conn_db_path(db_conn)

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
            patch("app.checker.prepare_initial_knowledge", side_effect=edit_during_llm),
        ):
            await initialize_new_topic(topic, settings, db_path=db_path)
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
        assert updated.status == TopicStatus.READY
        assert updated.init_attempts == 0
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

        async def mock_check(topic, settings, **kwargs):
            if topic.name == "Crash Topic":
                raise RuntimeError("Unexpected crash in check_topic")
            return await original_check_topic(topic, settings, **kwargs)

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
    """The delivery-intent drain: claim, send, apply."""

    async def test_successful_retry_records_the_delivery(self, db_conn: sqlite3.Connection) -> None:
        """A delivered intent becomes the ledger row, not a deleted one (AUG-153)."""
        topic = _make_topic(db_conn)
        create_pending_notification(
            db_conn,
            PendingNotification(topic_id=topic.id, title="Retry Title", body="Retry Body", url="json://localhost"),
        )
        db_conn.commit()

        settings = _make_settings()

        with patch("app.checker.send_single_notification", side_effect=_ok_send):
            await retry_pending_notifications(db_conn, settings)

        assert list_pending_notifications(db_conn) == []
        row = db_conn.execute("SELECT status, delivered_at FROM pending_notifications").fetchone()
        assert row["status"] == "sent"
        assert row["delivered_at"] is not None

    async def test_failed_retry_increments_count_and_schedules_the_next_attempt(
        self, db_conn: sqlite3.Connection
    ) -> None:
        topic = _make_topic(db_conn)
        create_pending_notification(
            db_conn,
            PendingNotification(topic_id=topic.id, title="T", body="B", url="json://localhost", retry_count=0),
        )
        db_conn.commit()

        settings = _make_settings()

        with patch("app.checker.send_single_notification", side_effect=_fail_send("unreachable")):
            await retry_pending_notifications(db_conn, settings)

        row = db_conn.execute("SELECT * FROM pending_notifications").fetchone()
        assert row["retry_count"] == 1
        assert row["status"] == "pending"
        assert row["last_error"] == "unreachable"
        # Backoff survives a restart because it is a stored due-time, not a timer.
        assert row["next_attempt_at"] is not None
        # ...and the drain honours it: this row is no longer due.
        assert list_due_notification_intents(db_conn, to_db_utc(datetime.now(UTC)), 10) == []

    async def test_expired_notifications_are_abandoned(self, db_conn: sqlite3.Connection) -> None:
        """Out-of-attempts intents become 'abandoned', keeping the record."""
        topic = _make_topic(db_conn)
        create_pending_notification(
            db_conn,
            PendingNotification(
                topic_id=topic.id,
                title="Expired",
                body="B",
                url="json://localhost",
                retry_count=3,
                max_retries=3,
            ),
        )
        db_conn.commit()

        settings = _make_settings()

        with patch("app.checker.send_single_notification", side_effect=_ok_send) as mock_send:
            await retry_pending_notifications(db_conn, settings)

        mock_send.assert_not_called()
        row = db_conn.execute("SELECT status FROM pending_notifications").fetchone()
        assert row["status"] == "abandoned"

    async def test_abandoned_notification_warns_with_ids(self, db_conn: sqlite3.Connection, caplog) -> None:  # noqa: ANN001
        """Abandoning an exhausted notification emits a WARNING naming topic/check ids (OVH-040)."""
        import logging

        topic = _make_topic(db_conn)
        create_pending_notification(
            db_conn,
            PendingNotification(
                topic_id=topic.id,
                check_result_id=777,
                title="Expired",
                body="B",
                url="json://localhost",
                retry_count=3,
                max_retries=3,
            ),
        )
        db_conn.commit()
        settings = _make_settings()

        with (
            caplog.at_level(logging.WARNING, logger="app.checker"),
            patch("app.checker.send_single_notification", side_effect=_ok_send),
        ):
            await retry_pending_notifications(db_conn, settings)

        abandon_logs = [r.getMessage() for r in caplog.records if "Abandoning notification" in r.getMessage()]
        assert len(abandon_logs) == 1
        msg = abandon_logs[0]
        assert f"topic_id={topic.id}" in msg
        assert "check_result_id=777" in msg

    async def test_empty_pending_is_noop(self, db_conn: sqlite3.Connection) -> None:
        """No pending intents means no send attempts."""
        settings = _make_settings()

        with patch("app.checker.send_single_notification", side_effect=_ok_send) as mock_send:
            await retry_pending_notifications(db_conn, settings)

        mock_send.assert_not_called()

    async def test_no_connection_held_across_send(self, db_conn: sqlite3.Connection) -> None:
        """The send must run with the claim connection already committed."""
        topic = _make_topic(db_conn)
        create_pending_notification(
            db_conn,
            PendingNotification(topic_id=topic.id, title="T", body="B", url="json://localhost"),
        )
        db_conn.commit()
        settings = _make_settings()

        in_transaction: list[bool] = []

        async def observe(title, body, url, timeout_s):  # noqa: ANN001
            in_transaction.append(db_conn.in_transaction)
            return NotificationDelivery(url=url, ok=True)

        with patch("app.checker.send_single_notification", side_effect=observe):
            await retry_pending_notifications(db_conn, settings)

        assert in_transaction == [False]

    async def test_crash_midloop_preserves_applied_results(self, db_conn: sqlite3.Connection, db_path: Path) -> None:
        """A crash applying item 2 must not roll back item 1's committed apply."""
        topic = _make_topic(db_conn)
        for i in range(2):
            create_pending_notification(
                db_conn,
                PendingNotification(topic_id=topic.id, title=f"T{i}", body="B", url="json://localhost"),
            )
        db_conn.commit()
        settings = _make_settings()

        pending = list_pending_notifications(db_conn)
        assert len(pending) == 2
        first_id, second_id = pending[0].id, pending[1].id

        from app.crud import apply_notification_outcome as real_apply

        # Every apply for the second row crashes — the recovery apply included, as
        # a genuinely unwritable database would.
        def crashing_apply(conn, intent_id, claim_token, **kwargs):  # noqa: ANN001
            if intent_id == second_id:
                raise RuntimeError("simulated crash applying item 2")
            return real_apply(conn, intent_id, claim_token, **kwargs)

        with (
            patch("app.checker.send_single_notification", side_effect=_ok_send),
            patch("app.checker.apply_notification_outcome", side_effect=crashing_apply),
        ):
            # AUG-263: the drain keeps ownership until every child settles, so the
            # sibling's crash is reported, not propagated out of a live gather.
            await retry_pending_notifications(db_conn, settings)

        statuses = {
            r["id"]: r["status"] for r in db_conn.execute("SELECT id, status FROM pending_notifications").fetchall()
        }
        assert statuses[first_id] == "sent"
        assert sorted(statuses.values()) == ["sending", "sent"]


class TestDeliveryIntentDurability:
    """The intent contract: created before the send, claimed once, fenced on apply."""

    async def test_crash_after_the_transition_still_delivers_exactly_once(
        self, db_conn: sqlite3.Connection, db_path: Path
    ) -> None:
        """TW-AUD-004: intents survive a crash between C3 and the first send."""
        topic = _make_topic(db_conn, name="CrashAfterC3")
        create_knowledge_state(db_conn, KnowledgeState(topic_id=topic.id, summary_text="Old.", token_count=10))
        db_conn.commit()
        settings = _make_settings()
        novelty = NoveltyResult(has_new_info=True, summary="New", confidence=0.9, relevance=0.9)

        async def crash(*_args, **_kwargs):
            raise RuntimeError("process died before any send")

        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=[_make_article(topic_id=topic.id)], total_feed_entries=1),
            ),
            patch("app.checker.analyze_articles", new_callable=AsyncMock, return_value=novelty),
            patch("app.checker.prepare_knowledge_update", new_callable=AsyncMock, return_value=_make_write_result()),
            patch("app.checker.deliver_notification_intents", side_effect=crash),
            pytest.raises(RuntimeError, match="process died"),
        ):
            await check_topic(topic, settings, db_path=db_path)

        # The intent is durable even though no send ever ran.
        pending = list_pending_notifications(db_conn)
        assert len(pending) == 1
        assert pending[0].url == "json://localhost"
        assert pending[0].check_result_id is not None

        sends: list[str] = []

        async def record(title, body, url, timeout_s):  # noqa: ANN001
            sends.append(url)
            return NotificationDelivery(url=url, ok=True)

        with patch("app.checker.send_single_notification", side_effect=record):
            await retry_pending_notifications(db_conn, settings)
            # A second drain must not re-deliver the ledger row.
            await retry_pending_notifications(db_conn, settings)

        assert sends == ["json://localhost"]

    def test_claim_rejects_exhausted_undue_and_already_claimed_rows(self, db_conn: sqlite3.Connection) -> None:
        """TW-AUD-006: every eligibility rule lives inside the atomic claim."""
        topic = _make_topic(db_conn)
        now = datetime.now(UTC)
        now_iso = to_db_utc(now)

        exhausted = create_pending_notification(
            db_conn,
            PendingNotification(topic_id=topic.id, title="X", body="B", url="json://a", retry_count=3, max_retries=3),
        )
        not_due = create_pending_notification(
            db_conn,
            PendingNotification(
                topic_id=topic.id,
                title="Y",
                body="B",
                url="json://b",
                next_attempt_at=to_db_utc(now + timedelta(hours=1)),
            ),
        )
        free = create_pending_notification(
            db_conn, PendingNotification(topic_id=topic.id, title="Z", body="B", url="json://c")
        )
        db_conn.commit()

        assert claim_notification_intent(db_conn, exhausted.id, "tok", now_iso) is False
        assert claim_notification_intent(db_conn, not_due.id, "tok", now_iso) is False
        assert claim_notification_intent(db_conn, free.id, "winner", now_iso) is True
        # A second claimant loses even though its own snapshot said the row was free.
        assert claim_notification_intent(db_conn, free.id, "loser", now_iso) is False

    def test_late_apply_with_a_stale_token_is_a_noop_across_clock_jumps(self, db_conn: sqlite3.Connection) -> None:
        """AUG-277: the fence is identity, not elapsed time.

        Liveness is judged monotonically while due-times are wall clock, so the
        fence has to hold when the clock steps forward (the stale release fires
        early on a live claim) and when it steps back (nothing looks stale at all).
        """
        topic = _make_topic(db_conn)
        for jump in (timedelta(hours=6), timedelta(hours=-6)):
            intent = create_pending_notification(
                db_conn, PendingNotification(topic_id=topic.id, title="T", body="B", url="json://x")
            )
            db_conn.commit()

            assert claim_notification_intent(db_conn, intent.id, "owner-A", to_db_utc(datetime.now(UTC)))
            # The clock jumps; the stale sweep releases A's live claim and B takes it.
            release_stale_notification_claims(db_conn, to_db_utc(datetime.now(UTC) + jump + timedelta(hours=1)))
            claimed_b = claim_notification_intent(db_conn, intent.id, "owner-B", to_db_utc(datetime.now(UTC) + jump))
            db_conn.commit()

            if not claimed_b:
                # A backward jump leaves A's claim intact — also correct, and A's
                # own apply below is the one that must win.
                assert apply_notification_outcome(db_conn, intent.id, "owner-A", sent=True) is True
                continue

            # A finally comes back. Its apply must change nothing.
            assert apply_notification_outcome(db_conn, intent.id, "owner-A", sent=True) is False
            row = db_conn.execute("SELECT status FROM pending_notifications WHERE id = ?", (intent.id,)).fetchone()
            assert row["status"] == "sending"
            assert apply_notification_outcome(db_conn, intent.id, "owner-B", sent=True) is True

    async def test_timed_out_send_stays_claimed_then_stale_release_rearms_it(self, db_conn: sqlite3.Connection) -> None:
        """AUG-071/TW-AUD-004: an unknown outcome is recorded as unknown."""
        topic = _make_topic(db_conn)
        create_pending_notification(
            db_conn, PendingNotification(topic_id=topic.id, title="T", body="B", url="json://slow")
        )
        db_conn.commit()
        settings = _make_settings()

        async def timeout(title, body, url, timeout_s):  # noqa: ANN001
            return NotificationDelivery(url=url, ok=False, error="timed out", timed_out=True)

        with patch("app.checker.send_single_notification", side_effect=timeout):
            await retry_pending_notifications(db_conn, settings)

        row = db_conn.execute("SELECT status, retry_count FROM pending_notifications").fetchone()
        assert row["status"] == "sending"
        assert row["retry_count"] == 0

        released = release_stale_notification_claims(db_conn, to_db_utc(datetime.now(UTC) + timedelta(hours=1)))
        db_conn.commit()
        assert released == 1
        assert len(list_pending_notifications(db_conn)) == 1

    async def test_an_escaping_send_error_still_records_a_retryable_failure(self, db_conn: sqlite3.Connection) -> None:
        """An exception must land an outcome, or the row is stuck 'sending' for good.

        Nothing else can free it: retry_count never moves so it is never
        abandoned, retention prunes only terminal rows, and the UI queue hides
        'sending'. Every later drain re-sends it.
        """
        topic = _make_topic(db_conn)
        create_pending_notification(
            db_conn, PendingNotification(topic_id=topic.id, title="T", body="B", url="json://x", max_retries=3)
        )
        db_conn.commit()
        settings = _make_settings()

        async def boom(title, body, url, timeout_s):  # noqa: ANN001
            raise ValueError("Invalid IPv6 URL")

        with patch("app.checker.send_single_notification", side_effect=boom):
            await retry_pending_notifications(db_conn, settings)

            row = db_conn.execute("SELECT * FROM pending_notifications").fetchone()
            assert row["status"] == "pending"
            assert row["retry_count"] == 1
            assert row["last_error"] == "ValueError"
            assert row["next_attempt_at"] is not None

            # The budget is spent like any other failure, so the row leaves the queue.
            for _ in range(2):
                db_conn.execute("UPDATE pending_notifications SET next_attempt_at = NULL")
                db_conn.commit()
                await retry_pending_notifications(db_conn, settings)

        row = db_conn.execute("SELECT status, retry_count FROM pending_notifications").fetchone()
        assert row["status"] == "abandoned"
        assert row["retry_count"] == 3

    async def test_a_failed_apply_after_a_delivered_send_is_not_re_sent(self, db_conn: sqlite3.Connection) -> None:
        """Exactly-once holds when the apply write fails, not just when the send does.

        A locked database (the daily VACUUM outliving busy_timeout) after a send
        that did deliver must not re-arm the row: the user would get the alert
        twice. The claim fence makes re-applying the delivered outcome a no-op if
        the first apply landed after all.
        """
        topic = _make_topic(db_conn)
        create_pending_notification(
            db_conn, PendingNotification(topic_id=topic.id, title="T", body="B", url="json://x")
        )
        db_conn.commit()
        settings = _make_settings()

        sends: list[str] = []

        async def record(title, body, url, timeout_s):  # noqa: ANN001
            sends.append(url)
            return NotificationDelivery(url=url, ok=True)

        applies = {"n": 0}

        def flaky_apply(*args, **kwargs):  # noqa: ANN002, ANN003
            applies["n"] += 1
            if applies["n"] == 1:
                raise sqlite3.OperationalError("database is locked")
            return apply_notification_outcome(*args, **kwargs)

        with (
            patch("app.checker.send_single_notification", side_effect=record),
            patch("app.checker.apply_notification_outcome", side_effect=flaky_apply),
        ):
            await retry_pending_notifications(db_conn, settings)
            # The stale-claim window elapses; a re-armed row would send again.
            release_stale_notification_claims(db_conn, to_db_utc(datetime.now(UTC) + timedelta(hours=1)))
            db_conn.commit()
            await retry_pending_notifications(db_conn, settings)

        assert sends == ["json://x"]
        row = db_conn.execute("SELECT status FROM pending_notifications").fetchone()
        assert row["status"] == "sent"

    async def test_rollup_is_true_only_when_every_intent_sent(self, db_conn: sqlite3.Connection, db_path: Path) -> None:
        topic = _make_topic(db_conn, name="Rollup")
        create_knowledge_state(db_conn, KnowledgeState(topic_id=topic.id, summary_text="Old.", token_count=10))
        db_conn.commit()
        settings = _make_settings(notifications=NotificationSettings(urls=["json://a", "json://b"]))
        novelty = NoveltyResult(has_new_info=True, summary="New", confidence=0.9, relevance=0.9)

        async def one_fails(title, body, url, timeout_s):  # noqa: ANN001
            return NotificationDelivery(url=url, ok=url == "json://a", error=None if url == "json://a" else "down")

        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=[_make_article(topic_id=topic.id)], total_feed_entries=1),
            ),
            patch("app.checker.analyze_articles", new_callable=AsyncMock, return_value=novelty),
            patch("app.checker.prepare_knowledge_update", new_callable=AsyncMock, return_value=_make_write_result()),
            patch("app.checker.send_single_notification", side_effect=one_fails),
        ):
            result = await check_topic(topic, settings, db_path=db_path)

        assert result.notification_sent is False
        statuses = sorted(r["status"] for r in db_conn.execute("SELECT status FROM pending_notifications"))
        assert statuses == ["pending", "sent"]
        # The delivered channel is never re-sent on the next drain.
        assert [i.url for i in list_pending_notifications(db_conn)] == ["json://b"]


class TestSourcesFailedSurfacing:
    """Mode-agnostic all-sources-failed surfacing on the check and init paths."""

    async def _check(
        self,
        db_conn: sqlite3.Connection,
        topic: Topic,
        *,
        feeds_total: int,
        feeds_failed: int,
        feeds_skipped: int = 0,
    ):
        db_path = conn_db_path(db_conn)
        settings = _make_settings()
        with patch(
            "app.checker.fetch_new_articles_for_topic",
            new_callable=AsyncMock,
            return_value=FetchResult(
                articles=[],
                total_feed_entries=0,
                feeds_total=feeds_total,
                feeds_failed=feeds_failed,
                feeds_skipped=feeds_skipped,
            ),
        ):
            return await check_topic(topic, settings, db_path=db_path)

    async def test_check_all_sources_failed_sets_stage_error(self, db_conn: sqlite3.Connection) -> None:
        topic = _make_topic(db_conn)
        result = await self._check(db_conn, topic, feeds_total=1, feeds_failed=1)
        assert result.stage_error is not None
        assert result.stage_error.startswith("sources_failed")
        # Persisted on the row so the detail page surfaces it.
        row = db_conn.execute("SELECT stage_error FROM check_results WHERE id = ?", (result.id,)).fetchone()
        assert row["stage_error"].startswith("sources_failed")

    async def test_check_healthy_empty_no_stage_error(self, db_conn: sqlite3.Connection) -> None:
        topic = _make_topic(db_conn)
        result = await self._check(db_conn, topic, feeds_total=1, feeds_failed=0)
        assert result.stage_error is None

    async def test_check_nothing_attempted_sets_sources_unavailable(self, db_conn: sqlite3.Connection) -> None:
        """feeds_total=0 (Exa disabled, all feeds backed off, empty feed_urls) is invisible silence."""
        topic = _make_topic(db_conn)
        result = await self._check(db_conn, topic, feeds_total=0, feeds_failed=0)
        assert result.stage_error is not None
        assert result.stage_error.startswith("sources_unavailable")
        row = db_conn.execute("SELECT stage_error FROM check_results WHERE id = ?", (result.id,)).fetchone()
        assert row["stage_error"].startswith("sources_unavailable")

    async def test_sources_unavailable_names_backoff_skips(self, db_conn: sqlite3.Connection) -> None:
        topic = _make_topic(db_conn)
        result = await self._check(db_conn, topic, feeds_total=0, feeds_failed=0, feeds_skipped=2)
        assert "2 feed(s) in backoff" in result.stage_error

    async def test_sources_unavailable_is_mode_agnostic(self, db_conn: sqlite3.Connection) -> None:
        """An AUTO topic with nothing attempted gets the same marker (no FeedMode coupling)."""
        topic = _make_topic(db_conn, name="AutoNothing", feed_mode=FeedMode.AUTO, feed_urls=[])
        result = await self._check(db_conn, topic, feeds_total=0, feeds_failed=0)
        assert result.stage_error.startswith("sources_unavailable")

    async def test_check_surfacing_is_mode_agnostic(self, db_conn: sqlite3.Connection) -> None:
        """An AUTO topic with the all-failed shape ALSO gets sources_failed (no FeedMode coupling)."""
        topic = _make_topic(db_conn, name="AutoFail", feed_mode=FeedMode.AUTO, feed_urls=[])
        result = await self._check(db_conn, topic, feeds_total=1, feeds_failed=1)
        assert result.stage_error is not None
        assert result.stage_error.startswith("sources_failed")

    async def test_init_all_sources_failed_message(self, db_conn: sqlite3.Connection, db_path: Path) -> None:
        from app.checker import initialize_new_topic

        topic = _make_topic(db_conn, name="ExaInitFail", feed_mode=FeedMode.EXA, feed_urls=[], status=TopicStatus.NEW)
        settings = _make_settings()
        with patch(
            "app.checker.fetch_new_articles_for_topic",
            new_callable=AsyncMock,
            return_value=FetchResult(articles=[], total_feed_entries=0, feeds_total=1, feeds_failed=1),
        ):
            await initialize_new_topic(topic, settings, db_path=db_path)
        updated = get_topic(db_conn, topic.id)
        assert updated.status == TopicStatus.ERROR
        assert updated.error_message.startswith("All feed source(s) failed")

    async def test_init_empty_result_keeps_generic_message(self, db_conn: sqlite3.Connection, db_path: Path) -> None:
        """A healthy source that returned nothing keeps the generic message."""
        from app.checker import initialize_new_topic

        topic = _make_topic(db_conn, name="ExaInitEmpty", feed_mode=FeedMode.EXA, feed_urls=[], status=TopicStatus.NEW)
        settings = _make_settings()
        with patch(
            "app.checker.fetch_new_articles_for_topic",
            new_callable=AsyncMock,
            return_value=FetchResult(articles=[], total_feed_entries=0, feeds_total=1, feeds_failed=0),
        ):
            await initialize_new_topic(topic, settings, db_path=db_path)
        updated = get_topic(db_conn, topic.id)
        assert updated.status == TopicStatus.ERROR
        assert updated.error_message == "No articles found during initialization"

    async def test_init_with_no_source_attempted_says_so(self, db_conn: sqlite3.Connection, db_path: Path) -> None:
        """Nothing ran, so the error must not blame an empty source (AUG-135)."""
        from app.checker import initialize_new_topic

        topic = _make_topic(
            db_conn, name="ExaInitNoSource", feed_mode=FeedMode.EXA, feed_urls=[], status=TopicStatus.NEW
        )
        settings = _make_settings()
        with patch(
            "app.checker.fetch_new_articles_for_topic",
            new_callable=AsyncMock,
            return_value=FetchResult(articles=[], total_feed_entries=0, feeds_total=0, feeds_failed=0),
        ):
            await initialize_new_topic(topic, settings, db_path=db_path)
        updated = get_topic(db_conn, topic.id)
        assert updated.status == TopicStatus.ERROR
        assert updated.error_message == "No source attempted during initialization (no source configured or enabled)"

    async def test_init_names_backed_off_feeds(self, db_conn: sqlite3.Connection, db_path: Path) -> None:
        """Every feed sitting in a backoff window is a diagnosis, not an empty result."""
        from app.checker import initialize_new_topic

        topic = _make_topic(db_conn, name="ManualInitBackoff", status=TopicStatus.NEW)
        settings = _make_settings()
        with patch(
            "app.checker.fetch_new_articles_for_topic",
            new_callable=AsyncMock,
            return_value=FetchResult(articles=[], total_feed_entries=0, feeds_total=0, feeds_failed=0, feeds_skipped=2),
        ):
            await initialize_new_topic(topic, settings, db_path=db_path)
        updated = get_topic(db_conn, topic.id)
        assert updated.status == TopicStatus.ERROR
        assert updated.error_message == "No source attempted during initialization (2 feed(s) in backoff)"


def _heartbeat_sender(*, ok: bool = True, error: str | None = None, fails: tuple[str, ...] = ()) -> AsyncMock:
    """AsyncMock for app.checker.send_single_notification, one call per target.

    Patched at the per-target send rather than at ``deliver_notification_intents``
    so the intent rows reach their real terminal status: the recovery notice is
    addressed from the ledger of alerts that actually went out, so a stub that
    leaves every row 'pending' would test nothing.
    """

    async def _send(title: str, body: str, url: str, timeout_s: float) -> NotificationDelivery:
        if url in fails:
            return NotificationDelivery(url=url, ok=False, error=error or "unreachable")
        return NotificationDelivery(url=url, ok=ok, error=None if ok else (error or "unreachable"))

    return AsyncMock(side_effect=_send)


def _titles(send: AsyncMock) -> list[str]:
    return [call.args[0] for call in send.await_args_list]


def _intent_rows(conn: sqlite3.Connection, topic_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT kind, status, url, latch_value, title FROM pending_notifications WHERE topic_id = ? ORDER BY id",
        (topic_id,),
    ).fetchall()


class TestSilenceHeartbeatPipeline:
    """Heartbeat behaviour driven end-to-end through check_topic."""

    async def _failing_check(
        self,
        db_conn: sqlite3.Connection,
        topic: Topic,
        send,
        *,
        threshold: int = 3,
        settings: Settings | None = None,
    ):
        settings = settings or _make_settings(silence_heartbeat_checks=threshold)
        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=[], total_feed_entries=0, feeds_total=1, feeds_failed=1),
            ),
            patch("app.checker.send_single_notification", send),
        ):
            return await check_topic(get_topic(db_conn, topic.id), settings, db_path=conn_db_path(db_conn))

    async def _healthy_empty_check(
        self,
        db_conn: sqlite3.Connection,
        topic: Topic,
        send,
        *,
        threshold: int = 3,
        settings: Settings | None = None,
    ):
        settings = settings or _make_settings(silence_heartbeat_checks=threshold)
        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=[], total_feed_entries=0, feeds_total=1, feeds_failed=0),
            ),
            patch("app.checker.send_single_notification", send),
        ):
            return await check_topic(get_topic(db_conn, topic.id), settings, db_path=conn_db_path(db_conn))

    async def test_alert_fires_once_at_the_threshold(self, db_conn: sqlite3.Connection) -> None:
        topic = _make_topic(db_conn)
        send = _heartbeat_sender()

        for _ in range(2):
            await self._failing_check(db_conn, topic, send)
        assert send.await_count == 0

        await self._failing_check(db_conn, topic, send)
        assert send.await_count == 1
        assert "sources failing" in _titles(send)[0]
        assert get_topic(db_conn, topic.id).heartbeat_alerted_at is not None

        for _ in range(3):
            await self._failing_check(db_conn, topic, send)
        assert send.await_count == 1

    async def test_fetch_exception_path_never_alerts_on_the_sources(self, db_conn: sqlite3.Connection) -> None:
        """A pipeline crash is recorded, but it is not evidence about the feeds (AUG-133)."""
        topic = _make_topic(db_conn)
        send = _heartbeat_sender()
        settings = _make_settings(silence_heartbeat_checks=2)
        for _ in range(3):
            with (
                patch(
                    "app.checker.fetch_new_articles_for_topic",
                    new_callable=AsyncMock,
                    side_effect=RuntimeError("boom"),
                ),
                patch("app.checker.send_single_notification", send),
            ):
                result = await check_topic(get_topic(db_conn, topic.id), settings, db_path=conn_db_path(db_conn))
        assert result.stage_error.startswith("pipeline_failed")
        assert send.await_count == 0
        assert get_topic(db_conn, topic.id).heartbeat_alerted_at is None

    async def test_pipeline_crash_does_not_clear_an_announced_outage(self, db_conn: sqlite3.Connection) -> None:
        """The latch survives a check that never reached the sources."""
        topic = _make_topic(db_conn)
        send = _heartbeat_sender()
        for _ in range(3):
            await self._failing_check(db_conn, topic, send)
        assert send.await_count == 1

        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                side_effect=RuntimeError("boom"),
            ),
            patch("app.checker.send_single_notification", send),
        ):
            await check_topic(
                get_topic(db_conn, topic.id),
                _make_settings(silence_heartbeat_checks=3),
                db_path=conn_db_path(db_conn),
            )
        assert send.await_count == 1
        assert get_topic(db_conn, topic.id).heartbeat_alerted_at is not None

    async def test_recovery_notice_after_the_outage(self, db_conn: sqlite3.Connection) -> None:
        topic = _make_topic(db_conn)
        send = _heartbeat_sender()
        for _ in range(3):
            await self._failing_check(db_conn, topic, send)
        assert send.await_count == 1

        await self._healthy_empty_check(db_conn, topic, send)
        assert send.await_count == 2
        assert "recovered" in _titles(send)[1]
        assert get_topic(db_conn, topic.id).heartbeat_alerted_at is None

        await self._healthy_empty_check(db_conn, topic, send)
        assert send.await_count == 2

    async def test_recovery_on_the_main_analysis_path(self, db_conn: sqlite3.Connection) -> None:
        """A check that reaches analysis also clears the outage."""
        topic = _make_topic(db_conn)
        send = _heartbeat_sender()
        for _ in range(3):
            await self._failing_check(db_conn, topic, send)
        assert send.await_count == 1

        novelty = NoveltyResult(has_new_info=False, confidence=0.9, reasoning="nothing new")
        settings = _make_settings(silence_heartbeat_checks=3)
        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=[_make_article()], total_feed_entries=1),
            ),
            patch("app.checker.analyze_articles", new_callable=AsyncMock, return_value=novelty),
            patch("app.checker.send_single_notification", send),
        ):
            await check_topic(get_topic(db_conn, topic.id), settings, db_path=conn_db_path(db_conn))

        assert send.await_count == 2
        assert "recovered" in _titles(send)[1]

    async def test_disabled_by_zero(self, db_conn: sqlite3.Connection) -> None:
        topic = _make_topic(db_conn)
        send = _heartbeat_sender()
        for _ in range(4):
            await self._failing_check(db_conn, topic, send, threshold=0)
        assert send.await_count == 0
        assert get_topic(db_conn, topic.id).heartbeat_alerted_at is None

    async def test_disabling_clears_an_outstanding_latch_silently(self, db_conn: sqlite3.Connection) -> None:
        """Turning the feature off must reset state, not park a phantom recovery."""
        topic = _make_topic(db_conn)
        send = _heartbeat_sender()
        for _ in range(3):
            await self._failing_check(db_conn, topic, send)
        assert send.await_count == 1
        assert get_topic(db_conn, topic.id).heartbeat_alerted_at is not None

        await self._failing_check(db_conn, topic, send, threshold=0)
        assert send.await_count == 1
        assert get_topic(db_conn, topic.id).heartbeat_alerted_at is None

        # Re-enabled later: a healthy check must NOT announce a stale recovery.
        await self._healthy_empty_check(db_conn, topic, send)
        assert send.await_count == 1

    async def test_failed_heartbeat_delivery_is_queued_and_drains(self, db_conn: sqlite3.Connection) -> None:
        topic = _make_topic(db_conn)
        send = _heartbeat_sender(ok=False, error="unreachable")
        for _ in range(3):
            await self._failing_check(db_conn, topic, send)

        rows = db_conn.execute(
            "SELECT title, kind, status FROM pending_notifications WHERE topic_id = ?", (topic.id,)
        ).fetchall()
        assert len(rows) == 1
        assert "sources failing" in rows[0]["title"]
        # The intent carries its heartbeat kind, so the revocation can find it.
        assert rows[0]["kind"] == "heartbeat_alert"
        assert rows[0]["status"] == "pending"
        # The latch is claimed with the intent, so a dead channel never re-alerts.
        assert get_topic(db_conn, topic.id).heartbeat_alerted_at is not None

        # The failed attempt scheduled a backoff; let that window elapse.
        db_conn.execute("UPDATE pending_notifications SET next_attempt_at = NULL WHERE topic_id = ?", (topic.id,))
        db_conn.commit()
        with patch("app.checker.send_single_notification", side_effect=_ok_send):
            await retry_pending_notifications(db_conn, _make_settings(silence_heartbeat_checks=3))
        remaining = db_conn.execute(
            "SELECT status FROM pending_notifications WHERE topic_id = ?", (topic.id,)
        ).fetchone()
        # Retained as the delivery ledger, not deleted (AUG-153).
        assert remaining["status"] == "sent"

    async def test_heartbeat_failure_does_not_break_the_check(self, db_conn: sqlite3.Connection, db_path: Path) -> None:
        topic = _make_topic(db_conn)
        settings = _make_settings(silence_heartbeat_checks=1)
        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=[], total_feed_entries=0, feeds_total=1, feeds_failed=1),
            ),
            patch("app.checker.evaluate_heartbeat", side_effect=RuntimeError("boom")),
        ):
            result = await check_topic(topic, settings, db_path=db_path)
        assert result.id is not None
        assert result.stage_error.startswith("sources_failed")

    async def test_non_ready_topic_never_heartbeats(self, db_conn: sqlite3.Connection) -> None:
        """A non-READY check must neither alert nor claim recovery."""
        topic = _make_topic(db_conn)
        send = _heartbeat_sender()
        for _ in range(3):
            await self._failing_check(db_conn, topic, send)
        assert send.await_count == 1

        topic = get_topic(db_conn, topic.id)
        topic.status = TopicStatus.ERROR
        update_topic(db_conn, topic)
        db_conn.commit()

        with patch("app.checker.send_single_notification", send):
            await check_topic(topic, _make_settings(silence_heartbeat_checks=3), db_path=conn_db_path(db_conn))
        assert send.await_count == 1
        assert get_topic(db_conn, topic.id).heartbeat_alerted_at is not None


class TestHeartbeatTransitionIsAtomic:
    """AUG-019/130/131/132: the latch, its intents and its revocations are one commit."""

    def _pipeline(self) -> TestSilenceHeartbeatPipeline:
        return TestSilenceHeartbeatPipeline()

    async def _outage(self, db_conn: sqlite3.Connection, topic: Topic, send, **kwargs) -> None:
        for _ in range(3):
            await self._pipeline()._failing_check(db_conn, topic, send, **kwargs)

    async def test_zero_targets_never_consume_the_latch(self, db_conn: sqlite3.Connection) -> None:
        """No configured Apprise target means no announcement, so nothing to latch (AUG-130)."""
        topic = _make_topic(db_conn)
        send = _heartbeat_sender()
        settings = _make_settings(silence_heartbeat_checks=3, notifications=NotificationSettings(urls=[]))
        await self._outage(db_conn, topic, send, settings=settings)

        assert send.await_count == 0
        assert get_topic(db_conn, topic.id).heartbeat_alerted_at is None
        assert _intent_rows(db_conn, topic.id) == []

        # A target configured during the same outage still gets the alert.
        await self._pipeline()._failing_check(db_conn, topic, send)
        assert send.await_count == 1
        assert get_topic(db_conn, topic.id).heartbeat_alerted_at is not None

    async def test_intent_failure_takes_the_latch_with_it(self, db_conn: sqlite3.Connection) -> None:
        """A crash between latch and intents must leave neither (AUG-019)."""
        topic = _make_topic(db_conn)
        send = _heartbeat_sender()
        with patch("app.checker.create_notification_intents", side_effect=sqlite3.OperationalError("disk I/O error")):
            await self._outage(db_conn, topic, send)

        assert send.await_count == 0
        assert get_topic(db_conn, topic.id).heartbeat_alerted_at is None
        assert _intent_rows(db_conn, topic.id) == []

        # The next check re-runs the whole transition cleanly.
        await self._pipeline()._failing_check(db_conn, topic, send)
        assert send.await_count == 1
        assert get_topic(db_conn, topic.id).heartbeat_alerted_at is not None

    async def test_crash_before_send_leaves_a_deliverable_intent(self, db_conn: sqlite3.Connection) -> None:
        """The send is outside the commit, so a death mid-send costs no message (AUG-019)."""
        topic = _make_topic(db_conn)
        send = _heartbeat_sender()
        with patch("app.checker.deliver_notification_intents", side_effect=RuntimeError("process died")):
            await self._outage(db_conn, topic, send)

        rows = _intent_rows(db_conn, topic.id)
        assert [(r["kind"], r["status"]) for r in rows] == [("heartbeat_alert", "pending")]
        assert (
            rows[0]["latch_value"]
            == db_conn.execute("SELECT heartbeat_alerted_at FROM topics WHERE id = ?", (topic.id,)).fetchone()[0]
        )

        with patch("app.checker.send_single_notification", send):
            await retry_pending_notifications(db_conn, _make_settings(silence_heartbeat_checks=3))
        assert send.await_count == 1
        assert _intent_rows(db_conn, topic.id)[0]["status"] == "sent"

    async def test_a_newer_check_invalidates_a_stale_decision(self, db_conn: sqlite3.Connection) -> None:
        """Two interleaved checks: the decision from check N cannot land after N+1 (AUG-131)."""
        from app.crud import create_check_result
        from app.heartbeat import evaluate_heartbeat as real_evaluate
        from app.models import CheckResult

        topic = _make_topic(db_conn)
        send = _heartbeat_sender()
        db_path = conn_db_path(db_conn)

        def _evaluate_then_race(conn, topic_arg, threshold):
            decision = real_evaluate(conn, topic_arg, threshold)
            if decision is not None:
                # A concurrent CLI check commits a newer result before we write.
                create_check_result(
                    db_conn,
                    CheckResult(topic_id=topic.id, checked_at=datetime.now(UTC), stage_error=None),
                )
                db_conn.commit()
            return decision

        with patch("app.checker.evaluate_heartbeat", side_effect=_evaluate_then_race):
            for _ in range(3):
                with (
                    patch(
                        "app.checker.fetch_new_articles_for_topic",
                        new_callable=AsyncMock,
                        return_value=FetchResult(articles=[], total_feed_entries=0, feeds_total=1, feeds_failed=1),
                    ),
                    patch("app.checker.send_single_notification", send),
                ):
                    await check_topic(
                        get_topic(db_conn, topic.id),
                        _make_settings(silence_heartbeat_checks=3),
                        db_path=db_path,
                    )

        assert send.await_count == 0
        assert get_topic(db_conn, topic.id).heartbeat_alerted_at is None
        assert _intent_rows(db_conn, topic.id) == []

    async def test_recovery_only_addresses_targets_that_got_the_alert(self, db_conn: sqlite3.Connection) -> None:
        """A target that never received the outage notice is not told it ended."""
        topic = _make_topic(db_conn)
        settings = _make_settings(
            silence_heartbeat_checks=3,
            notifications=NotificationSettings(urls=["json://good.example.com", "json://bad.example.com"]),
        )
        send = _heartbeat_sender(fails=("json://bad.example.com",))
        await self._outage(db_conn, topic, send, settings=settings)
        assert send.await_count == 2

        await self._pipeline()._healthy_empty_check(db_conn, topic, send, settings=settings)

        recovery = [r for r in _intent_rows(db_conn, topic.id) if r["kind"] == "heartbeat_recovery"]
        assert [r["url"] for r in recovery] == ["json://good.example.com"]
        assert get_topic(db_conn, topic.id).heartbeat_alerted_at is None

    async def test_recovery_revokes_the_superseded_alert(self, db_conn: sqlite3.Connection) -> None:
        """A queued alert must never arrive after the recovery that contradicts it (AUG-132)."""
        topic = _make_topic(db_conn)
        failing = _heartbeat_sender(ok=False, error="unreachable")
        await self._outage(db_conn, topic, failing)
        assert [r["status"] for r in _intent_rows(db_conn, topic.id)] == ["pending"]

        send = _heartbeat_sender()
        await self._pipeline()._healthy_empty_check(db_conn, topic, send)

        rows = _intent_rows(db_conn, topic.id)
        assert [(r["kind"], r["status"]) for r in rows] == [("heartbeat_alert", "revoked")]
        # Nobody received the alert, so there is nobody to tell about the recovery —
        # but the latch is released either way, so the next outage still announces.
        assert send.await_count == 0
        assert get_topic(db_conn, topic.id).heartbeat_alerted_at is None

        with patch("app.checker.send_single_notification", send):
            await retry_pending_notifications(db_conn, _make_settings(silence_heartbeat_checks=3))
        assert send.await_count == 0

    async def test_disabling_revokes_queued_heartbeat_messages(self, db_conn: sqlite3.Connection) -> None:
        """A queued alert cannot arrive after the feature was switched off (AUG-132)."""
        topic = _make_topic(db_conn)
        failing = _heartbeat_sender(ok=False, error="unreachable")
        await self._outage(db_conn, topic, failing)
        assert [r["status"] for r in _intent_rows(db_conn, topic.id)] == ["pending"]

        send = _heartbeat_sender()
        await self._pipeline()._failing_check(db_conn, topic, send, threshold=0)

        assert [r["status"] for r in _intent_rows(db_conn, topic.id)] == ["revoked"]
        assert get_topic(db_conn, topic.id).heartbeat_alerted_at is None
        with patch("app.checker.send_single_notification", send):
            await retry_pending_notifications(db_conn, _make_settings(silence_heartbeat_checks=3))
        assert send.await_count == 0

    async def test_corrupt_latch_clears_instead_of_wedging(self, db_conn: sqlite3.Connection) -> None:
        """Unparseable latch text must not suppress both transitions forever (AUG-144)."""
        topic = _make_topic(db_conn)
        send = _heartbeat_sender()
        await self._outage(db_conn, topic, send)
        db_conn.execute("UPDATE topics SET heartbeat_alerted_at = 'corrupt' WHERE id = ?", (topic.id,))
        db_conn.commit()

        # Still latched: no second alert while the outage continues.
        await self._pipeline()._failing_check(db_conn, topic, send)
        assert send.await_count == 1

        await self._pipeline()._healthy_empty_check(db_conn, topic, send)
        assert (
            db_conn.execute("SELECT heartbeat_alerted_at FROM topics WHERE id = ?", (topic.id,)).fetchone()[0] is None
        )

        # And the topic can announce its next outage normally.
        await self._outage(db_conn, topic, send)
        assert "sources failing" in _titles(send)[-1]


class TestAnalysisFailureIsResumable:
    """A failed analysis leaves its articles retryable, and bounded (TW-AUD-001)."""

    async def _failing_analysis_check(
        self,
        db_conn: sqlite3.Connection,
        db_path: Path,
        topic: Topic,
        articles: list[Article],
    ):
        failed = NoveltyResult(has_new_info=False, confidence=0.0, error="LLM analysis failed")
        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=articles, total_feed_entries=len(articles)),
            ),
            patch("app.checker.analyze_articles", new_callable=AsyncMock, return_value=failed),
            patch("app.checker.deliver_notification_intents", _per_url_mock(ok=True)),
        ):
            return await check_topic(get_topic(db_conn, topic.id), _make_settings(), db_path=db_path)

    async def test_failed_analysis_leaves_the_article_unprocessed(
        self, db_conn: sqlite3.Connection, db_path: Path
    ) -> None:
        """The articles were never evaluated, so they must stay queued for the next cycle."""
        topic = _make_topic(db_conn, name="RetryMe")
        article = create_article(db_conn, _make_article(id=None, topic_id=topic.id))
        db_conn.commit()

        await self._failing_analysis_check(db_conn, db_path, topic, [article])

        row = db_conn.execute(
            "SELECT processed, analysis_attempts FROM articles WHERE id = ?", (article.id,)
        ).fetchone()
        assert row["processed"] == 0
        assert row["analysis_attempts"] == 1

    async def test_repeated_failures_abandon_the_article_at_the_cap(
        self, db_conn: sqlite3.Connection, db_path: Path
    ) -> None:
        """An article that fails MAX_ANALYSIS_ATTEMPTS times is abandoned, not retried forever."""
        topic = _make_topic(db_conn, name="GiveUp")
        article = create_article(db_conn, _make_article(id=None, topic_id=topic.id))
        db_conn.commit()

        for _ in range(MAX_ANALYSIS_ATTEMPTS):
            await self._failing_analysis_check(db_conn, db_path, topic, [article])

        row = db_conn.execute(
            "SELECT processed, analysis_attempts FROM articles WHERE id = ?", (article.id,)
        ).fetchone()
        assert row["analysis_attempts"] == MAX_ANALYSIS_ATTEMPTS
        assert row["processed"] == 1

    async def test_unprocessed_articles_are_re_analyzed_on_a_later_cycle(
        self, db_conn: sqlite3.Connection, db_path: Path
    ) -> None:
        """A stranded row is fed back into analysis even when the feed has nothing new.

        The scraper dedups against stored hashes, so a failed cycle's articles never
        arrive again from the fetch — without an explicit re-select they were dead
        work nobody would ever look at (TW-AUD-001).
        """
        topic = _make_topic(db_conn, name="Stranded")
        article = create_article(db_conn, _make_article(id=None, topic_id=topic.id))
        db_conn.commit()

        await self._failing_analysis_check(db_conn, db_path, topic, [article])

        novelty = NoveltyResult(has_new_info=False, confidence=0.9)
        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=[], total_feed_entries=0, feeds_total=1, feeds_failed=0),
            ),
            patch("app.checker.analyze_articles", new_callable=AsyncMock, return_value=novelty) as mock_analyze,
            patch("app.checker.deliver_notification_intents", _per_url_mock(ok=True)),
        ):
            await check_topic(get_topic(db_conn, topic.id), _make_settings(), db_path=db_path)

        mock_analyze.assert_awaited_once()
        analyzed = mock_analyze.await_args.args[0]
        assert [a.id for a in analyzed] == [article.id]
        row = db_conn.execute("SELECT processed FROM articles WHERE id = ?", (article.id,)).fetchone()
        assert row["processed"] == 1


class TestInsufficientKnowledgeIsRecorded:
    """A soft knowledge rejection is not a clean notified success (TW-AUD-003)."""

    async def _check_with_insufficient_merge(
        self,
        db_conn: sqlite3.Connection,
        db_path: Path,
        topic: Topic,
        article: Article,
        settings: Settings,
        send,
    ):
        novelty = NoveltyResult(
            has_new_info=True,
            summary="Something new",
            confidence=0.9,
            relevance=0.9,
            importance=3,
        )
        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=[article], total_feed_entries=1),
            ),
            patch("app.checker.analyze_articles", new_callable=AsyncMock, return_value=novelty),
            patch(
                "app.checker.prepare_knowledge_update",
                new_callable=AsyncMock,
                return_value=_make_write_result(sufficient_data=False),
            ),
            patch("app.checker.deliver_notification_intents", send),
            patch("app.checker.deliver_webhook_intents", new_callable=AsyncMock, return_value=0),
        ):
            return await check_topic(get_topic(db_conn, topic.id), settings, db_path=db_path)

    async def test_notifies_but_records_the_unchanged_knowledge(
        self, db_conn: sqlite3.Connection, db_path: Path
    ) -> None:
        """The alert still fires; the row says the baseline never absorbed it."""
        topic = _make_topic(db_conn, name="ThinMerge")
        create_knowledge_state(
            db_conn,
            KnowledgeState(topic_id=topic.id, summary_text="Baseline.", token_count=9, version=1),
        )
        article = create_article(db_conn, _make_article(id=None, topic_id=topic.id))
        db_conn.commit()
        send = _per_url_mock(ok=True)

        result = await self._check_with_insufficient_merge(db_conn, db_path, topic, article, _make_settings(), send)

        send.assert_awaited_once()
        assert result.notification_sent is True
        assert result.id is not None
        assert result.stage_error is not None
        assert result.stage_error.startswith("knowledge_insufficient")
        assert result.notify_disposition == NotifyDisposition.PENDING_KNOWLEDGE_STALE
        stored = db_conn.execute(
            "SELECT stage_error, notify_disposition FROM check_results WHERE id = ?", (result.id,)
        ).fetchone()
        assert stored["stage_error"].startswith("knowledge_insufficient")
        assert stored["notify_disposition"] == "pending_knowledge_stale"

        # The prior knowledge is preserved, with no revision claiming otherwise.
        state = db_conn.execute(
            "SELECT summary_text, version FROM knowledge_states WHERE topic_id = ?", (topic.id,)
        ).fetchone()
        assert state["summary_text"] == "Baseline."
        assert state["version"] == 1
        assert db_conn.execute("SELECT COUNT(*) FROM knowledge_revisions").fetchone()[0] == 0

        # The evidence stays queued for another attempt rather than being consumed.
        row = db_conn.execute(
            "SELECT processed, analysis_attempts FROM articles WHERE id = ?", (article.id,)
        ).fetchone()
        assert row["processed"] == 0
        assert row["analysis_attempts"] == 1

    async def test_importance_suppressed_check_still_records_the_rejection(
        self, db_conn: sqlite3.Connection, db_path: Path
    ) -> None:
        """No send to describe, so the disposition stays the suppression reason."""
        topic = _make_topic(db_conn, name="ThinAndQuiet", importance_threshold=5)
        create_knowledge_state(
            db_conn,
            KnowledgeState(topic_id=topic.id, summary_text="Baseline.", token_count=9, version=1),
        )
        article = create_article(db_conn, _make_article(id=None, topic_id=topic.id))
        db_conn.commit()
        send = _per_url_mock(ok=True)

        result = await self._check_with_insufficient_merge(db_conn, db_path, topic, article, _make_settings(), send)

        send.assert_not_awaited()
        assert result.notify_disposition == NotifyDisposition.SUPPRESSED_IMPORTANCE
        assert result.stage_error is not None
        assert result.stage_error.startswith("knowledge_insufficient")

    async def test_suppressed_check_creates_no_delivery_intent(
        self, db_conn: sqlite3.Connection, db_path: Path
    ) -> None:
        """AUG-129: the suppression decision is durable and no channel is owed a send.

        The decision used to live only in memory, so nothing downstream could tell a
        suppressed result from a delivered one. It is now recorded on the row AND
        expressed as the absence of any delivery intent — both channels read the
        same decision because both are driven by the same intents.
        """
        topic = _make_topic(db_conn, name="SuppressedNoIntent", importance_threshold=5)
        create_knowledge_state(
            db_conn,
            KnowledgeState(topic_id=topic.id, summary_text="Baseline.", token_count=9, version=1),
        )
        article = create_article(db_conn, _make_article(id=None, topic_id=topic.id))
        db_conn.commit()
        settings = _make_settings(notifications=NotificationSettings(urls=["json://a"], webhook_urls=["https://h/x"]))
        novelty = NoveltyResult(has_new_info=True, summary="Trivia", confidence=0.9, relevance=0.9, importance=1)

        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=[article], total_feed_entries=1),
            ),
            patch("app.checker.analyze_articles", new_callable=AsyncMock, return_value=novelty),
            patch("app.checker.prepare_knowledge_update", new_callable=AsyncMock, return_value=_make_write_result()),
        ):
            result = await check_topic(topic, settings, db_path=db_path)

        assert result.notify_disposition == NotifyDisposition.SUPPRESSED_IMPORTANCE
        assert result.notification_sent is False
        assert db_conn.execute("SELECT COUNT(*) FROM pending_notifications").fetchone()[0] == 0
        assert db_conn.execute("SELECT COUNT(*) FROM pending_webhooks").fetchone()[0] == 0
        # The stored decision survives a re-read, which is what history must render.
        stored = db_conn.execute("SELECT notify_disposition FROM check_results WHERE id = ?", (result.id,)).fetchone()
        assert stored["notify_disposition"] == "suppressed_importance"


class TestOnlyAnalyzedArticlesAreProcessed:
    """Articles trimmed out of an over-budget prompt were never evaluated."""

    async def test_dropped_articles_stay_queued_and_are_re_analyzed(
        self, db_conn: sqlite3.Connection, db_path: Path
    ) -> None:
        topic = _make_topic(db_conn, name="BigBatch")
        kept = create_article(db_conn, _make_article(id=None, topic_id=topic.id, content_hash="k1"))
        dropped = create_article(
            db_conn,
            _make_article(id=None, topic_id=topic.id, content_hash="d1", url="https://example.com/article-2"),
        )
        db_conn.commit()

        partial = NoveltyResult(has_new_info=False, confidence=0.9, analyzed_article_ids=[kept.id])
        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=[kept, dropped], total_feed_entries=2),
            ),
            patch("app.checker.analyze_articles", new_callable=AsyncMock, return_value=partial),
            patch("app.checker.deliver_notification_intents", _per_url_mock(ok=True)),
        ):
            await check_topic(get_topic(db_conn, topic.id), _make_settings(), db_path=db_path)

        rows = {
            row["id"]: row
            for row in db_conn.execute("SELECT id, processed, analysis_attempts FROM articles").fetchall()
        }
        assert rows[kept.id]["processed"] == 1
        assert rows[dropped.id]["processed"] == 0
        assert rows[dropped.id]["analysis_attempts"] == 1

        # And the next cycle actually feeds it back to the model.
        novelty = NoveltyResult(has_new_info=False, confidence=0.9)
        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=[], total_feed_entries=0, feeds_total=1),
            ),
            patch("app.checker.analyze_articles", new_callable=AsyncMock, return_value=novelty) as mock_analyze,
            patch("app.checker.deliver_notification_intents", _per_url_mock(ok=True)),
        ):
            await check_topic(get_topic(db_conn, topic.id), _make_settings(), db_path=db_path)

        assert [a.id for a in mock_analyze.await_args.args[0]] == [dropped.id]

    async def test_silent_analysis_still_processes_the_whole_batch(
        self, db_conn: sqlite3.Connection, db_path: Path
    ) -> None:
        """Without a reported subset the batch is assumed read, as before."""
        topic = _make_topic(db_conn, name="NoSignal")
        article = create_article(db_conn, _make_article(id=None, topic_id=topic.id))
        db_conn.commit()

        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=[article], total_feed_entries=1),
            ),
            patch(
                "app.checker.analyze_articles",
                new_callable=AsyncMock,
                return_value=NoveltyResult(has_new_info=False, confidence=0.9),
            ),
            patch("app.checker.deliver_notification_intents", _per_url_mock(ok=True)),
        ):
            await check_topic(get_topic(db_conn, topic.id), _make_settings(), db_path=db_path)

        row = db_conn.execute("SELECT processed FROM articles WHERE id = ?", (article.id,)).fetchone()
        assert row["processed"] == 1

    async def test_initialization_only_processes_the_articles_it_used(
        self, db_conn: sqlite3.Connection, db_path: Path
    ) -> None:
        """A baseline built from a truncated corpus must not consume the rest."""
        from app.checker import initialize_new_topic

        topic = _make_topic(db_conn, name="TruncatedInit", status=TopicStatus.NEW)
        used = create_article(db_conn, _make_article(id=None, topic_id=topic.id, content_hash="u1"))
        unused = create_article(
            db_conn,
            _make_article(id=None, topic_id=topic.id, content_hash="x1", url="https://example.com/article-2"),
        )
        db_conn.commit()

        plan = KnowledgeUpdatePlan(
            summary_text="Baseline.",
            token_count=3,
            usage=TokenUsage(),
            analyzed_article_ids=[used.id],
        )
        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=[used, unused], total_feed_entries=2),
            ),
            patch("app.checker.prepare_initial_knowledge", new_callable=AsyncMock, return_value=plan),
        ):
            await initialize_new_topic(topic, _make_settings(), db_path=db_path)

        assert get_topic(db_conn, topic.id).status == TopicStatus.READY
        rows = {row["id"]: row["processed"] for row in db_conn.execute("SELECT id, processed FROM articles")}
        assert rows[used.id] == 1
        assert rows[unused.id] == 0

    async def test_a_real_context_overrun_leaves_the_dropped_article_for_next_cycle(
        self, db_conn: sqlite3.Connection, db_path: Path
    ) -> None:
        """End to end through the real analysis layer, on a genuinely small model.

        gpt-4's 8k window minus the requested output and the schema reserve leaves
        an input budget this batch cannot fit, so the fit ladder drops real
        articles — no hand-built subset anywhere in this test.
        """
        settings = _make_settings(
            llm=LLMSettings(model="openai/gpt-4", api_key="test-key"),
            max_articles_per_check=16,
        )
        topic = _make_topic(db_conn, name="Overrun")
        create_knowledge_state(db_conn, KnowledgeState(topic_id=topic.id, summary_text="Known.", token_count=2))
        body = " ".join(f"word{i}" for i in range(4_000))
        batch = [
            create_article(
                db_conn,
                _make_article(
                    id=None,
                    topic_id=topic.id,
                    content_hash=f"o{i}",
                    url=f"https://example.com/overrun-{i}",
                    raw_content=body,
                ),
            )
            for i in range(12)
        ]
        db_conn.commit()

        client = MagicMock()
        client.chat.completions.create_with_completion = AsyncMock(
            return_value=(
                NoveltyResponse(has_new_info=False, confidence=0.9, relevance=0.1, importance=1),
                MagicMock(usage=None),
            )
        )
        fetched = FetchResult(articles=batch, total_feed_entries=len(batch))
        with (
            patch("app.checker.fetch_new_articles_for_topic", new_callable=AsyncMock, return_value=fetched),
            patch("app.analysis.llm._get_client", return_value=client),
            patch("app.checker.deliver_notification_intents", _per_url_mock(ok=True)),
        ):
            await check_topic(get_topic(db_conn, topic.id), settings, db_path=db_path)

        sent = client.chat.completions.create_with_completion.await_args.kwargs["messages"][1]["content"]
        analyzed = [a for a in batch if a.url in sent]
        dropped = [a for a in batch if a.url not in sent]
        assert analyzed and dropped  # the ladder genuinely dropped articles

        rows = {row["id"]: row["processed"] for row in db_conn.execute("SELECT id, processed FROM articles")}
        assert all(rows[a.id] == 1 for a in analyzed)
        assert all(rows[a.id] == 0 for a in dropped)

        # Next cycle: the articles the model never saw are the ones it is handed.
        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=[], total_feed_entries=0, feeds_total=1),
            ),
            patch(
                "app.checker.analyze_articles",
                new_callable=AsyncMock,
                return_value=NoveltyResult(has_new_info=False, confidence=0.9),
            ) as mock_analyze,
            patch("app.checker.deliver_notification_intents", _per_url_mock(ok=True)),
        ):
            await check_topic(get_topic(db_conn, topic.id), settings, db_path=db_path)

        assert sorted(a.id for a in mock_analyze.await_args.args[0]) == sorted(a.id for a in dropped)
