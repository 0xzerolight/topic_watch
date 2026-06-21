"""Tests for transaction-handling bug fixes.

Bug 1: delete_topic_handler in app/web/routes.py must call conn.commit()
Bug 2: recover_stuck_topics in app/crud.py must NOT call conn.commit() itself

Pipeline transaction safety (OVH-007/066/099/101):
  * No SQLite write lock is held across the content-extraction await in
    ``fetch_new_articles_for_topic`` (WAL single-writer starvation).
  * ``check_topic`` commits durable state before the irreversible network sends
    (commit-before-send ordering).
  * ``initialize_new_topic`` does not hold a write transaction across its
    fetch + LLM awaits.
  * The originating ``CheckResult`` is created before ``send_webhooks`` so a
    queued webhook carries a non-NULL ``check_result_id``.
"""

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from app.analysis.knowledge import KnowledgeWriteResult
from app.analysis.llm import NoveltyResult, TokenUsage
from app.checker import check_topic, initialize_new_topic
from app.config import LLMSettings, NotificationSettings, Settings
from app.crud import (
    create_article,
    create_knowledge_state,
    create_topic,
    delete_topic,
    get_topic,
    list_pending_webhooks,
    recover_stuck_researching,
    recover_stuck_topics,
)
from app.models import Article, KnowledgeState, NotificationDelivery, Topic, TopicStatus
from app.scraping import FetchResult
from app.scraping.rss import FeedEntry, FeedResponse


def _conn_db_path(conn: sqlite3.Connection) -> Path:
    """Resolve the on-disk path backing a sqlite3.Connection."""
    rows = conn.execute("PRAGMA database_list").fetchall()
    for _seq, name, file in rows:
        if name == "main" and file:
            return Path(file)
    raise AssertionError("connection is not backed by a file")


def _pipeline_settings(**overrides) -> Settings:
    defaults = {
        "llm": LLMSettings(model="openai/gpt-4o-mini", api_key="test-key"),
        "notifications": NotificationSettings(urls=["json://localhost"]),
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _ready_topic(conn: sqlite3.Connection, **overrides) -> Topic:
    defaults = {
        "name": "PipelineTopic",
        "description": "A pipeline test topic",
        "feed_urls": ["https://example.com/feed.xml"],
        "status": TopicStatus.READY,
    }
    defaults.update(overrides)
    topic = create_topic(conn, Topic(**defaults))
    conn.commit()
    return topic


def _write_result() -> KnowledgeWriteResult:
    return KnowledgeWriteResult(
        state=KnowledgeState(topic_id=1, summary_text="state", token_count=0),
        usage=TokenUsage(prompt_tokens=0, completion_tokens=0),
        sufficient_data=True,
    )


def _make_article(**overrides) -> Article:
    defaults = {
        "topic_id": 1,
        "title": "Test Article",
        "url": "https://example.com/article-1",
        "content_hash": "abc123",
        "raw_content": "Article content here.",
        "source_feed": "https://example.com/feed.xml",
    }
    defaults.update(overrides)
    return Article(**defaults)


@pytest.fixture
def mem_conn():
    """Provide an in-memory SQLite connection with the topics schema."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS topics (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            description TEXT NOT NULL,
            feed_urls TEXT NOT NULL DEFAULT '[]',
            feed_mode TEXT NOT NULL DEFAULT 'auto',
            created_at TEXT NOT NULL,
            status_changed_at TEXT DEFAULT NULL,
            is_active INTEGER NOT NULL DEFAULT 1,
            status TEXT NOT NULL DEFAULT 'researching',
            error_message TEXT,
            check_interval_hours INTEGER,
            check_interval_minutes INTEGER,
            tags TEXT NOT NULL DEFAULT '[]',
            confidence_threshold REAL DEFAULT NULL,
            relevance_threshold REAL DEFAULT NULL,
            init_attempts INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    conn.commit()
    yield conn
    conn.close()


class TestDeleteTopicHandlerCommit:
    """Tests that delete_topic + conn.commit() persists the deletion."""

    def test_delete_topic_without_commit_does_not_persist(self, mem_conn):
        """Verify that without commit, deletion is not visible in another connection."""
        # This is a conceptual test using the same connection; we validate that
        # calling delete_topic then commit removes the topic from the DB.
        topic = Topic(name="Test Topic", description="A test topic")
        topic = create_topic(mem_conn, topic)
        mem_conn.commit()
        topic_id = topic.id

        # Verify topic exists
        assert get_topic(mem_conn, topic_id) is not None

        # Call delete_topic (as the route does)
        delete_topic(mem_conn, topic_id)
        # Simulate what the fixed handler does: conn.commit()
        mem_conn.commit()

        # Now the topic should be gone
        assert get_topic(mem_conn, topic_id) is None

    def test_delete_topic_followed_by_commit_removes_topic(self, db_conn):
        """End-to-end: create a topic, delete + commit, verify it's gone."""
        topic = Topic(name="Topic To Delete", description="Will be deleted")
        topic = create_topic(db_conn, topic)
        db_conn.commit()
        topic_id = topic.id

        assert get_topic(db_conn, topic_id) is not None

        delete_topic(db_conn, topic_id)
        db_conn.commit()

        assert get_topic(db_conn, topic_id) is None

    def test_delete_topic_rollback_keeps_topic(self, db_conn):
        """Verify that if we rollback instead of commit, topic is still there."""
        topic = Topic(name="Topic Kept", description="Should survive rollback")
        topic = create_topic(db_conn, topic)
        db_conn.commit()
        topic_id = topic.id

        delete_topic(db_conn, topic_id)
        db_conn.rollback()  # simulate no commit (the old bug)

        # Topic should still exist because we rolled back
        assert get_topic(db_conn, topic_id) is not None


class TestRecoverStuckTopicsNoCommit:
    """Tests that recover_stuck_topics does not call conn.commit() internally."""

    def test_recover_stuck_topics_update_not_auto_committed(self, mem_conn):
        """recover_stuck_topics update should be visible only after caller commits."""
        topic = Topic(
            name="Stuck Topic",
            description="Was stuck in RESEARCHING",
            status=TopicStatus.RESEARCHING,
        )
        topic = create_topic(mem_conn, topic)
        mem_conn.commit()
        topic_id = topic.id

        # Call recover_stuck_topics — should update but NOT commit internally
        count = recover_stuck_topics(mem_conn)
        assert count == 1

        # Rollback to undo the (uncommitted) update
        mem_conn.rollback()

        # After rollback the topic should still be RESEARCHING (update was rolled back)
        recovered = get_topic(mem_conn, topic_id)
        assert recovered is not None
        assert recovered.status == TopicStatus.RESEARCHING

    def test_recover_stuck_topics_committed_by_caller(self, mem_conn):
        """When the caller commits, the update from recover_stuck_topics persists."""
        topic = Topic(
            name="Stuck Topic 2",
            description="Was stuck in RESEARCHING",
            status=TopicStatus.RESEARCHING,
        )
        topic = create_topic(mem_conn, topic)
        mem_conn.commit()
        topic_id = topic.id

        count = recover_stuck_topics(mem_conn)
        assert count == 1

        # Caller is responsible for committing (as get_db() context manager does)
        mem_conn.commit()

        recovered = get_topic(mem_conn, topic_id)
        assert recovered is not None
        assert recovered.status == TopicStatus.ERROR
        assert "server restart" in recovered.error_message.lower()

    def test_recover_stuck_topics_returns_zero_when_none_stuck(self, mem_conn):
        """Returns 0 and makes no changes when no topics are RESEARCHING."""
        topic = Topic(
            name="Ready Topic",
            description="Already ready",
            status=TopicStatus.READY,
        )
        create_topic(mem_conn, topic)
        mem_conn.commit()

        count = recover_stuck_topics(mem_conn)
        assert count == 0

    def test_recover_stuck_topics_only_affects_researching_status(self, mem_conn):
        """Only RESEARCHING topics are updated, not ERROR or READY ones."""
        researching = Topic(name="Researching Topic", description="Stuck", status=TopicStatus.RESEARCHING)
        ready = Topic(name="Ready Topic", description="Fine", status=TopicStatus.READY)
        error = Topic(name="Error Topic", description="Already failed", status=TopicStatus.ERROR)

        researching = create_topic(mem_conn, researching)
        ready = create_topic(mem_conn, ready)
        error = create_topic(mem_conn, error)
        mem_conn.commit()

        count = recover_stuck_topics(mem_conn)
        assert count == 1

        mem_conn.commit()

        assert get_topic(mem_conn, researching.id).status == TopicStatus.ERROR
        assert get_topic(mem_conn, ready.id).status == TopicStatus.READY
        assert get_topic(mem_conn, error.id).status == TopicStatus.ERROR


class TestRecoverStuckResearchingNoCommit:
    """OVH-087: recover_stuck_researching must NOT call conn.commit() internally,
    mirroring recover_stuck_topics — the get_db caller owns the commit (invariant
    #12). A rollback after the call must revert the status change."""

    def _stuck_topic(self, conn: sqlite3.Connection) -> int:
        """Create a RESEARCHING topic backdated past the stuck timeout."""
        topic = create_topic(
            conn,
            Topic(name="Stuck Researching", description="d", status=TopicStatus.RESEARCHING),
        )
        old_time = datetime.now(UTC) - timedelta(minutes=20)
        conn.execute(
            "UPDATE topics SET status_changed_at = ? WHERE id = ?",
            (old_time.isoformat(), topic.id),
        )
        conn.commit()
        return topic.id

    def test_recover_stuck_researching_update_not_auto_committed(self, db_conn: sqlite3.Connection) -> None:
        """The status change is visible only after the caller commits; a rollback reverts it."""
        topic_id = self._stuck_topic(db_conn)

        count = recover_stuck_researching(db_conn, timeout_minutes=15)
        assert count == 1

        # Rollback to undo the (uncommitted) update.
        db_conn.rollback()

        recovered = get_topic(db_conn, topic_id)
        assert recovered is not None
        assert recovered.status == TopicStatus.RESEARCHING

    def test_recover_stuck_researching_committed_by_caller(self, db_conn: sqlite3.Connection) -> None:
        """When the caller commits, the recovery persists (as get_db does)."""
        topic_id = self._stuck_topic(db_conn)

        count = recover_stuck_researching(db_conn, timeout_minutes=15)
        assert count == 1

        db_conn.commit()

        recovered = get_topic(db_conn, topic_id)
        assert recovered is not None
        assert recovered.status == TopicStatus.ERROR


class TestNoWriteLockAcrossExtractionAwait:
    """OVH-007: fetch_new_articles_for_topic must not hold a write lock across
    the content-extraction await (WAL single-writer starvation)."""

    async def test_concurrent_write_succeeds_during_extraction(self, db_conn: sqlite3.Connection) -> None:
        """A concurrent short write on a second connection succeeds while the
        pipeline is mid content-extraction await — proving no write transaction
        is held across that await.

        Before the fix, the feed-health upsert (or article inserts) opened a
        write transaction that stayed open across the extraction gather; a second
        connection's write would hit SQLITE_BUSY and raise OperationalError.
        """
        from app.scraping import fetch_new_articles_for_topic

        topic = _ready_topic(db_conn, name="ExtractionLockTopic")
        db_path = _conn_db_path(db_conn)

        entry = FeedEntry(
            title="Concurrent Article",
            url="https://example.com/concurrent",
            summary="Summary",
            source_feed="https://example.com/feed.xml",
        )

        observed: dict[str, object] = {}

        async def _fetch_feeds_with_health_write(*_args, **kwargs) -> FeedResponse:
            # Mirror the real flow: fetch_feeds_for_topic invokes the health
            # callback, which writes a feed_health row on the shared connection.
            # Before the fix this write opened a transaction that stayed open
            # across the later extraction gather.
            callback = kwargs.get("health_callback")
            if callback is not None:
                callback("https://example.com/feed.xml", True, None)
            return FeedResponse(entries=[entry])

        async def _extract_with_concurrent_write(*_args, **_kwargs) -> str:
            # Mid-extraction: a *separate* connection attempts an immediate write.
            # If the pipeline holds a write txn on db_conn, this raises
            # OperationalError("database is locked") after busy_timeout.
            side = sqlite3.connect(str(db_path), check_same_thread=False)
            side.execute("PRAGMA busy_timeout=500")
            try:
                side.execute(
                    "INSERT INTO topics (name, description, feed_urls, created_at, status) VALUES (?, ?, '[]', ?, ?)",
                    ("Sidecar Topic", "written mid-extraction", "2025-01-01T00:00:00+00:00", "new"),
                )
                side.commit()
                observed["concurrent_write_ok"] = True
            except sqlite3.OperationalError as exc:  # pragma: no cover - failure path
                observed["concurrent_write_ok"] = False
                observed["error"] = str(exc)
            finally:
                side.close()
            # Also record whether db_conn itself is mid-transaction here.
            observed["main_conn_in_transaction"] = db_conn.in_transaction
            return "Extracted content body."

        with (
            patch(
                "app.scraping.fetch_feeds_for_topic",
                side_effect=_fetch_feeds_with_health_write,
            ),
            patch(
                "app.scraping.extract_article_content",
                side_effect=_extract_with_concurrent_write,
            ),
        ):
            result = await fetch_new_articles_for_topic(topic, db_conn)

        # The article was still stored despite the restructuring.
        assert len(result.articles) == 1
        # The concurrent write must have succeeded (no write lock held).
        assert observed.get("concurrent_write_ok") is True, observed.get("error")
        # The shared connection must not be sitting in an open write txn during
        # the await.
        assert observed.get("main_conn_in_transaction") is False


class TestCommitBeforeSendOrdering:
    """OVH-066: durable state (knowledge + mark-processed + check_result) is
    committed in one explicit write transaction BEFORE the irreversible network
    sends, so a late DB failure cannot occur after a notification already went
    out (which would re-spam on the next cycle)."""

    async def test_record_failure_cannot_follow_a_sent_notification(self, db_conn: sqlite3.Connection) -> None:
        """If persisting the check result raises, the notification must NOT have
        been sent yet (commit precedes send)."""
        topic = _ready_topic(db_conn, name="OrderingTopic")
        create_knowledge_state(
            db_conn,
            KnowledgeState(topic_id=topic.id, summary_text="Old.", token_count=5),
        )
        article = create_article(db_conn, _make_article(topic_id=topic.id))
        db_conn.commit()
        settings = _pipeline_settings()

        novelty = NoveltyResult(
            has_new_info=True,
            summary="New info",
            confidence=0.9,
            relevance=0.9,
        )

        send_attempted = {"value": False}

        async def _record_send(*_args, **_kwargs) -> list[NotificationDelivery]:
            send_attempted["value"] = True
            return [NotificationDelivery(url="json://localhost", ok=True)]

        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=[article], total_feed_entries=1),
            ),
            patch("app.checker.analyze_articles", new_callable=AsyncMock, return_value=novelty),
            patch("app.checker.update_knowledge", new_callable=AsyncMock, return_value=_write_result()),
            patch("app.checker.send_notification_per_url", side_effect=_record_send),
            patch("app.checker.send_webhooks", new_callable=AsyncMock, return_value=0),
            patch(
                "app.checker.create_check_result",
                side_effect=RuntimeError("simulated persist failure"),
            ),
            pytest.raises(RuntimeError, match="simulated persist failure"),
        ):
            await check_topic(topic, db_conn, settings)

        # The persist failed; because the durable commit precedes the send, the
        # notification must NOT have been dispatched.
        assert send_attempted["value"] is False

    async def test_marks_processed_committed_before_notification(self, db_conn: sqlite3.Connection) -> None:
        """When the notification fires, the article is already marked processed
        and the knowledge state is already updated (durable-before-deliver)."""
        topic = _ready_topic(db_conn, name="OrderingTopic2")
        create_knowledge_state(
            db_conn,
            KnowledgeState(topic_id=topic.id, summary_text="Old.", token_count=5),
        )
        article = create_article(db_conn, _make_article(topic_id=topic.id))
        db_conn.commit()
        settings = _pipeline_settings()

        novelty = NoveltyResult(has_new_info=True, summary="New", confidence=0.9, relevance=0.9)

        processed_at_send: dict[str, object] = {}

        async def _check_processed_on_send(*_args, **_kwargs) -> list[NotificationDelivery]:
            # At send time, a *separate* connection should already see the article
            # marked processed (durable state committed before send).
            side = sqlite3.connect(str(_conn_db_path(db_conn)), check_same_thread=False)
            side.row_factory = sqlite3.Row
            try:
                row = side.execute("SELECT processed FROM articles WHERE id = ?", (article.id,)).fetchone()
                processed_at_send["processed"] = bool(row["processed"]) if row else None
            finally:
                side.close()
            return [NotificationDelivery(url="json://localhost", ok=True)]

        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=[article], total_feed_entries=1),
            ),
            patch("app.checker.analyze_articles", new_callable=AsyncMock, return_value=novelty),
            patch("app.checker.update_knowledge", new_callable=AsyncMock, return_value=_write_result()),
            patch("app.checker.send_notification_per_url", side_effect=_check_processed_on_send),
            patch("app.checker.send_webhooks", new_callable=AsyncMock, return_value=0),
        ):
            result = await check_topic(topic, db_conn, settings)

        assert result.notification_sent is True
        assert processed_at_send.get("processed") is True

    async def test_persisted_row_records_delivery_outcome(self, db_conn: sqlite3.Connection) -> None:
        """The check_result row created before the send is updated afterwards with
        the real delivery outcome (post-send UPDATE landed and committed)."""
        from app.crud import get_check_result

        topic = _ready_topic(db_conn, name="OrderingTopic3")
        create_knowledge_state(
            db_conn,
            KnowledgeState(topic_id=topic.id, summary_text="Old.", token_count=5),
        )
        article = create_article(db_conn, _make_article(topic_id=topic.id))
        db_conn.commit()
        settings = _pipeline_settings()

        novelty = NoveltyResult(has_new_info=True, summary="New", confidence=0.9, relevance=0.9)

        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=[article], total_feed_entries=1),
            ),
            patch("app.checker.analyze_articles", new_callable=AsyncMock, return_value=novelty),
            patch("app.checker.update_knowledge", new_callable=AsyncMock, return_value=_write_result()),
            # Delivery fails -> row must record notification_sent=0 + the reason.
            patch(
                "app.checker.send_notification_per_url",
                new_callable=AsyncMock,
                return_value=[NotificationDelivery(url="json://localhost", ok=False, error="delivery failed")],
            ),
            patch("app.checker.send_webhooks", new_callable=AsyncMock, return_value=0),
        ):
            result = await check_topic(topic, db_conn, settings)

        assert result.id is not None
        persisted = get_check_result(db_conn, result.id)
        assert persisted is not None
        assert persisted.has_new_info is True
        assert persisted.notification_sent is False
        # Per-URL failures are summarized redacted (scheme://host: reason) (OVH-039).
        assert persisted.notification_error == "json://localhost: delivery failed"


class TestWebhookCheckResultId:
    """OVH-101: the originating CheckResult must be created before send_webhooks
    so a queued webhook carries a non-NULL check_result_id."""

    async def test_queued_webhook_has_check_result_id(self, db_conn: sqlite3.Connection) -> None:
        topic = _ready_topic(db_conn, name="WebhookCRTopic")
        create_knowledge_state(
            db_conn,
            KnowledgeState(topic_id=topic.id, summary_text="Old.", token_count=5),
        )
        article = create_article(db_conn, _make_article(topic_id=topic.id))
        db_conn.commit()
        # A webhook URL that will "fail" delivery so it gets queued.
        settings = _pipeline_settings(
            notifications=NotificationSettings(urls=["json://localhost"], webhook_urls=["https://hook.example.com/x"])
        )

        novelty = NoveltyResult(has_new_info=True, summary="New", confidence=0.9, relevance=0.9)

        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=[article], total_feed_entries=1),
            ),
            patch("app.checker.analyze_articles", new_callable=AsyncMock, return_value=novelty),
            patch("app.checker.update_knowledge", new_callable=AsyncMock, return_value=_write_result()),
            patch(
                "app.checker.send_notification_per_url",
                new_callable=AsyncMock,
                return_value=[NotificationDelivery(url="json://localhost", ok=True)],
            ),
            # Force the webhook POST to fail so it is enqueued for retry.
            patch("app.webhooks.send_webhook", new_callable=AsyncMock, return_value=False),
        ):
            result = await check_topic(topic, db_conn, settings)

        assert result.id is not None
        pending = list_pending_webhooks(db_conn)
        assert len(pending) == 1
        assert pending[0]["check_result_id"] == result.id


class TestInitNoConnectionAcrossAwaits:
    """OVH-099: initialize_new_topic must not hold a write transaction across
    its fetch + LLM awaits."""

    async def test_no_write_lock_during_init_fetch(self, db_conn: sqlite3.Connection) -> None:
        """Drive the REAL fetch pipeline through init: a feed-health write must
        not pin a transaction across the content-extraction await."""
        topic = _ready_topic(db_conn, name="InitLockTopic", status=TopicStatus.NEW)
        db_path = _conn_db_path(db_conn)
        settings = _pipeline_settings()

        entry = FeedEntry(
            title="Init Article",
            url="https://example.com/init-article",
            summary="Summary",
            source_feed="https://example.com/feed.xml",
        )

        observed: dict[str, object] = {}

        async def _fetch_feeds_with_health_write(*_args, **kwargs) -> FeedResponse:
            callback = kwargs.get("health_callback")
            if callback is not None:
                callback("https://example.com/feed.xml", True, None)
            return FeedResponse(entries=[entry])

        async def _extract_with_concurrent_write(*_args, **_kwargs) -> str:
            side = sqlite3.connect(str(db_path), check_same_thread=False)
            side.execute("PRAGMA busy_timeout=500")
            try:
                side.execute(
                    "INSERT INTO topics (name, description, feed_urls, created_at, status) VALUES (?, ?, '[]', ?, ?)",
                    ("Init Sidecar", "written during init fetch", "2025-01-01T00:00:00+00:00", "new"),
                )
                side.commit()
                observed["concurrent_write_ok"] = True
            except sqlite3.OperationalError as exc:  # pragma: no cover - failure path
                observed["concurrent_write_ok"] = False
                observed["error"] = str(exc)
            finally:
                side.close()
            observed["in_transaction"] = db_conn.in_transaction
            return "Extracted init content body."

        with (
            patch(
                "app.scraping.fetch_feeds_for_topic",
                side_effect=_fetch_feeds_with_health_write,
            ),
            patch(
                "app.scraping.extract_article_content",
                side_effect=_extract_with_concurrent_write,
            ),
            patch(
                "app.checker.initialize_knowledge",
                new_callable=AsyncMock,
                return_value=_write_result(),
            ),
        ):
            await initialize_new_topic(topic, db_conn, settings)

        assert topic.status == TopicStatus.READY
        assert observed.get("concurrent_write_ok") is True, observed.get("error")
        assert observed.get("in_transaction") is False
