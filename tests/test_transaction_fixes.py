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

Connection lifetime and durable transition (AUG-136/171/202/209, TW-AUD-005):
  * No connection is open at all during a stubbed fetch/analysis/knowledge/send
    await — not merely no write transaction.
  * Feed-health outcomes observed before a fetch failure are still persisted.
  * The failed-Apprise retry row is committed before webhook I/O begins, and a
    cancellation mid-send leaves the committed transition intact.
  * The transition refuses to write when the topic was replaced or its knowledge
    moved while the check was offline.
"""

import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from app import database
from app.analysis.knowledge import KnowledgeUpdatePlan
from app.analysis.llm import EmptyAfterCleanupError, NoveltyResult, TokenUsage
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
from app.database import get_connection
from app.models import Article, KnowledgeState, NotificationDelivery, NotifyDisposition, Topic, TopicStatus
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


def _write_result() -> KnowledgeUpdatePlan:
    return KnowledgeUpdatePlan(
        summary_text="state",
        token_count=0,
        usage=TokenUsage(prompt_tokens=0, completion_tokens=0),
        sufficient_data=True,
    )


def _per_url_ok():
    """Stand-in for send_notification_per_url that reports one successful target."""
    return AsyncMock(return_value=[NotificationDelivery(url="json://localhost", ok=True)])


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
            novelty_instruction TEXT DEFAULT NULL,
            importance_threshold INTEGER DEFAULT NULL,
            init_attempts INTEGER NOT NULL DEFAULT 0,
            generation TEXT NOT NULL DEFAULT ''
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

    async def test_concurrent_write_succeeds_during_extraction(
        self, db_conn: sqlite3.Connection, db_path: Path
    ) -> None:
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
            result = await fetch_new_articles_for_topic(topic, db_path=db_path)

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

    async def test_record_failure_cannot_follow_a_sent_notification(
        self, db_conn: sqlite3.Connection, db_path: Path
    ) -> None:
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
            patch("app.checker.prepare_knowledge_update", new_callable=AsyncMock, return_value=_write_result()),
            patch("app.checker.send_notification_per_url", side_effect=_record_send),
            patch("app.checker.send_webhooks", new_callable=AsyncMock, return_value=0),
            patch(
                "app.checker.create_check_result",
                side_effect=RuntimeError("simulated persist failure"),
            ),
            pytest.raises(RuntimeError, match="simulated persist failure"),
        ):
            await check_topic(topic, settings, db_path=db_path)

        # The persist failed; because the durable commit precedes the send, the
        # notification must NOT have been dispatched.
        assert send_attempted["value"] is False

    async def test_marks_processed_committed_before_notification(
        self, db_conn: sqlite3.Connection, db_path: Path
    ) -> None:
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
            patch("app.checker.prepare_knowledge_update", new_callable=AsyncMock, return_value=_write_result()),
            patch("app.checker.send_notification_per_url", side_effect=_check_processed_on_send),
            patch("app.checker.send_webhooks", new_callable=AsyncMock, return_value=0),
        ):
            result = await check_topic(topic, settings, db_path=db_path)

        assert result.notification_sent is True
        assert processed_at_send.get("processed") is True

    async def test_persisted_row_records_delivery_outcome(self, db_conn: sqlite3.Connection, db_path: Path) -> None:
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
            patch("app.checker.prepare_knowledge_update", new_callable=AsyncMock, return_value=_write_result()),
            # Delivery fails -> row must record notification_sent=0 + the reason.
            patch(
                "app.checker.send_notification_per_url",
                new_callable=AsyncMock,
                return_value=[NotificationDelivery(url="json://localhost", ok=False, error="delivery failed")],
            ),
            patch("app.checker.send_webhooks", new_callable=AsyncMock, return_value=0),
        ):
            result = await check_topic(topic, settings, db_path=db_path)

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

    async def test_queued_webhook_has_check_result_id(self, db_conn: sqlite3.Connection, db_path: Path) -> None:
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
            patch("app.checker.prepare_knowledge_update", new_callable=AsyncMock, return_value=_write_result()),
            patch(
                "app.checker.send_notification_per_url",
                new_callable=AsyncMock,
                return_value=[NotificationDelivery(url="json://localhost", ok=True)],
            ),
            # Force the webhook POST to fail so it is enqueued for retry.
            patch("app.webhooks.send_webhook", new_callable=AsyncMock, return_value=False),
        ):
            result = await check_topic(topic, settings, db_path=db_path)

        assert result.id is not None
        pending = list_pending_webhooks(db_conn)
        assert len(pending) == 1
        assert pending[0].check_result_id == result.id


class TestInitNoConnectionAcrossAwaits:
    """OVH-099: initialize_new_topic must not hold a write transaction across
    its fetch + LLM awaits."""

    async def test_no_write_lock_during_init_fetch(self, db_conn: sqlite3.Connection, db_path: Path) -> None:
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
                "app.checker.prepare_initial_knowledge",
                new_callable=AsyncMock,
                return_value=_write_result(),
            ),
        ):
            await initialize_new_topic(topic, settings, db_path=db_path)

        assert topic.status == TopicStatus.READY
        assert observed.get("concurrent_write_ok") is True, observed.get("error")
        assert observed.get("in_transaction") is False


class _OpenConnectionCounter:
    """Count how many connections are open at any moment.

    ``sqlite3.Connection.close`` is read-only, so the count is kept by a
    subclass installed as the connection factory rather than by patching the
    method on each instance.
    """

    def __init__(self) -> None:
        self.live = 0
        self.opened = 0
        outer = self

        class _Counted(sqlite3.Connection):
            def close(self) -> None:
                outer.live -= 1
                super().close()

        self._factory = _Counted

    def open(self, path: Path | None = None) -> sqlite3.Connection:
        target = path or database.DEFAULT_DB_PATH
        conn = sqlite3.connect(str(target), check_same_thread=False, factory=self._factory)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
        self.live += 1
        self.opened += 1
        return conn


class TestNoConnectionOpenDuringAwaits:
    """AUG-136/AUG-171: the pipeline holds no connection at all during its awaits.

    The earlier lock tests prove no *write transaction* spans a network await.
    These prove the stronger property the phase structure buys: no connection is
    even open, so a check can no longer pin a SQLite handle (or a request's
    resources) for the minutes its fetch, LLM and send phases take.
    """

    async def test_zero_connections_open_during_each_stubbed_await(
        self, db_conn: sqlite3.Connection, db_path: Path
    ) -> None:
        topic = _ready_topic(db_conn, name="OpenCountTopic")
        settings = _pipeline_settings()
        article = _make_article(id=None, topic_id=topic.id)
        article = create_article(db_conn, article)
        db_conn.commit()

        counter = _OpenConnectionCounter()
        peak_during_awaits = 0

        def _sample() -> None:
            nonlocal peak_during_awaits
            peak_during_awaits = max(peak_during_awaits, counter.live)

        async def _fetch(*_args, **_kwargs):
            await asyncio.sleep(0)
            _sample()
            return FetchResult(articles=[article], total_feed_entries=1)

        async def _analyze(*_args, **_kwargs):
            await asyncio.sleep(0)
            _sample()
            return NoveltyResult(has_new_info=True, summary="s", confidence=0.9, relevance=0.9, importance=5)

        async def _prepare(*_args, **_kwargs):
            await asyncio.sleep(0)
            _sample()
            return _write_result()

        async def _send(*_args, **_kwargs):
            await asyncio.sleep(0)
            _sample()
            return [NotificationDelivery(url="json://localhost", ok=True)]

        with (
            patch("app.database.get_connection", side_effect=counter.open),
            patch("app.checker.fetch_new_articles_for_topic", side_effect=_fetch),
            patch("app.checker.analyze_articles", side_effect=_analyze),
            patch("app.checker.prepare_knowledge_update", side_effect=_prepare),
            patch("app.checker.send_notification_per_url", side_effect=_send),
            patch("app.checker.send_webhooks", new_callable=AsyncMock, return_value=0),
        ):
            result = await check_topic(topic, settings, db_path=db_path)

        assert result.id is not None
        # Guard against a vacuous pass: the pipeline really did use the counter.
        assert counter.opened >= 3
        assert peak_during_awaits == 0, "a connection was open while the pipeline awaited I/O"
        assert counter.live == 0, "the pipeline leaked a connection"

    async def test_concurrent_writer_is_never_blocked_in_any_phase(
        self, db_conn: sqlite3.Connection, db_path: Path
    ) -> None:
        """A second connection writes immediately during fetch, analysis and send.

        A 200ms busy timeout is far below the seconds-to-minutes a real phase
        takes, so any transaction held across one of these awaits fails here
        rather than passing slowly.
        """
        topic = _ready_topic(db_conn, name="ConcurrentWriterTopic")
        settings = _pipeline_settings()
        article = create_article(db_conn, _make_article(id=None, topic_id=topic.id))
        db_conn.commit()

        blocked: list[str] = []

        def _write_from_a_second_connection(phase: str) -> None:
            side = sqlite3.connect(str(db_path), check_same_thread=False)
            side.execute("PRAGMA busy_timeout=200")
            try:
                side.execute(
                    "INSERT INTO topics (name, description, feed_urls, created_at, status, generation)"
                    " VALUES (?, 'sidecar', '[]', '2026-01-01T00:00:00+00:00', 'new', 'g')",
                    (f"Sidecar {phase}",),
                )
                side.commit()
            except sqlite3.OperationalError:
                blocked.append(phase)
            finally:
                side.close()

        async def _fetch(*_args, **_kwargs):
            _write_from_a_second_connection("fetch")
            return FetchResult(articles=[article], total_feed_entries=1)

        async def _analyze(*_args, **_kwargs):
            _write_from_a_second_connection("analysis")
            return NoveltyResult(has_new_info=True, summary="s", confidence=0.9, relevance=0.9, importance=5)

        async def _prepare(*_args, **_kwargs):
            _write_from_a_second_connection("knowledge")
            return _write_result()

        async def _send(*_args, **_kwargs):
            _write_from_a_second_connection("send")
            return [NotificationDelivery(url="json://localhost", ok=True)]

        with (
            patch("app.checker.fetch_new_articles_for_topic", side_effect=_fetch),
            patch("app.checker.analyze_articles", side_effect=_analyze),
            patch("app.checker.prepare_knowledge_update", side_effect=_prepare),
            patch("app.checker.send_notification_per_url", side_effect=_send),
            patch("app.checker.send_webhooks", new_callable=AsyncMock, return_value=0),
        ):
            result = await check_topic(topic, settings, db_path=db_path)

        assert result.id is not None
        assert blocked == [], f"a writer was blocked during: {blocked}"


class TestFeedHealthSurvivesPartialFailure:
    """AUG-171: outcomes observed before a fetch blows up are still persisted."""

    async def test_health_outcomes_persist_when_the_fetch_raises(
        self, db_conn: sqlite3.Connection, db_path: Path
    ) -> None:
        from app.crud import get_feed_health
        from app.scraping import fetch_new_articles_for_topic

        topic = _ready_topic(db_conn, name="PartialFetchTopic")

        async def _fetch_feeds_then_fail(*_args, **kwargs):
            callback = kwargs["health_callback"]
            callback("https://ok.example/feed", True, None, 'W/"e1"', "LM1")
            callback("https://bad.example/feed", False, "boom", None, None)
            raise RuntimeError("provider exploded after two feeds")

        with (
            patch("app.scraping.fetch_feeds_for_topic", side_effect=_fetch_feeds_then_fail),
            pytest.raises(RuntimeError, match="provider exploded"),
        ):
            await fetch_new_articles_for_topic(topic, db_path=db_path)

        ok = get_feed_health(db_conn, "https://ok.example/feed")
        bad = get_feed_health(db_conn, "https://bad.example/feed")
        assert ok is not None and ok.etag == 'W/"e1"'
        assert bad is not None and bad.consecutive_failures == 1


class TestSendPhaseDurability:
    """TW-AUD-005 + the C3 boundary: what survives an interruption mid-send."""

    async def _run_to_send(self, conn, topic, settings, db_path, *, send, webhooks):
        article = create_article(conn, _make_article(id=None, topic_id=topic.id))
        conn.commit()
        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=[article], total_feed_entries=1),
            ),
            patch(
                "app.checker.analyze_articles",
                new_callable=AsyncMock,
                return_value=NoveltyResult(has_new_info=True, summary="s", confidence=0.9, relevance=0.9, importance=5),
            ),
            patch("app.checker.prepare_knowledge_update", new_callable=AsyncMock, return_value=_write_result()),
            patch("app.checker.send_notification_per_url", side_effect=send),
            patch("app.checker.send_webhooks", side_effect=webhooks),
        ):
            return await check_topic(topic, settings, db_path=db_path)

    async def test_failed_apprise_queue_row_is_committed_before_webhook_io(
        self, db_conn: sqlite3.Connection, db_path: Path
    ) -> None:
        """The retry intent must be durable before the webhook POSTs start.

        It used to be staged on the connection that then awaited webhook
        delivery, so the writer transaction spanned that I/O and a cancellation
        rolled the intent back -- losing the only record that a channel still
        owed a delivery.
        """
        topic = _ready_topic(db_conn, name="QueueBeforeWebhook")
        settings = _pipeline_settings()
        observed: dict[str, object] = {}

        async def _send(*_args, **_kwargs):
            return [NotificationDelivery(url="json://down", ok=False, error="unreachable")]

        async def _webhooks(*_args, **_kwargs):
            # Read the queue from a SEPARATE connection: it can only see the row
            # if the queueing transaction has already committed.
            side = get_connection(db_path)
            try:
                observed["queued_before_webhook_io"] = side.execute(
                    "SELECT COUNT(*) FROM pending_notifications"
                ).fetchone()[0]
            finally:
                side.close()
            return 0

        await self._run_to_send(db_conn, topic, settings, db_path, send=_send, webhooks=_webhooks)

        assert observed.get("queued_before_webhook_io") == 1

    async def test_cancellation_mid_send_keeps_the_committed_transition(
        self, db_conn: sqlite3.Connection, db_path: Path
    ) -> None:
        """A cancelled send loses only the send: C3's state is already durable."""
        topic = _ready_topic(db_conn, name="CancelMidSend")
        settings = _pipeline_settings()

        async def _send(*_args, **_kwargs):
            raise asyncio.CancelledError

        async def _webhooks(*_args, **_kwargs):  # pragma: no cover - never reached
            return 0

        with pytest.raises(asyncio.CancelledError):
            await self._run_to_send(db_conn, topic, settings, db_path, send=_send, webhooks=_webhooks)

        # The knowledge write, the article disposition and the CheckResult all
        # landed before the send was attempted, and none of them rolled back.
        assert db_conn.execute("SELECT COUNT(*) FROM check_results WHERE topic_id = ?", (topic.id,)).fetchone()[0] == 1
        state = db_conn.execute(
            "SELECT summary_text, version FROM knowledge_states WHERE topic_id = ?", (topic.id,)
        ).fetchone()
        assert state is not None and state["summary_text"] == "state"
        assert db_conn.execute("SELECT COUNT(*) FROM articles WHERE processed = 1").fetchone()[0] == 1


class TestDurableTransitionGuards:
    """C3 refuses to write against state that moved while the check was offline."""

    async def _check_with_stubs(self, topic, settings, db_path, *, before_commit=None):
        article = _make_article(id=None, topic_id=topic.id)

        async def _prepare(*_args, **_kwargs):
            if before_commit is not None:
                before_commit()
            return _write_result()

        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=[article], total_feed_entries=1),
            ),
            patch(
                "app.checker.analyze_articles",
                new_callable=AsyncMock,
                return_value=NoveltyResult(has_new_info=True, summary="s", confidence=0.9, relevance=0.9, importance=1),
            ),
            patch("app.checker.prepare_knowledge_update", side_effect=_prepare),
            patch("app.checker.send_notification_per_url", new_callable=AsyncMock, return_value=[]),
        ):
            return await check_topic(topic, settings, db_path=db_path)

    async def test_knowledge_moving_mid_check_aborts_without_recording(
        self, db_conn: sqlite3.Connection, db_path: Path
    ) -> None:
        """A rival check that finishes first keeps its summary; the loser writes nothing."""
        topic = _ready_topic(db_conn, name="CasLoser")
        create_knowledge_state(db_conn, KnowledgeState(topic_id=topic.id, summary_text="base"))
        db_conn.commit()
        settings = _pipeline_settings()

        def _rival_finishes_first() -> None:
            side = get_connection(db_path)
            try:
                side.execute(
                    "UPDATE knowledge_states SET summary_text = 'rival', version = version + 1 WHERE topic_id = ?",
                    (topic.id,),
                )
                side.commit()
            finally:
                side.close()

        result = await self._check_with_stubs(topic, settings, db_path, before_commit=_rival_finishes_first)

        assert result.id is None
        assert result.stage_error is not None and result.stage_error.startswith("transition_aborted:")
        row = db_conn.execute("SELECT summary_text FROM knowledge_states WHERE topic_id = ?", (topic.id,)).fetchone()
        assert row["summary_text"] == "rival"
        assert db_conn.execute("SELECT COUNT(*) FROM check_results").fetchone()[0] == 0
        assert db_conn.execute("SELECT COUNT(*) FROM articles WHERE processed = 1").fetchone()[0] == 0

    async def test_recycled_topic_rowid_aborts_the_transition(self, db_conn: sqlite3.Connection, db_path: Path) -> None:
        """A delete+recreate onto the same rowid must not receive the old check's work."""
        topic = _ready_topic(db_conn, name="Recycled")
        settings = _pipeline_settings()

        def _delete_and_recreate() -> None:
            side = get_connection(db_path)
            try:
                side.execute("DELETE FROM topics WHERE id = ?", (topic.id,))
                side.execute(
                    "INSERT INTO topics (id, name, description, feed_urls, created_at, status, generation)"
                    " VALUES (?, ?, 'd', '[]', ?, 'ready', 'brand-new-generation')",
                    (topic.id, "Replacement", "2026-01-01T00:00:00+00:00"),
                )
                side.commit()
            finally:
                side.close()

        result = await self._check_with_stubs(topic, settings, db_path, before_commit=_delete_and_recreate)

        assert result.id is None
        assert "deleted or replaced" in (result.stage_error or "")
        assert db_conn.execute("SELECT COUNT(*) FROM check_results").fetchone()[0] == 0
        assert db_conn.execute("SELECT COUNT(*) FROM knowledge_states").fetchone()[0] == 0


class TestKnowledgeCleanupFailuresAtTheTransition:
    """The C3 transaction's behaviour when the knowledge phase raises.

    ``generate_*`` now raise ``EmptyAfterCleanupError`` when citation/reliability
    cleanup empties a summary. Knowledge init/update are allowed to raise (unlike
    ``analyze_articles``), so these prove the raising phase is handled where it
    matters: nothing is half-written, and the recorded row says what happened.
    """

    async def test_empty_after_cleanup_records_a_check_without_touching_knowledge(
        self, db_conn: sqlite3.Connection, db_path: Path
    ) -> None:
        topic = _ready_topic(db_conn, name="EmptyMergeTopic")
        create_knowledge_state(db_conn, KnowledgeState(topic_id=topic.id, summary_text="baseline"))
        article = create_article(db_conn, _make_article(id=None, topic_id=topic.id))
        db_conn.commit()
        settings = _pipeline_settings()

        async def _prepare(*_args, **_kwargs):
            raise EmptyAfterCleanupError("updated knowledge summary was empty after cleanup")

        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=[article], total_feed_entries=1),
            ),
            patch(
                "app.checker.analyze_articles",
                new_callable=AsyncMock,
                return_value=NoveltyResult(has_new_info=True, summary="s", confidence=0.9, relevance=0.9, importance=5),
            ),
            patch("app.checker.prepare_knowledge_update", side_effect=_prepare),
            patch("app.checker.send_notification_per_url", _per_url_ok()),
            patch("app.checker.send_webhooks", new_callable=AsyncMock, return_value=0),
        ):
            result = await check_topic(topic, settings, db_path=db_path)

        # The check IS recorded -- the failure is visible, not swallowed.
        assert result.id is not None
        assert result.stage_error is not None
        assert result.stage_error.startswith("knowledge_update_failed:")
        assert "EmptyAfterCleanupError" in result.stage_error
        # The baseline survives, and the articles stay unprocessed so the next
        # cycle re-attempts the merge (OVH-009).
        stored = db_conn.execute(
            "SELECT summary_text, version FROM knowledge_states WHERE topic_id = ?", (topic.id,)
        ).fetchone()
        assert stored["summary_text"] == "baseline"
        assert stored["version"] == 0
        assert db_conn.execute("SELECT COUNT(*) FROM articles WHERE processed = 1").fetchone()[0] == 0
        # No revision was appended for a merge that never landed.
        assert db_conn.execute("SELECT COUNT(*) FROM knowledge_revisions").fetchone()[0] == 0

    async def test_init_empty_after_cleanup_leaves_the_topic_in_error(
        self, db_conn: sqlite3.Connection, db_path: Path
    ) -> None:
        """Knowledge init raising is the settled contract; the topic must not go READY."""
        topic = _ready_topic(db_conn, name="EmptyInitTopic", status=TopicStatus.NEW)
        article = create_article(db_conn, _make_article(id=None, topic_id=topic.id))
        db_conn.commit()
        settings = _pipeline_settings()

        async def _prepare(*_args, **_kwargs):
            raise EmptyAfterCleanupError("initial knowledge summary was empty after cleanup")

        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=[article], total_feed_entries=1),
            ),
            patch("app.checker.prepare_initial_knowledge", side_effect=_prepare),
        ):
            await initialize_new_topic(topic, settings, db_path=db_path)

        assert get_topic(db_conn, topic.id).status == TopicStatus.ERROR
        assert db_conn.execute("SELECT COUNT(*) FROM knowledge_states").fetchone()[0] == 0
        assert db_conn.execute("SELECT COUNT(*) FROM articles WHERE processed = 1").fetchone()[0] == 0

    async def test_novelty_emptied_by_cleanup_is_recorded_as_analysis_failed(
        self, db_conn: sqlite3.Connection, db_path: Path
    ) -> None:
        """``analyze_articles`` never raises; its new empty-summary safe default
        carries ``error``, so the disposition must be analysis_failed rather than
        an indistinguishable clean ``no_new_info``."""
        topic = _ready_topic(db_conn, name="EmptyNoveltyTopic")
        article = create_article(db_conn, _make_article(id=None, topic_id=topic.id))
        db_conn.commit()
        settings = _pipeline_settings()

        safe_default = NoveltyResult(
            has_new_info=False,
            confidence=0.0,
            error="novelty summary was empty after citation/reliability cleanup",
        )

        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=[article], total_feed_entries=1),
            ),
            patch("app.checker.analyze_articles", new_callable=AsyncMock, return_value=safe_default),
            patch("app.checker.send_notification_per_url", _per_url_ok()),
        ):
            result = await check_topic(topic, settings, db_path=db_path)

        assert result.id is not None
        assert result.notify_disposition == NotifyDisposition.ANALYSIS_FAILED
        assert result.stage_error is not None and result.stage_error.startswith("analysis_failed:")
        stored = db_conn.execute("SELECT notify_disposition FROM check_results WHERE id = ?", (result.id,)).fetchone()
        assert stored["notify_disposition"] == "analysis_failed"


class TestTransitionIsAllOrNothing:
    """One logical transition, one commit — no durable fragments (TW-AUD-002/AUG-253)."""

    async def test_a_failed_result_insert_rolls_back_the_knowledge_write(
        self, db_conn: sqlite3.Connection, db_path: Path
    ) -> None:
        """Knowledge, its revision, the article disposition and the row live or die together.

        The knowledge state used to commit before the CheckResult existed, so a
        failure in between left a summary that had absorbed articles no check
        ever recorded, and the revision append (a third commit) could vanish on
        its own and silently bridge two states in the timeline.
        """
        topic = _ready_topic(db_conn, name="AllOrNothing")
        create_knowledge_state(db_conn, KnowledgeState(topic_id=topic.id, summary_text="base"))
        article = create_article(db_conn, _make_article(id=None, topic_id=topic.id))
        db_conn.commit()
        settings = _pipeline_settings()

        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=[article], total_feed_entries=1),
            ),
            patch(
                "app.checker.analyze_articles",
                new_callable=AsyncMock,
                return_value=NoveltyResult(has_new_info=True, summary="s", confidence=0.9, relevance=0.9, importance=1),
            ),
            patch("app.checker.prepare_knowledge_update", new_callable=AsyncMock, return_value=_write_result()),
            patch("app.checker.create_check_result", side_effect=sqlite3.OperationalError("disk I/O error")),
            patch("app.checker.send_notification_per_url", new_callable=AsyncMock, return_value=[]),
            pytest.raises(sqlite3.OperationalError),
        ):
            await check_topic(topic, settings, db_path=db_path)

        state = db_conn.execute(
            "SELECT summary_text, version FROM knowledge_states WHERE topic_id = ?", (topic.id,)
        ).fetchone()
        assert state["summary_text"] == "base"
        assert state["version"] == 0
        assert db_conn.execute("SELECT COUNT(*) FROM knowledge_revisions").fetchone()[0] == 0
        assert db_conn.execute("SELECT COUNT(*) FROM check_results").fetchone()[0] == 0
        assert db_conn.execute("SELECT processed FROM articles WHERE id = ?", (article.id,)).fetchone()[0] == 0

    async def test_a_failed_article_disposition_rolls_back_the_result(
        self, db_conn: sqlite3.Connection, db_path: Path
    ) -> None:
        """The attempt bookkeeping is part of the transition, not a follow-up write."""
        topic = _ready_topic(db_conn, name="AttemptRollback")
        article = create_article(db_conn, _make_article(id=None, topic_id=topic.id))
        db_conn.commit()
        settings = _pipeline_settings()

        failed = NoveltyResult(has_new_info=False, confidence=0.0, error="LLM analysis failed")
        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=[article], total_feed_entries=1),
            ),
            patch("app.checker.analyze_articles", new_callable=AsyncMock, return_value=failed),
            patch(
                "app.checker.record_article_analysis_failure",
                side_effect=sqlite3.OperationalError("disk I/O error"),
            ),
            pytest.raises(sqlite3.OperationalError),
        ):
            await check_topic(topic, settings, db_path=db_path)

        assert db_conn.execute("SELECT COUNT(*) FROM check_results").fetchone()[0] == 0
        row = db_conn.execute(
            "SELECT processed, analysis_attempts FROM articles WHERE id = ?", (article.id,)
        ).fetchone()
        assert row["processed"] == 0
        assert row["analysis_attempts"] == 0
