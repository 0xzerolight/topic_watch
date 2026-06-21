"""Tests for database operations: schema, CRUD, and dedup."""

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from app.crud import (
    article_hash_exists,
    create_article,
    create_check_result,
    create_knowledge_state,
    create_pending_notification,
    create_topic,
    delete_expired_notifications,
    delete_pending_notification,
    delete_topic,
    get_check_result,
    get_dashboard_data,
    get_knowledge_state,
    get_topic,
    get_topic_by_name,
    increment_notification_retry,
    list_articles_for_topic,
    list_check_results,
    list_pending_notifications,
    list_topics,
    mark_articles_processed,
    recover_stuck_topics,
    update_knowledge_state,
    update_topic,
)
from app.database import run_migrations
from app.models import (
    Article,
    CheckResult,
    KnowledgeState,
    PendingNotification,
    Topic,
    TopicStatus,
)


class TestSchema:
    """Test that the database schema is created correctly."""

    def test_tables_exist(self, db_conn: sqlite3.Connection) -> None:
        tables = db_conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
        table_names = {row["name"] for row in tables}
        assert "topics" in table_names
        assert "articles" in table_names
        assert "knowledge_states" in table_names
        assert "check_results" in table_names

    def test_wal_mode(self, db_conn: sqlite3.Connection) -> None:
        mode = db_conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"

    def test_foreign_keys_enabled(self, db_conn: sqlite3.Connection) -> None:
        fk = db_conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk == 1


class TestTopicCRUD:
    """Test CRUD operations for topics."""

    def test_create_and_get_topic(self, db_conn: sqlite3.Connection) -> None:
        topic = Topic(
            name="Test Topic",
            description="A test topic description",
            feed_urls=["https://example.com/feed.xml"],
        )
        created = create_topic(db_conn, topic)
        db_conn.commit()
        assert created.id is not None

        retrieved = get_topic(db_conn, created.id)
        assert retrieved is not None
        assert retrieved.name == "Test Topic"
        assert retrieved.feed_urls == ["https://example.com/feed.xml"]
        assert retrieved.status == TopicStatus.RESEARCHING

    def test_get_nonexistent_topic(self, db_conn: sqlite3.Connection) -> None:
        assert get_topic(db_conn, 9999) is None

    def test_get_topic_by_name(self, db_conn: sqlite3.Connection) -> None:
        topic = Topic(name="Named Topic", description="desc")
        create_topic(db_conn, topic)
        db_conn.commit()

        found = get_topic_by_name(db_conn, "Named Topic")
        assert found is not None
        assert found.name == "Named Topic"

    def test_get_topic_by_name_not_found(self, db_conn: sqlite3.Connection) -> None:
        assert get_topic_by_name(db_conn, "Nonexistent") is None

    def test_list_topics(self, db_conn: sqlite3.Connection) -> None:
        create_topic(db_conn, Topic(name="A", description="a"))
        create_topic(db_conn, Topic(name="B", description="b"))
        db_conn.commit()

        topics = list_topics(db_conn)
        assert len(topics) == 2
        assert topics[0].name == "A"
        assert topics[1].name == "B"

    def test_list_active_topics(self, db_conn: sqlite3.Connection) -> None:
        create_topic(db_conn, Topic(name="Active", description="a", is_active=True))
        create_topic(db_conn, Topic(name="Inactive", description="b", is_active=False))
        db_conn.commit()

        active = list_topics(db_conn, active_only=True)
        assert len(active) == 1
        assert active[0].name == "Active"

    def test_update_topic(self, db_conn: sqlite3.Connection) -> None:
        topic = Topic(name="Original", description="desc")
        created = create_topic(db_conn, topic)
        db_conn.commit()

        created.name = "Updated"
        created.status = TopicStatus.READY
        update_topic(db_conn, created)
        db_conn.commit()

        retrieved = get_topic(db_conn, created.id)
        assert retrieved is not None
        assert retrieved.name == "Updated"
        assert retrieved.status == TopicStatus.READY

    def test_delete_topic(self, db_conn: sqlite3.Connection) -> None:
        topic = Topic(name="ToDelete", description="desc")
        created = create_topic(db_conn, topic)
        db_conn.commit()

        assert delete_topic(db_conn, created.id) is True
        db_conn.commit()
        assert get_topic(db_conn, created.id) is None

    def test_delete_nonexistent_returns_false(self, db_conn: sqlite3.Connection) -> None:
        assert delete_topic(db_conn, 9999) is False

    def test_unique_topic_name(self, db_conn: sqlite3.Connection) -> None:
        create_topic(db_conn, Topic(name="Unique", description="a"))
        db_conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            create_topic(db_conn, Topic(name="Unique", description="b"))

    def test_topic_feed_urls_roundtrip(self, db_conn: sqlite3.Connection) -> None:
        urls = [
            "https://example.com/feed1.xml",
            "https://example.com/feed2.xml",
            "https://reddit.com/r/test/search.rss?q=test&sort=new",
        ]
        topic = Topic(name="Feeds", description="desc", feed_urls=urls)
        created = create_topic(db_conn, topic)
        db_conn.commit()

        retrieved = get_topic(db_conn, created.id)
        assert retrieved is not None
        assert retrieved.feed_urls == urls


class TestArticleCRUD:
    """Test CRUD operations for articles."""

    def _make_topic(self, conn: sqlite3.Connection) -> Topic:
        topic = create_topic(conn, Topic(name="ArtTopic", description="desc"))
        conn.commit()
        return topic

    def test_create_and_list_articles(self, db_conn: sqlite3.Connection) -> None:
        topic = self._make_topic(db_conn)
        article = Article(
            topic_id=topic.id,
            title="Test Article",
            url="https://example.com/article1",
            content_hash="abc123",
            source_feed="https://example.com/feed.xml",
        )
        created = create_article(db_conn, article)
        db_conn.commit()
        assert created.id is not None

        articles = list_articles_for_topic(db_conn, topic.id)
        assert len(articles) == 1
        assert articles[0].title == "Test Article"

    def test_dedup_by_hash(self, db_conn: sqlite3.Connection) -> None:
        topic = self._make_topic(db_conn)
        a1 = Article(
            topic_id=topic.id,
            title="A",
            url="url1",
            content_hash="dup_hash",
            source_feed="feed",
        )
        create_article(db_conn, a1)
        db_conn.commit()

        assert article_hash_exists(db_conn, topic.id, "dup_hash") is True
        assert article_hash_exists(db_conn, topic.id, "new_hash") is False

        a2 = Article(
            topic_id=topic.id,
            title="B",
            url="url2",
            content_hash="dup_hash",
            source_feed="feed",
        )
        with pytest.raises(sqlite3.IntegrityError):
            create_article(db_conn, a2)

    def test_same_hash_different_topics(self, db_conn: sqlite3.Connection) -> None:
        """Same content_hash is allowed across different topics."""
        t1 = create_topic(db_conn, Topic(name="T1", description="d"))
        t2 = create_topic(db_conn, Topic(name="T2", description="d"))
        db_conn.commit()

        create_article(
            db_conn,
            Article(
                topic_id=t1.id,
                title="A",
                url="url1",
                content_hash="same",
                source_feed="f",
            ),
        )
        create_article(
            db_conn,
            Article(
                topic_id=t2.id,
                title="A",
                url="url1",
                content_hash="same",
                source_feed="f",
            ),
        )
        db_conn.commit()
        # No error — hash uniqueness is scoped to topic

    def test_list_unprocessed_articles(self, db_conn: sqlite3.Connection) -> None:
        topic = self._make_topic(db_conn)
        create_article(
            db_conn,
            Article(
                topic_id=topic.id,
                title="Processed",
                url="url1",
                content_hash="h1",
                source_feed="feed",
                processed=True,
            ),
        )
        create_article(
            db_conn,
            Article(
                topic_id=topic.id,
                title="Unprocessed",
                url="url2",
                content_hash="h2",
                source_feed="feed",
                processed=False,
            ),
        )
        db_conn.commit()

        unprocessed = list_articles_for_topic(db_conn, topic.id, unprocessed_only=True)
        assert len(unprocessed) == 1
        assert unprocessed[0].title == "Unprocessed"

    def test_mark_articles_processed(self, db_conn: sqlite3.Connection) -> None:
        topic = self._make_topic(db_conn)
        a1 = create_article(
            db_conn,
            Article(
                topic_id=topic.id,
                title="A",
                url="url1",
                content_hash="h1",
                source_feed="feed",
            ),
        )
        a2 = create_article(
            db_conn,
            Article(
                topic_id=topic.id,
                title="B",
                url="url2",
                content_hash="h2",
                source_feed="feed",
            ),
        )
        db_conn.commit()

        mark_articles_processed(db_conn, [a1.id, a2.id])
        db_conn.commit()

        unprocessed = list_articles_for_topic(db_conn, topic.id, unprocessed_only=True)
        assert len(unprocessed) == 0

    def test_mark_empty_list(self, db_conn: sqlite3.Connection) -> None:
        """Marking an empty list should not error."""
        mark_articles_processed(db_conn, [])


class TestKnowledgeStateCRUD:
    """Test CRUD operations for knowledge states."""

    def test_create_and_get(self, db_conn: sqlite3.Connection) -> None:
        topic = create_topic(db_conn, Topic(name="KSTopic", description="desc"))
        db_conn.commit()

        state = KnowledgeState(
            topic_id=topic.id,
            summary_text="Initial knowledge summary",
            token_count=150,
        )
        created = create_knowledge_state(db_conn, state)
        db_conn.commit()
        assert created.id is not None

        retrieved = get_knowledge_state(db_conn, topic.id)
        assert retrieved is not None
        assert retrieved.summary_text == "Initial knowledge summary"
        assert retrieved.token_count == 150

    def test_get_nonexistent(self, db_conn: sqlite3.Connection) -> None:
        assert get_knowledge_state(db_conn, 9999) is None

    def test_update(self, db_conn: sqlite3.Connection) -> None:
        topic = create_topic(db_conn, Topic(name="KSUpdate", description="desc"))
        state = KnowledgeState(topic_id=topic.id, summary_text="V1", token_count=50)
        created = create_knowledge_state(db_conn, state)
        db_conn.commit()

        created.summary_text = "V2 - updated with new info"
        created.token_count = 120
        update_knowledge_state(db_conn, created)
        db_conn.commit()

        retrieved = get_knowledge_state(db_conn, topic.id)
        assert retrieved is not None
        assert retrieved.summary_text == "V2 - updated with new info"
        assert retrieved.token_count == 120

    def test_one_per_topic_upsert(self, db_conn: sqlite3.Connection) -> None:
        """Duplicate insert for same topic replaces existing state (INSERT OR REPLACE)."""
        topic = create_topic(db_conn, Topic(name="KSUnique", description="desc"))
        create_knowledge_state(
            db_conn,
            KnowledgeState(topic_id=topic.id, summary_text="First", token_count=10),
        )
        db_conn.commit()

        create_knowledge_state(
            db_conn,
            KnowledgeState(topic_id=topic.id, summary_text="Second", token_count=20),
        )
        db_conn.commit()

        state = get_knowledge_state(db_conn, topic.id)
        assert state is not None
        assert state.summary_text == "Second"
        assert state.token_count == 20


class TestCheckResultCRUD:
    """Test CRUD operations for check results."""

    def test_create_and_list(self, db_conn: sqlite3.Connection) -> None:
        topic = create_topic(db_conn, Topic(name="CRTopic", description="desc"))
        db_conn.commit()

        create_check_result(
            db_conn,
            CheckResult(
                topic_id=topic.id,
                articles_found=5,
                articles_new=2,
                has_new_info=True,
            ),
        )
        create_check_result(
            db_conn,
            CheckResult(
                topic_id=topic.id,
                articles_found=3,
                articles_new=0,
                has_new_info=False,
            ),
        )
        db_conn.commit()

        results = list_check_results(db_conn, topic.id)
        assert len(results) == 2

    def test_update_delivery_outcome(self, db_conn: sqlite3.Connection) -> None:
        """update_check_result_delivery records the post-send delivery outcome
        onto an already-created CheckResult row (OVH-066)."""
        from app.crud import update_check_result_delivery

        topic = create_topic(db_conn, Topic(name="CRDelivery", description="desc"))
        db_conn.commit()

        created = create_check_result(
            db_conn,
            CheckResult(topic_id=topic.id, has_new_info=True),
        )
        db_conn.commit()
        assert created.id is not None
        # Created before the send: delivery fields default to "not sent".
        assert get_check_result(db_conn, created.id).notification_sent is False

        update_check_result_delivery(
            db_conn,
            created.id,
            notification_sent=True,
            notification_error=None,
        )
        db_conn.commit()

        refreshed = get_check_result(db_conn, created.id)
        assert refreshed is not None
        assert refreshed.notification_sent is True
        assert refreshed.notification_error is None

        # And a failure outcome is recorded too.
        update_check_result_delivery(
            db_conn,
            created.id,
            notification_sent=False,
            notification_error="Delivery failed",
        )
        db_conn.commit()
        refreshed = get_check_result(db_conn, created.id)
        assert refreshed is not None
        assert refreshed.notification_sent is False
        assert refreshed.notification_error == "Delivery failed"

    def test_ordered_newest_first(self, db_conn: sqlite3.Connection) -> None:
        topic = create_topic(db_conn, Topic(name="CROrder", description="desc"))
        db_conn.commit()

        now = datetime.now(UTC)
        create_check_result(
            db_conn,
            CheckResult(topic_id=topic.id, checked_at=now - timedelta(hours=1)),
        )
        create_check_result(db_conn, CheckResult(topic_id=topic.id, checked_at=now))
        db_conn.commit()

        results = list_check_results(db_conn, topic.id, limit=1)
        assert len(results) == 1

    def test_limit(self, db_conn: sqlite3.Connection) -> None:
        topic = create_topic(db_conn, Topic(name="CRLimit", description="desc"))
        db_conn.commit()

        for _ in range(5):
            create_check_result(db_conn, CheckResult(topic_id=topic.id))
        db_conn.commit()

        results = list_check_results(db_conn, topic.id, limit=3)
        assert len(results) == 3

    def test_boolean_fields_roundtrip(self, db_conn: sqlite3.Connection) -> None:
        topic = create_topic(db_conn, Topic(name="CRBool", description="desc"))
        db_conn.commit()

        create_check_result(
            db_conn,
            CheckResult(
                topic_id=topic.id,
                has_new_info=True,
                notification_sent=True,
                notification_error="test error",
            ),
        )
        db_conn.commit()

        results = list_check_results(db_conn, topic.id)
        assert results[0].has_new_info is True
        assert results[0].notification_sent is True
        assert results[0].notification_error == "test error"


class TestCascadeDelete:
    """Test that deleting a topic cascades to related records."""

    def test_cascade_deletes_articles(self, db_conn: sqlite3.Connection) -> None:
        topic = create_topic(db_conn, Topic(name="Cascade", description="desc"))
        create_article(
            db_conn,
            Article(
                topic_id=topic.id,
                title="Art",
                url="url",
                content_hash="h",
                source_feed="feed",
            ),
        )
        db_conn.commit()

        delete_topic(db_conn, topic.id)
        db_conn.commit()

        articles = list_articles_for_topic(db_conn, topic.id)
        assert len(articles) == 0

    def test_cascade_deletes_knowledge_state(self, db_conn: sqlite3.Connection) -> None:
        topic = create_topic(db_conn, Topic(name="CascadeKS", description="desc"))
        create_knowledge_state(
            db_conn,
            KnowledgeState(topic_id=topic.id, summary_text="text", token_count=10),
        )
        db_conn.commit()

        delete_topic(db_conn, topic.id)
        db_conn.commit()

        assert get_knowledge_state(db_conn, topic.id) is None

    def test_cascade_deletes_check_results(self, db_conn: sqlite3.Connection) -> None:
        topic = create_topic(db_conn, Topic(name="CascadeCR", description="desc"))
        create_check_result(db_conn, CheckResult(topic_id=topic.id))
        db_conn.commit()

        delete_topic(db_conn, topic.id)
        db_conn.commit()

        results = list_check_results(db_conn, topic.id)
        assert len(results) == 0


class TestMigrations:
    """Tests for the database migration system."""

    def test_migrations_applied(self, db_conn: sqlite3.Connection) -> None:
        """All migrations are applied after init_db."""
        row = db_conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        assert row[0] is not None
        assert row[0] >= 3  # At least m001 + m002 + m003

    def test_migrations_idempotent(self, db_conn: sqlite3.Connection) -> None:
        """Running migrations twice does not error."""
        run_migrations(db_conn)
        run_migrations(db_conn)
        row = db_conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        assert row[0] >= 3

    def test_pending_notifications_table_exists(self, db_conn: sqlite3.Connection) -> None:
        """Migration m002 creates the pending_notifications table."""
        tables = db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='pending_notifications'"
        ).fetchall()
        assert len(tables) == 1

    def test_pending_webhooks_table_exists(self, db_conn: sqlite3.Connection) -> None:
        """Migration m010 creates the pending_webhooks table."""
        tables = db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='pending_webhooks'"
        ).fetchall()
        assert len(tables) == 1

    def test_topic_threshold_columns_exist(self, db_conn: sqlite3.Connection) -> None:
        """Migration m011 adds nullable confidence/relevance threshold columns."""
        columns = {row[1]: row for row in db_conn.execute("PRAGMA table_info(topics)").fetchall()}
        assert "confidence_threshold" in columns
        assert "relevance_threshold" in columns

    def test_check_result_token_columns_exist(self, db_conn: sqlite3.Connection) -> None:
        """Migration m012 adds prompt/completion token columns to check_results."""
        columns = {row[1] for row in db_conn.execute("PRAGMA table_info(check_results)").fetchall()}
        assert "prompt_tokens" in columns
        assert "completion_tokens" in columns

    def test_topic_init_attempts_column_exists(self, db_conn: sqlite3.Connection) -> None:
        """Migration m013 adds init_attempts column to topics."""
        columns = {row[1] for row in db_conn.execute("PRAGMA table_info(topics)").fetchall()}
        assert "init_attempts" in columns

    def test_check_result_stage_error_column_exists(self, db_conn: sqlite3.Connection) -> None:
        """Migration m015 adds nullable stage_error column to check_results."""
        columns = {row[1]: row for row in db_conn.execute("PRAGMA table_info(check_results)").fetchall()}
        assert "stage_error" in columns
        # Column is nullable (notnull flag, index 3, is 0).
        assert columns["stage_error"][3] == 0

    def test_check_result_stage_error_roundtrip(self, db_conn: sqlite3.Connection) -> None:
        """CheckResult.stage_error persists and loads back (None and a value)."""
        topic = create_topic(db_conn, Topic(name="StageErr", description="d"))
        db_conn.commit()

        with_err = create_check_result(
            db_conn,
            CheckResult(topic_id=topic.id, stage_error="knowledge_update_failed: boom"),
        )
        without_err = create_check_result(db_conn, CheckResult(topic_id=topic.id))
        db_conn.commit()

        loaded_err = get_check_result(db_conn, with_err.id)
        loaded_none = get_check_result(db_conn, without_err.id)
        assert loaded_err is not None
        assert loaded_err.stage_error == "knowledge_update_failed: boom"
        assert loaded_none is not None
        assert loaded_none.stage_error is None

    def test_pending_claimed_at_columns_exist(self, db_conn: sqlite3.Connection) -> None:
        """Migration m016 adds nullable claimed_at to both retry queues."""
        for table in ("pending_notifications", "pending_webhooks"):
            columns = {row[1]: row for row in db_conn.execute(f"PRAGMA table_info({table})").fetchall()}
            assert "claimed_at" in columns, f"{table} missing claimed_at"
            # Column is nullable (notnull flag, index 3, is 0).
            assert columns["claimed_at"][3] == 0

    def test_pending_notification_url_last_error_columns_exist(self, db_conn: sqlite3.Connection) -> None:
        """Migration m017 adds nullable url + last_error to pending_notifications."""
        columns = {row[1]: row for row in db_conn.execute("PRAGMA table_info(pending_notifications)").fetchall()}
        for col in ("url", "last_error"):
            assert col in columns, f"pending_notifications missing {col}"
            # Column is nullable (notnull flag, index 3, is 0).
            assert columns[col][3] == 0

    def test_pending_notification_url_last_error_roundtrip(self, db_conn: sqlite3.Connection) -> None:
        """url + last_error persist and load back (None and a value)."""
        topic = create_topic(db_conn, Topic(name="NotifUrl", description="d"))
        db_conn.commit()

        scoped = create_pending_notification(
            db_conn,
            PendingNotification(topic_id=topic.id, title="T", body="B", url="json://b", last_error="HTTP 500"),
        )
        legacy = create_pending_notification(db_conn, PendingNotification(topic_id=topic.id, title="T2", body="B2"))
        db_conn.commit()

        rows = {r.id: r for r in list_pending_notifications(db_conn)}
        assert rows[scoped.id].url == "json://b"
        assert rows[scoped.id].last_error == "HTTP 500"
        assert rows[legacy.id].url is None
        assert rows[legacy.id].last_error is None

    def test_topic_threshold_roundtrip(self, db_conn: sqlite3.Connection) -> None:
        """Per-topic thresholds and init_attempts persist and load back."""
        topic = create_topic(
            db_conn,
            Topic(
                name="Thresholds",
                description="d",
                confidence_threshold=0.9,
                relevance_threshold=0.5,
                init_attempts=2,
            ),
        )
        db_conn.commit()
        loaded = get_topic(db_conn, topic.id)
        assert loaded is not None
        assert loaded.confidence_threshold == 0.9
        assert loaded.relevance_threshold == 0.5
        assert loaded.init_attempts == 2

    def test_check_result_token_roundtrip(self, db_conn: sqlite3.Connection) -> None:
        """CheckResult token columns persist and load back."""
        topic = create_topic(db_conn, Topic(name="Tok", description="d"))
        db_conn.commit()
        result = create_check_result(
            db_conn,
            CheckResult(topic_id=topic.id, prompt_tokens=123, completion_tokens=45),
        )
        db_conn.commit()
        loaded = get_check_result(db_conn, result.id)
        assert loaded is not None
        assert loaded.prompt_tokens == 123
        assert loaded.completion_tokens == 45

    def test_perf_indexes_exist(self, db_conn: sqlite3.Connection) -> None:
        """Migration m014 adds performance indexes on the articles table."""
        index_names = {
            row[0]
            for row in db_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='articles'"
            ).fetchall()
        }
        assert "idx_articles_content_hash_lookup" in index_names
        assert "idx_articles_fetched_at" in index_names
        assert "idx_articles_topic_fetched_at" in index_names

    def test_perf_index_content_hash_used(self, db_conn: sqlite3.Connection) -> None:
        """content_hash lookup uses an index (SEARCH not SCAN)."""
        plan = db_conn.execute(
            "EXPLAIN QUERY PLAN SELECT topic_id, raw_content FROM articles WHERE content_hash = ?",
            ("abc",),
        ).fetchall()
        detail = " ".join(str(row[-1]) for row in plan)
        assert "USING INDEX" in detail
        assert "SCAN articles" not in detail

    def test_perf_index_topic_fetched_at_used(self, db_conn: sqlite3.Connection) -> None:
        """topic-scoped fetched_at ORDER BY is index-ordered (no temp B-tree)."""
        plan = db_conn.execute(
            "EXPLAIN QUERY PLAN SELECT * FROM articles WHERE topic_id = ? ORDER BY fetched_at DESC LIMIT 10",
            (1,),
        ).fetchall()
        detail = " ".join(str(row[-1]) for row in plan)
        assert "USING INDEX" in detail
        assert "USE TEMP B-TREE" not in detail

    def test_perf_indexes_idempotent(self, db_conn: sqlite3.Connection) -> None:
        """Re-running migrations does not error and keeps the indexes present."""
        run_migrations(db_conn)
        run_migrations(db_conn)
        index_names = {
            row[0]
            for row in db_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='articles'"
            ).fetchall()
        }
        assert "idx_articles_content_hash_lookup" in index_names
        assert "idx_articles_fetched_at" in index_names
        assert "idx_articles_topic_fetched_at" in index_names

    def test_failing_migration_logs_version_and_backup_then_reraises(self, tmp_path, caplog, monkeypatch) -> None:
        """OVH-047: a failing migration logs its version + backup path before re-raising."""
        import app.migrations as migrations_mod
        from app.database import get_connection, init_db

        db_path = tmp_path / "fail.db"
        init_db(db_path)  # establish an existing DB so a backup is created

        def _boom(_conn: sqlite3.Connection) -> None:
            raise ValueError("simulated migration failure")

        # Inject a pending migration with a version above any real one.
        # run_migrations imports MIGRATIONS from app.migrations at call time.
        bad_version = 9999
        monkeypatch.setattr(migrations_mod, "MIGRATIONS", [(bad_version, "intentionally broken", _boom)])

        conn = get_connection(db_path)
        try:
            with caplog.at_level("ERROR"), pytest.raises(ValueError, match="simulated migration failure"):
                run_migrations(conn, db_path=db_path)
        finally:
            conn.close()

        records = [r for r in caplog.records if r.levelname == "ERROR"]
        assert records, "Expected an ERROR log for the failed migration"
        msg = records[-1].getMessage()
        assert str(bad_version) in msg
        assert "intentionally broken" in msg
        # The backup path must be referenced (backups live under data/backups dir).
        assert "backup" in msg.lower()
        # The migration version must NOT have been recorded as applied.
        conn_after = get_connection(db_path)
        try:
            applied = conn_after.execute(
                "SELECT version FROM schema_version WHERE version=?", (bad_version,)
            ).fetchone()
        finally:
            conn_after.close()
        assert applied is None

    def test_partial_failure_commits_prior_migrations_and_resumes(self, tmp_path, monkeypatch) -> None:
        """OVH-060: a crash mid-sequence durably records the migrations that DID succeed,
        and a re-run resumes from there without re-running already-applied migrations."""
        import app.migrations as migrations_mod
        from app.database import get_connection, init_db

        db_path = tmp_path / "partial.db"
        init_db(db_path)  # establish an existing DB at the real head version

        good_version = 9001
        bad_version = 9002

        good_calls: list[int] = []

        def _good(conn: sqlite3.Connection) -> None:
            good_calls.append(1)
            conn.execute("CREATE TABLE IF NOT EXISTS ovh060_marker (id INTEGER PRIMARY KEY)")

        def _boom(_conn: sqlite3.Connection) -> None:
            raise ValueError("simulated mid-sequence migration failure")

        # First migration succeeds, second fails — the second must NOT undo the first.
        monkeypatch.setattr(
            migrations_mod,
            "MIGRATIONS",
            [
                (good_version, "good migration", _good),
                (bad_version, "broken migration", _boom),
            ],
        )

        conn = get_connection(db_path)
        try:
            with pytest.raises(ValueError, match="simulated mid-sequence migration failure"):
                run_migrations(conn, db_path=db_path)
        finally:
            conn.close()

        assert good_calls == [1], "good migration should have run exactly once"

        # The good migration's progress must be durable on a FRESH connection
        # (proves it was committed, not merely buffered on the failed connection).
        conn_check = get_connection(db_path)
        try:
            good_recorded = conn_check.execute(
                "SELECT version FROM schema_version WHERE version=?", (good_version,)
            ).fetchone()
            bad_recorded = conn_check.execute(
                "SELECT version FROM schema_version WHERE version=?", (bad_version,)
            ).fetchone()
            marker = conn_check.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='ovh060_marker'"
            ).fetchone()
        finally:
            conn_check.close()
        assert good_recorded is not None, "good migration version must be durably committed"
        assert bad_recorded is None, "failed migration version must not be recorded"
        assert marker is not None, "good migration's DDL must be durable"

        # Re-run with the second migration now fixed: the good one must NOT re-run.
        good_calls.clear()
        second_calls: list[int] = []

        def _now_fixed(conn: sqlite3.Connection) -> None:
            second_calls.append(1)
            conn.execute("CREATE TABLE IF NOT EXISTS ovh060_marker2 (id INTEGER PRIMARY KEY)")

        monkeypatch.setattr(
            migrations_mod,
            "MIGRATIONS",
            [
                (good_version, "good migration", _good),
                (bad_version, "now fixed migration", _now_fixed),
            ],
        )

        conn2 = get_connection(db_path)
        try:
            run_migrations(conn2, db_path=db_path)
        finally:
            conn2.close()

        assert good_calls == [], "already-applied migration must not re-run on resume"
        assert second_calls == [1], "the previously-failed migration must run on resume"

        conn_final = get_connection(db_path)
        try:
            applied = {
                r[0]
                for r in conn_final.execute(
                    "SELECT version FROM schema_version WHERE version IN (?, ?)",
                    (good_version, bad_version),
                ).fetchall()
            }
        finally:
            conn_final.close()
        assert applied == {good_version, bad_version}


class TestRecoverStuckTopics:
    """Tests for recover_stuck_topics."""

    def test_marks_researching_as_error(self, db_conn: sqlite3.Connection) -> None:
        """RESEARCHING topics are marked as ERROR."""
        topic = create_topic(
            db_conn,
            Topic(name="Stuck", description="desc", status=TopicStatus.RESEARCHING),
        )
        db_conn.commit()

        count = recover_stuck_topics(db_conn)
        assert count == 1

        updated = get_topic(db_conn, topic.id)
        assert updated.status == TopicStatus.ERROR
        assert "restart" in updated.error_message.lower()

    def test_does_not_affect_ready_topics(self, db_conn: sqlite3.Connection) -> None:
        """READY topics are not affected."""
        topic = create_topic(
            db_conn,
            Topic(name="Ready", description="desc", status=TopicStatus.READY),
        )
        db_conn.commit()

        count = recover_stuck_topics(db_conn)
        assert count == 0

        updated = get_topic(db_conn, topic.id)
        assert updated.status == TopicStatus.READY

    def test_does_not_affect_error_topics(self, db_conn: sqlite3.Connection) -> None:
        """ERROR topics are not affected."""
        topic = create_topic(
            db_conn,
            Topic(
                name="Errored",
                description="desc",
                status=TopicStatus.ERROR,
                error_message="Previous error",
            ),
        )
        db_conn.commit()

        count = recover_stuck_topics(db_conn)
        assert count == 0

        updated = get_topic(db_conn, topic.id)
        assert updated.error_message == "Previous error"


class TestPendingNotificationCRUD:
    """Test CRUD operations for pending notifications."""

    def _make_topic_and_check(self, conn: sqlite3.Connection) -> tuple[Topic, CheckResult]:
        topic = create_topic(conn, Topic(name="NotifTopic", description="d"))
        cr = create_check_result(conn, CheckResult(topic_id=topic.id))
        conn.commit()
        return topic, cr

    def test_create_and_list(self, db_conn: sqlite3.Connection) -> None:
        topic, cr = self._make_topic_and_check(db_conn)
        notif = PendingNotification(
            topic_id=topic.id,
            check_result_id=cr.id,
            title="Test Title",
            body="Test Body",
        )
        created = create_pending_notification(db_conn, notif)
        db_conn.commit()
        assert created.id is not None

        pending = list_pending_notifications(db_conn)
        assert len(pending) == 1
        assert pending[0].title == "Test Title"
        assert pending[0].body == "Test Body"
        assert pending[0].retry_count == 0

    def test_list_excludes_maxed_out_retries(self, db_conn: sqlite3.Connection) -> None:
        """Notifications at max retries are excluded from the pending list."""
        topic, cr = self._make_topic_and_check(db_conn)
        create_pending_notification(
            db_conn,
            PendingNotification(topic_id=topic.id, title="Fresh", body="B", retry_count=0),
        )
        create_pending_notification(
            db_conn,
            PendingNotification(
                topic_id=topic.id,
                title="Exhausted",
                body="B",
                retry_count=3,
                max_retries=3,
            ),
        )
        db_conn.commit()

        pending = list_pending_notifications(db_conn)
        assert len(pending) == 1
        assert pending[0].title == "Fresh"

    def test_increment_retry(self, db_conn: sqlite3.Connection) -> None:
        topic, _ = self._make_topic_and_check(db_conn)
        notif = create_pending_notification(
            db_conn,
            PendingNotification(topic_id=topic.id, title="T", body="B"),
        )
        db_conn.commit()

        increment_notification_retry(db_conn, notif.id)
        db_conn.commit()

        pending = list_pending_notifications(db_conn)
        assert pending[0].retry_count == 1

    def test_delete(self, db_conn: sqlite3.Connection) -> None:
        topic, _ = self._make_topic_and_check(db_conn)
        notif = create_pending_notification(
            db_conn,
            PendingNotification(topic_id=topic.id, title="T", body="B"),
        )
        db_conn.commit()

        delete_pending_notification(db_conn, notif.id)
        db_conn.commit()

        assert list_pending_notifications(db_conn) == []

    def test_delete_expired(self, db_conn: sqlite3.Connection) -> None:
        """delete_expired_notifications removes only maxed-out entries."""
        topic, _ = self._make_topic_and_check(db_conn)
        create_pending_notification(
            db_conn,
            PendingNotification(topic_id=topic.id, title="Active", body="B", retry_count=1),
        )
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

        deleted = delete_expired_notifications(db_conn)
        db_conn.commit()
        # Returns the abandoned rows (not just a count) so the prune site can
        # log what was permanently dropped (OVH-040).
        assert len(deleted) == 1
        assert deleted[0].title == "Expired"
        assert deleted[0].topic_id == topic.id

        # Only the active one remains
        remaining = list_pending_notifications(db_conn)
        assert len(remaining) == 1
        assert remaining[0].title == "Active"

    def test_cascade_deletes_with_topic(self, db_conn: sqlite3.Connection) -> None:
        """Deleting a topic cascades to its pending notifications."""
        topic, _ = self._make_topic_and_check(db_conn)
        create_pending_notification(
            db_conn,
            PendingNotification(topic_id=topic.id, title="T", body="B"),
        )
        db_conn.commit()

        delete_topic(db_conn, topic.id)
        db_conn.commit()

        assert list_pending_notifications(db_conn) == []


class TestGetDashboardData:
    """Test the aggregate dashboard query."""

    def test_empty_returns_empty(self, db_conn: sqlite3.Connection) -> None:
        assert get_dashboard_data(db_conn) == []

    def test_topic_with_no_checks(self, db_conn: sqlite3.Connection) -> None:
        create_topic(db_conn, Topic(name="NoChecks", description="d"))
        db_conn.commit()

        data = get_dashboard_data(db_conn)
        assert len(data) == 1
        assert data[0]["topic"].name == "NoChecks"
        assert data[0]["last_check"] is None
        assert data[0]["article_count"] == 0

    def test_topic_with_check_and_articles(self, db_conn: sqlite3.Connection) -> None:
        topic = create_topic(db_conn, Topic(name="WithData", description="d"))
        create_article(
            db_conn,
            Article(
                topic_id=topic.id,
                title="Art1",
                url="url1",
                content_hash="h1",
                source_feed="f",
            ),
        )
        create_article(
            db_conn,
            Article(
                topic_id=topic.id,
                title="Art2",
                url="url2",
                content_hash="h2",
                source_feed="f",
            ),
        )
        create_check_result(
            db_conn,
            CheckResult(topic_id=topic.id, articles_found=5, has_new_info=True),
        )
        db_conn.commit()

        data = get_dashboard_data(db_conn)
        assert len(data) == 1
        assert data[0]["article_count"] == 2
        assert data[0]["last_check"] is not None
        assert data[0]["last_check"].articles_found == 5
        assert data[0]["last_check"].has_new_info is True

    def test_returns_only_latest_check(self, db_conn: sqlite3.Connection) -> None:
        """When multiple checks exist, only the most recent is returned."""
        topic = create_topic(db_conn, Topic(name="Multi", description="d"))
        now = datetime.now(UTC)
        create_check_result(
            db_conn,
            CheckResult(
                topic_id=topic.id,
                checked_at=now - timedelta(hours=2),
                articles_found=3,
            ),
        )
        create_check_result(
            db_conn,
            CheckResult(
                topic_id=topic.id,
                checked_at=now,
                articles_found=7,
            ),
        )
        db_conn.commit()

        data = get_dashboard_data(db_conn)
        assert data[0]["last_check"].articles_found == 7

    def test_multiple_topics_sorted_by_name(self, db_conn: sqlite3.Connection) -> None:
        create_topic(db_conn, Topic(name="Zeta", description="d"))
        create_topic(db_conn, Topic(name="Alpha", description="d"))
        db_conn.commit()

        data = get_dashboard_data(db_conn)
        assert len(data) == 2
        assert data[0]["topic"].name == "Alpha"
        assert data[1]["topic"].name == "Zeta"


class TestGetNewTopics:
    """Tests for get_new_topics (OPML gradual init)."""

    def test_returns_new_topics_oldest_first(self, db_conn: sqlite3.Connection) -> None:
        from app.crud import get_new_topics

        t1 = Topic(name="First", description="d", status=TopicStatus.NEW)
        t2 = Topic(name="Second", description="d", status=TopicStatus.NEW)
        create_topic(db_conn, t1)
        create_topic(db_conn, t2)
        db_conn.commit()

        result = get_new_topics(db_conn, limit=1)
        assert len(result) == 1
        assert result[0].name == "First"

    def test_ignores_non_new_topics(self, db_conn: sqlite3.Connection) -> None:
        from app.crud import get_new_topics

        create_topic(db_conn, Topic(name="Ready", description="d", status=TopicStatus.READY))
        create_topic(db_conn, Topic(name="New", description="d", status=TopicStatus.NEW))
        db_conn.commit()

        result = get_new_topics(db_conn, limit=10)
        assert len(result) == 1
        assert result[0].name == "New"

    def test_empty_when_no_new_topics(self, db_conn: sqlite3.Connection) -> None:
        from app.crud import get_new_topics

        create_topic(db_conn, Topic(name="Ready", description="d", status=TopicStatus.READY))
        db_conn.commit()

        result = get_new_topics(db_conn)
        assert result == []


class TestGetAllFeedUrls:
    """Tests for get_all_feed_urls (OPML dedup)."""

    def test_returns_all_feed_urls(self, db_conn: sqlite3.Connection) -> None:
        from app.crud import get_all_feed_urls

        create_topic(
            db_conn,
            Topic(
                name="T1",
                description="d",
                feed_urls=["https://a.com/feed", "https://b.com/feed"],
            ),
        )
        create_topic(
            db_conn,
            Topic(
                name="T2",
                description="d",
                feed_urls=["https://c.com/feed"],
            ),
        )
        db_conn.commit()

        urls = get_all_feed_urls(db_conn)
        assert urls == {"https://a.com/feed", "https://b.com/feed", "https://c.com/feed"}

    def test_empty_when_no_topics(self, db_conn: sqlite3.Connection) -> None:
        from app.crud import get_all_feed_urls

        urls = get_all_feed_urls(db_conn)
        assert urls == set()


class TestGetAllTopicNames:
    """Tests for get_all_topic_names (OPML name-collision dedup)."""

    def test_returns_all_topic_names(self, db_conn: sqlite3.Connection) -> None:
        from app.crud import get_all_topic_names

        create_topic(db_conn, Topic(name="Alpha", description="d", feed_urls=["https://a.com/feed"]))
        create_topic(db_conn, Topic(name="Beta", description="d", feed_urls=["https://b.com/feed"]))
        db_conn.commit()

        assert get_all_topic_names(db_conn) == {"Alpha", "Beta"}

    def test_empty_when_no_topics(self, db_conn: sqlite3.Connection) -> None:
        from app.crud import get_all_topic_names

        assert get_all_topic_names(db_conn) == set()


class TestGetDashboardStats:
    """Tests for get_dashboard_stats."""

    def test_stats_with_no_data(self, db_conn: sqlite3.Connection) -> None:
        from app.crud import get_dashboard_stats

        stats = get_dashboard_stats(db_conn)
        assert stats.total_topics == 0
        assert stats.active_topics == 0
        assert stats.checks_24h == 0
        assert stats.checks_total == 0
        assert stats.new_info_24h == 0
        assert stats.new_info_total == 0
        assert stats.last_notification_at is None

    def test_stats_with_data(self, db_conn: sqlite3.Connection) -> None:
        from app.crud import get_dashboard_stats

        t1 = create_topic(db_conn, Topic(name="Active", description="d", status=TopicStatus.READY))
        create_topic(db_conn, Topic(name="Inactive", description="d", status=TopicStatus.READY, is_active=False))
        create_check_result(
            db_conn, CheckResult(topic_id=t1.id, articles_found=5, has_new_info=True, notification_sent=True)
        )
        create_check_result(db_conn, CheckResult(topic_id=t1.id, articles_found=3, has_new_info=False))
        db_conn.commit()

        stats = get_dashboard_stats(db_conn)
        assert stats.total_topics == 2
        assert stats.active_topics == 1
        assert stats.checks_total == 2
        assert stats.new_info_total == 1
        assert stats.last_notification_at is not None

    def test_24h_window_excludes_25h_old_check(self, db_conn: sqlite3.Connection) -> None:
        """OVH-021: a 25h-old check must be excluded from the 24h window.

        ``checked_at`` is stored as timezone-aware ISO (``T``/``+00:00``); a raw
        string compare against SQLite's space-separated ``datetime('now', ...)``
        over-counts rows beyond the intended window. Wrapping the column in
        ``datetime()`` makes the boundary correct.
        """
        from app.crud import get_dashboard_stats

        t1 = create_topic(db_conn, Topic(name="T", description="d", status=TopicStatus.READY))
        now = datetime.now(UTC)

        # Inside the window: 1h old, with new info.
        create_check_result(
            db_conn,
            CheckResult(topic_id=t1.id, checked_at=now - timedelta(hours=1), has_new_info=True),
        )
        # Outside the window: 25h old, with new info — must NOT be counted.
        create_check_result(
            db_conn,
            CheckResult(topic_id=t1.id, checked_at=now - timedelta(hours=25), has_new_info=True),
        )
        db_conn.commit()

        stats = get_dashboard_stats(db_conn)
        assert stats.checks_total == 2
        assert stats.new_info_total == 2
        assert stats.checks_24h == 1
        assert stats.new_info_24h == 1
