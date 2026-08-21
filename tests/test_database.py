"""Tests for database operations: schema, CRUD, and dedup."""

import sqlite3
from datetime import UTC, datetime, timedelta, timezone

import pytest

from app.crud import (
    abandon_expired_notifications,
    apply_notification_outcome,
    article_hash_exists,
    claim_notification_intent,
    create_article,
    create_check_result,
    create_knowledge_state,
    create_pending_notification,
    create_topic,
    delete_topic,
    get_article,
    get_check_result,
    get_dashboard_data,
    get_knowledge_state,
    get_topic,
    get_topic_by_name,
    list_articles_for_topic,
    list_check_results,
    list_pending_notifications,
    list_topics,
    mark_articles_processed,
    recover_stuck_topics,
    topic_generation_matches,
    update_knowledge_state,
    update_knowledge_state_cas,
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


class TestHeartbeatLatch:
    """The Silence Heartbeat latch is claimed/cleared exactly once, and survives edits."""

    def test_heartbeat_latch_claim_is_exactly_once(self, db_conn: sqlite3.Connection) -> None:
        from app.crud import claim_heartbeat_alert, clear_heartbeat_alert

        topic = create_topic(db_conn, Topic(name="Latch", description="d"))
        db_conn.commit()
        stamp = datetime(2026, 8, 3, 10, 0, tzinfo=UTC)

        assert claim_heartbeat_alert(db_conn, topic.id, stamp) is True
        assert claim_heartbeat_alert(db_conn, topic.id, stamp) is False  # second caller loses
        assert get_topic(db_conn, topic.id).heartbeat_alerted_at == stamp

        assert clear_heartbeat_alert(db_conn, topic.id) is True
        assert clear_heartbeat_alert(db_conn, topic.id) is False  # nothing left to clear
        assert get_topic(db_conn, topic.id).heartbeat_alerted_at is None

    def test_update_topic_leaves_the_heartbeat_latch_alone(self, db_conn: sqlite3.Connection) -> None:
        from app.crud import claim_heartbeat_alert

        topic = create_topic(db_conn, Topic(name="LatchEdit", description="d"))
        db_conn.commit()
        stale = get_topic(db_conn, topic.id)  # loaded before the alert fired

        claim_heartbeat_alert(db_conn, topic.id, datetime(2026, 8, 3, 10, 0, tzinfo=UTC))
        stale.description = "edited"
        update_topic(db_conn, stale)
        db_conn.commit()

        assert get_topic(db_conn, topic.id).heartbeat_alerted_at is not None


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


class TestKnowledgeStateCAS:
    """update_knowledge_state_cas rejects a write built on a stale snapshot."""

    def _state(self, conn: sqlite3.Connection) -> tuple[int, int]:
        topic = create_topic(conn, Topic(name="CAS", description="d"))
        assert topic.id is not None
        create_knowledge_state(conn, KnowledgeState(topic_id=topic.id, summary_text="v0"))
        conn.commit()
        stored = get_knowledge_state(conn, topic.id)
        assert stored is not None
        return topic.id, stored.version

    def test_matching_version_writes_and_bumps(self, db_conn: sqlite3.Connection) -> None:
        topic_id, version = self._state(db_conn)
        assert update_knowledge_state_cas(db_conn, topic_id, summary_text="v1", token_count=7, expected_version=version)
        db_conn.commit()
        stored = get_knowledge_state(db_conn, topic_id)
        assert stored is not None
        assert stored.summary_text == "v1"
        assert stored.token_count == 7
        assert stored.version == version + 1

    def test_stale_version_is_rejected_and_leaves_the_winner_intact(self, db_conn: sqlite3.Connection) -> None:
        topic_id, version = self._state(db_conn)
        # Winner writes first; the loser still holds the pre-write version.
        assert update_knowledge_state_cas(
            db_conn, topic_id, summary_text="winner", token_count=1, expected_version=version
        )
        assert not update_knowledge_state_cas(
            db_conn, topic_id, summary_text="loser", token_count=2, expected_version=version
        )
        db_conn.commit()
        stored = get_knowledge_state(db_conn, topic_id)
        assert stored is not None
        assert stored.summary_text == "winner"

    def test_updated_at_is_written_as_canonical_utc(self, db_conn: sqlite3.Connection) -> None:
        topic_id, version = self._state(db_conn)
        local = datetime(2026, 8, 20, 14, 0, tzinfo=timezone(timedelta(hours=3)))
        assert update_knowledge_state_cas(
            db_conn, topic_id, summary_text="v1", token_count=1, expected_version=version, updated_at=local
        )
        db_conn.commit()
        raw = db_conn.execute("SELECT updated_at FROM knowledge_states WHERE topic_id = ?", (topic_id,)).fetchone()[0]
        assert raw == "2026-08-20T11:00:00+00:00"


class TestTopicGenerationFence:
    """topic_generation_matches is the fence for post-await durable writes."""

    def test_matches_the_live_row(self, db_conn: sqlite3.Connection) -> None:
        topic = create_topic(db_conn, Topic(name="Fenced", description="d"))
        db_conn.commit()
        assert topic.id is not None
        assert topic_generation_matches(db_conn, topic.id, topic.generation)

    def test_recycled_rowid_does_not_match_the_old_generation(self, db_conn: sqlite3.Connection) -> None:
        topic = create_topic(db_conn, Topic(name="Original", description="d"))
        db_conn.commit()
        assert topic.id is not None
        original_id, original_generation = topic.id, topic.generation

        delete_topic(db_conn, original_id)
        db_conn.commit()
        # Force the replacement onto the freed rowid, the situation the fence exists for.
        db_conn.execute(
            "INSERT INTO topics (id, name, description, created_at, generation) VALUES (?, ?, ?, ?, ?)",
            (original_id, "Replacement", "d", datetime.now(UTC).isoformat(), "replacement-generation"),
        )
        db_conn.commit()

        assert not topic_generation_matches(db_conn, original_id, original_generation)
        assert topic_generation_matches(db_conn, original_id, "replacement-generation")

    def test_blank_generation_fails_closed(self, db_conn: sqlite3.Connection) -> None:
        topic = create_topic(db_conn, Topic(name="Blank", description="d"))
        db_conn.commit()
        assert topic.id is not None
        db_conn.execute("UPDATE topics SET generation = '' WHERE id = ?", (topic.id,))
        db_conn.commit()
        assert not topic_generation_matches(db_conn, topic.id, "")

    def test_missing_topic_does_not_match(self, db_conn: sqlite3.Connection) -> None:
        assert not topic_generation_matches(db_conn, 999_999, "anything")


class TestCheckResultCRUD:
    """Test CRUD operations for check results."""

    def test_list_recent_check_stage_errors_orders_newest_first(self, db_conn: sqlite3.Connection) -> None:
        from app.crud import list_recent_check_stage_errors

        topic = create_topic(db_conn, Topic(name="Streak", description="d"))
        stamp = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)
        # Identical timestamps: ordering must fall back to id DESC, not scan order.
        for stage_error in ("sources_failed: a", None, "sources_failed: b"):
            create_check_result(db_conn, CheckResult(topic_id=topic.id, checked_at=stamp, stage_error=stage_error))
        db_conn.commit()

        assert list_recent_check_stage_errors(db_conn, topic.id, limit=10) == [
            "sources_failed: b",
            None,
            "sources_failed: a",
        ]
        assert list_recent_check_stage_errors(db_conn, topic.id, limit=2) == ["sources_failed: b", None]
        assert list_recent_check_stage_errors(db_conn, 9999, limit=10) == []

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

    def test_migration_024_adds_heartbeat_column_and_is_idempotent(self, db_conn: sqlite3.Connection) -> None:
        """m024 adds the latch column; re-running it is a no-op that keeps data."""
        from app.migrations.m024_topic_heartbeat_alerted_at import up as m024_up

        columns = {row[1] for row in db_conn.execute("PRAGMA table_info(topics)").fetchall()}
        assert "heartbeat_alerted_at" in columns

        create_topic(db_conn, Topic(name="Survivor", description="d"))
        db_conn.commit()
        m024_up(db_conn)
        assert db_conn.execute("SELECT COUNT(*) FROM topics WHERE name = 'Survivor'").fetchone()[0] == 1

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

    def test_migration_027_canonicalizes_stored_tags(self, db_conn: sqlite3.Connection) -> None:
        """AUG-338: rows written before the validator existed are repaired once.

        The SQL tag filter matches stored JSON exactly, so a variant left behind
        would be permanently unselectable from a chip or ``?tag=``.
        """
        import json

        from app.migrations.m027_normalize_topic_tags import up as m027_up

        topic = create_topic(db_conn, Topic(name="Legacy Tags", description="d"))
        db_conn.execute(
            "UPDATE topics SET tags = ? WHERE id = ?",
            (json.dumps(["  Tech   News ", "Tech News", ""]), topic.id),
        )
        db_conn.commit()

        m027_up(db_conn)
        m027_up(db_conn)  # idempotent

        stored = db_conn.execute("SELECT tags FROM topics WHERE id = ?", (topic.id,)).fetchone()[0]
        assert json.loads(stored) == ["Tech News"]

    def test_migration_027_leaves_non_array_tags_alone(self, db_conn: sqlite3.Connection) -> None:
        """A hand-edited or future-version value must not be destroyed."""
        from app.migrations.m027_normalize_topic_tags import up as m027_up

        topic = create_topic(db_conn, Topic(name="Odd Tags", description="d"))
        db_conn.execute("UPDATE topics SET tags = ? WHERE id = ?", ('{"a": 1}', topic.id))
        db_conn.commit()

        m027_up(db_conn)

        assert db_conn.execute("SELECT tags FROM topics WHERE id = ?", (topic.id,)).fetchone()[0] == '{"a": 1}'

    def test_migration_026_adds_every_wave_a_column(self, db_conn: sqlite3.Connection) -> None:
        """m026 lands the whole wave-A schema in one pass."""
        expected = {
            "knowledge_states": {"version"},
            "topics": {"generation"},
            "articles": {"analysis_attempts"},
            "check_results": {"notify_disposition"},
            "pending_notifications": {
                "status",
                "kind",
                "claim_token",
                "next_attempt_at",
                "latch_value",
                "delivered_at",
            },
            "pending_webhooks": {"status", "claim_token", "next_attempt_at", "last_error", "delivered_at"},
        }
        for table, columns in expected.items():
            actual = {row[1] for row in db_conn.execute(f"PRAGMA table_info({table})").fetchall()}
            assert columns <= actual, f"{table} missing {columns - actual}"

        indexes = {row[0] for row in db_conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()}
        assert "idx_pending_notifications_due" in indexes
        assert "idx_pending_webhooks_due" in indexes

    def test_migration_026_is_idempotent_and_backfills_generation(self, db_conn: sqlite3.Connection) -> None:
        """Re-running m026 keeps data, and every topic row carries a distinct generation."""
        from app.migrations.m026_wave_a_durability import up as m026_up

        create_topic(db_conn, Topic(name="Gen A", description="d"))
        create_topic(db_conn, Topic(name="Gen B", description="d"))
        # Simulate pre-migration rows whose generation the backfill must mint.
        db_conn.execute("UPDATE topics SET generation = ''")
        db_conn.commit()

        m026_up(db_conn)
        m026_up(db_conn)

        generations = [r[0] for r in db_conn.execute("SELECT generation FROM topics").fetchall()]
        assert len(generations) == 2
        assert all(g for g in generations)
        assert len(set(generations)) == 2

    def test_migration_026_defaults_existing_intents_to_pending(self, db_conn: sqlite3.Connection) -> None:
        """Rows queued before the upgrade are undelivered, so 'pending' is correct."""
        topic = create_topic(db_conn, Topic(name="Queued", description="d"))
        assert topic.id is not None
        db_conn.execute(
            "INSERT INTO pending_notifications (topic_id, title, body, created_at) VALUES (?, ?, ?, ?)",
            (topic.id, "t", "b", datetime.now(UTC).isoformat()),
        )
        db_conn.commit()
        row = db_conn.execute("SELECT status, kind FROM pending_notifications").fetchone()
        assert row["status"] == "pending"
        assert row["kind"] == "novelty"

    def test_topic_novelty_instruction_column_exists(self, db_conn: sqlite3.Connection) -> None:
        """Migration m022 adds the nullable novelty_instruction column."""
        columns = {row[1] for row in db_conn.execute("PRAGMA table_info(topics)").fetchall()}
        assert "novelty_instruction" in columns

    def test_topic_importance_threshold_column_exists(self, db_conn: sqlite3.Connection) -> None:
        """Migration m023 adds the nullable importance_threshold column."""
        columns = {row[1] for row in db_conn.execute("PRAGMA table_info(topics)").fetchall()}
        assert "importance_threshold" in columns

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

    def test_topic_novelty_instruction_roundtrip(self, db_conn: sqlite3.Connection) -> None:
        """The per-topic novelty instruction persists on create and update."""
        topic = create_topic(
            db_conn,
            Topic(name="Instructed", description="d", novelty_instruction="Only official announcements."),
        )
        db_conn.commit()
        loaded = get_topic(db_conn, topic.id)
        assert loaded is not None
        assert loaded.novelty_instruction == "Only official announcements."

        loaded.novelty_instruction = None
        update_topic(db_conn, loaded)
        db_conn.commit()
        reloaded = get_topic(db_conn, topic.id)
        assert reloaded is not None
        assert reloaded.novelty_instruction is None

    def test_topic_importance_threshold_roundtrip(self, db_conn: sqlite3.Connection) -> None:
        """The per-topic importance threshold persists on create and update."""
        topic = create_topic(db_conn, Topic(name="Important", description="d", importance_threshold=4))
        db_conn.commit()
        loaded = get_topic(db_conn, topic.id)
        assert loaded is not None
        assert loaded.importance_threshold == 4

        loaded.importance_threshold = None
        update_topic(db_conn, loaded)
        db_conn.commit()
        reloaded = get_topic(db_conn, topic.id)
        assert reloaded is not None
        assert reloaded.importance_threshold is None

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

    def test_article_published_at_column_exists(self, db_conn: sqlite3.Connection) -> None:
        """Migration m018 adds nullable published_at column to articles."""
        columns = {row[1]: row for row in db_conn.execute("PRAGMA table_info(articles)").fetchall()}
        assert "published_at" in columns, "articles missing published_at"
        # Column is nullable (notnull flag, index 3, is 0).
        assert columns["published_at"][3] == 0

    def test_article_published_at_roundtrip(self, db_conn: sqlite3.Connection) -> None:
        """published_at persists and loads back (None and an ISO timestamp)."""
        topic = create_topic(db_conn, Topic(name="PubAt", description="d"))
        db_conn.commit()

        ts = datetime(2025, 3, 10, 9, 0, 0, tzinfo=UTC)
        with_pub = create_article(
            db_conn,
            Article(
                topic_id=topic.id,
                title="With date",
                url="https://example.com/with-date",
                content_hash="hash-with",
                source_feed="https://feed.example.com/rss",
                published_at=ts,
            ),
        )
        without_pub = create_article(
            db_conn,
            Article(
                topic_id=topic.id,
                title="No date",
                url="https://example.com/no-date",
                content_hash="hash-none",
                source_feed="https://feed.example.com/rss",
            ),
        )
        db_conn.commit()

        loaded_with = get_article(db_conn, with_pub.id)
        loaded_without = get_article(db_conn, without_pub.id)
        assert loaded_with is not None
        assert loaded_with.published_at == ts
        assert loaded_without is not None
        assert loaded_without.published_at is None

    def test_m018_registered_in_migrations_list(self) -> None:
        """Version 18 is registered with the expected description."""
        from app.migrations import MIGRATIONS

        entry = next((m for m in MIGRATIONS if m[0] == 18), None)
        assert entry is not None, "m018 not found in MIGRATIONS"
        assert "published_at" in entry[1]

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
        real = list(migrations_mod.MIGRATIONS)

        def _boom(_conn: sqlite3.Connection) -> None:
            raise ValueError("simulated migration failure")

        # Append a pending migration with a version above any real one. The real
        # registry stays in place because the ledger validator (TW-AUD-011)
        # requires every applied version to be registered.
        # run_migrations imports MIGRATIONS from app.migrations at call time.
        bad_version = 9999
        monkeypatch.setattr(migrations_mod, "MIGRATIONS", [*real, (bad_version, "intentionally broken", _boom)])

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

    def test_failing_migration_does_not_claim_a_restore(self, tmp_path, caplog, monkeypatch) -> None:
        """The failure log must not say the DB was restored — nothing restores it.

        ``_backup_db`` only copies. On failure the connection is rolled back and
        the DB is left at the last committed version, which is correct, but the
        message told operators a rollback-to-backup had happened.
        """
        import app.migrations as migrations_mod
        from app.database import get_connection, init_db

        db_path = tmp_path / "noclaim.db"
        init_db(db_path)
        real = list(migrations_mod.MIGRATIONS)

        def _boom(_conn: sqlite3.Connection) -> None:
            raise ValueError("simulated migration failure")

        monkeypatch.setattr(migrations_mod, "MIGRATIONS", [*real, (9999, "intentionally broken", _boom)])

        conn = get_connection(db_path)
        try:
            with caplog.at_level("ERROR"), pytest.raises(ValueError):
                run_migrations(conn, db_path=db_path)
        finally:
            conn.close()

        msg = [r for r in caplog.records if r.levelname == "ERROR"][-1].getMessage()
        assert "restored" not in msg.lower(), f"log claims a restore that never happens: {msg!r}"

    def test_failing_migration_without_backup_reports_no_backup(self, tmp_path, caplog, monkeypatch) -> None:
        """With no backup taken, the log must say so rather than print 'None'.

        Reachable whenever run_migrations gets a db_path that does not exist —
        including the default-path form used by callers that pass no db_path.
        """
        import app.migrations as migrations_mod
        from app.database import get_connection

        conn_path = tmp_path / "live.db"
        missing_path = tmp_path / "not-created-yet.db"

        def _boom(_conn: sqlite3.Connection) -> None:
            raise ValueError("simulated migration failure")

        monkeypatch.setattr(migrations_mod, "MIGRATIONS", [(9999, "intentionally broken", _boom)])

        conn = get_connection(conn_path)
        try:
            with caplog.at_level("ERROR"), pytest.raises(ValueError):
                run_migrations(conn, db_path=missing_path)
        finally:
            conn.close()

        msg = [r for r in caplog.records if r.levelname == "ERROR"][-1].getMessage()
        assert "None" not in msg, f"log prints a null backup path: {msg!r}"
        assert "backup" in msg.lower()

    def test_partial_failure_commits_prior_migrations_and_resumes(self, tmp_path, monkeypatch) -> None:
        """OVH-060: a crash mid-sequence durably records the migrations that DID succeed,
        and a re-run resumes from there without re-running already-applied migrations."""
        import app.migrations as migrations_mod
        from app.database import get_connection, init_db

        db_path = tmp_path / "partial.db"
        init_db(db_path)  # establish an existing DB at the real head version
        real = list(migrations_mod.MIGRATIONS)

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
                *real,
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
                *real,
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

    def test_applies_in_version_order_regardless_of_list_order(self, tmp_path, monkeypatch) -> None:
        """OVH-109: an out-of-order MIGRATIONS list still applies in numeric order.

        The runner must sort pending migrations by version, so a future
        append-only migration inserted out of position cannot apply/record out of
        order (which would then silently skip a lower version, since current=MAX).
        """
        import app.migrations as migrations_mod
        from app.database import get_connection, init_db

        db_path = tmp_path / "order.db"
        init_db(db_path)  # establish an existing DB at the real head version
        real = list(migrations_mod.MIGRATIONS)

        applied_order: list[int] = []

        def _make(version: int):
            def _up(_conn: sqlite3.Connection) -> None:
                applied_order.append(version)

            return _up

        v_lo = 9101
        v_hi = 9102
        # Deliberately list the higher version FIRST.
        monkeypatch.setattr(
            migrations_mod,
            "MIGRATIONS",
            [
                *real,
                (v_hi, "higher version listed first", _make(v_hi)),
                (v_lo, "lower version listed second", _make(v_lo)),
            ],
        )

        conn = get_connection(db_path)
        try:
            run_migrations(conn, db_path=db_path)
        finally:
            conn.close()

        # Despite list order, the lower version must apply first.
        assert applied_order == [v_lo, v_hi]

        # Both must be recorded (no silent skip of the lower version).
        conn_check = get_connection(db_path)
        try:
            recorded = {
                r[0]
                for r in conn_check.execute(
                    "SELECT version FROM schema_version WHERE version IN (?, ?)",
                    (v_lo, v_hi),
                ).fetchall()
            }
        finally:
            conn_check.close()
        assert recorded == {v_lo, v_hi}


class TestMigrationAtomicity:
    """TW-AUD-010: a migration body and its ledger row are one transaction."""

    def test_ddl_does_not_survive_a_failing_migration_body(self, tmp_path, monkeypatch) -> None:
        """A migration that ALTERs and then raises must leave no schema change.

        SQLite DDL runs in autocommit unless a transaction is already open, so the
        ALTER used to persist while its version row never landed — the next start
        then re-ran the same ALTER against a column that already existed and failed
        with an unrecorded schema change on disk.
        """
        import app.migrations as migrations_mod
        from app.database import get_connection, init_db

        db_path = tmp_path / "atomic.db"
        init_db(db_path)
        real = list(migrations_mod.MIGRATIONS)

        def _ddl_then_boom(conn: sqlite3.Connection) -> None:
            conn.execute("ALTER TABLE topics ADD COLUMN tw_aud_010_probe TEXT")
            raise ValueError("simulated failure after DDL")

        monkeypatch.setattr(migrations_mod, "MIGRATIONS", [*real, (9201, "ddl then boom", _ddl_then_boom)])

        conn = get_connection(db_path)
        try:
            with pytest.raises(ValueError, match="simulated failure after DDL"):
                run_migrations(conn, db_path=db_path)
        finally:
            conn.close()

        after = get_connection(db_path)
        try:
            columns = {row[1] for row in after.execute("PRAGMA table_info(topics)").fetchall()}
            recorded = after.execute("SELECT version FROM schema_version WHERE version = 9201").fetchone()
        finally:
            after.close()
        assert "tw_aud_010_probe" not in columns, "DDL persisted without its ledger row"
        assert recorded is None

    def test_ddl_does_not_survive_a_failing_ledger_insert(self, tmp_path, monkeypatch) -> None:
        """The ledger INSERT failing must roll the migration body back with it."""
        import app.migrations as migrations_mod
        from app.database import get_connection, init_db

        db_path = tmp_path / "ledger-fail.db"
        init_db(db_path)
        real = list(migrations_mod.MIGRATIONS)

        def _ddl_then_block_ledger(conn: sqlite3.Connection) -> None:
            conn.execute("ALTER TABLE topics ADD COLUMN tw_aud_010_probe2 TEXT")
            # Occupy the primary key so the runner's own INSERT collides.
            conn.execute("INSERT INTO schema_version (version) VALUES (9202)")

        monkeypatch.setattr(migrations_mod, "MIGRATIONS", [*real, (9202, "ledger collision", _ddl_then_block_ledger)])

        conn = get_connection(db_path)
        try:
            with pytest.raises(sqlite3.IntegrityError):
                run_migrations(conn, db_path=db_path)
        finally:
            conn.close()

        after = get_connection(db_path)
        try:
            columns = {row[1] for row in after.execute("PRAGMA table_info(topics)").fetchall()}
            recorded = after.execute("SELECT version FROM schema_version WHERE version = 9202").fetchone()
        finally:
            after.close()
        assert "tw_aud_010_probe2" not in columns
        assert recorded is None


class TestSchemaLedgerValidation:
    """TW-AUD-011: the applied ledger must be a contiguous registered prefix."""

    def test_gapped_ledger_is_rejected(self, tmp_path) -> None:
        """A missing intermediate version means missing schema — never 'current'."""
        from app.database import SchemaLedgerError, get_connection, init_db

        db_path = tmp_path / "gap.db"
        init_db(db_path)

        conn = get_connection(db_path)
        try:
            conn.execute("DELETE FROM schema_version WHERE version = 5")
            conn.commit()
            with pytest.raises(SchemaLedgerError, match="5"):
                run_migrations(conn, db_path=db_path)
        finally:
            conn.close()

    def test_future_version_is_rejected(self, tmp_path) -> None:
        """A ledger written by a newer binary must not run against this one."""
        from app.database import SchemaLedgerError, get_connection, init_db

        db_path = tmp_path / "future.db"
        init_db(db_path)

        conn = get_connection(db_path)
        try:
            conn.execute("INSERT INTO schema_version (version) VALUES (99999)")
            conn.commit()
            with pytest.raises(SchemaLedgerError, match="99999"):
                run_migrations(conn, db_path=db_path)
        finally:
            conn.close()

    def test_invalid_version_value_is_rejected(self, tmp_path) -> None:
        """A non-positive version is not a migration this runner ever wrote."""
        from app.database import SchemaLedgerError, get_connection, init_db

        db_path = tmp_path / "invalid.db"
        init_db(db_path)

        conn = get_connection(db_path)
        try:
            conn.execute("INSERT INTO schema_version (version) VALUES (0)")
            conn.commit()
            with pytest.raises(SchemaLedgerError):
                run_migrations(conn, db_path=db_path)
        finally:
            conn.close()

    def test_clean_ledger_is_accepted(self, tmp_path) -> None:
        """The normal fully-migrated ledger passes validation and is a no-op."""
        from app.database import get_connection, get_schema_version, init_db

        db_path = tmp_path / "clean.db"
        init_db(db_path)

        conn = get_connection(db_path)
        try:
            run_migrations(conn, db_path=db_path)
            from app.migrations import MIGRATIONS

            assert get_schema_version(conn) == max(v for v, _, _ in MIGRATIONS)
        finally:
            conn.close()


class TestOnlineBackup:
    """TW-AUD-012 / AUG-149: WAL-safe backups, verified, owner-only."""

    def test_backup_captures_commits_still_in_the_wal(self, tmp_path) -> None:
        """A committed row that has not been checkpointed must reach the backup."""
        import shutil

        from app.database import backup_database, get_connection, init_db

        db_path = tmp_path / "live.db"
        init_db(db_path)

        conn = get_connection(db_path)
        try:
            create_topic(conn, Topic(name="WAL Survivor", description="d"))
            conn.commit()

            # A plain file copy of the main DB misses it: the commit is in the -wal.
            naive = tmp_path / "naive-copy.db"
            shutil.copy2(db_path, naive)
            naive_conn = sqlite3.connect(str(naive))
            try:
                copied = naive_conn.execute("SELECT COUNT(*) FROM topics WHERE name = 'WAL Survivor'").fetchone()[0]
            finally:
                naive_conn.close()
            assert copied == 0, "precondition: the commit must still be uncheckpointed"

            dest = tmp_path / "backups" / "online.db"
            backup_database(conn, dest)
        finally:
            conn.close()

        backup_conn = sqlite3.connect(str(dest))
        try:
            found = backup_conn.execute("SELECT COUNT(*) FROM topics WHERE name = 'WAL Survivor'").fetchone()[0]
            integrity = backup_conn.execute("PRAGMA integrity_check").fetchone()[0]
        finally:
            backup_conn.close()
        assert found == 1
        assert integrity == "ok"

    def test_backup_file_and_directory_are_owner_only(self, tmp_path) -> None:
        """AUG-149: backups hold the same secrets as the DB."""
        import stat

        from app.database import backup_database, get_connection, init_db

        db_path = tmp_path / "modes.db"
        init_db(db_path)
        dest = tmp_path / "backups" / "modes.backup.db"

        conn = get_connection(db_path)
        try:
            backup_database(conn, dest)
        finally:
            conn.close()

        assert stat.S_IMODE(dest.stat().st_mode) == 0o600
        assert stat.S_IMODE(dest.parent.stat().st_mode) == 0o700

    def test_migration_backup_is_readable_and_pruned(self, tmp_path, monkeypatch) -> None:
        """The pre-migration backup is a real, openable database."""
        import app.migrations as migrations_mod
        from app.database import get_connection, init_db

        db_path = tmp_path / "premig.db"
        init_db(db_path)
        real = list(migrations_mod.MIGRATIONS)

        conn = get_connection(db_path)
        try:
            create_topic(conn, Topic(name="Before Migration", description="d"))
            conn.commit()

            monkeypatch.setattr(
                migrations_mod,
                "MIGRATIONS",
                [*real, (9203, "marker", lambda c: c.execute("CREATE TABLE IF NOT EXISTS mig_marker (id INTEGER)"))],
            )
            run_migrations(conn, db_path=db_path)
        finally:
            conn.close()

        backups = sorted((tmp_path / "backups").glob("topic_watch.*.db"))
        assert backups, "a pre-migration backup must exist"
        backup_conn = sqlite3.connect(str(backups[-1]))
        try:
            found = backup_conn.execute("SELECT COUNT(*) FROM topics WHERE name = 'Before Migration'").fetchone()[0]
        finally:
            backup_conn.close()
        assert found == 1


class TestDatabaseFileModes:
    """AUG-149: the database and its sidecars are owner-only."""

    def test_new_database_and_sidecars_are_owner_only(self, tmp_path) -> None:
        import stat

        from app.database import get_connection, init_db

        db_path = tmp_path / "perm.db"
        init_db(db_path)
        assert stat.S_IMODE(db_path.stat().st_mode) == 0o600

        conn = get_connection(db_path)
        try:
            create_topic(conn, Topic(name="Sidecar", description="d"))
            conn.commit()
            for suffix in ("-wal", "-shm"):
                sidecar = db_path.with_name(db_path.name + suffix)
                assert sidecar.exists(), f"expected {suffix} sidecar"
                assert stat.S_IMODE(sidecar.stat().st_mode) == 0o600
        finally:
            conn.close()

    def test_startup_tightens_a_world_readable_database(self, tmp_path) -> None:
        """An install that predates this hardening is fixed on the next start."""
        import stat

        from app.database import init_db

        db_path = tmp_path / "loose.db"
        init_db(db_path)
        db_path.chmod(0o644)

        init_db(db_path)
        assert stat.S_IMODE(db_path.stat().st_mode) == 0o600


class TestRealUpgradePath:
    """AUG-045: a real existing-user v23 -> v24 upgrade, not a no-op re-invoke."""

    def test_v23_database_upgrades_to_head_and_keeps_its_topics(self, tmp_path, monkeypatch) -> None:
        import app.migrations as migrations_mod
        from app.database import get_connection, get_schema_version, init_db

        real = list(migrations_mod.MIGRATIONS)
        db_path = tmp_path / "v23.db"

        # Build a genuine v23 database through the registered runner.
        monkeypatch.setattr(migrations_mod, "MIGRATIONS", [m for m in real if m[0] <= 23])
        init_db(db_path)

        conn = get_connection(db_path)
        try:
            assert get_schema_version(conn) == 23
            columns = {row[1] for row in conn.execute("PRAGMA table_info(topics)").fetchall()}
            assert "heartbeat_alerted_at" not in columns, "precondition: v23 predates the latch column"
            conn.execute(
                "INSERT INTO topics (name, description, feed_urls, feed_mode, created_at, status)"
                " VALUES ('Existing User', 'd', '[]', 'manual', ?, 'ready')",
                (datetime.now(UTC).isoformat(),),
            )
            conn.commit()
        finally:
            conn.close()

        # Now upgrade with the real registry, exactly as an existing install would.
        monkeypatch.setattr(migrations_mod, "MIGRATIONS", real)
        init_db(db_path)

        conn = get_connection(db_path)
        try:
            assert get_schema_version(conn) == max(v for v, _, _ in real)
            row = conn.execute("SELECT * FROM topics WHERE name = 'Existing User'").fetchone()
            assert row is not None, "the existing topic must survive the upgrade"
            assert row["heartbeat_alerted_at"] is None, "the new latch column defaults to NULL"
            topic = Topic.from_row(row)
            assert topic.heartbeat_alerted_at is None
            assert topic.generation, "m026 backfills a generation for existing rows"
        finally:
            conn.close()


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

        assert claim_notification_intent(db_conn, notif.id, "tok", "2999-01-01T00:00:00+00:00") is True
        apply_notification_outcome(db_conn, notif.id, "tok", sent=False, error="down")
        db_conn.commit()

        pending = list_pending_notifications(db_conn)
        assert pending[0].retry_count == 1
        assert pending[0].last_error == "down"

    def test_delete(self, db_conn: sqlite3.Connection) -> None:
        topic, _ = self._make_topic_and_check(db_conn)
        notif = create_pending_notification(
            db_conn,
            PendingNotification(topic_id=topic.id, title="T", body="B"),
        )
        db_conn.commit()

        assert claim_notification_intent(db_conn, notif.id, "tok", "2999-01-01T00:00:00+00:00") is True
        apply_notification_outcome(db_conn, notif.id, "tok", sent=True)
        db_conn.commit()

        # Delivered intents leave the queue but stay as the delivery ledger.
        assert list_pending_notifications(db_conn) == []
        row = db_conn.execute("SELECT status FROM pending_notifications WHERE id = ?", (notif.id,)).fetchone()
        assert row["status"] == "sent"

    def test_abandon_expired(self, db_conn: sqlite3.Connection) -> None:
        """abandon_expired_notifications retires only maxed-out entries."""
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

        abandoned = abandon_expired_notifications(db_conn)
        db_conn.commit()
        # Returns the abandoned rows (not just a count) so the caller can log
        # what was permanently dropped (OVH-040).
        assert len(abandoned) == 1
        assert abandoned[0].title == "Expired"
        assert abandoned[0].topic_id == topic.id

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

    def test_legacy_naive_checked_at_loads_as_aware_utc(self, db_conn: sqlite3.Connection) -> None:
        """OVH-108: the dashboard path uses the model's coercion, not a raw parse.

        A cell written before the canonical ``+00:00`` spelling existed still
        renders — it is a valid timestamp, and hydration normalizes it.
        """
        topic = create_topic(db_conn, Topic(name="Legacy", description="d"))
        cr = create_check_result(db_conn, CheckResult(topic_id=topic.id, articles_found=4))
        db_conn.execute("UPDATE check_results SET checked_at = ? WHERE id = ?", ("2026-06-13T12:00:00", cr.id))
        db_conn.commit()

        data = get_dashboard_data(db_conn)
        assert len(data) == 1
        last_check = data[0]["last_check"]
        assert last_check is not None
        assert last_check.checked_at == datetime(2026, 6, 13, 12, 0, tzinfo=UTC)
        assert last_check.articles_found == 4

    def test_corrupt_checked_at_fails_loud(self, db_conn: sqlite3.Connection) -> None:
        """TW-AUD-013 reverses OVH-108's degrade-to-now for this column.

        Rendering the dashboard with an invented check time is worse than not
        rendering it: the row claims the topic was just checked, the "last checked"
        ordering silently reshuffles, and nothing says the cell is broken. The
        error names the column so the operator can repair or restore the row.
        """
        from app.models import CorruptTimestampError

        topic = create_topic(db_conn, Topic(name="Corrupt", description="d"))
        cr = create_check_result(db_conn, CheckResult(topic_id=topic.id, articles_found=4))
        db_conn.execute("UPDATE check_results SET checked_at = ? WHERE id = ?", ("not-a-date", cr.id))
        db_conn.commit()

        with pytest.raises(CorruptTimestampError, match="checked_at"):
            get_dashboard_data(db_conn)


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
