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

    def test_one_per_topic(self, db_conn: sqlite3.Connection) -> None:
        """Only one knowledge state per topic (UNIQUE constraint)."""
        topic = create_topic(db_conn, Topic(name="KSUnique", description="desc"))
        create_knowledge_state(
            db_conn,
            KnowledgeState(topic_id=topic.id, summary_text="First", token_count=10),
        )
        db_conn.commit()

        with pytest.raises(sqlite3.IntegrityError):
            create_knowledge_state(
                db_conn,
                KnowledgeState(topic_id=topic.id, summary_text="Second", token_count=20),
            )


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
        assert deleted == 1

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
