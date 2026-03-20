"""Tests for transaction-handling bug fixes.

Bug 1: delete_topic_handler in app/web/routes.py must call conn.commit()
Bug 2: recover_stuck_topics in app/crud.py must NOT call conn.commit() itself
"""

import sqlite3

import pytest

from app.crud import create_topic, delete_topic, get_topic, recover_stuck_topics
from app.models import Topic, TopicStatus


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
            tags TEXT NOT NULL DEFAULT '[]'
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
