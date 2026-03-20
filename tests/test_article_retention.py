"""Tests for the article retention/cleanup functionality."""

import sqlite3
from datetime import UTC, datetime, timedelta

from app.crud import create_article, create_topic, delete_old_articles, list_articles_for_topic
from app.models import Article, Topic


def _insert_article(conn: sqlite3.Connection, topic_id: int, hash_suffix: str, fetched_at: datetime) -> Article:
    """Helper to insert an article with a specific fetched_at timestamp."""
    article = Article(
        topic_id=topic_id,
        title=f"Article {hash_suffix}",
        url=f"https://example.com/{hash_suffix}",
        content_hash=f"hash_{hash_suffix}",
        source_feed="https://example.com/feed.xml",
        fetched_at=fetched_at,
    )
    created = create_article(conn, article)
    conn.commit()
    return created


class TestDeleteOldArticles:
    """Tests for the delete_old_articles CRUD function."""

    def _make_topic(self, conn: sqlite3.Connection, name: str = "TestTopic") -> Topic:
        topic = create_topic(conn, Topic(name=name, description="desc"))
        conn.commit()
        return topic

    def test_deletes_articles_older_than_retention(self, db_conn: sqlite3.Connection) -> None:
        """Articles older than retention_days are deleted."""
        topic = self._make_topic(db_conn)
        now = datetime.now(UTC)

        # Old article: 100 days ago
        _insert_article(db_conn, topic.id, "old", now - timedelta(days=100))
        # Recent article: 10 days ago
        _insert_article(db_conn, topic.id, "recent", now - timedelta(days=10))

        deleted = delete_old_articles(db_conn, retention_days=90)
        db_conn.commit()

        assert deleted == 1
        remaining = list_articles_for_topic(db_conn, topic.id)
        assert len(remaining) == 1
        assert remaining[0].content_hash == "hash_recent"

    def test_preserves_articles_within_retention(self, db_conn: sqlite3.Connection) -> None:
        """Articles within the retention window are not deleted."""
        topic = self._make_topic(db_conn)
        now = datetime.now(UTC)

        _insert_article(db_conn, topic.id, "a", now - timedelta(days=30))
        _insert_article(db_conn, topic.id, "b", now - timedelta(days=60))
        _insert_article(db_conn, topic.id, "c", now - timedelta(days=89))

        deleted = delete_old_articles(db_conn, retention_days=90)
        db_conn.commit()

        assert deleted == 0
        remaining = list_articles_for_topic(db_conn, topic.id)
        assert len(remaining) == 3

    def test_returns_correct_count(self, db_conn: sqlite3.Connection) -> None:
        """The return value equals the number of deleted rows."""
        topic = self._make_topic(db_conn)
        now = datetime.now(UTC)

        _insert_article(db_conn, topic.id, "old1", now - timedelta(days=200))
        _insert_article(db_conn, topic.id, "old2", now - timedelta(days=150))
        _insert_article(db_conn, topic.id, "old3", now - timedelta(days=100))
        _insert_article(db_conn, topic.id, "new1", now - timedelta(days=45))

        deleted = delete_old_articles(db_conn, retention_days=90)
        db_conn.commit()

        assert deleted == 3

    def test_returns_zero_when_no_articles_qualify(self, db_conn: sqlite3.Connection) -> None:
        """Returns 0 when no articles exceed the retention period."""
        topic = self._make_topic(db_conn)
        now = datetime.now(UTC)

        _insert_article(db_conn, topic.id, "fresh", now - timedelta(days=1))

        deleted = delete_old_articles(db_conn, retention_days=90)
        db_conn.commit()

        assert deleted == 0

    def test_returns_zero_with_empty_table(self, db_conn: sqlite3.Connection) -> None:
        """Returns 0 when the articles table is empty."""
        deleted = delete_old_articles(db_conn, retention_days=90)
        assert deleted == 0

    def test_deletes_only_from_matching_articles(self, db_conn: sqlite3.Connection) -> None:
        """Old articles for multiple topics are all cleaned up."""
        topic1 = self._make_topic(db_conn, "Topic1")
        topic2 = self._make_topic(db_conn, "Topic2")
        now = datetime.now(UTC)

        _insert_article(db_conn, topic1.id, "t1old", now - timedelta(days=120))
        _insert_article(db_conn, topic1.id, "t1new", now - timedelta(days=5))
        _insert_article(db_conn, topic2.id, "t2old", now - timedelta(days=95))
        _insert_article(db_conn, topic2.id, "t2new", now - timedelta(days=20))

        deleted = delete_old_articles(db_conn, retention_days=90)
        db_conn.commit()

        assert deleted == 2

        remaining1 = list_articles_for_topic(db_conn, topic1.id)
        remaining2 = list_articles_for_topic(db_conn, topic2.id)
        assert len(remaining1) == 1
        assert remaining1[0].content_hash == "hash_t1new"
        assert len(remaining2) == 1
        assert remaining2[0].content_hash == "hash_t2new"

    def test_boundary_article_exactly_at_cutoff(self, db_conn: sqlite3.Connection) -> None:
        """An article fetched exactly at retention_days ago is deleted (strictly less than)."""
        topic = self._make_topic(db_conn)
        now = datetime.now(UTC)

        # Insert article at exactly 91 days ago — older than the 90 day limit
        _insert_article(db_conn, topic.id, "boundary_old", now - timedelta(days=91))
        # Insert article at exactly 89 days ago — within the 90 day limit
        _insert_article(db_conn, topic.id, "boundary_new", now - timedelta(days=89))

        deleted = delete_old_articles(db_conn, retention_days=90)
        db_conn.commit()

        assert deleted == 1
        remaining = list_articles_for_topic(db_conn, topic.id)
        assert len(remaining) == 1
        assert remaining[0].content_hash == "hash_boundary_new"

    def test_short_retention_period(self, db_conn: sqlite3.Connection) -> None:
        """A retention period of 1 day removes articles older than 1 day."""
        topic = self._make_topic(db_conn)
        now = datetime.now(UTC)

        _insert_article(db_conn, topic.id, "old", now - timedelta(days=2))
        _insert_article(db_conn, topic.id, "new", now - timedelta(hours=12))

        deleted = delete_old_articles(db_conn, retention_days=1)
        db_conn.commit()

        assert deleted == 1
        remaining = list_articles_for_topic(db_conn, topic.id)
        assert len(remaining) == 1
        assert remaining[0].content_hash == "hash_new"
