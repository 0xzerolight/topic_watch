"""Tests for cross-topic article deduplication.

Verifies that:
- find_article_by_hash() works correctly across topics
- fetch_new_articles_for_topic() reuses content when a cross-topic match exists
- fetch_new_articles_for_topic() fetches normally when no cross-topic match exists
- Within-topic dedup still prevents duplicate articles for the same topic
"""

import sqlite3
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

from app.crud import create_article, create_topic, find_article_by_hash
from app.models import Article, FeedMode, Topic
from app.scraping import fetch_new_articles_for_topic
from app.scraping.rss import FeedEntry, compute_article_hash

# ============================================================
# Helpers
# ============================================================


def _make_topic(conn: sqlite3.Connection, name: str = "Topic A") -> Topic:
    topic = create_topic(conn, Topic(name=name, description="d", feed_mode=FeedMode.MANUAL))
    conn.commit()
    return topic


def _make_article(
    conn: sqlite3.Connection,
    topic_id: int,
    url: str = "https://example.com/article",
    title: str = "Article Title",
    raw_content: str | None = "Some article body text.",
    fetched_at: str | None = None,
) -> Article:
    content_hash = compute_article_hash(url, title)
    article = Article(
        topic_id=topic_id,
        title=title,
        url=url,
        content_hash=content_hash,
        raw_content=raw_content,
        source_feed="https://example.com/feed.xml",
        fetched_at=fetched_at or datetime.now(UTC).isoformat(),
    )
    created = create_article(conn, article)
    conn.commit()
    return created


# ============================================================
# Tests for find_article_by_hash
# ============================================================


class TestFindArticleByHash:
    def test_returns_none_when_no_match(self, db_conn: sqlite3.Connection) -> None:
        result = find_article_by_hash(db_conn, "nonexistent_hash_abcdef1234567890")
        assert result is None

    def test_returns_article_when_match_exists(self, db_conn: sqlite3.Connection) -> None:
        topic = _make_topic(db_conn, "Topic A")
        article = _make_article(db_conn, topic.id)

        result = find_article_by_hash(db_conn, article.content_hash)

        assert result is not None
        assert result.content_hash == article.content_hash
        assert result.topic_id == topic.id
        assert result.title == article.title

    def test_returns_article_from_different_topic(self, db_conn: sqlite3.Connection) -> None:
        topic_a = _make_topic(db_conn, "Topic A")
        topic_b = _make_topic(db_conn, "Topic B")

        # Article only exists in topic_a
        article_a = _make_article(db_conn, topic_a.id)

        result = find_article_by_hash(db_conn, article_a.content_hash)

        assert result is not None
        assert result.topic_id == topic_a.id
        # topic_b has no article with this hash, but find_article_by_hash finds it anyway
        assert result.topic_id != topic_b.id

    def test_returns_most_recent_when_multiple_topics_have_same_hash(self, db_conn: sqlite3.Connection) -> None:
        topic_a = _make_topic(db_conn, "Topic A")
        topic_b = _make_topic(db_conn, "Topic B")

        url = "https://example.com/shared-article"
        title = "Shared Article Title"
        content_hash = compute_article_hash(url, title)

        # Insert for topic_a with an earlier timestamp
        article_a = Article(
            topic_id=topic_a.id,
            title=title,
            url=url,
            content_hash=content_hash,
            raw_content="Content from topic A",
            source_feed="feed",
            fetched_at="2025-01-01T10:00:00+00:00",
        )
        create_article(db_conn, article_a)

        # Insert for topic_b with a later timestamp
        article_b = Article(
            topic_id=topic_b.id,
            title=title,
            url=url,
            content_hash=content_hash,
            raw_content="Content from topic B (newer)",
            source_feed="feed",
            fetched_at="2025-01-02T10:00:00+00:00",
        )
        create_article(db_conn, article_b)
        db_conn.commit()

        result = find_article_by_hash(db_conn, content_hash)

        assert result is not None
        # Should return the most recent (topic_b)
        assert result.topic_id == topic_b.id
        assert result.raw_content == "Content from topic B (newer)"

    def test_raw_content_is_preserved(self, db_conn: sqlite3.Connection) -> None:
        topic = _make_topic(db_conn, "Topic A")
        article = _make_article(db_conn, topic.id, raw_content="The full article body text here.")

        result = find_article_by_hash(db_conn, article.content_hash)

        assert result is not None
        assert result.raw_content == "The full article body text here."

    def test_returns_article_with_none_raw_content(self, db_conn: sqlite3.Connection) -> None:
        topic = _make_topic(db_conn, "Topic A")
        article = _make_article(db_conn, topic.id, raw_content=None)

        result = find_article_by_hash(db_conn, article.content_hash)

        assert result is not None
        assert result.raw_content is None


# ============================================================
# Tests for fetch_new_articles_for_topic cross-topic dedup
# ============================================================


class TestFetchNewArticlesCrossTopicDedup:
    def _make_entry(
        self,
        url: str = "https://example.com/article",
        title: str = "Article Title",
        summary: str = "Summary text",
    ) -> FeedEntry:
        return FeedEntry(
            title=title,
            url=url,
            summary=summary,
            source_feed="https://example.com/feed.xml",
        )

    async def test_reuses_content_from_another_topic(self, db_conn: sqlite3.Connection) -> None:
        """When another topic already fetched an article, reuse its content."""
        topic_a = _make_topic(db_conn, "Topic A")
        topic_b = _make_topic(db_conn, "Topic B")

        url = "https://example.com/shared"
        title = "Shared Article"
        content_hash = compute_article_hash(url, title)

        # Pre-store the article for topic_a with content
        existing = Article(
            topic_id=topic_a.id,
            title=title,
            url=url,
            content_hash=content_hash,
            raw_content="Pre-fetched content from topic A",
            source_feed="https://example.com/feed.xml",
        )
        create_article(db_conn, existing)
        db_conn.commit()

        entry = self._make_entry(url=url, title=title)

        extract_mock = AsyncMock(return_value="Freshly fetched content")

        with (
            patch("app.scraping.fetch_feeds_for_topic", return_value=[entry]),
            patch("app.scraping.extract_article_content", extract_mock),
        ):
            stored = (await fetch_new_articles_for_topic(topic_b, db_conn)).articles

        # Article should be created for topic_b
        assert len(stored) == 1
        assert stored[0].topic_id == topic_b.id
        assert stored[0].content_hash == content_hash
        # Content should be reused from topic_a, NOT freshly fetched
        assert stored[0].raw_content == "Pre-fetched content from topic A"
        # HTTP fetch should NOT have been called
        extract_mock.assert_not_called()

    async def test_fetches_normally_when_no_cross_topic_match(self, db_conn: sqlite3.Connection) -> None:
        """When no cross-topic article exists, content is fetched normally."""
        topic = _make_topic(db_conn, "Topic A")
        entry = self._make_entry()

        extract_mock = AsyncMock(return_value="Freshly fetched content")

        with (
            patch("app.scraping.fetch_feeds_for_topic", return_value=[entry]),
            patch("app.scraping.extract_article_content", extract_mock),
        ):
            stored = (await fetch_new_articles_for_topic(topic, db_conn)).articles

        assert len(stored) == 1
        assert stored[0].raw_content == "Freshly fetched content"
        extract_mock.assert_called_once()

    async def test_does_not_reuse_when_existing_has_no_raw_content(self, db_conn: sqlite3.Connection) -> None:
        """If the cross-topic article has no raw_content, fetch content normally."""
        topic_a = _make_topic(db_conn, "Topic A")
        topic_b = _make_topic(db_conn, "Topic B")

        url = "https://example.com/shared"
        title = "Shared Article"
        content_hash = compute_article_hash(url, title)

        # Pre-store article for topic_a WITHOUT content
        existing = Article(
            topic_id=topic_a.id,
            title=title,
            url=url,
            content_hash=content_hash,
            raw_content=None,
            source_feed="https://example.com/feed.xml",
        )
        create_article(db_conn, existing)
        db_conn.commit()

        entry = self._make_entry(url=url, title=title)
        extract_mock = AsyncMock(return_value="Freshly fetched content")

        with (
            patch("app.scraping.fetch_feeds_for_topic", return_value=[entry]),
            patch("app.scraping.extract_article_content", extract_mock),
        ):
            stored = (await fetch_new_articles_for_topic(topic_b, db_conn)).articles

        assert len(stored) == 1
        assert stored[0].raw_content == "Freshly fetched content"
        # Should have fetched because existing had no content
        extract_mock.assert_called_once()

    async def test_within_topic_dedup_still_works(self, db_conn: sqlite3.Connection) -> None:
        """Cross-topic dedup must NOT create duplicates within the same topic."""
        topic = _make_topic(db_conn, "Topic A")

        url = "https://example.com/article"
        title = "Article Title"

        # Pre-store the article for this same topic
        _make_article(db_conn, topic.id, url=url, title=title)

        entry = self._make_entry(url=url, title=title)
        extract_mock = AsyncMock(return_value="Fresh content")

        with (
            patch("app.scraping.fetch_feeds_for_topic", return_value=[entry]),
            patch("app.scraping.extract_article_content", extract_mock),
        ):
            stored = (await fetch_new_articles_for_topic(topic, db_conn)).articles

        # Should be skipped — already exists for this topic
        assert len(stored) == 0
        extract_mock.assert_not_called()

    async def test_within_topic_dedup_takes_priority_over_cross_topic(self, db_conn: sqlite3.Connection) -> None:
        """article_hash_exists check fires before find_article_by_hash."""
        topic_a = _make_topic(db_conn, "Topic A")
        topic_b = _make_topic(db_conn, "Topic B")

        url = "https://example.com/shared"
        title = "Shared Article"
        content_hash = compute_article_hash(url, title)

        # Store article for BOTH topics (topic_b already has it)
        for tid in (topic_a.id, topic_b.id):
            art = Article(
                topic_id=tid,
                title=title,
                url=url,
                content_hash=content_hash,
                raw_content="Content",
                source_feed="https://example.com/feed.xml",
            )
            create_article(db_conn, art)
        db_conn.commit()

        entry = self._make_entry(url=url, title=title)
        extract_mock = AsyncMock(return_value="Fresh content")

        with (
            patch("app.scraping.fetch_feeds_for_topic", return_value=[entry]),
            patch("app.scraping.extract_article_content", extract_mock),
        ):
            stored = (await fetch_new_articles_for_topic(topic_b, db_conn)).articles

        # topic_b already has it — must be skipped
        assert len(stored) == 0
        extract_mock.assert_not_called()

    async def test_only_matching_articles_reuse_content(self, db_conn: sqlite3.Connection) -> None:
        """Only the cross-topic article is reused; others are fetched normally."""
        topic_a = _make_topic(db_conn, "Topic A")
        topic_b = _make_topic(db_conn, "Topic B")

        shared_url = "https://example.com/shared"
        shared_title = "Shared Article"
        new_url = "https://example.com/new"
        new_title = "New Article"

        shared_hash = compute_article_hash(shared_url, shared_title)

        # Pre-store shared article in topic_a only
        create_article(
            db_conn,
            Article(
                topic_id=topic_a.id,
                title=shared_title,
                url=shared_url,
                content_hash=shared_hash,
                raw_content="Shared content from topic A",
                source_feed="feed",
            ),
        )
        db_conn.commit()

        entries = [
            self._make_entry(url=shared_url, title=shared_title, summary="Shared summary"),
            self._make_entry(url=new_url, title=new_title, summary="New summary"),
        ]
        extract_mock = AsyncMock(return_value="Freshly fetched new content")

        with (
            patch("app.scraping.fetch_feeds_for_topic", return_value=entries),
            patch("app.scraping.extract_article_content", extract_mock),
        ):
            stored = (await fetch_new_articles_for_topic(topic_b, db_conn)).articles

        assert len(stored) == 2

        stored_by_hash = {a.content_hash: a for a in stored}
        new_hash = compute_article_hash(new_url, new_title)

        # Shared article reused from topic_a
        assert stored_by_hash[shared_hash].raw_content == "Shared content from topic A"
        # New article fetched normally
        assert stored_by_hash[new_hash].raw_content == "Freshly fetched new content"

        # Only the new article needed an HTTP fetch
        extract_mock.assert_called_once()
