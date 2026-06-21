"""Tests for cross-topic article deduplication.

Verifies that:
- find_article_by_hash() works correctly across topics
- fetch_new_articles_for_topic() reuses content when a cross-topic match exists
- fetch_new_articles_for_topic() fetches normally when no cross-topic match exists
- Within-topic dedup still prevents duplicate articles for the same topic
"""

import logging
import sqlite3
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

from app.crud import create_article, create_topic, find_article_by_hash
from app.models import Article, FeedMode, Topic
from app.scraping import fetch_new_articles_for_topic
from app.scraping.rss import FeedEntry, FeedResponse, compute_article_hash

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
            patch("app.scraping.fetch_feeds_for_topic", return_value=FeedResponse(entries=[entry])),
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

    async def test_reused_article_carries_resolved_url(self, db_conn: sqlite3.Connection) -> None:
        """OVH-025: reuse path stores the resolved publisher URL, not the Google redirect.

        Both topics fetch the same Google News entry, so the content_hash (computed
        from the redirect URL) matches. Topic A's article URL was later resolved to the
        real publisher URL. Topic B must reuse that resolved URL, while keeping the hash.
        """
        topic_a = _make_topic(db_conn, "Topic A")
        topic_b = _make_topic(db_conn, "Topic B")

        redirect_url = "https://news.google.com/rss/articles/ABC123?oc=5"
        resolved_url = "https://publisher.example.com/real-article"
        title = "Shared Article"
        # Hash is computed from the (unresolved) redirect URL that both feeds emit.
        content_hash = compute_article_hash(redirect_url, title)

        # Topic A already stored the article with the RESOLVED url but the redirect hash.
        existing = Article(
            topic_id=topic_a.id,
            title=title,
            url=resolved_url,
            content_hash=content_hash,
            raw_content="Pre-fetched content from topic A",
            source_feed="https://news.google.com/rss/search?q=x",
        )
        create_article(db_conn, existing)
        db_conn.commit()

        # Topic B's feed entry still carries the unresolved redirect URL.
        entry = self._make_entry(url=redirect_url, title=title)
        extract_mock = AsyncMock(return_value="Freshly fetched content")

        with (
            patch("app.scraping.fetch_feeds_for_topic", return_value=FeedResponse(entries=[entry])),
            patch("app.scraping.extract_article_content", extract_mock),
        ):
            stored = (await fetch_new_articles_for_topic(topic_b, db_conn)).articles

        assert len(stored) == 1
        # The stored URL is the resolved publisher URL, not the Google redirect.
        assert stored[0].url == resolved_url
        # Dedup must be preserved: the hash is unchanged.
        assert stored[0].content_hash == content_hash
        assert stored[0].raw_content == "Pre-fetched content from topic A"
        extract_mock.assert_not_called()

    async def test_reused_article_keeps_new_entry_provenance(self, db_conn: sqlite3.Connection) -> None:
        """OVH-084: reuse inherits ONLY raw_content; url/title/source_feed stay the new entry's.

        The point of cross-topic dedup is to reuse the expensive raw_content while
        keeping the rest of the row tied to THIS topic's own feed entry. The
        pre-stored article is given a different source_feed and a case-variant
        title that still hashes equal (the hash lowercases url|title before
        hashing), so a regression copying the matched article's fields would be
        detectable:

          * url      -> the resolved existing.url (per OVH-025), NOT the new entry's
          * title    -> the NEW entry's title (correct, not the stored case-variant)
          * source_feed -> the NEW entry's feed (feed-health attribution stays correct)
          * raw_content -> inherited from the existing article (the only reuse)
        """
        topic_a = _make_topic(db_conn, "Topic A")
        topic_b = _make_topic(db_conn, "Topic B")

        redirect_url = "https://news.google.com/rss/articles/XYZ789?oc=5"
        resolved_url = "https://publisher.example.com/the-real-article"
        new_entry_title = "Shared Headline"
        # A case variant of the title — lowercases to the same string, so
        # compute_article_hash (which lowercases url|title) yields the SAME hash.
        stored_variant_title = "shared HEADLINE"
        new_entry_feed = "https://topic-b-own-feed.example.com/rss"
        existing_feed = "https://topic-a-other-feed.example.com/rss"

        content_hash = compute_article_hash(redirect_url, new_entry_title)
        # Sanity: the case/whitespace-variant hashes identically (drives the match).
        assert compute_article_hash(redirect_url, stored_variant_title) == content_hash

        existing = Article(
            topic_id=topic_a.id,
            title=stored_variant_title,
            url=resolved_url,
            content_hash=content_hash,
            raw_content="Expensive pre-fetched body from topic A",
            source_feed=existing_feed,
        )
        create_article(db_conn, existing)
        db_conn.commit()

        entry = FeedEntry(
            title=new_entry_title,
            url=redirect_url,
            summary="Summary text",
            source_feed=new_entry_feed,
        )
        extract_mock = AsyncMock(return_value="Freshly fetched content")

        with (
            patch("app.scraping.fetch_feeds_for_topic", return_value=FeedResponse(entries=[entry])),
            patch("app.scraping.extract_article_content", extract_mock),
        ):
            stored = (await fetch_new_articles_for_topic(topic_b, db_conn)).articles

        assert len(stored) == 1
        reused = stored[0]
        # Row belongs to the new topic.
        assert reused.topic_id == topic_b.id
        # Only raw_content is inherited from the matched article.
        assert reused.raw_content == "Expensive pre-fetched body from topic A"
        # Title is the NEW entry's, not the stored case-variant.
        assert reused.title == new_entry_title
        # source_feed is the NEW entry's feed (correct feed-health attribution).
        assert reused.source_feed == new_entry_feed
        # url is the resolved existing.url (OVH-025), and the hash is preserved.
        assert reused.url == resolved_url
        assert reused.content_hash == content_hash
        # No HTTP fetch happened — content was reused.
        extract_mock.assert_not_called()

    async def test_reused_article_carries_originating_provider(self, db_conn: sqlite3.Connection) -> None:
        """OVH-114: a reused row keeps the ORIGINATING provider, not the current one.

        source_provider (m009) records which provider actually fetched the content.
        Cross-topic dedup copies bytes fetched by another topic's provider, so the
        reused row's attribution must point at that originating provider — not the
        provider that produced this topic's feed entry. A freshly fetched row, by
        contrast, is stamped with the current topic's provider.
        """
        topic_a = _make_topic(db_conn, "Topic A")
        topic_b = _make_topic(db_conn, "Topic B")

        url = "https://example.com/shared-provider"
        title = "Shared Provider Article"
        content_hash = compute_article_hash(url, title)

        # Pre-stored content was fetched via Bing for topic A.
        existing = Article(
            topic_id=topic_a.id,
            title=title,
            url=url,
            content_hash=content_hash,
            raw_content="Body fetched via Bing",
            source_feed="https://topic-a-feed.example.com/rss",
            source_provider="Bing News",
        )
        create_article(db_conn, existing)
        db_conn.commit()

        entry = self._make_entry(url=url, title=title)
        extract_mock = AsyncMock(return_value="Freshly fetched content")

        # Topic B's feed comes from Google News this cycle.
        with (
            patch(
                "app.scraping.fetch_feeds_for_topic",
                return_value=FeedResponse(entries=[entry], provider_name="Google News"),
            ),
            patch("app.scraping.extract_article_content", extract_mock),
        ):
            stored = (await fetch_new_articles_for_topic(topic_b, db_conn)).articles

        assert len(stored) == 1
        reused = stored[0]
        assert reused.topic_id == topic_b.id
        assert reused.raw_content == "Body fetched via Bing"
        # Attribution carries the ORIGINATING provider, not topic B's Google News.
        assert reused.source_provider == "Bing News"
        extract_mock.assert_not_called()

    async def test_fetches_normally_when_no_cross_topic_match(self, db_conn: sqlite3.Connection) -> None:
        """When no cross-topic article exists, content is fetched normally."""
        topic = _make_topic(db_conn, "Topic A")
        entry = self._make_entry()

        extract_mock = AsyncMock(return_value="Freshly fetched content")

        with (
            patch("app.scraping.fetch_feeds_for_topic", return_value=FeedResponse(entries=[entry])),
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
            patch("app.scraping.fetch_feeds_for_topic", return_value=FeedResponse(entries=[entry])),
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
            patch("app.scraping.fetch_feeds_for_topic", return_value=FeedResponse(entries=[entry])),
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
            patch("app.scraping.fetch_feeds_for_topic", return_value=FeedResponse(entries=[entry])),
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
            patch("app.scraping.fetch_feeds_for_topic", return_value=FeedResponse(entries=entries)),
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


# ============================================================
# Tests for race-condition duplicate observability
# ============================================================


class TestDuplicateRaceObservable:
    """When a concurrent insert races and loses to UNIQUE(topic_id, content_hash),
    the dropped article must be OBSERVABLE — counted in the result and logged at
    WARNING — not silently discarded.
    """

    def _make_entry(self, url: str, title: str) -> FeedEntry:
        return FeedEntry(
            title=title,
            url=url,
            summary="Summary text",
            source_feed="https://example.com/feed.xml",
        )

    async def test_race_drop_is_counted_in_result(self, db_conn: sqlite3.Connection) -> None:
        """A losing concurrent insert (IntegrityError) is counted in dropped_duplicates."""
        topic = _make_topic(db_conn, "Topic A")
        entry = self._make_entry("https://example.com/raced", "Raced Article")

        real_create = create_article

        def racing_create(conn: sqlite3.Connection, article: Article) -> Article:
            # Simulate the other concurrent fetch winning: insert the same
            # (topic_id, content_hash) row first, so this insert hits UNIQUE.
            real_create(
                conn,
                Article(
                    topic_id=article.topic_id,
                    title=article.title,
                    url=article.url,
                    content_hash=article.content_hash,
                    raw_content="winner content",
                    source_feed=article.source_feed,
                ),
            )
            return real_create(conn, article)  # raises sqlite3.IntegrityError

        with (
            patch("app.scraping.fetch_feeds_for_topic", return_value=FeedResponse(entries=[entry])),
            patch("app.scraping.extract_article_content", AsyncMock(return_value="loser content")),
            patch("app.scraping.create_article", side_effect=racing_create),
        ):
            result = await fetch_new_articles_for_topic(topic, db_conn)

        # The racing insert lost — no article returned for it, but it must be counted.
        assert result.articles == []
        assert result.dropped_duplicates == 1

    async def test_race_drop_is_logged_at_warning(self, db_conn: sqlite3.Connection, caplog) -> None:
        """The dropped duplicate is logged at WARNING, not swallowed at DEBUG."""
        topic = _make_topic(db_conn, "Topic A")
        entry = self._make_entry("https://example.com/raced2", "Raced Article 2")

        real_create = create_article

        def racing_create(conn: sqlite3.Connection, article: Article) -> Article:
            real_create(
                conn,
                Article(
                    topic_id=article.topic_id,
                    title=article.title,
                    url=article.url,
                    content_hash=article.content_hash,
                    raw_content="winner content",
                    source_feed=article.source_feed,
                ),
            )
            return real_create(conn, article)

        with (
            patch("app.scraping.fetch_feeds_for_topic", return_value=FeedResponse(entries=[entry])),
            patch("app.scraping.extract_article_content", AsyncMock(return_value="loser content")),
            patch("app.scraping.create_article", side_effect=racing_create),
            caplog.at_level(logging.WARNING, logger="app.scraping"),
        ):
            await fetch_new_articles_for_topic(topic, db_conn)

        assert any(
            record.levelno >= logging.WARNING and "duplicate" in record.message.lower() for record in caplog.records
        ), "expected a WARNING-level log about the dropped duplicate"
