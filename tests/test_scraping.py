"""Tests for the scraping pipeline: RSS fetching, content extraction, orchestration."""

import sqlite3
from datetime import UTC, datetime
from unittest.mock import patch

import httpx

from app.crud import create_topic, list_articles_for_topic
from app.models import Article, FeedMode, Topic
from app.scraping import fetch_new_articles_for_topic
from app.scraping.content import _truncate, extract_article_content
from app.scraping.providers import GoogleNewsProvider
from app.scraping.rss import (
    FeedEntry,
    FeedResponse,
    _parse_entry,
    _parse_feed_date,
    _resolve_google_news_url,
    compute_article_hash,
    fetch_feed,
    fetch_feeds_for_topic,
)

# --- Sample RSS/Atom XML for mocking ---

_EMPTY_RSS = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Empty Feed</title>
  </channel>
</rss>"""

_SAMPLE_RSS = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Test Feed</title>
    <item>
      <title>Article One</title>
      <link>https://example.com/article-1</link>
      <pubDate>Thu, 01 Jan 2025 12:00:00 GMT</pubDate>
      <description>Summary of article one.</description>
    </item>
    <item>
      <title>Article Two</title>
      <link>https://example.com/article-2</link>
      <pubDate>Fri, 02 Jan 2025 12:00:00 GMT</pubDate>
      <description>Summary of article two.</description>
    </item>
  </channel>
</rss>"""

_SAMPLE_ATOM = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Reddit Feed</title>
  <entry>
    <title>Reddit Post</title>
    <link href="https://reddit.com/r/test/1"/>
    <updated>2025-01-03T10:00:00Z</updated>
    <content type="html">&lt;p&gt;Reddit content here.&lt;/p&gt;</content>
  </entry>
</feed>"""

_SAMPLE_HTML = """\
<!DOCTYPE html>
<html>
<head><title>Test Article</title></head>
<body>
<article>
<h1>Test Article</h1>
<p>This is the main article content that should be extracted by trafilatura.
It needs to be long enough for trafilatura to consider it real content.
Here is some more text to make it substantial enough for extraction.
The article discusses important topics in technology and science.
Multiple paragraphs help trafilatura identify this as article content.</p>
<p>Second paragraph with more details about the topic at hand.
This provides additional context and information for the reader.
We want to ensure trafilatura picks this up as meaningful content.</p>
</article>
</body>
</html>"""


# --- Helper to build mock httpx transport ---


def _mock_transport(responses: dict[str, tuple[int, str]]) -> httpx.MockTransport:
    """Build a MockTransport that returns canned responses by URL."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        for pattern, (status, body) in responses.items():
            if pattern in url:
                return httpx.Response(status, text=body)
        return httpx.Response(404, text="Not found")

    return httpx.MockTransport(handler)


# ============================================================
# TestComputeArticleHash
# ============================================================


class TestComputeArticleHash:
    def test_deterministic(self) -> None:
        h1 = compute_article_hash("https://example.com/a", "Title")
        h2 = compute_article_hash("https://example.com/a", "Title")
        assert h1 == h2

    def test_case_insensitive(self) -> None:
        h1 = compute_article_hash("https://Example.com/A", "TITLE")
        h2 = compute_article_hash("https://example.com/a", "title")
        assert h1 == h2

    def test_different_inputs_different_hashes(self) -> None:
        h1 = compute_article_hash("url1", "title1")
        h2 = compute_article_hash("url2", "title2")
        assert h1 != h2

    def test_hex_length(self) -> None:
        h = compute_article_hash("url", "title")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


# ============================================================
# TestParseFeedDate
# ============================================================


class TestParseFeedDate:
    def test_rss_published_parsed(self) -> None:
        from time import strptime

        entry = {"published_parsed": strptime("2025-01-15", "%Y-%m-%d")}
        result = _parse_feed_date(entry)
        assert result is not None
        assert result.year == 2025
        assert result.month == 1
        assert result.day == 15
        assert result.tzinfo == UTC

    def test_atom_updated_parsed(self) -> None:
        from time import strptime

        entry = {"updated_parsed": strptime("2025-06-01", "%Y-%m-%d")}
        result = _parse_feed_date(entry)
        assert result is not None
        assert result.month == 6

    def test_prefers_published_over_updated(self) -> None:
        from time import strptime

        entry = {
            "published_parsed": strptime("2025-01-01", "%Y-%m-%d"),
            "updated_parsed": strptime("2025-06-01", "%Y-%m-%d"),
        }
        result = _parse_feed_date(entry)
        assert result is not None
        assert result.month == 1  # published, not updated

    def test_missing_dates(self) -> None:
        assert _parse_feed_date({}) is None

    def test_none_values(self) -> None:
        entry = {"published_parsed": None, "updated_parsed": None}
        assert _parse_feed_date(entry) is None


# ============================================================
# TestParseEntry
# ============================================================


class TestParseEntry:
    def test_valid_rss_entry(self) -> None:
        raw = {
            "title": "Test Title",
            "link": "https://example.com/test",
            "summary": "A summary.",
        }
        entry = _parse_entry(raw, "https://example.com/feed.xml")
        assert entry is not None
        assert entry.title == "Test Title"
        assert entry.url == "https://example.com/test"
        assert entry.summary == "A summary."
        assert entry.source_feed == "https://example.com/feed.xml"

    def test_missing_title_returns_none(self) -> None:
        raw = {"link": "https://example.com/test"}
        assert _parse_entry(raw, "feed") is None

    def test_missing_link_returns_none(self) -> None:
        raw = {"title": "Title"}
        assert _parse_entry(raw, "feed") is None

    def test_empty_title_returns_none(self) -> None:
        raw = {"title": "  ", "link": "https://example.com/test"}
        assert _parse_entry(raw, "feed") is None

    def test_atom_content_as_summary(self) -> None:
        raw = {
            "title": "Reddit Post",
            "link": "https://reddit.com/r/test/1",
            "content": [{"value": "<p>Content from Atom feed</p>"}],
        }
        entry = _parse_entry(raw, "feed")
        assert entry is not None
        assert "Content from Atom feed" in entry.summary

    def test_summary_preferred_over_content(self) -> None:
        raw = {
            "title": "Post",
            "link": "https://example.com",
            "summary": "The summary",
            "content": [{"value": "The content"}],
        }
        entry = _parse_entry(raw, "feed")
        assert entry is not None
        assert entry.summary == "The summary"


# ============================================================
# TestFetchFeed (async, mocked httpx)
# ============================================================


class TestFetchFeed:
    async def test_successful_rss(self) -> None:
        transport = _mock_transport({"example.com/feed": (200, _SAMPLE_RSS)})
        async with httpx.AsyncClient(transport=transport) as client:
            entries = await fetch_feed("https://example.com/feed.xml", client)
        assert len(entries) == 2
        assert entries[0].title == "Article One"
        assert entries[1].title == "Article Two"

    async def test_successful_atom(self) -> None:
        transport = _mock_transport({"reddit.com/feed": (200, _SAMPLE_ATOM)})
        async with httpx.AsyncClient(transport=transport) as client:
            entries = await fetch_feed("https://reddit.com/feed.rss", client)
        assert len(entries) == 1
        assert entries[0].title == "Reddit Post"

    async def test_timeout_returns_empty(self) -> None:
        def timeout_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timeout")

        transport = httpx.MockTransport(timeout_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            entries = await fetch_feed("https://example.com/feed", client)
        assert entries == []

    async def test_http_error_returns_empty(self) -> None:
        transport = _mock_transport({"example.com": (500, "Server Error")})
        async with httpx.AsyncClient(transport=transport) as client:
            entries = await fetch_feed("https://example.com/feed", client)
        assert entries == []

    async def test_malformed_xml_returns_empty(self) -> None:
        transport = _mock_transport({"example.com": (200, "not xml at all {{{")})
        async with httpx.AsyncClient(transport=transport) as client:
            entries = await fetch_feed("https://example.com/feed", client)
        assert entries == []


# ============================================================
# TestFetchFeedsForTopic (async, mocked)
# ============================================================


class TestBuildGoogleNewsUrl:
    """Tests for GoogleNewsProvider URL building (moved from old build_google_news_url)."""

    def test_basic_query(self) -> None:
        provider = GoogleNewsProvider()
        topic = Topic(name="Elden Ring DLC", description="release date of the DLC", feed_urls=[])
        url = provider.build_feed_url(topic)
        assert "news.google.com/rss/search" in url
        assert "Elden+Ring+DLC" in url
        assert "hl=en-US" in url

    def test_description_supplements_query(self) -> None:
        provider = GoogleNewsProvider()
        topic = Topic(name="Solo Leveling", description="season 3 release date anime", feed_urls=[])
        url = provider.build_feed_url(topic)
        assert "Solo+Leveling" in url
        assert "season" in url
        assert "release" in url

    def test_empty_description_uses_name_only(self) -> None:
        provider = GoogleNewsProvider()
        topic = Topic(name="Elden Ring DLC", description="", feed_urls=[])
        url = provider.build_feed_url(topic)
        assert "Elden+Ring+DLC" in url

    def test_special_characters_encoded(self) -> None:
        provider = GoogleNewsProvider()
        topic = Topic(name="C++ news & updates", description="", feed_urls=[])
        url = provider.build_feed_url(topic)
        assert "C%2B%2B" in url
        assert "%26" in url

    def test_empty_name(self) -> None:
        provider = GoogleNewsProvider()
        topic = Topic(name="", description="", feed_urls=[])
        url = provider.build_feed_url(topic)
        assert "news.google.com/rss/search" in url

    def test_long_description_truncated(self) -> None:
        provider = GoogleNewsProvider()
        topic = Topic(
            name="Test",
            description="one two three four five six seven eight nine ten",
            feed_urls=[],
        )
        url = provider.build_feed_url(topic)
        # Only first 6 words of description should be used
        assert "seven" not in url
        assert "six" in url


class TestResolveGoogleNewsUrl:
    def test_non_google_url_unchanged(self) -> None:
        url = "https://example.com/article"
        assert _resolve_google_news_url(url, "") == url

    def test_extracts_real_url_from_description(self) -> None:
        google_url = "https://news.google.com/rss/articles/CBMiQ2h0dHBz..."
        description = '<a href="https://comicbook.com/anime/solo-leveling-s3" target="_blank">Title</a>'
        result = _resolve_google_news_url(google_url, description)
        assert result == "https://comicbook.com/anime/solo-leveling-s3"

    def test_ignores_google_self_links(self) -> None:
        google_url = "https://news.google.com/rss/articles/CBMiQ2h0dHBz..."
        description = '<a href="https://news.google.com/stories/123">Title</a>'
        result = _resolve_google_news_url(google_url, description)
        assert result == google_url  # falls back to original

    def test_falls_back_when_no_href(self) -> None:
        google_url = "https://news.google.com/rss/articles/CBMiQ2h0dHBz..."
        result = _resolve_google_news_url(google_url, "No links here")
        assert result == google_url

    def test_empty_description(self) -> None:
        google_url = "https://news.google.com/rss/articles/CBMiQ2h0dHBz..."
        result = _resolve_google_news_url(google_url, "")
        assert result == google_url

    def test_parse_entry_resolves_google_url(self) -> None:
        """_parse_entry should resolve Google News URLs in the parsed entry."""
        raw_entry = {
            "title": "Solo Leveling S3 Release Date",
            "link": "https://news.google.com/rss/articles/CBMiQ2h0dHBz...",
            "summary": '<a href="https://animenews.com/solo-leveling-s3" target="_blank">Title</a>',
        }
        entry = _parse_entry(raw_entry, "https://news.google.com/rss/search?q=test")
        assert entry is not None
        assert entry.url == "https://animenews.com/solo-leveling-s3"


class TestFetchFeedsForTopic:
    async def test_combines_multiple_feeds(self) -> None:
        transport = _mock_transport(
            {
                "feed1": (200, _SAMPLE_RSS),
                "feed2": (200, _SAMPLE_ATOM),
            }
        )
        topic = Topic(
            name="T",
            description="d",
            feed_mode=FeedMode.MANUAL,
            feed_urls=[
                "https://example.com/feed1.xml",
                "https://reddit.com/feed2.rss",
            ],
        )
        original_init = httpx.AsyncClient.__init__

        def patched_init(self_client, **kwargs):
            kwargs["transport"] = transport
            original_init(self_client, **kwargs)

        with patch.object(httpx.AsyncClient, "__init__", patched_init):
            response = await fetch_feeds_for_topic(topic)

        assert len(response.entries) == 3  # 2 from RSS + 1 from Atom
        assert response.provider_name is None  # MANUAL mode

    async def test_deduplicates_by_url(self) -> None:
        # Both feeds return the same RSS content
        transport = _mock_transport(
            {
                "feed1": (200, _SAMPLE_RSS),
                "feed2": (200, _SAMPLE_RSS),
            }
        )
        topic = Topic(
            name="T",
            description="d",
            feed_mode=FeedMode.MANUAL,
            feed_urls=[
                "https://example.com/feed1.xml",
                "https://example.com/feed2.xml",
            ],
        )
        original_init = httpx.AsyncClient.__init__

        def patched_init(self_client, **kwargs):
            kwargs["transport"] = transport
            original_init(self_client, **kwargs)

        with patch.object(httpx.AsyncClient, "__init__", patched_init):
            response = await fetch_feeds_for_topic(topic)

        # Should dedup: both feeds have same 2 URLs
        assert len(response.entries) == 2

    async def test_empty_feed_urls_manual_mode(self) -> None:
        topic = Topic(name="T", description="d", feed_mode=FeedMode.MANUAL, feed_urls=[])
        response = await fetch_feeds_for_topic(topic)
        assert response.entries == []

    async def test_one_feed_failure_doesnt_stop_others(self) -> None:
        transport = _mock_transport(
            {
                "good": (200, _SAMPLE_RSS),
                "bad": (500, "Error"),
            }
        )
        topic = Topic(
            name="T",
            description="d",
            feed_mode=FeedMode.MANUAL,
            feed_urls=[
                "https://example.com/good.xml",
                "https://example.com/bad.xml",
            ],
        )
        original_init = httpx.AsyncClient.__init__

        def patched_init(self_client, **kwargs):
            kwargs["transport"] = transport
            original_init(self_client, **kwargs)

        with patch.object(httpx.AsyncClient, "__init__", patched_init):
            response = await fetch_feeds_for_topic(topic)

        assert len(response.entries) == 2  # Got entries from the good feed

    async def test_auto_mode_uses_router(self) -> None:
        """Auto mode uses the router to select a provider (Bing first by default)."""
        from app.scraping.routing import ProviderRouter

        transport = _mock_transport({"bing.com": (200, _SAMPLE_RSS)})
        topic = Topic(name="Test Topic", description="d", feed_mode=FeedMode.AUTO)
        router = ProviderRouter()

        original_init = httpx.AsyncClient.__init__

        def patched_init(self_client, **kwargs):
            kwargs["transport"] = transport
            original_init(self_client, **kwargs)

        with patch.object(httpx.AsyncClient, "__init__", patched_init):
            response = await fetch_feeds_for_topic(topic, router=router)

        assert len(response.entries) == 2
        assert response.provider_name == "bing_news"
        assert response.needs_url_resolution is False

    async def test_manual_mode_ignores_auto_url(self) -> None:
        """Manual mode uses feed_urls, not auto-generated URL."""
        transport = _mock_transport({"example.com": (200, _SAMPLE_RSS)})
        topic = Topic(
            name="T",
            description="d",
            feed_mode=FeedMode.MANUAL,
            feed_urls=["https://example.com/feed.xml"],
        )
        original_init = httpx.AsyncClient.__init__

        def patched_init(self_client, **kwargs):
            kwargs["transport"] = transport
            original_init(self_client, **kwargs)

        with patch.object(httpx.AsyncClient, "__init__", patched_init):
            response = await fetch_feeds_for_topic(topic)

        assert len(response.entries) == 2
        assert response.provider_name is None  # MANUAL mode has no provider

    async def test_auto_fallback_on_empty(self) -> None:
        """When first provider returns empty, falls back to second."""
        from app.scraping.providers import BingNewsProvider, GoogleNewsProvider
        from app.scraping.routing import ProviderRouter

        transport = _mock_transport(
            {
                "bing.com": (200, _EMPTY_RSS),
                "news.google.com": (200, _SAMPLE_RSS),
            }
        )
        topic = Topic(name="Test", description="d", feed_mode=FeedMode.AUTO)
        router = ProviderRouter(providers=[BingNewsProvider(), GoogleNewsProvider()])

        original_init = httpx.AsyncClient.__init__

        def patched_init(self_client, **kwargs):
            kwargs["transport"] = transport
            original_init(self_client, **kwargs)

        with patch.object(httpx.AsyncClient, "__init__", patched_init):
            response = await fetch_feeds_for_topic(topic, router=router)

        assert len(response.entries) == 2
        assert response.provider_name == "google_news"
        assert response.needs_url_resolution is True

    async def test_auto_fallback_on_http_error(self) -> None:
        """When first provider HTTP errors, falls back to second."""
        from app.scraping.providers import BingNewsProvider, GoogleNewsProvider
        from app.scraping.routing import ProviderRouter

        transport = _mock_transport(
            {
                "bing.com": (500, "Error"),
                "news.google.com": (200, _SAMPLE_RSS),
            }
        )
        topic = Topic(name="Test", description="d", feed_mode=FeedMode.AUTO)
        router = ProviderRouter(providers=[BingNewsProvider(), GoogleNewsProvider()])

        original_init = httpx.AsyncClient.__init__

        def patched_init(self_client, **kwargs):
            kwargs["transport"] = transport
            original_init(self_client, **kwargs)

        with patch.object(httpx.AsyncClient, "__init__", patched_init):
            response = await fetch_feeds_for_topic(topic, router=router)

        assert len(response.entries) == 2
        assert response.provider_name == "google_news"

    async def test_auto_both_fail(self) -> None:
        """When both providers fail, returns empty and marks both unhealthy."""
        from app.scraping.providers import BingNewsProvider, GoogleNewsProvider
        from app.scraping.routing import ProviderRouter

        transport = _mock_transport(
            {
                "bing.com": (500, "Error"),
                "news.google.com": (500, "Error"),
            }
        )
        topic = Topic(name="Test", description="d", feed_mode=FeedMode.AUTO)
        router = ProviderRouter(providers=[BingNewsProvider(), GoogleNewsProvider()])

        original_init = httpx.AsyncClient.__init__

        def patched_init(self_client, **kwargs):
            kwargs["transport"] = transport
            original_init(self_client, **kwargs)

        with patch.object(httpx.AsyncClient, "__init__", patched_init):
            response = await fetch_feeds_for_topic(topic, router=router)

        assert response.entries == []
        # Both providers should have recorded a failure
        assert "bing_news" in router._health
        assert "google_news" in router._health


# ============================================================
# TestExtractArticleContent (async, mocked)
# ============================================================


class TestExtractArticleContent:
    async def test_extracts_from_html(self) -> None:
        transport = _mock_transport({"example.com": (200, _SAMPLE_HTML)})
        async with httpx.AsyncClient(transport=transport) as client:
            content = await extract_article_content("https://example.com/article", client=client)
        assert len(content) > 0
        assert "article content" in content.lower() or len(content) > 50

    async def test_fallback_on_fetch_failure(self) -> None:
        transport = _mock_transport({"example.com": (500, "Error")})
        async with httpx.AsyncClient(transport=transport) as client:
            content = await extract_article_content(
                "https://example.com/article",
                fallback_summary="Fallback text",
                client=client,
            )
        assert content == "Fallback text"

    async def test_fallback_on_empty_extraction(self) -> None:
        # Minimal HTML that trafilatura can't extract from
        transport = _mock_transport({"example.com": (200, "<html><body></body></html>")})
        async with httpx.AsyncClient(transport=transport) as client:
            content = await extract_article_content(
                "https://example.com/article",
                fallback_summary="Fallback",
                client=client,
            )
        assert content == "Fallback"

    async def test_truncates_long_content(self) -> None:
        long_html = "<html><body><article>" + "<p>" + "word " * 2000 + "</p>" * 5 + "</article></body></html>"
        transport = _mock_transport({"example.com": (200, long_html)})
        async with httpx.AsyncClient(transport=transport) as client:
            content = await extract_article_content(
                "https://example.com/article",
                client=client,
                max_content_length=100,
            )
        assert len(content) <= 104  # 100 + "..."

    async def test_favor_recall_fallback(self) -> None:
        """When favor_precision fails, favor_recall should be tried before RSS fallback."""
        transport = _mock_transport({"example.com": (200, "<html><body><p>Some content</p></body></html>")})
        call_count = 0

        def mock_extract(html, **kwargs):
            nonlocal call_count
            call_count += 1
            if kwargs.get("favor_precision"):
                return None  # Precision fails
            if kwargs.get("favor_recall"):
                return "Recall extracted content"
            return None

        async with httpx.AsyncClient(transport=transport) as client:
            with patch("app.scraping.content.trafilatura.extract", side_effect=mock_extract):
                content = await extract_article_content(
                    "https://example.com/article",
                    fallback_summary="RSS fallback",
                    client=client,
                )
        assert content == "Recall extracted content"
        assert call_count == 2

    async def test_both_extractions_fail_uses_summary(self) -> None:
        """When both precision and recall fail, RSS summary is used."""
        transport = _mock_transport({"example.com": (200, "<html><body></body></html>")})

        with patch("app.scraping.content.trafilatura.extract", return_value=None):
            async with httpx.AsyncClient(transport=transport) as client:
                content = await extract_article_content(
                    "https://example.com/article",
                    fallback_summary="RSS summary fallback",
                    client=client,
                )
        assert content == "RSS summary fallback"


# ============================================================
# TestTruncate
# ============================================================


class TestTruncate:
    def test_short_text_unchanged(self) -> None:
        assert _truncate("hello world", 100) == "hello world"

    def test_truncates_at_word_boundary(self) -> None:
        result = _truncate("hello beautiful world", 15)
        # text[:15] = "hello beautiful", rfind(" ") = 5 → "hello..."
        assert result == "hello..."

    def test_empty_string(self) -> None:
        assert _truncate("", 100) == ""

    def test_exact_length(self) -> None:
        assert _truncate("12345", 5) == "12345"

    def test_single_long_word(self) -> None:
        result = _truncate("abcdefghij", 5)
        # No space found, so rfind returns -1 (not > 0), uses full truncated slice
        assert result == "abcde..."


# ============================================================
# TestFetchNewArticlesForTopic (async, uses db_conn)
# ============================================================


class TestFetchNewArticlesForTopic:
    def _make_topic(self, conn: sqlite3.Connection) -> Topic:
        topic = create_topic(conn, Topic(name="ScrapeTopic", description="d"))
        conn.commit()
        return topic

    async def test_stores_new_articles(self, db_conn: sqlite3.Connection) -> None:
        topic = self._make_topic(db_conn)
        topic.feed_urls = ["https://example.com/feed.xml"]

        entries = [
            FeedEntry(
                title="New Article",
                url="https://example.com/new",
                summary="Summary",
                source_feed="https://example.com/feed.xml",
            )
        ]
        with (
            patch("app.scraping.fetch_feeds_for_topic", return_value=FeedResponse(entries=entries)),
            patch("app.scraping.extract_article_content", return_value="Extracted content"),
        ):
            stored = (await fetch_new_articles_for_topic(topic, db_conn)).articles

        assert len(stored) == 1
        assert stored[0].title == "New Article"
        assert stored[0].raw_content == "Extracted content"

        articles = list_articles_for_topic(db_conn, topic.id)
        assert len(articles) == 1

    async def test_skips_duplicates(self, db_conn: sqlite3.Connection) -> None:
        topic = self._make_topic(db_conn)

        entry = FeedEntry(
            title="Existing",
            url="https://example.com/existing",
            summary="Summary",
            source_feed="feed",
        )
        content_hash = compute_article_hash(entry.url, entry.title)

        # Pre-store the article
        from app.crud import create_article

        create_article(
            db_conn,
            Article(
                topic_id=topic.id,
                title="Existing",
                url="https://example.com/existing",
                content_hash=content_hash,
                source_feed="feed",
            ),
        )
        db_conn.commit()

        with patch("app.scraping.fetch_feeds_for_topic", return_value=FeedResponse(entries=[entry])):
            stored = (await fetch_new_articles_for_topic(topic, db_conn)).articles

        assert len(stored) == 0

    async def test_respects_max_articles(self, db_conn: sqlite3.Connection) -> None:
        topic = self._make_topic(db_conn)

        entries = [
            FeedEntry(
                title=f"Article {i}",
                url=f"https://example.com/{i}",
                summary=f"Summary {i}",
                source_feed="feed",
                published=datetime(2025, 1, i + 1, tzinfo=UTC),
            )
            for i in range(5)
        ]
        with (
            patch("app.scraping.fetch_feeds_for_topic", return_value=FeedResponse(entries=entries)),
            patch("app.scraping.extract_article_content", return_value="Content"),
        ):
            stored = (await fetch_new_articles_for_topic(topic, db_conn, max_articles=2)).articles

        assert len(stored) == 2

    async def test_no_feeds_returns_empty(self, db_conn: sqlite3.Connection) -> None:
        topic = self._make_topic(db_conn)

        with patch("app.scraping.fetch_feeds_for_topic", return_value=FeedResponse()):
            stored = (await fetch_new_articles_for_topic(topic, db_conn)).articles

        assert stored == []

    async def test_content_extraction_failure_uses_fallback(self, db_conn: sqlite3.Connection) -> None:
        topic = self._make_topic(db_conn)

        entries = [
            FeedEntry(
                title="Fail Article",
                url="https://example.com/fail",
                summary="Fallback summary text",
                source_feed="feed",
            )
        ]
        with (
            patch("app.scraping.fetch_feeds_for_topic", return_value=FeedResponse(entries=entries)),
            patch(
                "app.scraping.extract_article_content",
                side_effect=Exception("Network error"),
            ),
        ):
            stored = (await fetch_new_articles_for_topic(topic, db_conn)).articles

        assert len(stored) == 1
        assert stored[0].raw_content == "Fallback summary text"


# ============================================================
# TestFeedFetchRetry
# ============================================================


class TestFeedFetchRetry:
    """Tests for retry logic in feed fetching."""

    async def test_retries_on_timeout(self) -> None:
        """Feed fetch retries once on timeout, then succeeds."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.ReadTimeout("timeout")
            return httpx.Response(200, text=_SAMPLE_RSS)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            entries = await fetch_feed("https://example.com/feed.xml", client)

        assert len(entries) == 2
        assert call_count == 2

    async def test_retries_on_server_error(self) -> None:
        """Feed fetch retries once on 500, then succeeds."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(500, text="Server Error")
            return httpx.Response(200, text=_SAMPLE_RSS)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            entries = await fetch_feed("https://example.com/feed.xml", client)

        assert len(entries) == 2
        assert call_count == 2

    async def test_no_retry_on_client_error(self) -> None:
        """Feed fetch does not retry on 4xx errors."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(404, text="Not Found")

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            entries = await fetch_feed("https://example.com/feed.xml", client)

        assert entries == []
        assert call_count == 1

    async def test_returns_empty_after_max_retries(self) -> None:
        """Feed fetch returns empty after exhausting retries."""

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timeout")

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            entries = await fetch_feed("https://example.com/feed.xml", client)

        assert entries == []


# ============================================================
# TestSSRFProtection (scraping layer)
# ============================================================


class TestSSRFBlockInFetch:
    """Tests that SSRF checks block private URLs at fetch time."""

    async def test_blocks_localhost(self) -> None:
        entries = await fetch_feed("http://localhost/feed.xml")
        assert entries == []

    async def test_blocks_private_ip(self) -> None:
        entries = await fetch_feed("http://192.168.1.1/feed.xml")
        assert entries == []

    async def test_blocks_loopback(self) -> None:
        entries = await fetch_feed("http://127.0.0.1/feed.xml")
        assert entries == []

    async def test_blocks_ipv6_loopback(self) -> None:
        entries = await fetch_feed("http://[::1]/feed.xml")
        assert entries == []

    async def test_blocks_ipv6_ula(self) -> None:
        entries = await fetch_feed("http://[fd00::1]/feed.xml")
        assert entries == []

    async def test_blocks_ipv6_link_local(self) -> None:
        entries = await fetch_feed("http://[fe80::1]/feed.xml")
        assert entries == []

    async def test_blocks_ipv6_mapped_ipv4(self) -> None:
        entries = await fetch_feed("http://[::ffff:127.0.0.1]/feed.xml")
        assert entries == []
