"""Tests for the scraping pipeline: RSS fetching, content extraction, orchestration."""

import sqlite3
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import trafilatura

from app.analysis.prompts import _STUB_CONTENT_MIN_CHARS, _content_quality_tag
from app.crud import create_article, create_topic, list_articles_for_topic
from app.models import Article, FeedHealth, FeedMode, Topic
from app.scraping import fetch_new_articles_for_topic
from app.scraping.content import _truncate, extract_article_content
from app.scraping.providers import GoogleNewsProvider
from app.scraping.rss import (
    FeedEntry,
    FeedResponse,
    _parse_entry,
    _parse_feed_date,
    _resolve_bing_news_url,
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

    def test_html_summary_is_stripped_to_text(self) -> None:
        """OVH-112: HTML in an RSS summary is stripped to plain text before storage.

        Google News summaries are <ol><li><a> link lists; storing that raw as
        raw_content wastes the novelty-prompt budget on tag/href noise. The link
        text survives, the markup does not.
        """
        raw = {
            "title": "Roundup",
            "link": "https://example.com/roundup",
            "summary": (
                '<ol><li><a href="https://news.google.com/x">Headline One</a></li>'
                '<li><a href="https://news.google.com/y">Headline Two</a></li></ol>'
            ),
        }
        entry = _parse_entry(raw, "feed")
        assert entry is not None
        assert "Headline One" in entry.summary
        assert "Headline Two" in entry.summary
        # No markup, no href noise survives into the stored summary.
        assert "<" not in entry.summary and ">" not in entry.summary
        assert "href" not in entry.summary
        assert "news.google.com" not in entry.summary

    def test_html_entities_in_summary_are_unescaped(self) -> None:
        """OVH-112: HTML entities decode to their characters when tags are stripped."""
        raw = {
            "title": "Entities",
            "link": "https://example.com/e",
            "summary": "<p>Apple &amp; Google &lt;merge&gt;</p>",
        }
        entry = _parse_entry(raw, "feed")
        assert entry is not None
        assert entry.summary == "Apple & Google <merge>"

    def test_plain_text_summary_is_unchanged(self) -> None:
        """OVH-112: a tag-free summary passes through verbatim (no false rewrites)."""
        raw = {
            "title": "Plain",
            "link": "https://example.com/p",
            "summary": "Just a plain sentence with no markup.",
        }
        entry = _parse_entry(raw, "feed")
        assert entry is not None
        assert entry.summary == "Just a plain sentence with no markup."


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

    async def test_malformed_xml_is_detected_as_bozo_failure(self) -> None:
        """OVH-165: empty result alone is ambiguous — pin that a malformed feed is
        detected via feedparser's bozo flag (not just coincidentally zero entries).

        ``fetch_feed`` collapses to ``[]`` for both a healthy-empty feed and a
        malformed one. Reaching through ``fetch_feed_with_status`` asserts the
        malformed body is a *soft failure* (``fetch_ok=False`` + a health-failure
        callback carrying the bozo exception), so a persistently malformed feed is
        no longer silently counted as a healthy empty fetch.
        """
        from unittest.mock import MagicMock

        from app.scraping.rss import fetch_feed_with_status

        callback = MagicMock()
        transport = _mock_transport({"example.com": (200, "not xml at all {{{")})
        async with httpx.AsyncClient(transport=transport) as client:
            entries, fetch_ok = await fetch_feed_with_status(
                "https://example.com/feed", client, health_callback=callback
            )

        assert entries == []
        assert fetch_ok is False
        callback.assert_called_once()
        url_arg, ok_arg, reason_arg = callback.call_args[0][:3]
        assert url_arg == "https://example.com/feed"
        assert ok_arg is False
        assert reason_arg is not None  # carries the bozo exception text

    async def test_one_malformed_entry_keeps_other_entries(self, caplog: pytest.LogCaptureFixture) -> None:
        """OVH-024: a single bad entry must not discard the whole feed."""
        from app.scraping import rss

        real_parse_entry = rss._parse_entry

        def flaky_parse_entry(raw_entry: dict, source_feed: str) -> FeedEntry | None:
            # Blow up only on the first entry; parse the rest normally.
            if raw_entry.get("title") == "Article One":
                raise ValueError("boom: malformed entry")
            return real_parse_entry(raw_entry, source_feed)

        transport = _mock_transport({"example.com/feed": (200, _SAMPLE_RSS)})
        with (
            patch("app.scraping.rss._parse_entry", side_effect=flaky_parse_entry),
            caplog.at_level("WARNING"),
        ):
            async with httpx.AsyncClient(transport=transport) as client:
                entries = await fetch_feed("https://example.com/feed.xml", client)

        # The valid second entry survives; the malformed first one is skipped.
        assert [e.title for e in entries] == ["Article Two"]
        assert any("entry" in r.getMessage().lower() for r in caplog.records)


# ============================================================
# TestFeedBozoHandling (OVH-044)
# ============================================================


# A body feedparser flags bozo but from which it still recovers an entry.
_BOZO_RECOVERED_RSS = (
    '<?xml version="1.0"?><rss version="2.0"><channel><title>T</title>'
    "<item><title>Recovered & Co</title><link>https://example.com/ok</link></item>"
    "</channel></rss>"
)


class TestFeedBozoHandling:
    """OVH-044: feedparser bozo + zero entries is a soft failure."""

    async def test_bozo_empty_marks_unhealthy(self) -> None:
        """Bozo + zero entries => fetch_ok=False and a failure health callback."""
        from unittest.mock import MagicMock

        from app.scraping.rss import fetch_feed_with_status

        callback = MagicMock()
        transport = _mock_transport({"example.com/feed": (200, "not xml at all {{{")})
        async with httpx.AsyncClient(transport=transport) as client:
            entries, fetch_ok = await fetch_feed_with_status(
                "https://example.com/feed.xml", client, health_callback=callback
            )

        assert entries == []
        assert fetch_ok is False
        callback.assert_called_once()
        args = callback.call_args[0]
        assert args[0] == "https://example.com/feed.xml"
        assert args[1] is False
        assert args[2] is not None  # carries the bozo exception text

    async def test_bozo_with_recovered_entries_succeeds(self) -> None:
        """Bozo but entries recovered => proceed as success."""
        from unittest.mock import MagicMock

        from app.scraping.rss import fetch_feed_with_status

        callback = MagicMock()
        transport = _mock_transport({"example.com/feed": (200, _BOZO_RECOVERED_RSS)})
        async with httpx.AsyncClient(transport=transport) as client:
            entries, fetch_ok = await fetch_feed_with_status(
                "https://example.com/feed.xml", client, health_callback=callback
            )

        assert fetch_ok is True
        assert [e.title for e in entries] == ["Recovered & Co"]
        callback.assert_called_once_with("https://example.com/feed.xml", True, None, None, None)


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


class TestResolveBingNewsUrl:
    # Real Bing RSS link: an apiclick redirect with the publisher URL fully
    # percent-encoded in the ``url=`` query param (host is NOT encoded).
    _APICLICK = (
        "http://www.bing.com/news/apiclick.aspx?ref=FexRss&aid=&tid=ABC"
        "&url=https%3a%2f%2fpublisher.example%2fstory&c=123&mkt=es-es"
    )

    def test_unwraps_apiclick_to_real_url(self) -> None:
        assert _resolve_bing_news_url(self._APICLICK) == "https://publisher.example/story"

    def test_http_wrapper_keeps_https_target(self) -> None:
        # The http:// wrapper must not downgrade the decoded https:// target.
        assert _resolve_bing_news_url(self._APICLICK).startswith("https://")

    def test_non_bing_url_unchanged(self) -> None:
        url = "https://example.com/article"
        assert _resolve_bing_news_url(url) == url

    def test_google_news_url_unchanged(self) -> None:
        # The two resolvers run back-to-back in _parse_entry; a Google URL must
        # pass through the Bing resolver untouched.
        url = "https://news.google.com/rss/articles/CBMiQ2h0dHBz..."
        assert _resolve_bing_news_url(url) == url

    def test_apiclick_without_url_param_unchanged(self) -> None:
        link = "http://www.bing.com/news/apiclick.aspx?ref=FexRss&tid=ABC&c=1"
        assert _resolve_bing_news_url(link) == link

    def test_apiclick_empty_url_param_falls_back(self) -> None:
        # parse_qs drops blank values, so an empty url= yields no target.
        link = "http://www.bing.com/news/apiclick.aspx?url=&c=1"
        assert _resolve_bing_news_url(link) == link

    def test_non_http_target_rejected(self) -> None:
        link = "http://www.bing.com/news/apiclick.aspx?url=javascript%3aalert(1)&c=1"
        assert _resolve_bing_news_url(link) == link

    def test_self_referential_target_rejected(self) -> None:
        # A url= that decodes to another bing apiclick must not be adopted (loop guard).
        link = (
            "http://www.bing.com/news/apiclick.aspx?url=https%3a%2f%2fwww.bing.com%2fnews%2fapiclick.aspx%3furl%3dx&c=1"
        )
        assert _resolve_bing_news_url(link) == link

    def test_double_encoded_target_rejected(self) -> None:
        # parse_qs decodes exactly once; a double-encoded target stays scheme-less
        # ("https%3a%2f...") and is rejected, pinning the single-decode assumption.
        link = "http://www.bing.com/news/apiclick.aspx?url=https%253a%252f%252fa.test%252fx&c=1"
        assert _resolve_bing_news_url(link) == link

    def test_multiple_url_params_takes_first(self) -> None:
        link = (
            "http://www.bing.com/news/apiclick.aspx?url=https%3a%2f%2fa.test%2fone&url=https%3a%2f%2fb.test%2ftwo&c=1"
        )
        assert _resolve_bing_news_url(link) == "https://a.test/one"

    def test_uppercase_host_and_path_still_unwraps(self) -> None:
        link = "http://WWW.BING.COM/news/APICLICK.ASPX?url=https%3a%2f%2fa.test%2fx&c=1"
        assert _resolve_bing_news_url(link) == "https://a.test/x"

    def test_host_with_port_still_unwraps(self) -> None:
        # .hostname strips the port; a .netloc-based check would fail this.
        link = "http://www.bing.com:80/news/apiclick.aspx?url=https%3a%2f%2fa.test%2fx&c=1"
        assert _resolve_bing_news_url(link) == "https://a.test/x"


class TestBingStubRegression:
    """Locks the [STUB] fix end-to-end: a Bing apiclick entry must unwrap to the
    real publisher URL so extraction yields real content (>=200 chars, no STUB
    tag) instead of the short RSS-summary fallback the redirect forces."""

    _REAL_URL = "https://publisher.example/bing-real-article"
    _APICLICK = (
        "http://www.bing.com/news/apiclick.aspx?ref=FexRss&aid=&tid=ABC"
        "&url=https%3a%2f%2fpublisher.example%2fbing-real-article&c=123&mkt=es-es"
    )

    def test_parse_entry_unwraps_and_keys_hash_off_real_url(self) -> None:
        entry = _parse_entry(
            {"title": "Wrapped headline", "link": self._APICLICK, "summary": "short"},
            "https://www.bing.com/news/search?q=test&format=rss",
        )
        assert entry is not None
        assert entry.url == self._REAL_URL
        # Dedup now keys off the real URL, not the apiclick redirect.
        assert compute_article_hash(entry.url, entry.title) == compute_article_hash(self._REAL_URL, "Wrapped headline")
        assert compute_article_hash(entry.url, entry.title) != compute_article_hash(self._APICLICK, "Wrapped headline")

    async def test_unwrapped_url_clears_stub_tag(self) -> None:
        entry = _parse_entry(
            {"title": "Wrapped headline", "link": self._APICLICK, "summary": "short"},
            "https://www.bing.com/news/search?q=test&format=rss",
        )
        assert entry is not None
        # Mock keyed on the DECODED path ("/bing-real-article" with a real slash);
        # the apiclick URL carries it percent-encoded (%2f), so before the fix the
        # fetch would hit apiclick, miss the mock, and fall back to the short summary.
        transport = _mock_transport({"publisher.example/bing-real-article": (200, _SAMPLE_HTML)})
        with (
            patch("app.scraping.content.is_private_url", return_value=False),
            patch("app.url_validation.is_private_url", return_value=False),
        ):
            async with httpx.AsyncClient(transport=transport) as client:
                content = await extract_article_content(entry.url, fallback_summary=entry.summary, client=client)
        assert len(content) >= _STUB_CONTENT_MIN_CHARS
        assert _content_quality_tag(content) == ""


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

    async def test_manual_partial_failure_counts_feeds(self) -> None:
        """OVH-130: a manual fetch where some feeds fail reports feeds_total /
        feeds_failed so a degraded check is distinguishable from a healthy one."""
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

        assert response.feeds_total == 2
        assert response.feeds_failed == 1

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

    async def test_cascade_log_labels_failed_vs_empty(self, caplog: pytest.LogCaptureFixture) -> None:
        """OVH-133: the cascade log distinguishes a failed provider from an empty
        one and names the reason it is cascading."""
        import logging

        from app.scraping.providers import BingNewsProvider, GoogleNewsProvider
        from app.scraping.routing import ProviderRouter

        transport = _mock_transport(
            {
                "bing.com": (500, "Error"),  # failed, not empty
                "news.google.com": (200, _SAMPLE_RSS),
            }
        )
        topic = Topic(name="Test", description="d", feed_mode=FeedMode.AUTO)
        router = ProviderRouter(providers=[BingNewsProvider(), GoogleNewsProvider()])

        original_init = httpx.AsyncClient.__init__

        def patched_init(self_client, **kwargs):
            kwargs["transport"] = transport
            original_init(self_client, **kwargs)

        with (
            patch.object(httpx.AsyncClient, "__init__", patched_init),
            caplog.at_level(logging.INFO, logger="app.scraping.rss"),
        ):
            await fetch_feeds_for_topic(topic, router=router)

        cascade_logs = [r.getMessage() for r in caplog.records if "cascading to" in r.getMessage()]
        assert len(cascade_logs) == 1
        # Distinguishes 'fetch failed' from an empty result, and names the target.
        assert "fetch failed" in cascade_logs[0]
        assert "bing_news" in cascade_logs[0]
        assert "google_news" in cascade_logs[0]

    async def test_cascade_log_labels_empty_result(self, caplog: pytest.LogCaptureFixture) -> None:
        """OVH-133: an empty-but-OK first provider is labelled 'no entries', not 'failed'."""
        import logging

        from app.scraping.providers import BingNewsProvider, GoogleNewsProvider
        from app.scraping.routing import ProviderRouter

        transport = _mock_transport(
            {
                "bing.com": (200, _EMPTY_RSS),  # empty, not failed
                "news.google.com": (200, _SAMPLE_RSS),
            }
        )
        topic = Topic(name="Test", description="d", feed_mode=FeedMode.AUTO)
        router = ProviderRouter(providers=[BingNewsProvider(), GoogleNewsProvider()])

        original_init = httpx.AsyncClient.__init__

        def patched_init(self_client, **kwargs):
            kwargs["transport"] = transport
            original_init(self_client, **kwargs)

        with (
            patch.object(httpx.AsyncClient, "__init__", patched_init),
            caplog.at_level(logging.INFO, logger="app.scraping.rss"),
        ):
            await fetch_feeds_for_topic(topic, router=router)

        cascade_logs = [r.getMessage() for r in caplog.records if "cascading to" in r.getMessage()]
        assert len(cascade_logs) == 1
        assert "no entries" in cascade_logs[0]
        assert "fetch failed" not in cascade_logs[0]

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

    async def test_empty_but_ok_feed_does_not_mark_unhealthy(self) -> None:
        """A 200 response with zero entries is NOT a failure — provider health unchanged."""
        from app.scraping.providers import BingNewsProvider, GoogleNewsProvider
        from app.scraping.routing import ProviderRouter

        transport = _mock_transport(
            {
                "bing.com": (200, _EMPTY_RSS),
                "news.google.com": (200, _EMPTY_RSS),
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
        # Legitimately-empty feeds must NOT count as provider failures.
        assert "bing_news" not in router._health
        assert "google_news" not in router._health

    async def test_fetch_error_still_marks_unhealthy(self) -> None:
        """A real fetch error (HTTP 500) still marks the provider unhealthy."""
        from app.scraping.providers import BingNewsProvider, GoogleNewsProvider
        from app.scraping.routing import ProviderRouter

        transport = _mock_transport(
            {
                "bing.com": (500, "Error"),
                "news.google.com": (200, _EMPTY_RSS),
            }
        )
        topic = Topic(name="Test", description="d", feed_mode=FeedMode.AUTO)
        router = ProviderRouter(providers=[BingNewsProvider(), GoogleNewsProvider()])

        original_init = httpx.AsyncClient.__init__

        def patched_init(self_client, **kwargs):
            kwargs["transport"] = transport
            original_init(self_client, **kwargs)

        with patch.object(httpx.AsyncClient, "__init__", patched_init):
            await fetch_feeds_for_topic(topic, router=router)

        # Bing genuinely errored — must be marked unhealthy.
        assert "bing_news" in router._health
        # Google returned empty-but-OK — must NOT be marked unhealthy.
        assert "google_news" not in router._health


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

    async def test_extraction_nothing_logs_fallback(self, caplog: pytest.LogCaptureFixture) -> None:
        """OVH-045: html present but extraction empty logs the RSS-summary fallback."""
        transport = _mock_transport({"example.com": (200, "<html><body></body></html>")})

        with patch("app.scraping.content.trafilatura.extract", return_value=None):
            async with httpx.AsyncClient(transport=transport) as client:
                with caplog.at_level("INFO", logger="app.scraping.content"):
                    content = await extract_article_content(
                        "https://example.com/article",
                        fallback_summary="RSS summary",
                        client=client,
                    )

        assert content == "RSS summary"
        assert any(
            "extracted nothing" in r.getMessage() and "example.com/article" in r.getMessage() for r in caplog.records
        )

    async def test_parses_html_once_for_both_passes(self) -> None:
        """OVH-115: the raw HTML DOM is parsed once via load_html, not re-parsed per pass.

        Both the precision and recall extraction passes must run against the
        same parsed tree, so load_html is called exactly once and trafilatura
        never receives the raw HTML string for re-parsing.
        """
        raw_html = "<html><body><p>Some content</p></body></html>"
        transport = _mock_transport({"example.com": (200, raw_html)})

        sentinel_tree = object()
        extract_inputs: list[object] = []

        def mock_extract(parsed, **kwargs):
            extract_inputs.append(parsed)
            return None  # force both passes to run

        with (
            patch("app.scraping.content.trafilatura.load_html", return_value=sentinel_tree) as mock_load,
            patch("app.scraping.content.trafilatura.extract", side_effect=mock_extract),
        ):
            async with httpx.AsyncClient(transport=transport) as client:
                await extract_article_content(
                    "https://example.com/article",
                    fallback_summary="RSS",
                    client=client,
                )

        # Parsed exactly once.
        assert mock_load.call_count == 1
        assert mock_load.call_args.args[0] == raw_html
        # Both extraction passes ran against the single parsed tree, never the raw HTML.
        assert extract_inputs == [sentinel_tree, sentinel_tree]

    async def test_extraction_output_identical_with_single_parse(self) -> None:
        """OVH-115: parse-once must yield byte-identical content to the raw-HTML path."""
        transport = _mock_transport({"example.com": (200, _SAMPLE_HTML)})
        async with httpx.AsyncClient(transport=transport) as client:
            content = await extract_article_content("https://example.com/article", client=client)

        # Reference: extract directly from the same raw HTML the old impl used.
        reference = trafilatura.extract(_SAMPLE_HTML, favor_precision=True)
        assert reference is not None
        assert content == _truncate(reference, 5000)


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

    async def test_partial_feed_failure_surfaced(
        self, db_conn: sqlite3.Connection, caplog: pytest.LogCaptureFixture
    ) -> None:
        """OVH-130: a degraded check (some feeds failed) propagates the counts to
        the FetchResult and logs a WARNING so it is distinguishable from healthy."""
        import logging

        topic = self._make_topic(db_conn)
        entries = [
            FeedEntry(
                title="Survivor",
                url="https://example.com/survivor",
                summary="S",
                source_feed="feed",
            )
        ]
        # 1 of 3 feeds survived (2 failed) — total_feed_entries=1 alone hides that.
        response = FeedResponse(entries=entries, feeds_total=3, feeds_failed=2)
        with (
            patch("app.scraping.fetch_feeds_for_topic", return_value=response),
            patch("app.scraping.extract_article_content", return_value="content"),
            caplog.at_level(logging.WARNING, logger="app.scraping"),
        ):
            result = await fetch_new_articles_for_topic(topic, db_conn)

        assert result.feeds_total == 3
        assert result.feeds_failed == 2
        partial = [r.getMessage() for r in caplog.records if "partial feed-fetch failure" in r.getMessage()]
        assert len(partial) == 1
        assert "2 of 3" in partial[0]

    async def test_no_partial_warning_when_all_feeds_ok(
        self, db_conn: sqlite3.Connection, caplog: pytest.LogCaptureFixture
    ) -> None:
        """OVH-130: a fully-healthy fetch (no failures) must NOT log the degraded warning."""
        import logging

        topic = self._make_topic(db_conn)
        entries = [
            FeedEntry(title="A", url="https://example.com/a", summary="S", source_feed="feed"),
        ]
        response = FeedResponse(entries=entries, feeds_total=2, feeds_failed=0)
        with (
            patch("app.scraping.fetch_feeds_for_topic", return_value=response),
            patch("app.scraping.extract_article_content", return_value="content"),
            caplog.at_level(logging.WARNING, logger="app.scraping"),
        ):
            result = await fetch_new_articles_for_topic(topic, db_conn)

        assert result.feeds_failed == 0
        assert not [r for r in caplog.records if "partial feed-fetch failure" in r.getMessage()]

    async def test_feed_health_write_failure_warns(
        self, db_conn: sqlite3.Connection, caplog: pytest.LogCaptureFixture
    ) -> None:
        """OVH-132: a swallowed feed_health DB write surfaces at WARNING, not DEBUG —
        loss of health telemetry must be visible."""
        import logging

        topic = self._make_topic(db_conn)

        async def _fake_fetch(topic_arg, *, timeout, max_attempts, health_callback):
            # Simulate a successful feed fetch that triggers a health write.
            if health_callback:
                health_callback("https://example.com/feed.xml", True, None)
            return FeedResponse(entries=[])

        with (
            patch("app.scraping.fetch_feeds_for_topic", side_effect=_fake_fetch),
            patch("app.scraping.upsert_feed_health_success", side_effect=Exception("locked")),
            caplog.at_level(logging.WARNING, logger="app.scraping"),
        ):
            await fetch_new_articles_for_topic(topic, db_conn)

        warns = [r for r in caplog.records if "Failed to record feed health" in r.getMessage()]
        assert len(warns) == 1
        assert warns[0].levelno == logging.WARNING

    async def test_extraction_batch_reuses_one_pooled_client(self, db_conn: sqlite3.Connection) -> None:
        """OVH-128: the whole extraction batch shares ONE pooled httpx client.

        Every per-article extraction must receive the same non-None client (so
        connection pooling/keep-alive is preserved), and exactly one client is
        constructed for the batch (not one per article).
        """
        topic = self._make_topic(db_conn)
        entries = [
            FeedEntry(
                title=f"Article {i}",
                url=f"https://example.com/{i}",
                summary=f"Summary {i}",
                source_feed="feed",
                published=datetime(2025, 1, i + 1, tzinfo=UTC),
            )
            for i in range(3)
        ]

        clients_constructed = 0
        original_init = httpx.AsyncClient.__init__

        def counting_init(self, *args, **kwargs):
            nonlocal clients_constructed
            clients_constructed += 1
            original_init(self, *args, **kwargs)

        seen_clients: list[object] = []

        async def fake_extract(url, fallback_summary="", client=None, **kwargs):
            seen_clients.append(client)
            return "Extracted"

        with (
            patch("app.scraping.fetch_feeds_for_topic", return_value=FeedResponse(entries=entries)),
            patch("app.scraping.extract_article_content", side_effect=fake_extract),
            patch.object(httpx.AsyncClient, "__init__", counting_init),
        ):
            stored = (await fetch_new_articles_for_topic(topic, db_conn)).articles

        assert len(stored) == 3
        # Every extraction got a real (non-None) client...
        assert all(c is not None for c in seen_clients)
        # ...and they all shared the SAME client instance...
        assert len({id(c) for c in seen_clients}) == 1
        # ...constructed exactly once for the whole batch.
        assert clients_constructed == 1

    async def test_no_client_created_when_nothing_to_fetch(self, db_conn: sqlite3.Connection) -> None:
        """OVH-128: a reuse-only batch (no network fetches) builds no extraction client."""
        topic = self._make_topic(db_conn)

        # Seed a stored article in another topic so the entry is reused cross-topic.
        from app.crud import create_article

        other = create_topic(db_conn, Topic(name="Other", description="d"))
        db_conn.commit()
        entry = FeedEntry(
            title="Shared",
            url="https://example.com/shared",
            summary="s",
            source_feed="feed",
        )
        content_hash = compute_article_hash(entry.url, entry.title)
        create_article(
            db_conn,
            Article(
                topic_id=other.id,
                title="Shared",
                url="https://example.com/shared",
                content_hash=content_hash,
                raw_content="Reused body",
                source_feed="feed",
            ),
        )
        db_conn.commit()

        clients_constructed = 0
        original_init = httpx.AsyncClient.__init__

        def counting_init(self, *args, **kwargs):
            nonlocal clients_constructed
            clients_constructed += 1
            original_init(self, *args, **kwargs)

        with (
            patch("app.scraping.fetch_feeds_for_topic", return_value=FeedResponse(entries=[entry])),
            patch.object(httpx.AsyncClient, "__init__", counting_init),
        ):
            stored = (await fetch_new_articles_for_topic(topic, db_conn)).articles

        assert len(stored) == 1
        assert stored[0].raw_content == "Reused body"
        # Nothing to fetch → no extraction client constructed.
        assert clients_constructed == 0


# ============================================================
# TestPublishedAtPersistence
# ============================================================


class TestPublishedAtPersistence:
    """published_at flows from FeedEntry through _store_articles into the DB."""

    def _make_topic(self, conn: sqlite3.Connection) -> Topic:
        topic = create_topic(conn, Topic(name="PubAtTopic", description="d"))
        conn.commit()
        return topic

    async def test_fresh_fetch_carries_published_at(self, db_conn: sqlite3.Connection) -> None:
        """Fresh-fetch path: published datetime on FeedEntry lands on stored Article."""
        topic = self._make_topic(db_conn)
        pub = datetime(2025, 6, 1, 8, 0, 0, tzinfo=UTC)
        entry = FeedEntry(
            title="Dated Article",
            url="https://example.com/dated",
            summary="Summary",
            source_feed="https://feed.example.com/rss",
            published=pub,
        )
        with (
            patch("app.scraping.fetch_feeds_for_topic", return_value=FeedResponse(entries=[entry])),
            patch("app.scraping.extract_article_content", return_value="Body"),
        ):
            stored = (await fetch_new_articles_for_topic(topic, db_conn)).articles

        assert len(stored) == 1
        assert stored[0].published_at == pub

        # Confirm it survived the DB round-trip.
        loaded = list_articles_for_topic(db_conn, topic.id)
        assert loaded[0].published_at == pub

    async def test_fresh_fetch_none_published_stores_none(self, db_conn: sqlite3.Connection) -> None:
        """FeedEntry with published=None stores published_at=None without error."""
        topic = self._make_topic(db_conn)
        entry = FeedEntry(
            title="Undated Article",
            url="https://example.com/undated",
            summary="Summary",
            source_feed="https://feed.example.com/rss",
            published=None,
        )
        with (
            patch("app.scraping.fetch_feeds_for_topic", return_value=FeedResponse(entries=[entry])),
            patch("app.scraping.extract_article_content", return_value="Body"),
        ):
            stored = (await fetch_new_articles_for_topic(topic, db_conn)).articles

        assert len(stored) == 1
        assert stored[0].published_at is None

    async def test_cross_topic_reuse_carries_published_at(self, db_conn: sqlite3.Connection) -> None:
        """Cross-topic reuse path: published_at from the new FeedEntry is stored."""
        topic_a = create_topic(db_conn, Topic(name="PubAtA", description="d"))
        topic_b = create_topic(db_conn, Topic(name="PubAtB", description="d"))
        db_conn.commit()

        url = "https://example.com/shared-pub"
        title = "Shared Published Article"
        content_hash = compute_article_hash(url, title)
        pub = datetime(2025, 4, 20, 14, 0, 0, tzinfo=UTC)

        # Pre-store the article under topic A (the originating source).
        create_article(
            db_conn,
            Article(
                topic_id=topic_a.id,
                title=title,
                url=url,
                content_hash=content_hash,
                raw_content="Shared body",
                source_feed="https://feed-a.example.com/rss",
            ),
        )
        db_conn.commit()

        # Topic B sees the same URL in its feed — entry carries a published date.
        entry = FeedEntry(
            title=title,
            url=url,
            summary="Summary",
            source_feed="https://feed-b.example.com/rss",
            published=pub,
        )
        with (
            patch("app.scraping.fetch_feeds_for_topic", return_value=FeedResponse(entries=[entry])),
        ):
            stored = (await fetch_new_articles_for_topic(topic_b, db_conn)).articles

        assert len(stored) == 1
        reused = stored[0]
        assert reused.topic_id == topic_b.id
        assert reused.published_at == pub


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

    async def test_exhaustion_retries_full_max_attempts_then_sleeps_between(self) -> None:
        """OVH-074: exhaustion must actually RETRY, not just fail once.

        Pins the exhaustion-after-retrying contract the test name claims: the
        handler is invoked exactly ``max_attempts`` times and ``asyncio.sleep``
        is awaited the between-attempts count (``max_attempts - 1``). Without
        this, a regression silently setting ``max_attempts`` to 1 (or removing
        the retry loop) would pass the sibling 'returns empty' test unchanged.
        """
        max_attempts = 2
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            raise httpx.ReadTimeout("timeout")

        transport = httpx.MockTransport(handler)
        # Patch the backoff sleep in the rss module so the test does not actually
        # wait, and so we can assert it is awaited exactly between attempts.
        with patch("app.scraping.rss.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            async with httpx.AsyncClient(transport=transport) as client:
                entries = await fetch_feed("https://example.com/feed.xml", client, max_attempts=max_attempts)

        assert entries == []
        # Retried to exhaustion: one call per attempt.
        assert call_count == max_attempts
        # Slept only BETWEEN attempts, never after the final failure.
        assert mock_sleep.await_count == max_attempts - 1


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


# ============================================================
# TestSSRFRedirectProtection (redirect re-validation)
# ============================================================


class TestSSRFRedirectProtection:
    """A public URL that 3xx-redirects to a private/loopback host must be
    blocked: the private target is never fetched."""

    async def test_feed_fetch_blocks_redirect_to_loopback(self) -> None:
        """Uses the Phase-0 build_redirect_transport helper: a public feed URL
        302-redirects to loopback; the private target must never be fetched."""
        from tests.helpers.redirect_transport import build_redirect_transport

        target = "http://127.0.0.1/secret.xml"
        base = build_redirect_transport(target, match="public.example.com")
        fetched: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            fetched.append(str(request.url))
            return base.handler(request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False) as client:
            entries = await fetch_feed("https://public.example.com/feed.xml", client)

        assert entries == []
        # The loopback target must NOT have been fetched.
        assert not any("127.0.0.1" in u for u in fetched)

    async def test_feed_fetch_blocks_redirect_to_metadata_ip(self) -> None:
        fetched: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            fetched.append(url)
            if "169.254.169.254" in url:
                return httpx.Response(200, text=_SAMPLE_RSS)
            return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/meta-data"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False) as client:
            entries = await fetch_feed("https://public.example.com/feed.xml", client)

        assert entries == []
        assert not any("169.254.169.254" in u for u in fetched)

    async def test_feed_fetch_allows_public_no_redirect(self) -> None:
        """A normal, non-redirecting public fetch still works."""
        transport = httpx.MockTransport(lambda req: httpx.Response(200, text=_SAMPLE_RSS))
        async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
            entries = await fetch_feed("https://public.example.com/feed.xml", client)
        assert len(entries) == 2

    async def test_feed_fetch_follows_public_redirect(self) -> None:
        """A redirect to another PUBLIC host is followed and parsed."""

        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if "final.example.org" in url:
                return httpx.Response(200, text=_SAMPLE_RSS)
            return httpx.Response(302, headers={"location": "https://final.example.org/feed.xml"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False) as client:
            entries = await fetch_feed("https://public.example.com/feed.xml", client)
        assert len(entries) == 2

    async def test_content_fetch_blocks_redirect_to_loopback(self) -> None:
        fetched: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            fetched.append(url)
            if "127.0.0.1" in url:
                return httpx.Response(200, text=_SAMPLE_HTML)
            return httpx.Response(302, headers={"location": "http://127.0.0.1/secret"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False) as client:
            content = await extract_article_content(
                "https://public.example.com/article",
                fallback_summary="the fallback summary",
                client=client,
            )

        # Falls back to summary because the private redirect was blocked.
        assert content == "the fallback summary"
        assert not any("127.0.0.1" in u for u in fetched)

    async def test_content_fetch_allows_public_no_redirect(self) -> None:
        transport = httpx.MockTransport(lambda req: httpx.Response(200, text=_SAMPLE_HTML))
        async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
            content = await extract_article_content(
                "https://public.example.com/article",
                fallback_summary="fallback",
                client=client,
            )
        assert "additional context" in content.lower() or len(content) > 20


# ============================================================
# TestSafeSendRedirectEdgeCases (hop cap, 303 downgrade, scheme)
# ============================================================


class TestSafeSendRedirectEdgeCases:
    """Direct tests for safe_send redirect handling: hop cap, the 303 ->
    GET method/body downgrade, and rejection of non-http(s) redirect schemes."""

    async def test_exceeding_max_redirects_raises(self) -> None:
        """A redirect chain longer than max_redirects raises PrivateRedirectError."""
        from app.url_validation import PrivateRedirectError, safe_get

        hops: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            hops.append(str(request.url))
            # Always redirect to a fresh public URL, never terminating.
            n = len(hops)
            return httpx.Response(302, headers={"location": f"https://public.example.com/hop/{n}"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False) as client:
            with pytest.raises(PrivateRedirectError, match="Exceeded maximum"):
                await safe_get(client, "https://public.example.com/start", max_redirects=3)

        # Initial request + exactly max_redirects (3) followed hops = 4 sends.
        assert len(hops) == 4

    async def test_303_redirect_downgrades_to_get_and_strips_body(self) -> None:
        """A 303 on a POST must reissue as GET with no body and no content headers."""
        from app.url_validation import safe_send

        seen: list[tuple[str, str, bytes, str | None]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            seen.append((request.method, url, request.content, request.headers.get("content-type")))
            if "final.example.org" in url:
                return httpx.Response(200, text="done")
            return httpx.Response(303, headers={"location": "https://final.example.org/result"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False) as client:
            request = client.build_request(
                "POST",
                "https://public.example.com/submit",
                content=b"payload=1",
                headers={"content-type": "application/x-www-form-urlencoded"},
            )
            response = await safe_send(client, request)

        assert response.status_code == 200
        assert response.text == "done"
        # First hop: original POST with body.
        assert seen[0][0] == "POST"
        assert seen[0][2] == b"payload=1"
        # Second hop: downgraded to GET, body stripped, content-type removed.
        method, url, body, content_type = seen[1]
        assert method == "GET"
        assert "final.example.org" in url
        assert body == b""
        assert content_type is None

    async def test_redirect_to_non_http_scheme_is_rejected(self) -> None:
        """A redirect to a file:// (or other non-http) scheme is blocked and the
        target is never fetched, even though it has no netloc to flag as private."""
        from app.url_validation import PrivateRedirectError, safe_get

        fetched: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            fetched.append(str(request.url))
            return httpx.Response(302, headers={"location": "file:///etc/passwd"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False) as client:
            with pytest.raises(PrivateRedirectError, match="non-http"):
                await safe_get(client, "https://public.example.com/feed.xml")

        # Only the initial public URL was ever requested; file:// never fetched.
        assert fetched == ["https://public.example.com/feed.xml"]


class TestResolveRedirectUrls:
    """OVH-157: redirect resolution is gated by the provider abstraction.

    The orchestrator must not hardcode the Google News host: resolution only
    fires when the provider's ``needs_url_resolution`` is set, and which URLs to
    resolve is delegated to ``is_google_news_url`` / the resolver.
    """

    def _entry(self, url: str) -> FeedEntry:
        return FeedEntry(title="t", url=url, source_feed="https://feed.example/rss")

    async def test_skips_when_provider_does_not_need_resolution(self) -> None:
        from app.scraping import _resolve_redirect_urls

        gnews = "https://news.google.com/rss/articles/ABC123?oc=5"
        fetch_batch = [(self._entry(gnews), "h1")]
        response = FeedResponse(provider_name="bing_news", needs_url_resolution=False)

        with patch("app.scraping.resolve_google_news_urls", new_callable=AsyncMock) as mock_resolve:
            await _resolve_redirect_urls(fetch_batch, response, feed_fetch_timeout=5.0)
        mock_resolve.assert_not_called()
        # URL left untouched.
        assert fetch_batch[0][0].url == gnews

    async def test_resolves_only_google_article_urls_when_gated_on(self) -> None:
        from app.scraping import _resolve_redirect_urls

        gnews = "https://news.google.com/rss/articles/ABC123?oc=5"
        plain = "https://example.com/story"
        fetch_batch = [(self._entry(gnews), "h1"), (self._entry(plain), "h2")]
        response = FeedResponse(provider_name="google_news", needs_url_resolution=True)

        with patch(
            "app.scraping.resolve_google_news_urls",
            new_callable=AsyncMock,
            return_value={gnews: "https://real.example/article"},
        ) as mock_resolve:
            await _resolve_redirect_urls(fetch_batch, response, feed_fetch_timeout=5.0)

        # Only the Google article URL was offered for resolution (not the plain one).
        mock_resolve.assert_awaited_once()
        offered = mock_resolve.await_args.args[0]
        assert offered == [gnews]
        # Resolved URL applied in place; the plain URL is untouched.
        assert fetch_batch[0][0].url == "https://real.example/article"
        assert fetch_batch[1][0].url == plain

    async def test_no_resolver_call_when_no_google_urls(self) -> None:
        from app.scraping import _resolve_redirect_urls

        fetch_batch = [(self._entry("https://example.com/a"), "h1")]
        response = FeedResponse(provider_name="google_news", needs_url_resolution=True)

        with patch("app.scraping.resolve_google_news_urls", new_callable=AsyncMock) as mock_resolve:
            await _resolve_redirect_urls(fetch_batch, response, feed_fetch_timeout=5.0)
        mock_resolve.assert_not_called()


class TestFeedStateHelpers:
    """Phase 1: health callback forwards validators; state loader reads the row."""

    def test_health_callback_forwards_validators(self, db_conn: sqlite3.Connection) -> None:
        from app.crud import get_feed_health
        from app.scraping import _make_health_callback

        cb = _make_health_callback(db_conn)
        cb("https://ex.com/feed", True, None, 'W/"v1"', "LM1")
        db_conn.commit()
        h = get_feed_health(db_conn, "https://ex.com/feed")
        assert h is not None and h.etag == 'W/"v1"' and h.last_modified == "LM1"

    def test_feed_state_loader_returns_row(self, db_conn: sqlite3.Connection) -> None:
        from app.crud import upsert_feed_health_failure
        from app.scraping import _make_feed_state_loader

        upsert_feed_health_failure(db_conn, "https://ex.com/feed", "boom")
        db_conn.commit()
        loader = _make_feed_state_loader(db_conn)
        h = loader("https://ex.com/feed")
        assert h is not None and h.consecutive_failures == 1
        assert loader("https://missing.example/feed") is None


class TestConditionalGet:
    """Phase 1: conditional GET (ETag/Last-Modified) + 304 fast-path."""

    async def test_conditional_get_304_is_empty_but_ok(self) -> None:
        from app.scraping.rss import fetch_feed_with_status

        sent: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            sent.update(request.headers)
            return httpx.Response(304)

        calls: list[tuple] = []

        def cb(url, success, err, etag=None, lm=None):
            calls.append((url, success, err, etag, lm))

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
            entries, ok = await fetch_feed_with_status(
                "https://example.com/feed", client, health_callback=cb, etag='W/"v1"', last_modified="LM1"
            )

        assert entries == [] and ok is True
        assert sent.get("if-none-match") == 'W/"v1"'
        assert sent.get("if-modified-since") == "LM1"
        # success recorded, validators NOT overwritten (None, None preserves them via COALESCE)
        assert calls == [("https://example.com/feed", True, None, None, None)]

    async def test_conditional_get_200_forwards_validators(self) -> None:
        from app.scraping.rss import fetch_feed_with_status

        rss = (
            '<?xml version="1.0"?><rss version="2.0"><channel><title>T</title>'
            "<item><title>A</title><link>https://example.com/a</link></item></channel></rss>"
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=rss, headers={"ETag": 'W/"v2"', "Last-Modified": "LM2"})

        calls: list[tuple] = []

        def cb(url, success, err, etag=None, lm=None):
            calls.append((url, success, err, etag, lm))

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
            entries, ok = await fetch_feed_with_status("https://example.com/feed", client, health_callback=cb)

        assert ok is True and len(entries) == 1
        assert calls == [("https://example.com/feed", True, None, 'W/"v2"', "LM2")]


def _dead_feed_health(url: str) -> FeedHealth:
    return FeedHealth(feed_url=url, consecutive_failures=10, last_error_at=datetime.now(UTC))


class TestManualBackoffAndValidators:
    """Phase 1: MANUAL backoff-skip + validator threading (both modes)."""

    async def test_manual_skips_backed_off_feed(self) -> None:
        topic = Topic(
            name="T",
            description="d",
            feed_urls=["https://live.example/feed", "https://dead.example/feed"],
            feed_mode=FeedMode.MANUAL,
        )
        attempted: list[str] = []

        async def fake_fetch(
            url, client, *, timeout, max_attempts, health_callback=None, etag=None, last_modified=None
        ):
            attempted.append(url)
            return ([FeedEntry(title="A", url="https://live.example/a", source_feed=url)], True)

        def loader(url):
            return _dead_feed_health(url) if url == "https://dead.example/feed" else None

        with patch("app.scraping.rss.fetch_feed_with_status", side_effect=fake_fetch):
            resp = await fetch_feeds_for_topic(
                topic, feed_state_loader=loader, backoff_base_minutes=15, backoff_cap_hours=24
            )

        assert attempted == ["https://live.example/feed"]  # dead feed never fetched
        assert resp.feeds_skipped == 1
        assert resp.feeds_total == 1  # only the attempted feed counts

    async def test_manual_all_backed_off_returns_empty(self) -> None:
        topic = Topic(name="T", description="d", feed_urls=["https://dead.example/feed"], feed_mode=FeedMode.MANUAL)

        async def fake_fetch(*a, **k):  # must never be called
            raise AssertionError("backed-off feed must not be fetched")

        with patch("app.scraping.rss.fetch_feed_with_status", side_effect=fake_fetch):
            resp = await fetch_feeds_for_topic(
                topic, feed_state_loader=_dead_feed_health, backoff_base_minutes=15, backoff_cap_hours=24
            )

        assert resp.feeds_skipped == 1
        assert resp.feeds_total == 0
        assert resp.entries == []

    async def test_auto_sends_stored_validators(self) -> None:
        topic = Topic(name="T", description="d", feed_urls=[], feed_mode=FeedMode.AUTO)
        seen: list[tuple[str | None, str | None]] = []

        async def fake_fetch(
            url, client, *, timeout, max_attempts, health_callback=None, etag=None, last_modified=None
        ):
            seen.append((etag, last_modified))
            return ([FeedEntry(title="A", url="https://x/a", source_feed=url)], True)

        def loader(url):
            return FeedHealth(feed_url=url, etag='W/"auto"', last_modified="LM")

        with patch("app.scraping.rss.fetch_feed_with_status", side_effect=fake_fetch):
            await fetch_feeds_for_topic(topic, feed_state_loader=loader)

        assert seen and seen[0] == ('W/"auto"', "LM")  # AUTO threaded the stored validators

    async def test_auto_cascade_sends_fallback_validators(self) -> None:
        # Forced cascade: primary fails, fallback yields entries — assert the FALLBACK
        # fetch received the fallback URL's own stored validators (the cascade branch).
        from app.scraping.routing import ProviderRouter

        topic = Topic(name="T", description="d", feed_urls=[], feed_mode=FeedMode.AUTO)
        calls: list[tuple[str, str | None]] = []  # (url, etag)

        async def fake_fetch(
            url, client, *, timeout, max_attempts, health_callback=None, etag=None, last_modified=None
        ):
            calls.append((url, etag))
            if len(calls) == 1:  # primary: fetch failed, no entries -> cascade
                return ([], False)
            return ([FeedEntry(title="A", url="https://x/a", source_feed=url)], True)

        def loader(url):
            return FeedHealth(feed_url=url, etag=f"etag::{url[:20]}", last_modified="LM")

        with patch("app.scraping.rss.fetch_feed_with_status", side_effect=fake_fetch):
            # Fresh router so global provider-health state is not mutated.
            await fetch_feeds_for_topic(topic, router=ProviderRouter(), feed_state_loader=loader)

        assert len(calls) == 2  # primary + cascade fallback
        fb_url, fb_etag = calls[1]
        assert fb_etag == f"etag::{fb_url[:20]}"  # fallback got ITS url's stored validator
