"""Tests for Google News redirect URL resolution."""

import json
import sqlite3
from unittest.mock import patch

import httpx

from app.crud import create_topic
from app.models import FeedMode, Topic
from app.scraping import fetch_new_articles_for_topic
from app.scraping.google_news import (
    _extract_article_id,
    is_google_news_url,
    resolve_google_news_url,
    resolve_google_news_urls,
)
from app.scraping.rss import FeedEntry

# --- Mock HTML for Google News article page ---


def _google_article_html(signature: str = "test-sig-123", timestamp: str = "1700000000") -> str:
    """Build a minimal HTML response mimicking a Google News article page."""
    return f"""\
<!DOCTYPE html>
<html>
<body>
<c-wiz>
  <div jscontroller="abc" data-n-a-sg="{signature}" data-n-a-ts="{timestamp}">
  </div>
</c-wiz>
</body>
</html>"""


def _batchexecute_response(decoded_url: str) -> str:
    """Build a mock batchexecute response containing the decoded URL."""
    inner = json.dumps([None, decoded_url])
    # Real responses have trailing metadata items that get stripped by [:-2]
    data = [
        ["wrb.fr", "Fbv4je", inner, None, None, None, "generic"],
        ["di", 8],
        ["af.httprm", 8, "123", 12],
    ]
    return ")]}'\n\n" + json.dumps(data)


def _batchexecute_error_response() -> str:
    """Build a mock batchexecute response with no URL (error case)."""
    data = [
        ["wrb.fr", "Fbv4je", None, None, None, [3], "generic"],
        ["di", 8],
        ["af.httprm", 8, "123", 12],
    ]
    return ")]}'\n\n" + json.dumps(data)


# --- Helpers ---

_GOOGLE_ARTICLE_ID = "CBMiqgFBVV95cUxPVzdBUWVuQzFRS2Jz"
_GOOGLE_RSS_URL = f"https://news.google.com/rss/articles/{_GOOGLE_ARTICLE_ID}?oc=5"
_GOOGLE_ARTICLES_URL = f"https://news.google.com/articles/{_GOOGLE_ARTICLE_ID}"
_REAL_ARTICLE_URL = "https://comicbook.com/anime/solo-leveling-s3"


def _mock_transport(responses: dict[str, tuple[int, str]]) -> httpx.MockTransport:
    """Build a MockTransport that returns canned responses by URL pattern."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        for pattern, (status, body) in responses.items():
            if pattern in url:
                return httpx.Response(status, text=body)
        return httpx.Response(404, text="Not found")

    return httpx.MockTransport(handler)


# ============================================================
# TestIsGoogleNewsUrl
# ============================================================


class TestIsGoogleNewsUrl:
    def test_rss_articles_url(self) -> None:
        assert is_google_news_url("https://news.google.com/rss/articles/CBMi123?oc=5") is True

    def test_articles_url(self) -> None:
        assert is_google_news_url("https://news.google.com/articles/CBMi123") is True

    def test_read_url(self) -> None:
        assert is_google_news_url("https://news.google.com/read/CBMi123") is True

    def test_regular_url(self) -> None:
        assert is_google_news_url("https://example.com/article") is False

    def test_google_but_not_news(self) -> None:
        assert is_google_news_url("https://www.google.com/search?q=test") is False

    def test_google_news_homepage(self) -> None:
        assert is_google_news_url("https://news.google.com/") is False

    def test_google_news_rss_search(self) -> None:
        assert is_google_news_url("https://news.google.com/rss/search?q=test") is False


# ============================================================
# TestExtractArticleId
# ============================================================


class TestExtractArticleId:
    def test_rss_articles_path(self) -> None:
        url = "https://news.google.com/rss/articles/CBMi123abc?oc=5"
        assert _extract_article_id(url) == "CBMi123abc"

    def test_articles_path(self) -> None:
        url = "https://news.google.com/articles/CBMi456def"
        assert _extract_article_id(url) == "CBMi456def"

    def test_read_path(self) -> None:
        url = "https://news.google.com/read/CBMi789ghi"
        assert _extract_article_id(url) == "CBMi789ghi"

    def test_invalid_path(self) -> None:
        assert _extract_article_id("https://news.google.com/") is None

    def test_non_google_url_with_articles_path(self) -> None:
        # _extract_article_id doesn't check hostname (it's a private helper
        # only called after is_google_news_url filters). It just parses paths.
        assert _extract_article_id("https://example.com/articles/test") == "test"

    def test_no_articles_path(self) -> None:
        assert _extract_article_id("https://example.com/page/123") is None


# ============================================================
# TestResolveGoogleNewsUrl
# ============================================================


class TestResolveGoogleNewsUrl:
    async def test_successful_resolution(self) -> None:
        transport = _mock_transport(
            {
                f"/articles/{_GOOGLE_ARTICLE_ID}": (200, _google_article_html()),
                "batchexecute": (200, _batchexecute_response(_REAL_ARTICLE_URL)),
            }
        )
        async with httpx.AsyncClient(transport=transport) as client:
            result = await resolve_google_news_url(_GOOGLE_RSS_URL, client)
        assert result == _REAL_ARTICLE_URL

    async def test_non_google_url_returned_unchanged(self) -> None:
        async with httpx.AsyncClient() as client:
            result = await resolve_google_news_url("https://example.com/article", client)
        assert result == "https://example.com/article"

    async def test_article_page_fetch_failure(self) -> None:
        """Falls back to original URL when article page returns error."""
        transport = _mock_transport(
            {
                f"/articles/{_GOOGLE_ARTICLE_ID}": (500, "Error"),
                f"/rss/articles/{_GOOGLE_ARTICLE_ID}": (500, "Error"),
            }
        )
        async with httpx.AsyncClient(transport=transport) as client:
            result = await resolve_google_news_url(_GOOGLE_RSS_URL, client)
        assert result == _GOOGLE_RSS_URL

    async def test_missing_data_attributes(self) -> None:
        """Falls back when article page HTML lacks signature/timestamp."""
        html = "<html><body><c-wiz><div jscontroller='abc'></div></c-wiz></body></html>"
        transport = _mock_transport(
            {
                f"/articles/{_GOOGLE_ARTICLE_ID}": (200, html),
                f"/rss/articles/{_GOOGLE_ARTICLE_ID}": (200, html),
            }
        )
        async with httpx.AsyncClient(transport=transport) as client:
            result = await resolve_google_news_url(_GOOGLE_RSS_URL, client)
        assert result == _GOOGLE_RSS_URL

    async def test_batchexecute_failure(self) -> None:
        """Falls back when batchexecute returns error."""
        transport = _mock_transport(
            {
                f"/articles/{_GOOGLE_ARTICLE_ID}": (200, _google_article_html()),
                "batchexecute": (500, "Error"),
            }
        )
        async with httpx.AsyncClient(transport=transport) as client:
            result = await resolve_google_news_url(_GOOGLE_RSS_URL, client)
        assert result == _GOOGLE_RSS_URL

    async def test_batchexecute_returns_no_url(self) -> None:
        """Falls back when batchexecute response doesn't contain a URL."""
        transport = _mock_transport(
            {
                f"/articles/{_GOOGLE_ARTICLE_ID}": (200, _google_article_html()),
                "batchexecute": (200, _batchexecute_error_response()),
            }
        )
        async with httpx.AsyncClient(transport=transport) as client:
            result = await resolve_google_news_url(_GOOGLE_RSS_URL, client)
        assert result == _GOOGLE_RSS_URL

    async def test_rate_limit_429(self) -> None:
        """Falls back on 429 rate limiting."""
        transport = _mock_transport(
            {
                f"/articles/{_GOOGLE_ARTICLE_ID}": (429, "Too Many Requests"),
                f"/rss/articles/{_GOOGLE_ARTICLE_ID}": (429, "Too Many Requests"),
            }
        )
        async with httpx.AsyncClient(transport=transport) as client:
            result = await resolve_google_news_url(_GOOGLE_RSS_URL, client)
        assert result == _GOOGLE_RSS_URL

    async def test_fallback_to_rss_articles_path(self) -> None:
        """Uses /rss/articles/ path when /articles/ returns missing attributes."""
        html_no_attrs = "<html><body><c-wiz><div jscontroller='abc'></div></c-wiz></body></html>"

        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if "batchexecute" in url:
                return httpx.Response(200, text=_batchexecute_response(_REAL_ARTICLE_URL))
            if "/rss/articles/" in url:
                return httpx.Response(200, text=_google_article_html())
            if "/articles/" in url:
                return httpx.Response(200, text=html_no_attrs)
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await resolve_google_news_url(_GOOGLE_RSS_URL, client)
        assert result == _REAL_ARTICLE_URL


# ============================================================
# TestResolveGoogleNewsUrls (batch)
# ============================================================


class TestResolveGoogleNewsUrls:
    async def test_resolves_google_urls_only(self) -> None:
        """Only Google News URLs are resolved; others are left alone."""
        urls = [
            _GOOGLE_RSS_URL,
            "https://example.com/article",
        ]

        transport = _mock_transport(
            {
                "/articles/": (200, _google_article_html()),
                "batchexecute": (200, _batchexecute_response(_REAL_ARTICLE_URL)),
            }
        )

        with patch("app.scraping.google_news.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = lambda self: _make_async(httpx.AsyncClient(transport=transport))
            # Use direct patching of the internal client creation
            pass

        # Simpler approach: patch at the resolve level
        async def mock_resolve(url, client):
            if is_google_news_url(url):
                return _REAL_ARTICLE_URL
            return url

        with patch("app.scraping.google_news.resolve_google_news_url", side_effect=mock_resolve):
            resolved = await resolve_google_news_urls(urls)

        assert _GOOGLE_RSS_URL in resolved
        assert resolved[_GOOGLE_RSS_URL] == _REAL_ARTICLE_URL
        assert "https://example.com/article" not in resolved

    async def test_empty_list(self) -> None:
        resolved = await resolve_google_news_urls([])
        assert resolved == {}

    async def test_no_google_urls(self) -> None:
        resolved = await resolve_google_news_urls(["https://example.com/article"])
        assert resolved == {}

    async def test_resolution_failure_excluded_from_results(self) -> None:
        """URLs that fail to resolve are not in the returned dict."""

        async def mock_resolve(url, client):
            return url  # No resolution — returns same URL

        with patch("app.scraping.google_news.resolve_google_news_url", side_effect=mock_resolve):
            resolved = await resolve_google_news_urls([_GOOGLE_RSS_URL])

        assert resolved == {}


# ============================================================
# TestGoogleNewsIntegration (pipeline integration)
# ============================================================


class TestGoogleNewsIntegration:
    def _make_topic(self, conn: sqlite3.Connection) -> Topic:
        topic = create_topic(conn, Topic(name="Solo Leveling", description="Season 3", feed_mode=FeedMode.AUTO))
        conn.commit()
        return topic

    async def test_google_news_urls_resolved_before_content_extraction(self, db_conn: sqlite3.Connection) -> None:
        """Google News redirect URLs are resolved to actual article URLs before content extraction."""
        topic = self._make_topic(db_conn)

        entries = [
            FeedEntry(
                title="Solo Leveling S3 Update",
                url=_GOOGLE_RSS_URL,
                summary="Summary text",
                source_feed="https://news.google.com/rss/search?q=test",
            )
        ]

        extracted_urls: list[str] = []

        async def mock_extract(url, **kwargs):
            extracted_urls.append(url)
            return "Extracted article content"

        with (
            patch("app.scraping.fetch_feeds_for_topic", return_value=entries),
            patch("app.scraping.extract_article_content", side_effect=mock_extract),
            patch("app.scraping.resolve_google_news_urls", return_value={_GOOGLE_RSS_URL: _REAL_ARTICLE_URL}),
        ):
            result = await fetch_new_articles_for_topic(topic, db_conn)

        assert len(result.articles) == 1
        # Content extraction should have been called with the resolved URL, not the Google redirect
        assert extracted_urls[0] == _REAL_ARTICLE_URL
        # Stored article should have the resolved URL
        assert result.articles[0].url == _REAL_ARTICLE_URL

    async def test_unresolvable_urls_still_processed(self, db_conn: sqlite3.Connection) -> None:
        """Articles with unresolvable Google News URLs still go through the pipeline."""
        topic = self._make_topic(db_conn)

        entries = [
            FeedEntry(
                title="Some Article",
                url=_GOOGLE_RSS_URL,
                summary="Summary text",
                source_feed="https://news.google.com/rss/search?q=test",
            )
        ]

        with (
            patch("app.scraping.fetch_feeds_for_topic", return_value=entries),
            patch("app.scraping.extract_article_content", return_value="Some content"),
            patch("app.scraping.resolve_google_news_urls", return_value={}),  # Resolution fails
        ):
            result = await fetch_new_articles_for_topic(topic, db_conn)

        # Article should still be stored with original URL
        assert len(result.articles) == 1
        assert result.articles[0].url == _GOOGLE_RSS_URL


# Helper for async context manager mock
async def _make_async(obj):
    return obj
