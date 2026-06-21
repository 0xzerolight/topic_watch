"""Tests for the safe_href Jinja2 filter and ingestion-time scheme guards.

Regression coverage for OVH-014: an untrusted feed article/feed URL with a
``javascript:`` or ``data:text/html`` scheme must never render into an href
(Jinja autoescape does NOT neutralize the scheme), and must not survive feed
ingestion into the DB.
"""

import pytest

from app.scraping.rss import _parse_entry, _resolve_google_news_url
from app.web.routers.templates import _safe_href, templates


class TestSafeHref:
    def test_http_url_passes_through(self) -> None:
        url = "http://example.com/article"
        assert _safe_href(url) == url

    def test_https_url_passes_through(self) -> None:
        url = "https://example.com/article?q=1#frag"
        assert _safe_href(url) == url

    @pytest.mark.parametrize(
        "url",
        [
            "javascript:alert(1)",
            "JavaScript:alert(document.cookie)",
            "  javascript:alert(1)",  # leading whitespace must not bypass
            "data:text/html,<script>alert(1)</script>",
            "data:text/html;base64,PHNjcmlwdD4=",
            "vbscript:msgbox(1)",
            "file:///etc/passwd",
        ],
    )
    def test_dangerous_scheme_becomes_hash(self, url: str) -> None:
        assert _safe_href(url) == "#"

    def test_empty_and_none_become_hash(self) -> None:
        assert _safe_href("") == "#"
        assert _safe_href(None) == "#"

    def test_relative_or_schemeless_becomes_hash(self) -> None:
        # No explicit http(s) scheme -> not allowlisted.
        assert _safe_href("//evil.example.com/x") == "#"
        assert _safe_href("not-a-url") == "#"

    def test_filter_registered_on_environment(self) -> None:
        assert templates.env.filters.get("safe_href") is _safe_href

    def test_template_render_neutralizes_javascript_href(self) -> None:
        rendered = templates.env.from_string('<a href="{{ url|safe_href }}">x</a>').render(url="javascript:alert(1)")
        assert 'href="#"' in rendered
        assert "javascript:" not in rendered

    def test_template_render_keeps_http_href(self) -> None:
        rendered = templates.env.from_string('<a href="{{ url|safe_href }}">x</a>').render(url="https://example.com/a")
        assert 'href="https://example.com/a"' in rendered


class TestIngestionSchemeGuard:
    def test_parse_entry_drops_javascript_link(self) -> None:
        raw = {"title": "Evil", "link": "javascript:alert(1)", "summary": ""}
        assert _parse_entry(raw, "https://example.com/feed.xml") is None

    def test_parse_entry_drops_data_link(self) -> None:
        raw = {"title": "Evil", "link": "data:text/html,<script>1</script>", "summary": ""}
        assert _parse_entry(raw, "feed") is None

    def test_parse_entry_keeps_http_link(self) -> None:
        raw = {"title": "Good", "link": "https://example.com/x", "summary": ""}
        entry = _parse_entry(raw, "feed")
        assert entry is not None
        assert entry.url == "https://example.com/x"

    def test_google_resolver_rejects_javascript_href(self) -> None:
        google_url = "https://news.google.com/rss/articles/CBMiQ2h0dHBz..."
        description = '<a href="javascript:alert(1)">Title</a>'
        # Must not adopt the javascript: scheme; falls back to the safe redirect URL.
        result = _resolve_google_news_url(google_url, description)
        assert result == google_url

    def test_google_resolver_rejects_data_href(self) -> None:
        google_url = "https://news.google.com/rss/articles/CBMiQ2h0dHBz..."
        description = '<a href="data:text/html,<script>1</script>">Title</a>'
        result = _resolve_google_news_url(google_url, description)
        assert result == google_url

    def test_google_resolver_keeps_http_href(self) -> None:
        google_url = "https://news.google.com/rss/articles/CBMiQ2h0dHBz..."
        description = '<a href="https://real.example.com/a">Title</a>'
        result = _resolve_google_news_url(google_url, description)
        assert result == "https://real.example.com/a"
