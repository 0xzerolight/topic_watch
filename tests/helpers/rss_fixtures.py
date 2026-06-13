"""Factories for canned RSS/Atom feeds served over an httpx.MockTransport.

The production scraping layer (``app.scraping.rss`` / ``app.scraping.content``)
creates its own ``httpx.AsyncClient`` internally, so the established way to
intercept it (see ``tests/test_scraping.py``) is to patch
``httpx.AsyncClient.__init__`` to inject the transport these factories build.

Use public (non-private) hostnames such as ``https://example.com/...`` —
``app.url_validation.is_private_url`` blocks loopback/RFC-1918 targets before
any request is made, so a localhost feed would be silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from xml.sax.saxutils import escape

import httpx


@dataclass
class RssEntry:
    """A single canned feed entry (title/link/published/summary)."""

    title: str
    link: str
    summary: str = ""
    published: datetime | None = None


def _format_rfc822(dt: datetime) -> str:
    """Format a datetime as an RFC-822 date (RSS <pubDate>)."""
    return dt.strftime("%a, %d %b %Y %H:%M:%S GMT")


def _format_iso(dt: datetime) -> str:
    """Format a datetime as ISO-8601 (Atom <updated>)."""
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def build_rss_xml(entries: list[RssEntry], *, channel_title: str = "Test Feed") -> str:
    """Build an RSS 2.0 document string from the given entries."""
    items = []
    for entry in entries:
        parts = [
            f"    <title>{escape(entry.title)}</title>",
            f"    <link>{escape(entry.link)}</link>",
            f"    <description>{escape(entry.summary)}</description>",
        ]
        if entry.published is not None:
            parts.append(f"    <pubDate>{_format_rfc822(entry.published)}</pubDate>")
        items.append("  <item>\n" + "\n".join(parts) + "\n  </item>")
    body = "\n".join(items)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<rss version="2.0">\n'
        "  <channel>\n"
        f"    <title>{escape(channel_title)}</title>\n"
        f"{body}\n"
        "  </channel>\n"
        "</rss>"
    )


def build_atom_xml(entries: list[RssEntry], *, feed_title: str = "Test Feed") -> str:
    """Build an Atom 1.0 document string from the given entries."""
    items = []
    for entry in entries:
        parts = [
            f"    <title>{escape(entry.title)}</title>",
            f'    <link href="{escape(entry.link)}"/>',
            f'    <content type="html">{escape(entry.summary)}</content>',
        ]
        if entry.published is not None:
            parts.append(f"    <updated>{_format_iso(entry.published)}</updated>")
        items.append("  <entry>\n" + "\n".join(parts) + "\n  </entry>")
    body = "\n".join(items)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<feed xmlns="http://www.w3.org/2005/Atom">\n'
        f"  <title>{escape(feed_title)}</title>\n"
        f"{body}\n"
        "</feed>"
    )


@dataclass
class _RssTransportConfig:
    feeds: dict[str, str] = field(default_factory=dict)
    articles: dict[str, str] = field(default_factory=dict)
    default_status: int = 404


def build_rss_transport(
    feeds: dict[str, list[RssEntry]] | None = None,
    *,
    atom_feeds: dict[str, list[RssEntry]] | None = None,
    articles: dict[str, str] | None = None,
    default_status: int = 404,
) -> httpx.MockTransport:
    """Build a MockTransport serving canned feeds and article HTML by URL substring.

    Args:
        feeds: ``{url_substring: [RssEntry, ...]}`` served as RSS 2.0 XML.
        atom_feeds: ``{url_substring: [RssEntry, ...]}`` served as Atom 1.0 XML.
        articles: ``{url_substring: html_body}`` served as ``text/html`` (for
            content extraction). The body is returned verbatim.
        default_status: status code returned when no substring matches.

    Returns:
        An ``httpx.MockTransport``. Matching is by substring against the full
        request URL, longest-pattern-first so specific article URLs win over
        broad feed prefixes.
    """
    rendered: list[tuple[str, str, str]] = []  # (pattern, content_type, body)
    for pattern, entries in (feeds or {}).items():
        rendered.append((pattern, "application/rss+xml", build_rss_xml(entries)))
    for pattern, entries in (atom_feeds or {}).items():
        rendered.append((pattern, "application/atom+xml", build_atom_xml(entries)))
    for pattern, html in (articles or {}).items():
        rendered.append((pattern, "text/html", html))

    # Longest pattern first so a specific article URL is preferred over a
    # shorter feed-prefix substring that also matches.
    rendered.sort(key=lambda t: len(t[0]), reverse=True)

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        for pattern, content_type, body in rendered:
            if pattern in url:
                return httpx.Response(200, text=body, headers={"content-type": content_type})
        return httpx.Response(default_status, text="Not found")

    return httpx.MockTransport(handler)
