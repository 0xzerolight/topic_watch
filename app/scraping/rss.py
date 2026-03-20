"""RSS/Atom feed fetching and parsing.

Fetches feeds via httpx, parses with feedparser, and converts entries
to FeedEntry models ready for dedup and storage.
"""

import asyncio
import hashlib
import logging
from calendar import timegm
from collections.abc import Callable
from datetime import UTC, datetime
from time import struct_time
from urllib.parse import quote_plus

import feedparser
import httpx
from pydantic import BaseModel

from app.models import FeedMode, Topic
from app.url_validation import is_private_url

logger = logging.getLogger(__name__)

FeedHealthCallback = Callable[[str, bool, str | None], None]  # (feed_url, success, error_msg)

_USER_AGENT = "TopicWatch/0.1.0 (RSS reader)"
_FEED_FETCH_TIMEOUT = 15.0
_GOOGLE_NEWS_RSS_TEMPLATE = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"


def build_google_news_url(topic_name: str) -> str:
    """Build a Google News RSS search URL from a topic name."""
    return _GOOGLE_NEWS_RSS_TEMPLATE.format(query=quote_plus(topic_name))


class FeedEntry(BaseModel):
    """A single entry parsed from an RSS/Atom feed."""

    title: str
    url: str
    published: datetime | None = None
    summary: str = ""
    source_feed: str


def compute_article_hash(url: str, title: str) -> str:
    """Compute a deterministic, case-insensitive content hash."""
    raw = f"{url}|{title}".lower()
    return hashlib.sha256(raw.encode()).hexdigest()


def _parse_feed_date(entry: dict) -> datetime | None:
    """Extract a datetime from a feedparser entry's date fields."""
    for field in ("published_parsed", "updated_parsed"):
        val = entry.get(field)
        if isinstance(val, struct_time):
            try:
                return datetime.fromtimestamp(timegm(val), tz=UTC)
            except (ValueError, OverflowError):
                continue
    return None


def _parse_entry(raw_entry: dict, source_feed: str) -> FeedEntry | None:
    """Convert a feedparser entry dict to a FeedEntry, or None if invalid."""
    title = raw_entry.get("title", "").strip()
    url = raw_entry.get("link", "").strip()
    if not title or not url:
        return None

    # Atom/Reddit feeds store content in 'content' field
    summary = raw_entry.get("summary", "")
    if not summary:
        content_list = raw_entry.get("content", [])
        if content_list and isinstance(content_list, list):
            summary = content_list[0].get("value", "")

    return FeedEntry(
        title=title,
        url=url,
        published=_parse_feed_date(raw_entry),
        summary=summary,
        source_feed=source_feed,
    )


async def fetch_feed(
    feed_url: str,
    client: httpx.AsyncClient | None = None,
    timeout: float = _FEED_FETCH_TIMEOUT,
    max_attempts: int = 2,
    health_callback: FeedHealthCallback | None = None,
) -> list[FeedEntry]:
    """Fetch and parse a single RSS/Atom feed. Returns [] on any error."""
    if is_private_url(feed_url):
        logger.warning("Blocked fetch to private URL: %s", feed_url)
        return []
    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(
            headers={"User-Agent": _USER_AGENT},
            timeout=timeout,
            follow_redirects=True,
        )
    assert client is not None
    try:
        for attempt in range(max_attempts):
            try:
                response = await client.get(feed_url)
                response.raise_for_status()
                parsed = feedparser.parse(response.text)
                entries = []
                for raw in parsed.entries:
                    entry = _parse_entry(raw, feed_url)
                    if entry:
                        entries.append(entry)
                if health_callback:
                    health_callback(feed_url, True, None)
                return entries
            except httpx.TimeoutException as exc:
                if attempt < max_attempts - 1:
                    logger.debug("Timeout fetching feed (attempt %d): %s", attempt + 1, feed_url)
                    await asyncio.sleep(2)
                    continue
                logger.warning("Timeout fetching feed after %d attempts: %s", max_attempts, feed_url)
                if health_callback:
                    health_callback(feed_url, False, f"Timeout after {max_attempts} attempts: {exc}")
                return []
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code >= 500 and attempt < max_attempts - 1:
                    logger.debug(
                        "HTTP %d fetching feed (attempt %d): %s", exc.response.status_code, attempt + 1, feed_url
                    )
                    await asyncio.sleep(2)
                    continue
                logger.warning("HTTP %d fetching feed: %s", exc.response.status_code, feed_url)
                if health_callback:
                    health_callback(feed_url, False, f"HTTP {exc.response.status_code}")
                return []
            except httpx.NetworkError as exc:
                if attempt < max_attempts - 1:
                    logger.debug(
                        "Network error fetching feed (attempt %d): %s — %s",
                        attempt + 1,
                        feed_url,
                        type(exc).__name__,
                    )
                    await asyncio.sleep(2)
                    continue
                logger.warning(
                    "Network error fetching feed after %d attempts: %s — %s",
                    max_attempts,
                    feed_url,
                    type(exc).__name__,
                )
                if health_callback:
                    health_callback(feed_url, False, f"Network error: {type(exc).__name__}: {exc}")
                return []
            except Exception as exc:
                logger.warning("Error fetching feed: %s", feed_url, exc_info=True)
                if health_callback:
                    health_callback(feed_url, False, f"{type(exc).__name__}: {exc}")
                return []
        return []  # pragma: no cover
    finally:
        if owns_client:
            await client.aclose()


async def fetch_feeds_for_topic(
    topic: Topic,
    timeout: float = _FEED_FETCH_TIMEOUT,
    max_attempts: int = 2,
    health_callback: FeedHealthCallback | None = None,
) -> list[FeedEntry]:
    """Fetch all feeds for a topic concurrently, deduplicated by URL."""
    effective_urls = [build_google_news_url(topic.name)] if topic.feed_mode == FeedMode.AUTO else topic.feed_urls

    if not effective_urls:
        return []

    async with httpx.AsyncClient(
        headers={"User-Agent": _USER_AGENT},
        timeout=timeout,
        follow_redirects=True,
    ) as client:
        tasks = [
            fetch_feed(url, client, timeout=timeout, max_attempts=max_attempts, health_callback=health_callback)
            for url in effective_urls
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    seen_urls: set[str] = set()
    entries: list[FeedEntry] = []
    for result in results:
        if isinstance(result, BaseException):
            logger.warning("Feed fetch failed: %s", result)
            continue
        for entry in result:
            if entry.url not in seen_urls:
                seen_urls.add(entry.url)
                entries.append(entry)

    return entries
