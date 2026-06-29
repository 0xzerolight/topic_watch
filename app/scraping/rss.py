"""RSS/Atom feed fetching and parsing.

Fetches feeds via httpx, parses with feedparser, and converts entries
to FeedEntry models ready for dedup and storage.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from calendar import timegm
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from html.parser import HTMLParser
from time import struct_time
from typing import TYPE_CHECKING
from urllib.parse import ParseResult, parse_qs, urlparse

import feedparser
import httpx
from pydantic import BaseModel

from app.feed_backoff import BACKOFF_BASE_MINUTES, BACKOFF_CAP_HOURS, feed_backoff_until
from app.log_redaction import redact_url
from app.models import FeedMode, Topic
from app.url_validation import is_private_url, safe_get

if TYPE_CHECKING:
    from app.models import FeedHealth
    from app.scraping.routing import ProviderRouter

logger = logging.getLogger(__name__)

FeedHealthCallback = Callable[
    [str, bool, str | None, str | None, str | None], None
]  # (feed_url, success, error_msg, etag, last_modified)

_USER_AGENT = "TopicWatch/1.0.0 (RSS reader)"
_FEED_FETCH_TIMEOUT = 15.0


class FeedEntry(BaseModel):
    """A single entry parsed from an RSS/Atom feed."""

    title: str
    url: str
    published: datetime | None = None
    summary: str = ""
    source_feed: str


@dataclass
class FeedResponse:
    """Result of fetching feeds for a topic.

    Wraps the parsed entries with metadata about which provider was
    used, so downstream code can make provider-specific decisions
    (e.g. Google News URL resolution) without importing provider classes.

    ``feeds_total`` / ``feeds_failed`` expose per-fetch health so the check
    pipeline can distinguish a healthy partial yield from a degraded check where
    some sources silently dropped out (OVH-130). For AUTO mode a single provider
    is fetched (with at most one cascade), so the counts reflect that attempt;
    for MANUAL mode they count the topic's explicit feed URLs.
    """

    entries: list[FeedEntry] = field(default_factory=list)
    provider_name: str | None = None
    needs_url_resolution: bool = False
    feeds_total: int = 0
    feeds_failed: int = 0
    feeds_skipped: int = 0
    """MANUAL mode: feeds skipped this cycle because they are in a backoff window
    (persistently failing). For MANUAL mode ``feeds_total`` counts feeds ATTEMPTED
    (skipped feeds are excluded and surface here), so a backed-off feed is never
    miscounted as a partial failure."""


def compute_article_hash(url: str, title: str) -> str:
    """Compute a deterministic, case-insensitive content hash."""
    raw = f"{url}|{title}".lower()
    return hashlib.sha256(raw.encode()).hexdigest()


def _validators(state: FeedHealth | None) -> tuple[str | None, str | None]:
    """Return ``(etag, last_modified)`` for a feed-health row, or ``(None, None)``."""
    if state is None:
        return None, None
    return state.etag, state.last_modified


def _parse_feed_date(entry: dict) -> datetime | None:
    """Extract a datetime from a feedparser entry's date fields."""
    for date_field in ("published_parsed", "updated_parsed"):
        val = entry.get(date_field)
        if isinstance(val, struct_time):
            try:
                return datetime.fromtimestamp(timegm(val), tz=UTC)
            except (ValueError, OverflowError):
                continue
    return None


_GOOGLE_NEWS_HREF_RE = re.compile(r'<a[^>]+href=["\']([^"\']+)["\']', re.IGNORECASE)


def _resolve_google_news_url(link: str, description: str) -> str:
    """Extract the real article URL from a Google News RSS entry (fast path).

    Google News RSS entries use redirect URLs (news.google.com/rss/articles/...)
    as their <link>. Some entries embed the actual article URL as an <a href>
    in the description HTML. This is a zero-cost regex check that avoids HTTP
    requests. When it fails (e.g. Google embeds the same redirect URL in the
    description), the async resolver in google_news.py handles it later in
    the pipeline.
    """
    if "news.google.com/" not in link:
        return link
    match = _GOOGLE_NEWS_HREF_RE.search(description)
    if match:
        real_url = match.group(1)
        # Defense-in-depth (OVH-014): only adopt an http(s) href from the
        # untrusted description; a javascript:/data: href must fall back to the
        # safe Google redirect link rather than become the article URL.
        if (
            real_url
            and not real_url.startswith("https://news.google.com/")
            and urlparse(real_url).scheme.lower() in ("http", "https")
        ):
            return real_url
    return link


def _is_bing_apiclick(parsed: ParseResult) -> bool:
    """True if a parsed URL is a Bing News ``apiclick.aspx`` redirect.

    Matches on ``hostname`` (urlparse lowercases it and strips any port/userinfo,
    so a ``www.bing.com:80`` netloc still matches) and the case-folded path.
    """
    host = (parsed.hostname or "").lower()
    return (host == "bing.com" or host.endswith(".bing.com")) and parsed.path.lower() == "/news/apiclick.aspx"


def _resolve_bing_news_url(link: str) -> str:
    """Extract the real article URL from a Bing News apiclick redirect (fast path).

    Bing News RSS <link>s are ``www.bing.com/news/apiclick.aspx`` redirects that
    carry the real publisher URL, fully percent-encoded, in the ``url`` query
    param. Unlike Google News this needs no HTTP round-trip — the target decodes
    from the query string alone. Returns the decoded target, or the original
    ``link`` when it is not such a redirect, has no usable ``url`` value, or that
    value is non-http(s) or itself a Bing apiclick link (loop guard). Mirrors the
    OVH-014 scheme guard in ``_resolve_google_news_url``.

    Relies on Bing fully percent-encoding the target (true in all observed data);
    a target carrying an unencoded ``&`` would be truncated by ``parse_qs``.
    """
    parsed = urlparse(link)
    if not _is_bing_apiclick(parsed):
        return link
    targets = parse_qs(parsed.query).get("url")
    if not targets:
        return link
    real_url = targets[0]
    target = urlparse(real_url)
    if target.scheme.lower() in ("http", "https") and not _is_bing_apiclick(target):
        return real_url
    return link


class _HTMLTextExtractor(HTMLParser):
    """Collect only the text nodes of an HTML fragment, discarding all markup."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def text(self) -> str:
        return "".join(self._parts)


def _strip_html(value: str) -> str:
    """Reduce an HTML fragment to whitespace-collapsed plain text (OVH-112).

    RSS summary fallbacks (notably Google News' ``<ol><li><a>`` link lists) are
    HTML. Storing that raw as ``raw_content`` wastes the novelty-prompt budget on
    tag/href noise and inflates the ``[STUB]`` byte-count heuristic. This keeps the
    human-readable text, drops the markup, and unescapes entities. A tag-free
    string round-trips to (a whitespace-collapsed) copy of itself, so plain
    summaries are effectively untouched. Never raises: a malformed fragment falls
    back to the original input.
    """
    if not value or ("<" not in value and "&" not in value):
        return value
    try:
        parser = _HTMLTextExtractor()
        parser.feed(value)
        parser.close()
        text = parser.text()
    except Exception:
        logger.debug("HTML strip failed; keeping raw summary", exc_info=True)
        return value
    return re.sub(r"\s+", " ", text).strip()


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

    # Google News RSS uses redirect URLs — resolve to actual article URLs.
    # Done on the RAW summary because the resolver regex-extracts <a href>.
    url = _resolve_google_news_url(url, summary)
    # Bing News RSS uses apiclick.aspx redirects carrying the real URL in the
    # ``url`` query param — unwrap it (zero-network). Mutually exclusive with the
    # Google path by host, so the order is irrelevant.
    url = _resolve_bing_news_url(url)

    # OVH-112: strip HTML AFTER url resolution so the STORED summary (which becomes
    # raw_content when extraction fails) is plain text, not tag/href noise. The
    # content hash is url|title only, so dedup is unaffected.
    summary = _strip_html(summary)

    # Defense-in-depth (OVH-014): a non-http(s) scheme (javascript:, data:, ...)
    # must never reach the DB, where it would later render into an href.
    if urlparse(url).scheme.lower() not in ("http", "https"):
        logger.warning("Dropping feed entry with non-http(s) link scheme: %s", url)
        return None

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
    etag: str | None = None,
    last_modified: str | None = None,
) -> list[FeedEntry]:
    """Fetch and parse a single RSS/Atom feed. Returns [] on any error."""
    entries, _ = await fetch_feed_with_status(
        feed_url,
        client,
        timeout=timeout,
        max_attempts=max_attempts,
        health_callback=health_callback,
        etag=etag,
        last_modified=last_modified,
    )
    return entries


async def fetch_feed_with_status(
    feed_url: str,
    client: httpx.AsyncClient | None = None,
    timeout: float = _FEED_FETCH_TIMEOUT,
    max_attempts: int = 2,
    health_callback: FeedHealthCallback | None = None,
    etag: str | None = None,
    last_modified: str | None = None,
) -> tuple[list[FeedEntry], bool]:
    """Fetch and parse a single feed, also reporting whether the fetch succeeded.

    Returns ``(entries, fetch_ok)``. ``fetch_ok`` is True when the feed was
    fetched and parsed successfully — even if it legitimately contained zero
    entries — and False on any error (blocked URL, timeout, HTTP error, etc.).
    This lets callers distinguish "fetched OK but empty" from "fetch failed" so
    an empty-but-valid feed does not get treated as a provider failure.

    ``etag`` / ``last_modified`` are the feed's stored conditional-GET validators;
    when present they are sent as ``If-None-Match`` / ``If-Modified-Since`` and a
    304 returns ``([], True)`` (the empty-but-OK bucket) without re-parsing.
    """
    if await asyncio.to_thread(is_private_url, feed_url):
        logger.warning("Blocked fetch to private URL: %s", redact_url(feed_url))
        return [], False
    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(
            headers={"User-Agent": _USER_AGENT},
            timeout=timeout,
            follow_redirects=False,
        )
    assert client is not None
    cond_headers: dict[str, str] = {}
    if etag:
        cond_headers["If-None-Match"] = etag
    if last_modified:
        cond_headers["If-Modified-Since"] = last_modified
    try:
        for attempt in range(max_attempts):
            try:
                response = await safe_get(client, feed_url, headers=cond_headers or None)
                # 304 Not Modified: validators still valid. Treat as an empty-but-
                # successful fetch — the existing "([], True)" bucket that
                # _fetch_auto/_fetch_manual already handle. Pass (None, None) so the
                # stored validators are preserved (COALESCE), not wiped.
                if response.status_code == 304:
                    if health_callback:
                        health_callback(feed_url, True, None, None, None)
                    return [], True
                response.raise_for_status()
                parsed = feedparser.parse(response.text)
                entries = []
                for raw in parsed.entries:
                    # OVH-024: isolate each entry so one malformed entry does not
                    # discard the whole feed. The outer handlers below stay for
                    # genuine fetch/parse-level failures only.
                    try:
                        entry = _parse_entry(raw, feed_url)
                    except Exception:
                        logger.warning("Skipping malformed feed entry in %s", feed_url, exc_info=True)
                        continue
                    if entry:
                        entries.append(entry)
                # OVH-044: feedparser flags malformed/non-feed bodies as bozo. If
                # bozo with zero recovered entries, treat it as a soft failure so it
                # surfaces in feed_health and engages the provider cascade; if bozo
                # but entries were still recovered, just note it and proceed.
                if getattr(parsed, "bozo", 0):
                    bozo_exc = getattr(parsed, "bozo_exception", None)
                    if not entries:
                        logger.warning("Feed parse error (bozo) with no entries: %s — %s", feed_url, bozo_exc)
                        if health_callback:
                            health_callback(feed_url, False, f"Feed parse error: {bozo_exc}", None, None)
                        return [], False
                    logger.debug(
                        "Feed flagged bozo but %d entries recovered: %s — %s", len(entries), feed_url, bozo_exc
                    )
                if health_callback:
                    health_callback(
                        feed_url,
                        True,
                        None,
                        response.headers.get("etag"),
                        response.headers.get("last-modified"),
                    )
                return entries, True
            except httpx.TimeoutException as exc:
                if attempt < max_attempts - 1:
                    logger.debug("Timeout fetching feed (attempt %d): %s", attempt + 1, feed_url)
                    await asyncio.sleep(2)
                    continue
                logger.warning("Timeout fetching feed after %d attempts: %s", max_attempts, feed_url)
                if health_callback:
                    health_callback(feed_url, False, f"Timeout after {max_attempts} attempts: {exc}", None, None)
                return [], False
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code >= 500 and attempt < max_attempts - 1:
                    logger.debug(
                        "HTTP %d fetching feed (attempt %d): %s", exc.response.status_code, attempt + 1, feed_url
                    )
                    await asyncio.sleep(2)
                    continue
                logger.warning("HTTP %d fetching feed: %s", exc.response.status_code, feed_url)
                if health_callback:
                    health_callback(feed_url, False, f"HTTP {exc.response.status_code}", None, None)
                return [], False
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
                    health_callback(feed_url, False, f"Network error: {type(exc).__name__}: {exc}", None, None)
                return [], False
            except Exception as exc:
                logger.warning("Error fetching feed: %s", feed_url, exc_info=True)
                if health_callback:
                    health_callback(feed_url, False, f"{type(exc).__name__}: {exc}", None, None)
                return [], False
        return [], False  # pragma: no cover
    finally:
        if owns_client:
            await client.aclose()


async def fetch_feeds_for_topic(
    topic: Topic,
    timeout: float = _FEED_FETCH_TIMEOUT,
    max_attempts: int = 2,
    health_callback: FeedHealthCallback | None = None,
    router: ProviderRouter | None = None,
    feed_state_loader: Callable[[str], FeedHealth | None] | None = None,
    backoff_base_minutes: int = BACKOFF_BASE_MINUTES,
    backoff_cap_hours: int = BACKOFF_CAP_HOURS,
) -> FeedResponse:
    """Fetch all feeds for a topic, deduplicated by URL.

    For AUTO mode: uses the router to select a provider, with within-cycle
    fallback (max 1 retry with the next provider). For MANUAL mode: fetches all
    explicit feed URLs concurrently, skipping any in a backoff window.

    ``feed_state_loader`` supplies the stored ``FeedHealth`` per URL — used to
    send conditional-GET validators (both modes) and to skip backed-off feeds
    (MANUAL only; AUTO provider backoff is owned by ``ProviderRouter``).
    """
    if topic.feed_mode == FeedMode.AUTO:
        return await _fetch_auto(topic, timeout, max_attempts, health_callback, router, feed_state_loader)
    return await _fetch_manual(
        topic, timeout, max_attempts, health_callback, feed_state_loader, backoff_base_minutes, backoff_cap_hours
    )


async def _fetch_auto(
    topic: Topic,
    timeout: float,
    max_attempts: int,
    health_callback: FeedHealthCallback | None,
    router: ProviderRouter | None,
    feed_state_loader: Callable[[str], FeedHealth | None] | None = None,
) -> FeedResponse:
    """AUTO mode: try provider, fallback to next on empty/error."""
    if router is None:
        from app.scraping.routing import router as default_router

        router = default_router

    provider = router.get_provider()

    async with httpx.AsyncClient(
        headers={"User-Agent": _USER_AGENT},
        timeout=timeout,
        follow_redirects=False,
    ) as client:
        feed_url = provider.build_feed_url(topic)
        # Capture the health epoch before the fetch await so a success that races
        # with a concurrent failure is recognised as stale (OVH-127).
        provider_epoch = router.health_epoch(provider.name)
        p_etag, p_last_modified = _validators(feed_state_loader(feed_url) if feed_state_loader else None)
        entries, fetch_ok = await fetch_feed_with_status(
            feed_url,
            client,
            timeout=timeout,
            max_attempts=max_attempts,
            health_callback=health_callback,
            etag=p_etag,
            last_modified=p_last_modified,
        )

        if entries:
            if router.mark_healthy(provider.name, observed_epoch=provider_epoch):
                logger.info("Provider %s recovered (back to healthy)", provider.name)
            return FeedResponse(
                entries=entries,
                provider_name=provider.name,
                needs_url_resolution=provider.needs_url_resolution(),
                feeds_total=1,
                feeds_failed=0,
            )

        # No entries. Only a real fetch error marks the provider unhealthy —
        # a legitimately-empty-but-successful feed must not trigger cascade/cooldown.
        # Distinguish those two cases in the log so a silently-failing provider is
        # not indistinguishable from a genuinely-empty one (OVH-133).
        if not fetch_ok and router.mark_unhealthy(provider.name):
            logger.warning("Provider %s marked unhealthy (failure threshold reached)", provider.name)
        reason = "fetch failed" if not fetch_ok else "returned no entries (empty result)"
        next_provider = router.get_next_provider(provider)
        if next_provider is None:
            logger.warning("Provider %s %s; no fallback provider available", provider.name, reason)
            return FeedResponse(
                entries=[],
                provider_name=provider.name,
                needs_url_resolution=False,
                feeds_total=1,
                feeds_failed=1 if not fetch_ok else 0,
            )

        logger.info("Provider %s %s, cascading to %s", provider.name, reason, next_provider.name)
        feed_url = next_provider.build_feed_url(topic)
        next_epoch = router.health_epoch(next_provider.name)
        f_etag, f_last_modified = _validators(feed_state_loader(feed_url) if feed_state_loader else None)
        entries, next_fetch_ok = await fetch_feed_with_status(
            feed_url,
            client,
            timeout=timeout,
            max_attempts=max_attempts,
            health_callback=health_callback,
            etag=f_etag,
            last_modified=f_last_modified,
        )
        first_failed = 1 if not fetch_ok else 0

        if entries:
            if router.mark_healthy(next_provider.name, observed_epoch=next_epoch):
                logger.info("Provider %s recovered (back to healthy)", next_provider.name)
            return FeedResponse(
                entries=entries,
                provider_name=next_provider.name,
                needs_url_resolution=next_provider.needs_url_resolution(),
                feeds_total=2,
                feeds_failed=first_failed,
            )

        if not next_fetch_ok and router.mark_unhealthy(next_provider.name):
            logger.warning("Provider %s marked unhealthy (failure threshold reached)", next_provider.name)
        next_reason = "fetch failed" if not next_fetch_ok else "returned no entries (empty result)"
        logger.warning(
            "Provider cascade exhausted: %s %s, fallback %s %s",
            provider.name,
            reason,
            next_provider.name,
            next_reason,
        )
        return FeedResponse(
            entries=[],
            provider_name=next_provider.name,
            needs_url_resolution=False,
            feeds_total=2,
            feeds_failed=first_failed + (1 if not next_fetch_ok else 0),
        )


async def _fetch_manual(
    topic: Topic,
    timeout: float,
    max_attempts: int,
    health_callback: FeedHealthCallback | None,
    feed_state_loader: Callable[[str], FeedHealth | None] | None = None,
    backoff_base_minutes: int = BACKOFF_BASE_MINUTES,
    backoff_cap_hours: int = BACKOFF_CAP_HOURS,
) -> FeedResponse:
    """MANUAL mode: fetch explicit feed URLs concurrently, skipping backed-off ones."""
    if not topic.feed_urls:
        return FeedResponse()

    # Decide skips and load validators from ONE health lookup per URL.
    now = datetime.now(UTC)
    attempted: list[tuple[str, str | None, str | None]] = []  # (url, etag, last_modified)
    feeds_skipped = 0
    for url in topic.feed_urls:
        state = feed_state_loader(url) if feed_state_loader else None
        until = feed_backoff_until(state, base_minutes=backoff_base_minutes, cap_hours=backoff_cap_hours)
        if until is not None and until > now:
            feeds_skipped += 1
            logger.debug("Skipping backed-off feed %s (next retry %s)", url, until.isoformat())
            continue
        etag, last_modified = _validators(state)
        attempted.append((url, etag, last_modified))

    # Build the client only when something is actually attempted (the all-skipped
    # case returns here without opening a connection).
    if not attempted:
        return FeedResponse(feeds_total=0, feeds_failed=0, feeds_skipped=feeds_skipped)

    async with httpx.AsyncClient(
        headers={"User-Agent": _USER_AGENT},
        timeout=timeout,
        follow_redirects=False,
    ) as client:
        # fetch_feed_with_status reports per-feed success so a partial failure
        # (some of N feeds down) is countable, not just absorbed into a smaller
        # entry list (OVH-130).
        tasks = [
            fetch_feed_with_status(
                url,
                client,
                timeout=timeout,
                max_attempts=max_attempts,
                health_callback=health_callback,
                etag=etag,
                last_modified=last_modified,
            )
            for (url, etag, last_modified) in attempted
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    seen_urls: set[str] = set()
    entries: list[FeedEntry] = []
    feeds_total = len(attempted)
    feeds_failed = 0
    for result in results:
        if isinstance(result, BaseException):
            logger.warning("Feed fetch failed: %s", result)
            feeds_failed += 1
            continue
        feed_entries, fetch_ok = result
        if not fetch_ok:
            feeds_failed += 1
        for entry in feed_entries:
            if entry.url not in seen_urls:
                seen_urls.add(entry.url)
                entries.append(entry)

    return FeedResponse(
        entries=entries, feeds_total=feeds_total, feeds_failed=feeds_failed, feeds_skipped=feeds_skipped
    )
