"""RSS/Atom feed fetching and parsing.

Fetches feeds via httpx, parses with feedparser, and converts entries
to FeedEntry models ready for dedup and storage.
"""

from __future__ import annotations

import asyncio
import logging
import re
from calendar import timegm
from datetime import UTC, datetime
from html.parser import HTMLParser
from time import struct_time
from typing import TYPE_CHECKING
from urllib.parse import ParseResult, parse_qs, urlparse

import feedparser
import httpx

from app.feed_backoff import feed_backoff_until
from app.log_redaction import redact_url
from app.models import FeedMode, Topic
from app.scraping.google_news import GOOGLE_NEWS_HOST
from app.scraping.providers import provider_identity
from app.scraping.source import (
    DEADLINE_ERROR,
    Deadline,
    SourceDeadlineExceeded,
    SourceRequest,
    bounded,
    collapse_duplicate_entries,
    host_matches,
    register_source,
    url_hostname,
)
from app.scraping.source import FeedEntry as FeedEntry
from app.scraping.source import FeedHealthCallback as FeedHealthCallback
from app.scraping.source import FeedResponse as FeedResponse
from app.scraping.source import compute_article_hash as compute_article_hash
from app.scraping.source import fetch_feeds_for_topic as fetch_feeds_for_topic
from app.url_validation import is_private_url, safe_get

if TYPE_CHECKING:
    from app.models import FeedHealth

logger = logging.getLogger(__name__)

_RETRY_BACKOFF_SECONDS = 2.0

BING_HOST = "bing.com"
"""Bing News RSS and its apiclick redirects are served from this domain."""

_USER_AGENT = "TopicWatch/1.0.0 (RSS reader)"
_FEED_FETCH_TIMEOUT = 15.0


def _validators(state: FeedHealth | None) -> tuple[str | None, str | None]:
    """Return ``(etag, last_modified)`` for a feed-health row, or ``(None, None)``."""
    if state is None:
        return None, None
    return state.etag, state.last_modified


def _struct_time_to_datetime(val: object) -> datetime | None:
    """Convert one feedparser ``*_parsed`` field to UTC, or None if unusable."""
    if not isinstance(val, struct_time):
        return None
    try:
        return datetime.fromtimestamp(timegm(val), tz=UTC)
    except (ValueError, OverflowError):
        return None


def _parse_feed_date(entry: dict) -> datetime | None:
    """Extract a datetime from a feedparser entry's date fields."""
    for date_field in ("published_parsed", "updated_parsed"):
        stamp = _struct_time_to_datetime(entry.get(date_field))
        if stamp is not None:
            return stamp
    return None


def _parse_updated_date(entry: dict) -> datetime | None:
    """The feed's own revision stamp for this entry, when it publishes one.

    Read separately from ``_parse_feed_date`` (which falls back to it as a
    publication date): as a revision marker it only means anything when the feed
    states it in its own right (AUG-320). Read through ``dict.get`` rather than
    feedparser's own lookup, because that answers with ``published_parsed`` when
    an entry has no ``updated_parsed`` of its own — a deprecated fallback that
    would give every plain RSS item a revision stamp it never published.
    """
    return _struct_time_to_datetime(dict.get(entry, "updated_parsed"))


_GOOGLE_NEWS_HREF_RE = re.compile(r'<a[^>]+href=["\']([^"\']+)["\']', re.IGNORECASE)


def _resolve_google_news_url(link: str, description: str) -> str:
    """Extract the real article URL from a Google News RSS entry (fast path).

    Google News RSS entries use redirect URLs (news.google.com/rss/articles/...)
    as their <link>. Some entries embed the actual article URL as an <a href>
    in the description HTML. This is a zero-cost regex check that avoids HTTP
    requests. When it fails (e.g. Google embeds the same redirect URL in the
    description), the async resolver in google_news.py handles it later in
    the pipeline.

    Only entries whose link really is hosted on Google News take this path
    (TW-AUD-031): a raw-URL substring test let any feed mentioning
    ``news.google.com/`` in a path or query have its stored URL replaced by an
    arbitrary href from its own description.
    """
    if not host_matches(url_hostname(link), GOOGLE_NEWS_HOST):
        return link
    match = _GOOGLE_NEWS_HREF_RE.search(description)
    if match:
        real_url = match.group(1)
        # Defense-in-depth (OVH-014): only adopt an http(s) href from the
        # untrusted description; a javascript:/data: href must fall back to the
        # safe Google redirect link rather than become the article URL.
        if (
            real_url
            and not host_matches(url_hostname(real_url), GOOGLE_NEWS_HOST)
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
    return host_matches(host, BING_HOST) and parsed.path.lower() == "/news/apiclick.aspx"


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


# Tags that render as their own line/box, so the text on either side of them is
# two words, not one. Used to emit a boundary while stripping markup (AUG-185);
# every other tag is treated as inline and keeps its neighbours adjacent.
_BLOCK_LEVEL_TAGS = frozenset(
    {
        "address", "article", "aside", "blockquote", "br", "caption", "dd", "div", "dl", "dt",
        "fieldset", "figcaption", "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6",
        "header", "hr", "li", "main", "nav", "ol", "p", "pre", "section", "table", "tbody", "td",
        "tfoot", "th", "thead", "tr", "ul",
    }
)  # fmt: skip


class _HTMLTextExtractor(HTMLParser):
    """Collect only the text nodes of an HTML fragment, discarding all markup.

    Block-level tags emit a single space so adjacent blocks stay separate words
    (AUG-185): ``<li>Alpha</li><li>Beta</li>`` is ``Alpha Beta``, not ``AlphaBeta``.
    The boundary is emitted on both the start and the end tag, so an unclosed
    block in a malformed fragment still separates. Inline tags emit nothing, so
    ``<b>Al</b><i>pha</i>`` stays one word. The caller collapses the runs of
    whitespace this introduces.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []

    def _boundary(self, tag: str) -> None:
        if tag in _BLOCK_LEVEL_TAGS:
            self._parts.append(" ")

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._boundary(tag)

    def handle_endtag(self, tag: str) -> None:
        self._boundary(tag)

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


def _parser_headers(response: httpx.Response) -> dict[str, str]:
    """Transport metadata feedparser needs to decode and resolve the document (TW-AUD-019).

    ``response.text`` pre-decodes the body with httpx's own guess, discarding the
    XML encoding declaration, so a correctly declared non-UTF-8 feed arrives as
    mojibake. Handing over ``response.content`` instead makes feedparser's
    declaration/BOM detection authoritative, with ``content-type`` as the only
    transport-level charset hint.

    ``content-location`` is the base URI feedparser resolves document-relative
    entry links against. It is the URL the document actually came from — the
    final hop after redirects, already SSRF-validated per hop by ``safe_send`` —
    so a relative Atom link becomes an absolute URL instead of being dropped by
    the http(s) scheme guard. Resolved URLs still go through that guard.

    Only the three headers feedparser reads are forwarded; passing the whole
    response header set would hand it transport headers (e.g. ``content-encoding``
    for a body httpx has already decompressed) that are no longer true of these
    bytes.
    """
    headers = {"content-location": str(response.url)}
    for name in ("content-type", "content-language"):
        value = response.headers.get(name)
        if value:
            headers[name] = value
    return headers


def _is_wrong_media_type_only(parsed: object, bozo_exc: object) -> bool:
    """True when the only complaint is a non-XML ``Content-Type`` on a real feed.

    Forwarding ``content-type`` to feedparser (TW-AUD-019) also activates its
    media-type check, which flags every feed a server mislabels as ``text/plain``
    or ``text/html``. A legitimately empty one of those must stay in the
    healthy-empty bucket rather than becoming an OVH-044 soft failure and
    dragging the feed into backoff. ``parsed.version`` is set only when the
    document really parsed as RSS/Atom, so an HTML error page (no version) and a
    truncated feed (a SAX exception, not this one) are still failures.
    """
    return bool(isinstance(bozo_exc, feedparser.NonXMLContentType) and getattr(parsed, "version", ""))


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
        updated=_parse_updated_date(raw_entry),
        summary=summary,
        source_feed=source_feed,
    )


def _record_out_of_budget(feed_url: str, health_callback: FeedHealthCallback | None, reason: str) -> None:
    """Record the typed timeout outcome for a feed that ran out of budget."""
    logger.warning("Feed fetch out of budget: %s — %s", feed_url, reason)
    if health_callback:
        health_callback(feed_url, False, reason, None, None)


async def _retry_pause(budget: Deadline) -> bool:
    """Wait out the retry backoff; False when the budget cannot fund another try.

    The pause itself is charged against the deadline, so a feed that has burned
    its budget stops retrying instead of sleeping into the next scheduler tick.
    """
    if budget.expired():
        return False
    await asyncio.sleep(budget.slice(_RETRY_BACKOFF_SECONDS))
    return not budget.expired()


async def fetch_feed(
    feed_url: str,
    client: httpx.AsyncClient | None = None,
    timeout: float = _FEED_FETCH_TIMEOUT,
    max_attempts: int = 2,
    health_callback: FeedHealthCallback | None = None,
    etag: str | None = None,
    last_modified: str | None = None,
    deadline: Deadline | None = None,
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
        deadline=deadline,
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
    deadline: Deadline | None = None,
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

    ``deadline`` bounds this feed's whole share of the attempt — the DNS check,
    every retry and the sleeps between them — rather than each transport wait
    separately (TW-AUD-018). Running out is a typed outcome: the feed is recorded
    as failed with ``DEADLINE_ERROR`` and no further attempt is made.
    """
    budget = deadline if deadline is not None else Deadline.after()
    if budget.expired():
        _record_out_of_budget(feed_url, health_callback, f"{DEADLINE_ERROR} before the feed fetch")
        return [], False
    # The SSRF check resolves DNS, which is why it runs once per feed rather than
    # once per attempt; it carries its own hard resolve cap (OVH-148).
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
                response = await bounded(
                    budget, "the feed request", safe_get(client, feed_url, headers=cond_headers or None)
                )
                # 304 Not Modified: validators still valid. Treat as an empty-but-
                # successful fetch — the existing "([], True)" bucket that
                # _fetch_auto/_fetch_manual already handle. Pass (None, None) so the
                # stored validators are preserved (COALESCE), not wiped.
                if response.status_code == 304:
                    if health_callback:
                        health_callback(feed_url, True, None, None, None)
                    return [], True
                response.raise_for_status()
                parsed = feedparser.parse(response.content, response_headers=_parser_headers(response))
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
                    if not entries and not _is_wrong_media_type_only(parsed, bozo_exc):
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
            except SourceDeadlineExceeded as exc:
                _record_out_of_budget(feed_url, health_callback, str(exc))
                return [], False
            except httpx.TimeoutException as exc:
                if attempt < max_attempts - 1 and await _retry_pause(budget):
                    logger.debug("Timeout fetching feed (attempt %d): %s", attempt + 1, feed_url)
                    continue
                logger.warning("Timeout fetching feed after %d attempts: %s", max_attempts, feed_url)
                if health_callback:
                    health_callback(feed_url, False, f"Timeout after {max_attempts} attempts: {exc}", None, None)
                return [], False
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code >= 500 and attempt < max_attempts - 1 and await _retry_pause(budget):
                    logger.debug(
                        "HTTP %d fetching feed (attempt %d): %s", exc.response.status_code, attempt + 1, feed_url
                    )
                    continue
                logger.warning("HTTP %d fetching feed: %s", exc.response.status_code, feed_url)
                if health_callback:
                    health_callback(feed_url, False, f"HTTP {exc.response.status_code}", None, None)
                return [], False
            except httpx.NetworkError as exc:
                if attempt < max_attempts - 1 and await _retry_pause(budget):
                    logger.debug(
                        "Network error fetching feed (attempt %d): %s — %s",
                        attempt + 1,
                        feed_url,
                        type(exc).__name__,
                    )
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


async def _fetch_auto(topic: Topic, request: SourceRequest) -> FeedResponse:
    """AUTO mode: try provider, fallback to next on empty/error."""
    timeout = request.timeout
    max_attempts = request.max_attempts
    health_callback = request.health_callback
    feed_state_loader = request.feed_state_loader
    deadline = request.deadline
    router = request.router
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
            deadline=deadline,
        )

        if entries:
            if router.mark_healthy(provider.name, observed_epoch=provider_epoch):
                logger.info("Provider %s recovered (back to healthy)", provider.name)
            return FeedResponse.from_source(
                provider_identity(provider),
                entries=entries,
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
            return FeedResponse.from_source(
                provider_identity(provider),
                feeds_total=1,
                feeds_failed=1 if not fetch_ok else 0,
                needs_url_resolution=False,
            )

        if deadline.expired():
            # The cascade is a second full fetch; starting one with no budget left
            # only pushes the topic further past its slot.
            logger.warning("Provider %s %s; no budget left to cascade to %s", provider.name, reason, next_provider.name)
            return FeedResponse.from_source(
                provider_identity(provider),
                feeds_total=1,
                feeds_failed=1 if not fetch_ok else 0,
                needs_url_resolution=False,
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
            deadline=deadline,
        )
        first_failed = 1 if not fetch_ok else 0

        if entries:
            if router.mark_healthy(next_provider.name, observed_epoch=next_epoch):
                logger.info("Provider %s recovered (back to healthy)", next_provider.name)
            return FeedResponse.from_source(
                provider_identity(next_provider),
                entries=entries,
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
        return FeedResponse.from_source(
            provider_identity(next_provider),
            feeds_total=2,
            feeds_failed=first_failed + (1 if not next_fetch_ok else 0),
            needs_url_resolution=False,
        )


async def _fetch_manual(topic: Topic, request: SourceRequest) -> FeedResponse:
    """MANUAL mode: fetch explicit feed URLs concurrently, skipping backed-off ones."""
    timeout = request.timeout
    max_attempts = request.max_attempts
    health_callback = request.health_callback
    feed_state_loader = request.feed_state_loader
    backoff_base_minutes = request.backoff_base_minutes
    backoff_cap_hours = request.backoff_cap_hours
    deadline = request.deadline
    if not topic.feed_urls:
        return FeedResponse()

    # Decide skips and load validators from ONE health lookup per URL. A URL listed
    # twice is one feed: fetching it per occurrence duplicated the request, the
    # health write and every per-feed counter (AUG-188).
    now = datetime.now(UTC)
    attempted: list[tuple[str, str | None, str | None]] = []  # (url, etag, last_modified)
    feeds_skipped = 0
    for url in dict.fromkeys(topic.feed_urls):
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
                deadline=deadline,
            )
            for (url, etag, last_modified) in attempted
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

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
        entries.extend(feed_entries)

    # Two configured feeds carrying one story merge on the entries themselves —
    # newest revision, then the copy with text — rather than on which feed the
    # user happened to list first (AUG-322).
    return FeedResponse(
        entries=collapse_duplicate_entries(entries),
        feeds_total=feeds_total,
        feeds_failed=feeds_failed,
        feeds_skipped=feeds_skipped,
    )


# Registered at import: the package's __init__ imports this module (and exa), so
# every mode has a fetcher before the first dispatch.
register_source(FeedMode.AUTO, _fetch_auto)
register_source(FeedMode.MANUAL, _fetch_manual)
