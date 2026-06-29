"""Scraping pipeline: fetch feeds, extract content, dedup, store.

The main entry point is ``fetch_new_articles_for_topic``, which
orchestrates the full pipeline from RSS fetch through DB storage.
"""

import asyncio
import logging
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx

from app.crud import (
    article_hash_exists,
    create_article,
    find_article_by_hash,
    upsert_feed_health_failure,
    upsert_feed_health_success,
)
from app.models import Article, Topic
from app.scraping.content import extract_article_content
from app.scraping.google_news import is_google_news_url, resolve_google_news_urls
from app.scraping.rss import FeedEntry, compute_article_hash, fetch_feeds_for_topic
from app.scraping.rss import FeedResponse as FeedResponse

logger = logging.getLogger(__name__)

_CONTENT_FETCH_CONCURRENCY = 3


@dataclass
class FetchResult:
    """Result of fetching articles for a topic."""

    articles: list[Article]
    total_feed_entries: int
    dropped_duplicates: int = 0
    """Articles dropped because a concurrent insert won the UNIQUE race.

    Surfaces otherwise-silent article loss so callers/monitoring can detect it.
    Each increment corresponds to a WARNING-level log entry.
    """
    feeds_total: int = 0
    """Number of feed fetches attempted (manual: configured URLs; auto: provider
    attempts including cascade)."""
    feeds_failed: int = 0
    """How many of those fetches failed. ``0 < feeds_failed < feeds_total`` is a
    *degraded* check — some sources silently dropped out — which is logged at
    WARNING so it is not indistinguishable from a healthy partial yield (OVH-130).
    """


def _insert_or_count_dup(
    conn: sqlite3.Connection,
    article: Article,
    topic_name: str,
    stored: list[Article],
) -> bool:
    """Insert one article, handling the concurrent-insert UNIQUE race in one place.

    On success the created row is appended to ``stored`` and ``True`` is returned.
    If a concurrent insert already won the ``UNIQUE(topic_id, content_hash)`` race,
    the loss is logged at WARNING and ``False`` is returned so the caller can count
    it toward ``dropped_duplicates`` (the FetchResult observability signal).
    """
    try:
        stored.append(create_article(conn, article))
        return True
    except sqlite3.IntegrityError:
        logger.warning(
            "Dropped duplicate article (concurrent insert race) for topic '%s': %s",
            topic_name,
            article.url,
        )
        return False


def _make_health_callback(conn: sqlite3.Connection):
    """Build the per-feed health-recording callback used during feed fetch.

    The callback writes feed_health rows on ``conn``; a swallowed write is
    surfaced at WARNING because feed_health is the ONLY persisted record of
    per-feed failures, so silently losing it would leave the dashboard showing
    stale health while feeds break (OVH-132).
    """

    def callback(feed_url: str, success: bool, error_msg: str | None) -> None:
        try:
            if success:
                upsert_feed_health_success(conn, feed_url)
            else:
                upsert_feed_health_failure(conn, feed_url, error_msg or "Unknown error")
        except Exception:
            logger.warning("Failed to record feed health for %s", feed_url, exc_info=True)

    return callback


def _log_feed_coverage(topic: Topic, feeds_total: int, feeds_failed: int) -> None:
    """Log a degraded/total feed-fetch failure so partial coverage is visible.

    ``0 < feeds_failed < feeds_total`` is a *degraded* check — total_feed_entries
    only reflects the survivors, so it would otherwise look like a healthy partial
    yield (OVH-130). A full failure is logged distinctly.
    """
    if 0 < feeds_failed < feeds_total:
        logger.warning(
            "Topic '%s': partial feed-fetch failure — %d of %d feed fetch(es) failed",
            topic.name,
            feeds_failed,
            feeds_total,
        )
    elif feeds_total and feeds_failed >= feeds_total:
        logger.warning("Topic '%s': all %d feed fetch(es) failed", topic.name, feeds_total)


def _split_dedup_candidates(
    entries: list[FeedEntry],
    conn: sqlite3.Connection,
    topic_id: int,
) -> tuple[list[tuple[FeedEntry, str]], list[tuple[FeedEntry, str, str, str | None]]]:
    """Filter feed entries to those not already stored; split reuse vs. fetch-needed.

    Returns ``(new_entries, reuse_entries)``. ``new_entries`` are ``(entry, hash)``
    pairs whose content must be fetched. ``reuse_entries`` are
    ``(entry, hash, content, provider)`` for entries whose content already exists
    cross-topic (OVH-025/OVH-114): the reused row's RESOLVED url and ORIGINATING
    provider are adopted so attribution stays correct and the (already-computed)
    hash keeps dedup intact.
    """
    new_entries: list[tuple[FeedEntry, str]] = []
    reuse_entries: list[tuple[FeedEntry, str, str, str | None]] = []
    for entry in entries:
        content_hash = compute_article_hash(entry.url, entry.title)
        if article_hash_exists(conn, topic_id, content_hash):
            continue
        existing = find_article_by_hash(conn, content_hash)
        if existing and existing.raw_content:
            logger.info(
                "Cross-topic dedup: reusing content for '%s' (from topic_id=%d)",
                entry.title,
                existing.topic_id,
            )
            # OVH-025: adopt the originating article's RESOLVED url instead of this
            # entry's (possibly unresolved redirect). The hash was already computed
            # above from entry.url, so dedup stays intact.
            entry.url = existing.url
            reuse_entries.append((entry, content_hash, existing.raw_content, existing.source_provider))
        else:
            new_entries.append((entry, content_hash))
    return new_entries, reuse_entries


def _select_candidates(
    new_entries: list[tuple[FeedEntry, str]],
    reuse_entries: list[tuple[FeedEntry, str, str, str | None]],
    max_articles: int,
) -> tuple[list[tuple[FeedEntry, str, str | None, str | None]], list[tuple[FeedEntry, str]]]:
    """Combine reuse + fetch candidates, sort recency-first, apply the limit.

    Each candidate carries ``(entry, hash, reused_content, provider)``; provider is
    the originating one for reused rows and ``None`` for fresh fetches (stamped with
    this topic's provider later). Returns ``(reuse_batch, fetch_batch)`` after the
    limit, where ``fetch_batch`` is the ``(entry, hash)`` subset still needing a fetch.
    """
    datetime_min = datetime.min.replace(tzinfo=UTC)
    all_candidates: list[tuple[FeedEntry, str, str | None, str | None]] = [
        (e, h, c, p) for e, h, c, p in reuse_entries
    ] + [(e, h, None, None) for e, h in new_entries]
    all_candidates.sort(key=lambda t: t[0].published or datetime_min, reverse=True)
    all_candidates = all_candidates[:max_articles]

    reuse_batch: list[tuple[FeedEntry, str, str | None, str | None]] = [
        (e, h, c, p) for e, h, c, p in all_candidates if c is not None
    ]
    fetch_batch = [(e, h) for e, h, c, _ in all_candidates if c is None]
    return reuse_batch, fetch_batch


async def _resolve_redirect_urls(
    fetch_batch: list[tuple[FeedEntry, str]],
    response: FeedResponse,
    feed_fetch_timeout: float,
) -> None:
    """Resolve provider redirect URLs in-place for entries needing content fetch.

    Gated by the provider's ``needs_url_resolution`` (carried on the FeedResponse)
    rather than a hardcoded host substring (OVH-157): only providers that emit
    opaque redirects (Google News) opt in. The which-URLs-need-resolving decision
    is delegated to ``is_google_news_url``/``resolve_google_news_urls`` instead of
    leaking the ``news.google.com`` detail into the orchestrator. Done after
    dedup+limiting to minimize requests (typically ~10 URLs, not 100).
    """
    if not response.needs_url_resolution:
        return
    to_resolve = [e.url for e, _ in fetch_batch if is_google_news_url(e.url)]
    if not to_resolve:
        return
    resolved = await resolve_google_news_urls(to_resolve, timeout=feed_fetch_timeout)
    for entry, _ in fetch_batch:
        if entry.url in resolved:
            entry.url = resolved[entry.url]


async def _extract_contents(
    fetch_batch: list[tuple[FeedEntry, str]],
    article_fetch_timeout: float,
    concurrency: int,
) -> list[str | BaseException]:
    """Extract article content concurrently for the fetch batch.

    OVH-128: shares ONE pooled httpx client across the batch (keep-alive /
    connection reuse) instead of one client per article. The client is
    loop-confined and closed in finally, and mirrors the per-call config
    (timeout + follow_redirects=False) so the SSRF per-hop redirect checks in
    safe_get stay intact. Returns ``[]`` for an empty (reuse-only) batch so no
    client is built.
    """
    if not fetch_batch:
        return []
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=article_fetch_timeout, follow_redirects=False) as fetch_client:

        async def _extract(entry: FeedEntry) -> str:
            async with semaphore:
                return await extract_article_content(
                    entry.url,
                    fallback_summary=entry.summary,
                    client=fetch_client,
                    timeout=article_fetch_timeout,
                )

        content_tasks = [_extract(entry) for entry, _ in fetch_batch]
        return list(await asyncio.gather(*content_tasks, return_exceptions=True))


def _store_articles(
    reuse_batch: list[tuple[FeedEntry, str, str | None, str | None]],
    fetch_batch: list[tuple[FeedEntry, str]],
    contents: list[str | BaseException],
    topic: Topic,
    provider_name: str | None,
    conn: sqlite3.Connection,
) -> tuple[list[Article], int]:
    """Normalize both batches and run a single insert loop.

    Reused content is already resolved and carries its originating provider;
    freshly fetched rows carry provider=None ("this topic's provider") and need
    the BaseException -> summary -> None coercion. Returns
    ``(stored, dropped_duplicates)`` where dropped_duplicates counts rows lost to a
    concurrent UNIQUE(topic_id, content_hash) race (OVH-114 attribution preserved).
    """
    assert topic.id is not None
    pending: list[tuple[FeedEntry, str, str | None, str | None]] = list(reuse_batch)
    for (entry, content_hash), content in zip(fetch_batch, contents, strict=False):
        if isinstance(content, BaseException):
            logger.warning("Content extraction failed for %s: %s", entry.url, content)
            content = entry.summary
        resolved_content = content if isinstance(content, str) and content else None
        pending.append((entry, content_hash, resolved_content, None))

    stored: list[Article] = []
    dropped_duplicates = 0
    for entry, content_hash, resolved_content, origin_provider in pending:
        article = Article(
            topic_id=topic.id,
            title=entry.title,
            url=entry.url,
            content_hash=content_hash,
            raw_content=resolved_content,
            source_feed=entry.source_feed,
            # OVH-114: reused rows keep the originating provider; fresh rows (None)
            # are attributed to the provider that produced this topic's feed.
            source_provider=origin_provider if origin_provider is not None else provider_name,
            # Publication date is a property of the article itself, so even reused
            # rows take THIS feed entry's parsed date (not the originating row's) —
            # unlike source_provider above, which is about the origin fetch.
            published_at=entry.published,
        )
        if not _insert_or_count_dup(conn, article, topic.name, stored):
            dropped_duplicates += 1
    return stored, dropped_duplicates


async def fetch_new_articles_for_topic(
    topic: Topic,
    conn: sqlite3.Connection,
    max_articles: int = 10,
    feed_fetch_timeout: float = 15.0,
    article_fetch_timeout: float = 20.0,
    feed_max_retries: int = 2,
    concurrency: int = _CONTENT_FETCH_CONCURRENCY,
) -> FetchResult:
    """Fetch feeds, dedup against DB, extract content, and store new articles.

    Args:
        topic: The topic to fetch articles for (must have an id).
        conn: Database connection for dedup checks and article storage.
        max_articles: Maximum number of new articles to process per call.
        feed_fetch_timeout: Timeout in seconds for RSS feed fetches.
        article_fetch_timeout: Timeout in seconds for article content fetches.
        feed_max_retries: Maximum retry attempts for feed fetching.
        concurrency: Maximum number of concurrent article content fetches.

    Returns:
        FetchResult with stored articles and total feed entry count.
    """
    if topic.id is None:
        raise ValueError("Topic must have an ID")

    # 1. Fetch all feed entries. The health callback writes feed_health rows on
    # ``conn``; commit immediately afterwards so that write lock is NOT held
    # across the later content-extraction await (OVH-007: WAL single-writer
    # starvation). From here the connection performs only SELECTs until the
    # final article-insert phase, so no write transaction spans the awaits.
    response = await fetch_feeds_for_topic(
        topic,
        timeout=feed_fetch_timeout,
        max_attempts=feed_max_retries,
        health_callback=_make_health_callback(conn),
    )
    conn.commit()
    entries = response.entries

    feeds_total = response.feeds_total
    feeds_failed = response.feeds_failed
    _log_feed_coverage(topic, feeds_total, feeds_failed)

    if not entries:
        return FetchResult(
            articles=[],
            total_feed_entries=0,
            feeds_total=feeds_total,
            feeds_failed=feeds_failed,
        )

    # 2. Dedup against the DB and split into reuse vs. fetch-needed.
    new_entries, reuse_entries = _split_dedup_candidates(entries, conn, topic.id)
    if not new_entries and not reuse_entries:
        return FetchResult(
            articles=[],
            total_feed_entries=len(entries),
            feeds_total=feeds_total,
            feeds_failed=feeds_failed,
        )

    # 3. Combine, sort recency-first, and apply the limit.
    reuse_batch, fetch_batch = _select_candidates(new_entries, reuse_entries, max_articles)

    # 3b. Resolve provider redirect URLs for entries needing a content fetch.
    await _resolve_redirect_urls(fetch_batch, response, feed_fetch_timeout)

    # 4. Extract content concurrently (only for entries needing fetch).
    contents = await _extract_contents(fetch_batch, article_fetch_timeout, concurrency)

    # 5. Normalize both batches and run a single insert loop.
    stored, dropped_duplicates = _store_articles(
        reuse_batch, fetch_batch, contents, topic, response.provider_name, conn
    )

    conn.commit()
    if dropped_duplicates:
        logger.warning(
            "Topic '%s': %d article(s) dropped as duplicates during concurrent inserts",
            topic.name,
            dropped_duplicates,
        )
    logger.info(
        "Topic '%s': %d new articles stored (from %d feed entries)",
        topic.name,
        len(stored),
        len(entries),
    )
    return FetchResult(
        articles=stored,
        total_feed_entries=len(entries),
        dropped_duplicates=dropped_duplicates,
        feeds_total=feeds_total,
        feeds_failed=feeds_failed,
    )
