"""Scraping pipeline: fetch feeds, extract content, dedup, store.

The main entry point is ``fetch_new_articles_for_topic``, which
orchestrates the full pipeline from RSS fetch through DB storage.
"""

import asyncio
import logging
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime

from app.crud import (
    article_hash_exists,
    create_article,
    find_article_by_hash,
    upsert_feed_health_failure,
    upsert_feed_health_success,
)
from app.models import Article, Topic
from app.scraping.content import extract_article_content
from app.scraping.google_news import resolve_google_news_urls
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

    def _make_health_callback(conn):
        def callback(feed_url, success, error_msg):
            try:
                if success:
                    upsert_feed_health_success(conn, feed_url)
                else:
                    upsert_feed_health_failure(conn, feed_url, error_msg or "Unknown error")
            except Exception:
                # feed_health is the ONLY persisted record of per-feed failures, so
                # a swallowed write (locked DB under WAL contention, schema drift,
                # disk error) leaves the dashboard showing stale health while feeds
                # silently break. Surface it at WARNING — matching the dropped-
                # duplicates loss convention — not DEBUG (OVH-132).
                logger.warning("Failed to record feed health for %s", feed_url, exc_info=True)

        return callback

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
    # Surface a degraded check: when SOME but not all feeds failed, total_feed_entries
    # only reflects the survivors, so the check looks like a healthy partial yield.
    # Log a WARNING so degraded coverage is distinguishable from healthy (OVH-130).
    if 0 < feeds_failed < feeds_total:
        logger.warning(
            "Topic '%s': partial feed-fetch failure — %d of %d feed fetch(es) failed",
            topic.name,
            feeds_failed,
            feeds_total,
        )
    elif feeds_total and feeds_failed >= feeds_total:
        logger.warning(
            "Topic '%s': all %d feed fetch(es) failed",
            topic.name,
            feeds_total,
        )

    if not entries:
        return FetchResult(
            articles=[],
            total_feed_entries=0,
            feeds_total=feeds_total,
            feeds_failed=feeds_failed,
        )

    # 2. Filter to entries not already in DB; split into reuse vs. fetch-needed
    new_entries: list[tuple[FeedEntry, str]] = []
    reuse_entries: list[tuple[FeedEntry, str, str]] = []  # (entry, hash, reused_content)
    for entry in entries:
        content_hash = compute_article_hash(entry.url, entry.title)
        if article_hash_exists(conn, topic.id, content_hash):
            continue
        existing = find_article_by_hash(conn, content_hash)
        if existing and existing.raw_content:
            logger.info(
                "Cross-topic dedup: reusing content for '%s' (from topic_id=%d)",
                entry.title,
                existing.topic_id,
            )
            # OVH-025: adopt the originating article's RESOLVED url instead of this
            # entry's (possibly unresolved news.google.com redirect). The hash was
            # already computed above from entry.url, so dedup stays intact.
            entry.url = existing.url
            reuse_entries.append((entry, content_hash, existing.raw_content))
        else:
            new_entries.append((entry, content_hash))

    if not new_entries and not reuse_entries:
        return FetchResult(
            articles=[],
            total_feed_entries=len(entries),
            feeds_total=feeds_total,
            feeds_failed=feeds_failed,
        )

    # 3. Combine reuse + fetch candidates, sort by published date (newest first,
    # None dates last), and apply the limit. Selection is purely recency-first.
    datetime_min = datetime.min.replace(tzinfo=UTC)
    all_candidates: list[tuple[FeedEntry, str, str | None]] = [(e, h, c) for e, h, c in reuse_entries] + [
        (e, h, None) for e, h in new_entries
    ]
    all_candidates.sort(
        key=lambda t: t[0].published or datetime_min,
        reverse=True,
    )
    all_candidates = all_candidates[:max_articles]

    reuse_batch = [(e, h, c) for e, h, c in all_candidates if c is not None]
    fetch_batch = [(e, h) for e, h, c in all_candidates if c is None]

    # 3b. Resolve Google News redirect URLs for entries that need content fetching.
    # Done after dedup+limiting to minimize requests (typically ~10 URLs, not 100).
    if response.needs_url_resolution:
        google_urls = [e.url for e, _ in fetch_batch if "news.google.com/" in e.url]
        if google_urls:
            resolved = await resolve_google_news_urls(google_urls, timeout=feed_fetch_timeout)
            for entry, _ in fetch_batch:
                if entry.url in resolved:
                    entry.url = resolved[entry.url]

    # 4. Extract content concurrently with semaphore (only for entries needing fetch)
    semaphore = asyncio.Semaphore(concurrency)

    async def _extract(entry: FeedEntry) -> str:
        async with semaphore:
            return await extract_article_content(
                entry.url,
                fallback_summary=entry.summary,
                timeout=article_fetch_timeout,
            )

    content_tasks = [_extract(entry) for entry, _ in fetch_batch]
    contents = await asyncio.gather(*content_tasks, return_exceptions=True)

    # 5. Normalize both batches into a uniform (entry, content_hash, content) list,
    # then run a single insert loop. Reused content is already resolved; freshly
    # fetched content needs the BaseException -> summary -> None coercion first.
    pending: list[tuple[FeedEntry, str, str | None]] = list(reuse_batch)
    for (entry, content_hash), content in zip(fetch_batch, contents, strict=False):
        if isinstance(content, BaseException):
            logger.warning("Content extraction failed for %s: %s", entry.url, content)
            content = entry.summary
        resolved_content = content if isinstance(content, str) and content else None
        pending.append((entry, content_hash, resolved_content))

    stored: list[Article] = []
    dropped_duplicates = 0
    for entry, content_hash, resolved_content in pending:
        article = Article(
            topic_id=topic.id,
            title=entry.title,
            url=entry.url,
            content_hash=content_hash,
            raw_content=resolved_content,
            source_feed=entry.source_feed,
            source_provider=response.provider_name,
        )
        if _insert_or_count_dup(conn, article, topic.name, stored):
            continue
        dropped_duplicates += 1

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
