"""Scraping pipeline: fetch feeds, extract content, dedup, store.

The main entry point is ``fetch_new_articles_for_topic``, which
orchestrates the full pipeline from RSS fetch through DB storage.
"""

import asyncio
import logging
import sqlite3
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
from app.scraping.rss import FeedEntry, compute_article_hash, fetch_feeds_for_topic

logger = logging.getLogger(__name__)

_CONTENT_FETCH_CONCURRENCY = 3


async def fetch_new_articles_for_topic(
    topic: Topic,
    conn: sqlite3.Connection,
    max_articles: int = 10,
    feed_fetch_timeout: float = 15.0,
    article_fetch_timeout: float = 20.0,
    feed_max_retries: int = 2,
    concurrency: int = _CONTENT_FETCH_CONCURRENCY,
) -> list[Article]:
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
        List of newly stored Article objects.
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
                logger.debug("Failed to record feed health for %s", feed_url, exc_info=True)

        return callback

    # 1. Fetch all feed entries
    entries = await fetch_feeds_for_topic(
        topic,
        timeout=feed_fetch_timeout,
        max_attempts=feed_max_retries,
        health_callback=_make_health_callback(conn),
    )
    if not entries:
        return []

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
            reuse_entries.append((entry, content_hash, existing.raw_content))
        else:
            new_entries.append((entry, content_hash))

    if not new_entries and not reuse_entries:
        return []

    # 3. Sort by published date (newest first, None dates last), apply limit
    datetime_min = datetime.min.replace(tzinfo=UTC)
    new_entries.sort(
        key=lambda pair: pair[0].published or datetime_min,
        reverse=True,
    )
    reuse_entries.sort(
        key=lambda triple: triple[0].published or datetime_min,
        reverse=True,
    )
    # Combine and limit: reuse entries are cheap, prioritise by date across both
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

    # 5. Create and store articles
    stored: list[Article] = []

    # 5a. Articles with reused content
    for entry, content_hash, reused_content in reuse_batch:
        article = Article(
            topic_id=topic.id,
            title=entry.title,
            url=entry.url,
            content_hash=content_hash,
            raw_content=reused_content,
            source_feed=entry.source_feed,
        )
        try:
            created = create_article(conn, article)
            stored.append(created)
        except sqlite3.IntegrityError:
            logger.debug("Duplicate article (race condition): %s", entry.url)

    # 5b. Articles with freshly fetched content
    for (entry, content_hash), content in zip(fetch_batch, contents, strict=False):
        if isinstance(content, BaseException):
            logger.warning("Content extraction failed for %s: %s", entry.url, content)
            content = entry.summary

        article = Article(
            topic_id=topic.id,
            title=entry.title,
            url=entry.url,
            content_hash=content_hash,
            raw_content=content if isinstance(content, str) and content else None,
            source_feed=entry.source_feed,
        )
        try:
            created = create_article(conn, article)
            stored.append(created)
        except sqlite3.IntegrityError:
            logger.debug("Duplicate article (race condition): %s", entry.url)

    conn.commit()
    logger.info(
        "Topic '%s': %d new articles stored (from %d feed entries)",
        topic.name,
        len(stored),
        len(entries),
    )
    return stored
