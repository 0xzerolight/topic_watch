"""Scraping pipeline: fetch feeds, extract content, dedup, store.

The main entry point is ``fetch_new_articles_for_topic``, which
orchestrates the full pipeline from RSS fetch through DB storage.
"""

import asyncio
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from app.crud import (
    create_article,
    find_article_by_hash,
    get_feed_health,
    list_article_dedup_keys,
    topic_has_articles_from_feed,
    upsert_feed_health_aborted,
    upsert_feed_health_failure,
    upsert_feed_health_success,
)
from app.database import get_connection, get_db
from app.models import Article, FeedHealth, Topic
from app.scraping.content import extract_article_content
from app.scraping.exa import fetch_exa_source as fetch_exa_source
from app.scraping.google_news import is_google_news_url, resolve_google_news_urls
from app.scraping.rss import FeedEntry, fetch_feeds_for_topic
from app.scraping.rss import FeedResponse as FeedResponse
from app.scraping.source import Deadline as Deadline
from app.scraping.source import FeedHealthOutcome as FeedHealthOutcome
from app.scraping.source import FetchStatus as FetchStatus
from app.scraping.source import (
    article_identity,
    assign_ranking_dates,
    bounded,
    collapse_duplicate_entries,
    compute_article_hash,
    ranking_date,
    revision_marker,
    story_identity,
)

if TYPE_CHECKING:
    from app.config import ExaSettings

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
    feeds_skipped: int = 0
    """Manual feeds skipped this cycle because they are in a backoff window
    (persistently failing). Not a failure — excluded from feeds_total/failed."""


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


def apply_feed_health(conn: sqlite3.Connection, outcomes: list[FeedHealthOutcome]) -> None:
    """Persist collected feed-health outcomes. Contains no awaits; caller commits.

    A swallowed write is surfaced at WARNING because feed_health is the ONLY
    persisted record of per-feed failures, so silently losing it would leave the
    dashboard showing stale health while feeds break (OVH-132).

    The outcome's status decides which of the three writes applies. A 200 is the
    feed's current statement of which validators it issues, so its headers replace
    the stored pair exactly, clearing one the feed has stopped sending; only a
    304's silence means "unchanged" and preserves them (AUG-152). An aborted fetch
    is recorded for diagnostics without touching the failure streak.
    """
    for outcome in outcomes:
        try:
            if outcome.status is FetchStatus.ABORTED:
                upsert_feed_health_aborted(conn, outcome.feed_url, outcome.error_msg or "Unknown error")
            elif outcome.status.succeeded:
                upsert_feed_health_success(
                    conn,
                    outcome.feed_url,
                    outcome.etag,
                    outcome.last_modified,
                    replace_validators=outcome.status is not FetchStatus.NOT_MODIFIED,
                )
            else:
                upsert_feed_health_failure(conn, outcome.feed_url, outcome.error_msg or "Unknown error")
        except Exception:
            logger.warning("Failed to record feed health for %s", outcome.feed_url, exc_info=True)


def _commit_feed_health(db_path: Path | None, outcomes: list[FeedHealthOutcome]) -> None:
    """Apply and commit collected health outcomes on their own short connection."""
    if not outcomes:
        return
    try:
        with get_db(db_path) as conn:
            apply_feed_health(conn, outcomes)
    except Exception:
        logger.warning("Failed to persist feed health outcomes", exc_info=True)


def _make_health_collector(outcomes: list[FeedHealthOutcome]):
    """Build the per-feed health callback that appends to ``outcomes`` in memory."""

    def callback(outcome: FeedHealthOutcome) -> None:
        outcomes.append(outcome)

    return callback


def _make_feed_article_check(db_path: Path | None, topic_id: int):
    """Build the "does this topic already hold articles from this feed" probe.

    Answers the one question conditional requests need before they are safe to
    send: ``feed_health`` stores validators per feed URL, but articles belong to a
    topic, so a topic that has never received this feed's representation would get
    a 304 for articles it does not have (TW-AUD-020). Opens and closes its own
    connection per call for the same reason ``_make_feed_state_loader`` does — it
    is called from inside the fetch coroutine, between network awaits.
    """

    def check(feed_url: str) -> bool:
        conn = get_connection(db_path)
        try:
            return topic_has_articles_from_feed(conn, topic_id, feed_url)
        finally:
            conn.close()

    return check


def _make_feed_state_loader(db_path: Path | None):
    """Build a per-feed health-row loader (one SELECT, reused for validators + backoff).

    Returns the stored ``FeedHealth`` for a URL (or ``None`` if untracked), so the
    fetch layer can both send conditional-GET validators and decide backoff skips
    from a single lookup.

    Each lookup opens and closes its own connection. The loader is called from
    inside the fetch coroutine, interleaved with network awaits, so a connection
    held for the loader's lifetime would be a connection held across those awaits
    — the exact invariant AUG-136/AUG-171 exist to restore. Each individual call
    is synchronous and await-free, so nothing is ever open while I/O is pending;
    the cost is one sub-millisecond local open per configured feed.
    """

    def loader(feed_url: str) -> FeedHealth | None:
        conn = get_connection(db_path)
        try:
            return get_feed_health(conn, feed_url)
        finally:
            conn.close()

    return loader


def all_sources_failed(feeds_total: int, feeds_failed: int) -> bool:
    """True when every attempted feed source failed (source-agnostic).

    ``feeds_total == 0`` means nothing was attempted (e.g. an EXA topic with Exa
    disabled), which is NOT a fetch failure — so this returns False there.
    """
    return bool(feeds_total) and feeds_failed >= feeds_total


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
    elif all_sources_failed(feeds_total, feeds_failed):
        logger.warning("Topic '%s': all %d feed fetch(es) failed", topic.name, feeds_total)


def _prepare_entries(entries: list[FeedEntry]) -> list[FeedEntry]:
    """Normalize what the source returned before anything spends budget on it.

    Two source-agnostic corrections, applied once for every provider rather than
    per fetcher: impossible dates are discarded so they cannot own the top of the
    recency ranking (AUG-184), and entries describing one article collapse to
    their best copy so repeats stop consuming article slots and content fetches
    (AUG-179).

    ``updated`` goes through the same skew guard as ``published``, because it is
    the signal that says an article was revised: a publisher whose clock runs
    hours fast had every entry read as a revision, which bypasses the story rule
    permanently and re-stores the whole feed on every check.

    Ranking is settled here too, while feed order is still intact — it is the
    only place that knows which entry followed which.
    """
    assign_ranking_dates(entries)
    return collapse_duplicate_entries(entries)


@dataclass(frozen=True)
class _StoredArticles:
    """What a topic already holds, as the two keys dedup compares an entry against.

    Read once per check rather than queried per entry, because the story key is
    derived from a stored row's URL and title and so cannot be looked up directly
    without an index this schema does not have.
    """

    identities: frozenset[str]
    stories: frozenset[str]

    @classmethod
    def load(cls, conn: sqlite3.Connection, topic_id: int) -> "_StoredArticles":
        rows = list_article_dedup_keys(conn, topic_id)
        return cls(
            identities=frozenset(content_hash for content_hash, _, _ in rows),
            stories=frozenset(compute_article_hash(url, title) for _, url, title in rows),
        )

    def holds(self, entry: FeedEntry) -> bool:
        """True when this entry is an article the topic has already stored.

        Either the exact representation is stored, or the story is stored and the
        entry offers no evidence of a revision. The second rule is what makes a
        provider's redirect wrapper, a tracking variant and another provider's copy
        of one story a single article (AUG-180); the revision exception is what
        lets a correction through (AUG-320).
        """
        if article_identity(entry) in self.identities:
            return True
        return not revision_marker(entry) and story_identity(entry) in self.stories


def _split_dedup_candidates(
    entries: list[FeedEntry],
    stored: _StoredArticles,
    conn: sqlite3.Connection,
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
        content_hash = article_identity(entry)
        if stored.holds(entry):
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

    Entries are ranked by the date ``assign_ranking_dates`` settled for them
    (AUG-184), not by the one they claim. Ties keep source order, since the sort
    is stable.
    """
    all_candidates: list[tuple[FeedEntry, str, str | None, str | None]] = [
        (e, h, c, p) for e, h, c, p in reuse_entries
    ] + [(e, h, None, None) for e, h in new_entries]
    all_candidates.sort(key=lambda t: ranking_date(t[0]), reverse=True)
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
    deadline: Deadline,
) -> bool:
    """Resolve provider redirect URLs in-place for entries needing content fetch.

    Returns whether any URL actually changed, which is what tells the caller the
    identities computed before this point need re-checking.

    Gated by the provider's ``needs_url_resolution`` (carried on the FeedResponse)
    rather than a hardcoded host substring (OVH-157): only providers that emit
    opaque redirects (Google News) opt in. The which-URLs-need-resolving decision
    is delegated to ``is_google_news_url``/``resolve_google_news_urls`` instead of
    leaking the ``news.google.com`` detail into the orchestrator. Done after
    dedup+limiting to minimize requests (typically ~10 URLs, not 100).
    """
    if not response.needs_url_resolution:
        return False
    to_resolve = [e.url for e, _ in fetch_batch if is_google_news_url(e.url)]
    if not to_resolve:
        return False
    resolved = await resolve_google_news_urls(to_resolve, timeout=feed_fetch_timeout, deadline=deadline)
    changed = False
    for entry, _ in fetch_batch:
        if entry.url in resolved:
            entry.url = resolved[entry.url]
            changed = True
    return changed


def _drop_resolved_duplicates(
    fetch_batch: list[tuple[FeedEntry, str]],
    stored: _StoredArticles,
) -> list[tuple[FeedEntry, str]]:
    """Re-apply dedup to entries whose real URL only appeared during resolution.

    A Google News entry names an opaque wrapper until it is resolved, so the
    story it belongs to is unknown while the first dedup pass runs. Once the
    publisher URL is known, an entry the topic already holds — under a previous
    wrapper, or from the other provider, which hands that URL over directly — is
    dropped before its content is fetched (AUG-180). Entries that collapse onto
    each other inside the batch are dropped the same way.

    The stored key stays the one computed from what the feed handed over, so a
    later check recognises an unchanged entry without spending a resolution on it.
    """
    kept: list[tuple[FeedEntry, str]] = []
    seen_stories: set[str] = set()
    for entry, content_hash in fetch_batch:
        story = story_identity(entry)
        if stored.holds(entry) or story in seen_stories:
            logger.info("Resolved URL belongs to a story this topic already has: %s", entry.url)
            continue
        seen_stories.add(story)
        kept.append((entry, content_hash))
    return kept


async def _extract_contents(
    fetch_batch: list[tuple[FeedEntry, str]],
    article_fetch_timeout: float,
    concurrency: int,
    deadline: Deadline,
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
                if deadline.expired():
                    # Out of budget (TW-AUD-018): keep whatever the source already
                    # gave us instead of starting another fetch. Routing it through
                    # the prefetched short-circuit applies the same length cap the
                    # network path would have, with no request.
                    already_have = entry.content or entry.summary
                    if not already_have:
                        return ""
                    return await extract_article_content(entry.url, prefetched=already_have)
                # entry.content (Exa prefetched text) short-circuits the fetch; RSS and
                # empty-text entries carry content=None and fall through to the network path.
                return await bounded(
                    deadline,
                    "article extraction",
                    extract_article_content(
                        entry.url,
                        fallback_summary=entry.summary,
                        client=fetch_client,
                        timeout=article_fetch_timeout,
                        prefetched=entry.content,
                    ),
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
    *,
    db_path: Path | None = None,
    max_articles: int = 10,
    feed_fetch_timeout: float = 15.0,
    article_fetch_timeout: float = 20.0,
    feed_max_retries: int = 2,
    concurrency: int = _CONTENT_FETCH_CONCURRENCY,
    feed_backoff_base_minutes: int = 15,
    feed_backoff_cap_hours: int = 24,
    exa_settings: "ExaSettings | None" = None,
    deadline: Deadline | None = None,
) -> FetchResult:
    """Fetch feeds, dedup against DB, extract content, and store new articles.

    Connection lifetime (AUG-136/AUG-171): this function owns its database access
    and never accepts a caller's connection, because every phase boundary here is
    a network await. It opens exactly two short-lived connections — one to apply
    feed health and run the dedup SELECTs, one to insert the surviving articles —
    and holds neither across any I/O.

    Args:
        topic: The topic to fetch articles for (must have an id).
        db_path: Database path used to open the short-lived phase connections.
        max_articles: Maximum number of new articles to process per call.
        feed_fetch_timeout: Timeout in seconds for RSS feed fetches.
        article_fetch_timeout: Timeout in seconds for article content fetches.
        feed_max_retries: Maximum retry attempts for feed fetching.
        concurrency: Maximum number of concurrent article content fetches.
        exa_settings: Exa configuration, required for EXA-mode topics (ignored otherwise).
        deadline: One monotonic budget for this whole logical attempt (TW-AUD-018).
            Feed retries, redirect resolution and content extraction each used to
            draw their own, so their sum was unbounded and a merely-slow source
            could hold the topic slot past the next scheduler tick. Callers that
            own a larger unit of work pass theirs; otherwise one starts here.

    Returns:
        FetchResult with stored articles and total feed entry count.
    """
    if topic.id is None:
        raise ValueError("Topic must have an ID")
    budget = deadline if deadline is not None else Deadline.after()

    # --- P1: fetch all feed entries with NO connection open. Per-feed health
    # verdicts accumulate in memory; they are applied below in one short phase.
    health_outcomes: list[FeedHealthOutcome] = []
    try:
        response = await fetch_feeds_for_topic(
            topic,
            timeout=feed_fetch_timeout,
            max_attempts=feed_max_retries,
            health_callback=_make_health_collector(health_outcomes),
            feed_state_loader=_make_feed_state_loader(db_path),
            topic_holds_feed_articles=_make_feed_article_check(db_path, topic.id),
            backoff_base_minutes=feed_backoff_base_minutes,
            backoff_cap_hours=feed_backoff_cap_hours,
            exa_settings=exa_settings,
            max_results=max_articles,
            deadline=budget,
        )
    except BaseException:
        # A partial fetch still observed real per-feed failures. Persist what was
        # collected before propagating (including on cancellation), so an aborted
        # cycle does not silently reset the dashboard's only failure record.
        _commit_feed_health(db_path, health_outcomes)
        raise

    entries = _prepare_entries(response.entries)
    feeds_total = response.feeds_total
    feeds_failed = response.feeds_failed
    _log_feed_coverage(topic, feeds_total, feeds_failed)

    empty_result = FetchResult(
        articles=[],
        total_feed_entries=len(entries),
        feeds_total=feeds_total,
        feeds_failed=feeds_failed,
        feeds_skipped=response.feeds_skipped,
    )

    # --- C1: one short connection applies the health batch and runs the dedup
    # SELECTs, then closes before the redirect/extraction awaits below.
    with get_db(db_path) as conn:
        apply_feed_health(conn, health_outcomes)
        conn.commit()
        if not entries:
            return empty_result
        already_held = _StoredArticles.load(conn, topic.id)
        new_entries, reuse_entries = _split_dedup_candidates(entries, already_held, conn)

    if not new_entries and not reuse_entries:
        return empty_result

    # Combine, sort recency-first, and apply the limit.
    reuse_batch, fetch_batch = _select_candidates(new_entries, reuse_entries, max_articles)

    # --- P1b: redirect resolution and content extraction, still connection-free.
    if await _resolve_redirect_urls(fetch_batch, response, feed_fetch_timeout, budget):
        # Resolution revealed which story these entries belong to, so dedup runs
        # once more before their content is fetched (AUG-180). No connection is
        # needed: the topic's keys were read in the phase above.
        fetch_batch = _drop_resolved_duplicates(fetch_batch, already_held)
    contents = await _extract_contents(fetch_batch, article_fetch_timeout, concurrency, budget)

    # --- C2: one short connection normalizes both batches and inserts.
    with get_db(db_path) as conn:
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
        feeds_skipped=response.feeds_skipped,
    )
