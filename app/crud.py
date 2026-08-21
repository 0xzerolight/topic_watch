"""CRUD operations for all data models.

All functions accept a sqlite3.Connection as their first parameter
for explicit dependency injection and testability.
"""

import logging
import sqlite3
import unicodedata
from datetime import UTC, datetime, timedelta

from app.models import (
    Article,
    CheckResult,
    DashboardStats,
    FeedHealth,
    KnowledgeRevision,
    KnowledgeState,
    NotificationKind,
    PendingNotification,
    PendingWebhook,
    Topic,
    TopicStatus,
    to_db_utc,
    to_utc,
)

logger = logging.getLogger(__name__)


# --- Topic CRUD ---


def create_topic(conn: sqlite3.Connection, topic: Topic) -> Topic:
    """Insert a new topic and return it with the generated ID."""
    data = topic.to_insert_dict()
    cursor = conn.execute(
        """INSERT INTO topics (name, description, feed_urls, feed_mode,
           created_at, status_changed_at, is_active, status, error_message, check_interval_minutes, tags,
           confidence_threshold, relevance_threshold, novelty_instruction, importance_threshold, init_attempts,
           generation)
           VALUES (:name, :description, :feed_urls, :feed_mode,
           :created_at, :status_changed_at, :is_active, :status, :error_message, :check_interval_minutes, :tags,
           :confidence_threshold, :relevance_threshold, :novelty_instruction, :importance_threshold,
           :init_attempts, :generation)""",
        data,
    )
    topic.id = cursor.lastrowid
    logger.info("Created topic: %s (id=%d)", topic.name, topic.id)
    return topic


def get_topic(conn: sqlite3.Connection, topic_id: int) -> Topic | None:
    """Get a topic by ID, or None if not found."""
    row = conn.execute("SELECT * FROM topics WHERE id = ?", (topic_id,)).fetchone()
    return Topic.from_row(row) if row else None


def get_topic_by_name(conn: sqlite3.Connection, name: str) -> Topic | None:
    """Get a topic by name, or None if not found."""
    row = conn.execute("SELECT * FROM topics WHERE name = ?", (name,)).fetchone()
    return Topic.from_row(row) if row else None


def list_topics(
    conn: sqlite3.Connection,
    active_only: bool = False,
    tag: str | None = None,
    is_active: bool | None = None,
) -> list[Topic]:
    """List all topics, optionally filtering by active state and/or tag.

    The active state is a tri-state filter:
    - ``is_active=True``  -> only active topics  (``WHERE is_active = 1``)
    - ``is_active=False`` -> only inactive topics (``WHERE is_active = 0``)
    - ``is_active=None``  -> no active-state filter

    ``active_only=True`` is kept as a backwards-compatible one-way shorthand for
    ``is_active=True``; an explicit ``is_active`` value takes precedence.
    """
    where_clauses = []
    params: list = []

    if is_active is None and active_only:
        is_active = True

    if is_active is not None:
        where_clauses.append("t.is_active = ?")
        params.append(1 if is_active else 0)

    if tag:
        where_clauses.append("json_each.value = ?")
        params.append(tag)

    from_clause = "FROM topics t, json_each(t.tags)" if tag else "FROM topics t"

    where_sql = ""
    if where_clauses:
        where_sql = "WHERE " + " AND ".join(where_clauses)

    rows = conn.execute(
        f"SELECT DISTINCT t.* {from_clause} {where_sql} ORDER BY t.name",
        params,
    ).fetchall()
    return [Topic.from_row(row) for row in rows]


def update_topic(conn: sqlite3.Connection, topic: Topic) -> Topic:
    """Update an existing topic. The topic must have an ID."""
    if topic.id is None:
        raise ValueError("Cannot update a topic without an ID")
    data = topic.to_insert_dict()
    data["id"] = topic.id
    conn.execute(
        """UPDATE topics SET name=:name, description=:description,
           feed_urls=:feed_urls, feed_mode=:feed_mode,
           is_active=:is_active, status=:status, status_changed_at=:status_changed_at,
           error_message=:error_message, check_interval_minutes=:check_interval_minutes,
           tags=:tags, confidence_threshold=:confidence_threshold,
           relevance_threshold=:relevance_threshold, novelty_instruction=:novelty_instruction,
           importance_threshold=:importance_threshold, init_attempts=:init_attempts
           WHERE id=:id""",
        data,
    )
    logger.info("Updated topic: %s (id=%d)", topic.name, topic.id)
    return topic


def update_topic_config(conn: sqlite3.Connection, topic: Topic) -> Topic:
    """Update only the columns the edit form owns. Does not commit.

    The edit handler loads a ``Topic``, awaits DNS validation of the submitted
    feed URLs, and only then writes. A full-row ``update_topic`` from that
    pre-await snapshot also rewrites status, status_changed_at, error_message and
    init_attempts — so an initialization that finished during the await was undone,
    putting a READY topic back into RESEARCHING until stuck recovery marked it
    ERROR, or marking an in-flight one READY and letting checks run against
    knowledge that does not exist yet (AUG-022). Lifecycle columns are owned by the
    init/check paths and are left alone here; ``is_active`` belongs to the
    enable/disable command.
    """
    if topic.id is None:
        raise ValueError("Cannot update a topic without an ID")
    data = topic.to_insert_dict()
    data["id"] = topic.id
    conn.execute(
        """UPDATE topics SET name=:name, description=:description,
           feed_urls=:feed_urls, feed_mode=:feed_mode,
           check_interval_minutes=:check_interval_minutes, tags=:tags,
           confidence_threshold=:confidence_threshold,
           relevance_threshold=:relevance_threshold,
           novelty_instruction=:novelty_instruction,
           importance_threshold=:importance_threshold
           WHERE id=:id""",
        data,
    )
    logger.info("Updated topic configuration: %s (id=%d)", topic.name, topic.id)
    return topic


def update_topic_init_status(
    conn: sqlite3.Connection,
    topic_id: int,
    *,
    status: TopicStatus,
    status_changed_at: datetime,
    error_message: str | None,
    init_attempts: int,
    expected_status: TopicStatus | None = None,
    generation: str | None = None,
) -> bool:
    """Targeted UPDATE of only the init-lifecycle columns a topic init owns.

    Unlike ``update_topic`` (which rewrites the whole row from a possibly-stale
    in-memory ``Topic``), this writes only status/error/init_attempts so a
    concurrent UI edit to feeds/thresholds during the long init await is not
    clobbered (OVH-100). Does not commit; the caller owns the transaction.

    ``expected_status`` and ``generation`` fence the write to the claim the caller
    still owns. Without them, an initializer that stuck recovery had already given
    up on (RESEARCHING -> ERROR) could still land its terminal READY/ERROR minutes
    later, making the topic's final state last-writer-wins between live work and
    recovery (AUG-139). Returns True only when the row was actually updated.
    """
    sql = """UPDATE topics
             SET status = ?, status_changed_at = ?, error_message = ?, init_attempts = ?
             WHERE id = ?"""
    params: list = [status.value, to_db_utc(status_changed_at), error_message, init_attempts, topic_id]
    if expected_status is not None:
        sql += " AND status = ?"
        params.append(expected_status.value)
    if generation is not None:
        sql += " AND generation = ?"
        params.append(generation)
    cursor = conn.execute(sql, params)
    return cursor.rowcount == 1


def recover_stuck_topics(conn: sqlite3.Connection) -> int:
    """Mark all RESEARCHING topics as ERROR.

    Called at startup — any topic still in RESEARCHING status when the
    server starts is definitively stuck (the background task is dead).

    ``status_changed_at`` moves with the status: it means "when the current status
    was entered", so leaving it at the start of the abandoned research made a
    freshly recovered topic report an ERROR age of however long it had been
    researching (AUG-158).
    """
    cursor = conn.execute(
        "UPDATE topics SET status = ?, status_changed_at = ?, error_message = ? WHERE status = ?",
        (
            TopicStatus.ERROR.value,
            to_db_utc(datetime.now(UTC)),
            "Research interrupted by server restart. Click Retry.",
            TopicStatus.RESEARCHING.value,
        ),
    )
    count = cursor.rowcount
    if count:
        logger.warning("Recovered %d stuck topic(s) from previous run", count)
    return count


def recover_stuck_researching(conn: sqlite3.Connection, timeout_minutes: int = 15) -> int:
    """Mark RESEARCHING topics stuck longer than timeout_minutes as ERROR.

    Uses status_changed_at to determine how long a topic has been in
    RESEARCHING status. Topics that entered RESEARCHING more than
    timeout_minutes ago without completing are considered stuck.

    Does not commit; the caller owns the transaction (invariant #12), matching
    ``recover_stuck_topics``. The scheduler's ``_recover_stuck`` runs this inside
    a ``get_db`` block, which commits on success (OVH-087).

    Stamps ``status_changed_at`` with the recovery time for the same reason
    ``recover_stuck_topics`` does (AUG-158). The staleness predicate reads the old
    value before the SET applies, so the window is still measured from the start
    of the research.
    """
    cursor = conn.execute(
        """UPDATE topics SET status = ?, status_changed_at = ?, error_message = ?
           WHERE status = ?
             AND status_changed_at IS NOT NULL
             AND datetime(status_changed_at, '+' || ? || ' minutes') <= datetime('now')""",
        (
            TopicStatus.ERROR.value,
            to_db_utc(datetime.now(UTC)),
            "Research timed out (stuck detection). Click Retry.",
            TopicStatus.RESEARCHING.value,
            timeout_minutes,
        ),
    )
    count = cursor.rowcount
    if count:
        logger.warning("Recovered %d stuck researching topic(s)", count)
    return count


# Shared SELECT/JOIN for the dashboard listing: each topic joined to its most
# recent check result plus an article-count subquery. Only the WHERE clause
# varies between the unfiltered dashboard and the filtered search; values always
# flow through ``?`` placeholders.
#
# Confidence is read with SQLite ``json_extract`` so the dashboard renders the
# confidence badge from a single scalar instead of shipping the full
# ``llm_response`` blob (several KB) per topic and re-parsing it in Python
# (OVH-052). The blob is selected only on detail/export paths that need the
# payload. ``json_extract`` over a fixed column path is parameter-free SQL.
_DASHBOARD_SELECT = """
    SELECT t.*,
           cr.id AS cr_id,
           cr.checked_at AS cr_checked_at,
           cr.articles_found AS cr_articles_found,
           cr.articles_new AS cr_articles_new,
           cr.has_new_info AS cr_has_new_info,
           json_extract(cr.llm_response, '$.confidence') AS cr_confidence,
           cr.notification_sent AS cr_notification_sent,
           cr.notification_error AS cr_notification_error,
           cr.stage_error AS cr_stage_error,
           cr.seen_at AS cr_seen_at,
           (SELECT COUNT(*) FROM articles WHERE articles.topic_id = t.id) AS article_count
    FROM topics t
    LEFT JOIN check_results cr ON cr.id = (
        SELECT id FROM check_results
        WHERE topic_id = t.id
        ORDER BY checked_at DESC LIMIT 1
    )
"""


def _query_dashboard_rows(
    conn: sqlite3.Connection,
    where_sql: str,
    params: list,
) -> list[dict]:
    """Run the shared dashboard SELECT with an optional WHERE clause.

    ``where_sql`` is built only from which filters are present (clause
    *structure*); all filter *values* are bound via ``params`` placeholders.
    """
    rows = conn.execute(
        f"{_DASHBOARD_SELECT}{where_sql} ORDER BY t.name",
        params,
    ).fetchall()

    result = []
    for row in rows:
        topic = Topic.from_row(row)
        last_check = None
        if row["cr_id"] is not None and topic.id is not None:
            # Map the cr_-prefixed join aliases to the model via the shared helper
            # so the dashboard path no longer re-implements CheckResult's coercion
            # coupling inline (OVH-151). Confidence stays pre-extracted by SQL and
            # the full llm_response blob is intentionally not shipped here (OVH-052).
            last_check = CheckResult.from_dashboard_row(row, topic.id)
        result.append(
            {
                "topic": topic,
                "last_check": last_check,
                "article_count": row["article_count"],
            }
        )
    return result


def get_dashboard_data(conn: sqlite3.Connection) -> list[dict]:
    """Get all topics with last check and article count in a single query."""
    return _query_dashboard_rows(conn, "", [])


def _search_canonical(text: str) -> str:
    """NFC-normalize and casefold text for literal, Unicode-insensitive search matching.

    Used on both the query and the searched fields so ``%``/``_`` stay plain
    characters instead of SQL LIKE wildcards, and composed/decomposed or
    differently-cased Unicode text still matches (AUG-337).
    """
    return unicodedata.normalize("NFC", text).casefold()


def search_dashboard_data(
    conn: sqlite3.Connection,
    query: str | None = None,
    status: str | None = None,
) -> list[dict]:
    """Get topics with last check and article count, with optional name/status filters.

    ``query`` matches literally (no LIKE wildcards) against both name and
    description (AUG-103), NFC-normalized and casefolded on both sides so
    Unicode-equivalent and differently-cased text still matches, and trimmed
    of surrounding whitespace (AUG-337).
    """
    where_clauses = []
    params: list = []

    if status:
        where_clauses.append("t.status = ?")
        params.append(status)

    where_sql = ""
    if where_clauses:
        where_sql = " WHERE " + " AND ".join(where_clauses)

    rows = _query_dashboard_rows(conn, where_sql, params)

    query = query.strip() if query else None
    if query:
        needle = _search_canonical(query)
        rows = [
            item
            for item in rows
            if needle in _search_canonical(item["topic"].name)
            or (item["topic"].description and needle in _search_canonical(item["topic"].description))
        ]

    return rows


def delete_topic(conn: sqlite3.Connection, topic_id: int) -> bool:
    """Delete a topic by ID. Returns True if a row was deleted."""
    cursor = conn.execute("DELETE FROM topics WHERE id = ?", (topic_id,))
    deleted = cursor.rowcount > 0
    if deleted:
        logger.info("Deleted topic id=%d", topic_id)
    return deleted


def get_topics_due_for_check(conn: sqlite3.Connection, default_interval_minutes: int) -> list[Topic]:
    """Get active READY topics whose check interval has elapsed.

    Uses topic.check_interval_minutes if set, otherwise falls back to
    default_interval_minutes. Topics with no check results are always due.
    NULLIF guards against a stored 0 falling through COALESCE as non-NULL.
    """
    rows = conn.execute(
        """
        SELECT t.*
        FROM topics t
        LEFT JOIN (
            SELECT topic_id, MAX(checked_at) AS last_checked_at
            FROM check_results
            GROUP BY topic_id
        ) cr ON cr.topic_id = t.id
        WHERE t.is_active = 1
          AND t.status = 'ready'
          AND (
              cr.last_checked_at IS NULL
              OR datetime(cr.last_checked_at,
                  '+' || COALESCE(NULLIF(t.check_interval_minutes, 0), ?) || ' minutes'
              ) <= datetime('now')
          )
        ORDER BY t.name
        """,
        (default_interval_minutes,),
    ).fetchall()
    return [Topic.from_row(row) for row in rows]


# --- Article CRUD ---


def create_article(conn: sqlite3.Connection, article: Article) -> Article:
    """Insert a new article. Returns the article with generated ID."""
    data = article.to_insert_dict()
    cursor = conn.execute(
        """INSERT INTO articles (topic_id, title, url, content_hash,
           raw_content, source_feed, source_provider, published_at, fetched_at, processed)
           VALUES (:topic_id, :title, :url, :content_hash,
           :raw_content, :source_feed, :source_provider, :published_at, :fetched_at, :processed)""",
        data,
    )
    article.id = cursor.lastrowid
    return article


def get_article(conn: sqlite3.Connection, article_id: int) -> Article | None:
    """Get an article by ID, or None if not found."""
    row = conn.execute("SELECT * FROM articles WHERE id = ?", (article_id,)).fetchone()
    return Article.from_row(row) if row else None


def list_articles_for_topic(
    conn: sqlite3.Connection,
    topic_id: int,
    unprocessed_only: bool = False,
    limit: int | None = None,
    offset: int = 0,
) -> list[Article]:
    """List articles for a topic, optionally filtering to unprocessed."""
    if unprocessed_only:
        query = "SELECT * FROM articles WHERE topic_id = ? AND processed = 0 ORDER BY fetched_at DESC"
        params: list = [topic_id]
    else:
        query = "SELECT * FROM articles WHERE topic_id = ? ORDER BY fetched_at DESC"
        params = [topic_id]
    if limit is not None:
        query += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])
    rows = conn.execute(query, params).fetchall()
    return [Article.from_row(row) for row in rows]


def list_article_headers_for_topic(
    conn: sqlite3.Connection,
    topic_id: int,
    limit: int | None = None,
    offset: int = 0,
) -> list[Article]:
    """List articles for a topic without hydrating ``raw_content`` (AUG-038).

    Topic detail renders only title/url/source/provider/timestamps, but the
    full-row ``list_articles_for_topic`` (``SELECT *``) hydrates every capped
    ``raw_content`` body regardless — up to ~1MB of text the page discards
    immediately at the supported 200-row page size. This is that path's
    metadata-only counterpart; ``list_articles_for_topic`` stays exclusively for
    exports and the analysis pipeline, which need the body text. Returned
    ``Article`` objects carry ``raw_content=None`` (the model's default) — never
    render it from this path.
    """
    query = (
        "SELECT id, topic_id, title, url, content_hash, source_feed, source_provider, "
        "published_at, fetched_at, processed, analysis_attempts "
        "FROM articles WHERE topic_id = ? ORDER BY fetched_at DESC"
    )
    params: list = [topic_id]
    if limit is not None:
        query += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])
    rows = conn.execute(query, params).fetchall()
    return [Article.from_row(row) for row in rows]


def count_articles_for_topic(conn: sqlite3.Connection, topic_id: int) -> int:
    """Count total articles for a topic."""
    row = conn.execute("SELECT COUNT(*) FROM articles WHERE topic_id = ?", (topic_id,)).fetchone()
    return int(row[0])


def article_hash_exists(conn: sqlite3.Connection, topic_id: int, content_hash: str) -> bool:
    """Check if an article with this hash already exists for the topic."""
    row = conn.execute(
        "SELECT 1 FROM articles WHERE topic_id = ? AND content_hash = ?",
        (topic_id, content_hash),
    ).fetchone()
    return row is not None


def list_article_dedup_keys(conn: sqlite3.Connection, topic_id: int) -> list[tuple[str, str, str]]:
    """Return ``(content_hash, url, title)`` for every article stored for a topic.

    Dedup compares an incoming entry against two keys — the exact representation
    and the story it belongs to — so it reads the topic's keys once per check
    instead of running a lookup per feed entry. The row count is bounded by the
    article retention window.
    """
    rows = conn.execute(
        "SELECT content_hash, url, title FROM articles WHERE topic_id = ?",
        (topic_id,),
    ).fetchall()
    return [(row["content_hash"], row["url"], row["title"]) for row in rows]


def list_article_bodies_for_urls(
    conn: sqlite3.Connection, topic_id: int, urls: list[str]
) -> list[tuple[str, str, str | None, str | None]]:
    """Return ``(url, title, raw_content, source_provider)`` for a topic's rows at these URLs.

    The bodies a topic already holds are what tells a genuine revision from a
    source that merely moved its ``updated`` stamp again, so dedup reads them for
    the handful of URLs actually under contest rather than for the whole
    retention window. ``source_provider`` comes along because a body a source
    handed over pre-extracted is only comparable with bodies from that same
    source.
    """
    unique = list(dict.fromkeys(urls))
    if not unique:
        return []
    placeholders = ",".join("?" * len(unique))
    rows = conn.execute(
        f"SELECT url, title, raw_content, source_provider FROM articles WHERE topic_id = ? AND url IN ({placeholders})",  # noqa: S608 - placeholders only, values are bound
        [topic_id, *unique],
    ).fetchall()
    return [(row["url"], row["title"], row["raw_content"], row["source_provider"]) for row in rows]


def topic_has_articles_from_feed(conn: sqlite3.Connection, topic_id: int, feed_url: str) -> bool:
    """True when this topic already stores at least one article from this feed.

    Conditional-GET validators live in ``feed_health``, keyed by feed URL, while
    articles belong to a topic. A topic subscribing to a feed another topic
    already polls therefore inherits validators for a representation it has never
    received: the server answers 304 and it stores nothing (TW-AUD-020). This is
    the question that makes sending them safe.
    """
    row = conn.execute(
        "SELECT 1 FROM articles WHERE topic_id = ? AND source_feed = ? LIMIT 1",
        (topic_id, feed_url),
    ).fetchone()
    return row is not None


def find_article_by_hash(conn: sqlite3.Connection, content_hash: str) -> Article | None:
    """Find any article with this content hash, across all topics.

    Returns the most recent matching article, or None if not found.
    Used for cross-topic deduplication to reuse fetched content.
    """
    row = conn.execute(
        "SELECT * FROM articles WHERE content_hash = ? ORDER BY fetched_at DESC LIMIT 1",
        (content_hash,),
    ).fetchone()
    return Article.from_row(row) if row else None


def mark_articles_processed(conn: sqlite3.Connection, article_ids: list[int]) -> None:
    """Mark multiple articles as processed."""
    if not article_ids:
        return
    placeholders = ",".join("?" * len(article_ids))
    conn.execute(
        f"UPDATE articles SET processed = 1 WHERE id IN ({placeholders})",
        article_ids,
    )


# How many analysis attempts an article gets before it is given up on. A cap, not
# a preference: without one a permanently undecodable article is re-fetched and
# re-sent to the model on every cycle forever, and with one set too low a
# transient provider outage discards real news. Three cycles is long enough to
# outlive a restart or a rate-limit window.
MAX_ANALYSIS_ATTEMPTS = 3


def record_article_analysis_failure(
    conn: sqlite3.Connection,
    article_ids: list[int],
    max_attempts: int = MAX_ANALYSIS_ATTEMPTS,
) -> int:
    """Count a failed analysis attempt against articles. Does NOT commit.

    Articles whose analysis failed are deliberately NOT marked processed: the
    check never evaluated them, so leaving the flag clear is what lets the next
    cycle re-select and re-analyze them (TW-AUD-001). The attempt counter is the
    bound on that retry — once an article has burned ``max_attempts`` cycles it is
    abandoned by marking it processed, so a single poisonous row cannot re-enter
    every future check's prompt.

    Returns the number of articles abandoned by this call, for the caller's log.
    """
    if not article_ids:
        return 0
    placeholders = ",".join("?" * len(article_ids))
    conn.execute(
        f"UPDATE articles SET analysis_attempts = analysis_attempts + 1 WHERE id IN ({placeholders})",
        article_ids,
    )
    cursor = conn.execute(
        f"UPDATE articles SET processed = 1 "  # noqa: S608 - placeholders only, values are bound
        f"WHERE id IN ({placeholders}) AND processed = 0 AND analysis_attempts >= ?",
        [*article_ids, max_attempts],
    )
    return cursor.rowcount


# Bare-column compare so SQLite can use idx_articles_fetched_at (m014). Wrapping
# fetched_at in datetime() would force a full table SCAN (OVH-022/050). The bound
# is a precomputed tz-aware isoformat() string, matching how fetched_at is stored
# (Article.to_insert_dict), so the lexicographic comparison is exact.
_DELETE_OLD_ARTICLES_SQL = "DELETE FROM articles WHERE fetched_at < ?"


def delete_old_articles(conn: sqlite3.Connection, retention_days: int) -> int:
    """Delete articles older than retention_days. Returns count of deleted rows."""
    cutoff = datetime.now(UTC) - timedelta(days=retention_days)
    cursor = conn.execute(_DELETE_OLD_ARTICLES_SQL, (cutoff.isoformat(),))
    return cursor.rowcount


# --- KnowledgeState CRUD ---


def create_knowledge_state(conn: sqlite3.Connection, state: KnowledgeState) -> KnowledgeState:
    """Insert or replace knowledge state for a topic.

    Uses INSERT OR REPLACE so re-initialization of READY topics works
    atomically without a separate delete step.
    """
    data = state.to_insert_dict()
    cursor = conn.execute(
        """INSERT OR REPLACE INTO knowledge_states (topic_id, summary_text, token_count, updated_at, version)
           VALUES (:topic_id, :summary_text, :token_count, :updated_at, :version)""",
        data,
    )
    state.id = cursor.lastrowid
    return state


def delete_knowledge_state(conn: sqlite3.Connection, topic_id: int) -> bool:
    """Delete knowledge state for a topic. Returns True if deleted."""
    cursor = conn.execute("DELETE FROM knowledge_states WHERE topic_id = ?", (topic_id,))
    return cursor.rowcount > 0


def get_knowledge_state(conn: sqlite3.Connection, topic_id: int) -> KnowledgeState | None:
    """Get the current knowledge state for a topic."""
    row = conn.execute("SELECT * FROM knowledge_states WHERE topic_id = ?", (topic_id,)).fetchone()
    return KnowledgeState.from_row(row) if row else None


def update_knowledge_state(conn: sqlite3.Connection, state: KnowledgeState) -> KnowledgeState:
    """Update an existing knowledge state."""
    if state.id is None:
        raise ValueError("Cannot update a knowledge state without an ID")
    data = state.to_insert_dict()
    data["id"] = state.id
    conn.execute(
        """UPDATE knowledge_states SET summary_text=:summary_text,
           token_count=:token_count, updated_at=:updated_at
           WHERE id=:id""",
        data,
    )
    return state


def update_knowledge_state_cas(
    conn: sqlite3.Connection,
    topic_id: int,
    *,
    summary_text: str,
    token_count: int,
    expected_version: int,
    updated_at: datetime | None = None,
) -> bool:
    """Compare-and-swap a topic's knowledge state. Returns False on conflict.

    The knowledge summary is produced by a multi-second LLM round-trip that must
    run with no connection open, so the state read at snapshot time and the state
    written afterwards are separated by an unbounded gap. Guarding the UPDATE on
    the version observed at snapshot time turns a lost update into a visible
    ``False``: the loser's summary — built from a now-stale base — is discarded
    instead of overwriting the winner's.

    ``expected_version`` of 0 also matches a state row that predates migration 026
    (the column's DEFAULT), so the first post-upgrade write is not spuriously
    rejected. Does NOT commit — the caller owns the transaction.
    """
    stamp = to_db_utc(updated_at or datetime.now(UTC))
    cursor = conn.execute(
        """UPDATE knowledge_states
              SET summary_text = ?, token_count = ?, updated_at = ?, version = version + 1
            WHERE topic_id = ? AND version = ?""",
        (summary_text, token_count, stamp, topic_id, expected_version),
    )
    return cursor.rowcount > 0


def topic_generation_matches(conn: sqlite3.Connection, topic_id: int, generation: str) -> bool:
    """True when the live topic row still carries the generation captured earlier.

    The fence for every durable write that follows a long await: ``topic_id`` is a
    recyclable rowid, so a delete+recreate can put a different topic behind the id
    a worker is holding. A blank stored generation (a row written before migration
    026 backfilled it) never matches, which fails closed.
    """
    row = conn.execute("SELECT generation FROM topics WHERE id = ?", (topic_id,)).fetchone()
    return bool(row) and bool(generation) and row["generation"] == generation


# --- KnowledgeRevision CRUD ---


def create_knowledge_revision(conn: sqlite3.Connection, revision: KnowledgeRevision) -> KnowledgeRevision:
    """Append a knowledge revision. Never updates or replaces an existing row."""
    data = revision.to_insert_dict()
    cursor = conn.execute(
        """INSERT INTO knowledge_revisions
               (topic_id, summary_text, token_count, source, change_note, created_at)
           VALUES (:topic_id, :summary_text, :token_count, :source, :change_note, :created_at)""",
        data,
    )
    revision.id = cursor.lastrowid
    return revision


def list_knowledge_revision_headers(
    conn: sqlite3.Connection,
    topic_id: int,
    limit: int,
) -> list[KnowledgeRevision]:
    """List a topic's revisions newest-first WITHOUT their summary text.

    The timeline renders four metadata fields per row and lazy-loads each diff,
    so shipping ``summary_text`` would read up to ~40 KB per revision for text
    no template shows. ``''`` is selected as a literal rather than dropping the
    column so the shared ``from_row`` coercion still applies; the returned models
    carry an empty ``summary_text`` and must never be written back.

    Ordered by ``id DESC`` rather than ``created_at DESC``: two revisions written
    in the same second must still order deterministically, and ``id`` is the
    insertion order the diff timeline depends on. The column list matches
    ``idx_knowledge_revisions_topic`` so this is served from the index alone.
    """
    rows = conn.execute(
        """SELECT id, topic_id, '' AS summary_text, token_count, source,
                  NULL AS change_note, created_at
           FROM knowledge_revisions
           WHERE topic_id = ?
           ORDER BY id DESC
           LIMIT ?""",
        (topic_id, limit),
    ).fetchall()
    return [KnowledgeRevision.from_row(row) for row in rows]


def get_knowledge_revision(conn: sqlite3.Connection, revision_id: int) -> KnowledgeRevision | None:
    """Get a single revision, summary text included, or None."""
    row = conn.execute("SELECT * FROM knowledge_revisions WHERE id = ?", (revision_id,)).fetchone()
    return KnowledgeRevision.from_row(row) if row else None


def get_previous_knowledge_revision(
    conn: sqlite3.Connection,
    topic_id: int,
    revision_id: int,
) -> KnowledgeRevision | None:
    """Get the revision immediately preceding ``revision_id`` for this topic.

    ``None`` when ``revision_id`` is the topic's oldest retained revision — the
    diff view then renders the full snapshot. Pruning always removes the oldest
    rows, so retained revisions stay contiguous and this is never a
    non-adjacent comparison.
    """
    row = conn.execute(
        "SELECT * FROM knowledge_revisions WHERE topic_id = ? AND id < ? ORDER BY id DESC LIMIT 1",
        (topic_id, revision_id),
    ).fetchone()
    return KnowledgeRevision.from_row(row) if row else None


def prune_knowledge_revisions(conn: sqlite3.Connection, topic_id: int, keep: int) -> int:
    """Delete all but the newest ``keep`` revisions for a topic. Returns rows deleted.

    Bounds history growth: a busy topic would otherwise accumulate one full
    knowledge summary per new-info check forever.
    """
    cursor = conn.execute(
        """DELETE FROM knowledge_revisions
           WHERE topic_id = :topic_id
             AND id NOT IN (
                 SELECT id FROM knowledge_revisions
                 WHERE topic_id = :topic_id
                 ORDER BY id DESC
                 LIMIT :keep
             )""",
        {"topic_id": topic_id, "keep": keep},
    )
    return cursor.rowcount


# --- CheckResult CRUD ---


def create_check_result(conn: sqlite3.Connection, result: CheckResult) -> CheckResult:
    """Record a check result for a topic."""
    data = result.to_insert_dict()
    # ``seen_at`` is intentionally omitted from this column list: new rows are born
    # NULL/unseen so a freshly-detected "new info" badge shows until the topic is
    # opened. The surplus ``seen_at`` key in ``data`` is ignored by sqlite3
    # named-parameter binding.
    cursor = conn.execute(
        """INSERT INTO check_results (topic_id, checked_at, articles_found,
           articles_new, has_new_info, llm_response, notification_sent,
           notification_error, prompt_tokens, completion_tokens, stage_error,
           notify_disposition)
           VALUES (:topic_id, :checked_at, :articles_found, :articles_new,
           :has_new_info, :llm_response, :notification_sent,
           :notification_error, :prompt_tokens, :completion_tokens, :stage_error,
           :notify_disposition)""",
        data,
    )
    result.id = cursor.lastrowid
    return result


def update_check_result_delivery(
    conn: sqlite3.Connection,
    check_result_id: int,
    *,
    notification_sent: bool,
    notification_error: str | None,
) -> None:
    """Record the post-send delivery outcome on an existing check_result row.

    The CheckResult is created and committed *before* the irreversible network
    sends (OVH-066/OVH-101); this updates only the delivery-outcome columns
    afterwards so the durable novelty state never depends on a send succeeding.
    The caller commits.
    """
    conn.execute(
        "UPDATE check_results SET notification_sent = ?, notification_error = ? WHERE id = ?",
        (int(notification_sent), notification_error, check_result_id),
    )


def mark_latest_check_seen(conn: sqlite3.Connection, topic_id: int) -> None:
    """Stamp ``seen_at`` on a topic's latest check when it carries unseen new info.

    Called when the topic detail page is opened, to clear the dashboard's "new
    info" badge (gated on ``has_new_info AND seen_at IS NULL``). The WHERE clause
    scopes the write to the single latest row and guards on ``has_new_info = 1 AND
    seen_at IS NULL`` so re-views are no-ops (the timestamp never churns) and older
    rows are never touched. Uses a Python UTC-ISO timestamp for parity with every
    other datetime column (SQLite ``datetime('now')`` would diverge). The caller
    commits.
    """
    conn.execute(
        """
        UPDATE check_results SET seen_at = ?
        WHERE id = (
            SELECT id FROM check_results
            WHERE topic_id = ?
            ORDER BY checked_at DESC LIMIT 1
        )
          AND has_new_info = 1
          AND seen_at IS NULL
        """,
        (datetime.now(UTC).isoformat(), topic_id),
    )


def mark_check_seen(conn: sqlite3.Connection, topic_id: int, check_id: int) -> None:
    """Stamp ``seen_at`` on one specific check result (TW-AUD-024).

    Companion to :func:`mark_latest_check_seen`, keyed to an explicit ``check_id``
    instead of "whichever row is latest right now". The detail GET route no
    longer mutates on read (a prefetch, retry, or failed render used to clear the
    dashboard badge before the user ever saw the detail); this is called instead,
    once the page has actually rendered, acknowledging the exact check it
    displayed. The extra ``id =`` subquery guard means a check that lands after
    render is never silently marked seen by a late-arriving ack for a stale page.
    Same ``has_new_info`` / ``seen_at IS NULL`` guards as the sibling function
    keep re-acks a no-op. The caller commits.
    """
    conn.execute(
        """
        UPDATE check_results SET seen_at = ?
        WHERE id = ?
          AND topic_id = ?
          AND has_new_info = 1
          AND seen_at IS NULL
          AND id = (
              SELECT id FROM check_results
              WHERE topic_id = ?
              ORDER BY checked_at DESC LIMIT 1
          )
        """,
        (datetime.now(UTC).isoformat(), check_id, topic_id, topic_id),
    )


def get_check_result(conn: sqlite3.Connection, check_id: int) -> CheckResult | None:
    """Get a check result by ID, or None if not found."""
    row = conn.execute("SELECT * FROM check_results WHERE id = ?", (check_id,)).fetchone()
    return CheckResult.from_row(row) if row else None


def list_check_results(
    conn: sqlite3.Connection,
    topic_id: int,
    limit: int = 20,
    offset: int = 0,
    cutoff_id: int | None = None,
) -> list[CheckResult]:
    """Get recent check results for a topic, newest first.

    ``cutoff_id``, when given, restricts the set to ids at or below it so a
    multi-page OFFSET traversal stays stable even if a newer check commits
    between page requests: mint it from ``max_check_result_id`` on the first
    page and carry it on later ones (AUG-314).
    """
    if cutoff_id is not None:
        rows = conn.execute(
            "SELECT * FROM check_results WHERE topic_id = ? AND id <= ? ORDER BY checked_at DESC LIMIT ? OFFSET ?",
            (topic_id, cutoff_id, limit, offset),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM check_results WHERE topic_id = ? ORDER BY checked_at DESC LIMIT ? OFFSET ?",
            (topic_id, limit, offset),
        ).fetchall()
    return [CheckResult.from_row(row) for row in rows]


def count_check_results(conn: sqlite3.Connection, topic_id: int, cutoff_id: int | None = None) -> int:
    """Count total check results for a topic, optionally capped at cutoff_id (AUG-314)."""
    if cutoff_id is not None:
        row = conn.execute(
            "SELECT COUNT(*) FROM check_results WHERE topic_id = ? AND id <= ?",
            (topic_id, cutoff_id),
        ).fetchone()
    else:
        row = conn.execute("SELECT COUNT(*) FROM check_results WHERE topic_id = ?", (topic_id,)).fetchone()
    return int(row[0])


def max_check_result_id(conn: sqlite3.Connection, topic_id: int) -> int | None:
    """Highest ``check_results.id`` for a topic, or None if it has none.

    Mints a stable pagination cutoff: the first page of a traversal captures
    this value and later pages pass it back, so a check committing mid-browse
    cannot shift rows underneath an OFFSET-based page 2+ (AUG-314).
    """
    row = conn.execute("SELECT MAX(id) FROM check_results WHERE topic_id = ?", (topic_id,)).fetchone()
    return int(row[0]) if row[0] is not None else None


def list_recent_check_stage_errors(
    conn: sqlite3.Connection,
    topic_id: int,
    limit: int,
) -> list[tuple[int, str | None]]:
    """Return ``(id, stage_error)`` for a topic's newest ``limit`` checks, newest first.

    Only the two columns are selected, so the Silence Heartbeat never ships the
    ``llm_response`` blobs that ``list_check_results`` carries, and ties on
    ``checked_at`` break by ``id`` so back-to-back checks order deterministically
    (AUG-258). The id travels with the streak because the heartbeat's latch write
    is fenced to the exact check its decision was computed from (AUG-131).
    """
    rows = conn.execute(
        """SELECT id, stage_error FROM check_results
           WHERE topic_id = ?
           ORDER BY checked_at DESC, id DESC
           LIMIT ?""",
        (topic_id, limit),
    ).fetchall()
    return [(int(row["id"]), row["stage_error"]) for row in rows]


def sum_check_tokens(conn: sqlite3.Connection, topic_id: int) -> tuple[int, int]:
    """Return (total_prompt_tokens, total_completion_tokens) across all checks."""
    row = conn.execute(
        """SELECT COALESCE(SUM(prompt_tokens), 0), COALESCE(SUM(completion_tokens), 0)
           FROM check_results WHERE topic_id = ?""",
        (topic_id,),
    ).fetchone()
    return int(row[0]), int(row[1])


# --- Delivery intents ---
#
# A delivery intent is one durable row per (message, target), created INSIDE the
# check's transaction before any send begins (TW-AUD-004). The lifecycle is
# 'pending' -> 'sending' -> 'sent' | 'abandoned', with 'revoked' for an event that
# stopped being true before its message left.
#
# Two properties hold for every intent helper below and must not be regressed
# (TW-AUD-006 / AUG-277):
#
# 1. Eligibility lives INSIDE the claim predicate, never in the list query alone.
#    A drainer working from a snapshot taken before another drainer exhausted a
#    row must lose the claim, not physically retry past ``max_retries``.
# 2. Every apply is fenced by the immutable ``claim_token`` the winning claim
#    stamped. A worker whose claim was released as stale — including one released
#    because the wall clock jumped — cannot mutate the row its successor now owns.
#    Liveness is judged monotonically by the caller; the fence is identity-based,
#    so it is correct across a clock jump in either direction.

# How long a terminal intent is kept as the delivery ledger before the daily
# maintenance tick prunes it. A constant, not a setting: it is the read window for
# the dashboard's "last notified", not a preference (AUG-153).
DELIVERY_INTENT_RETENTION_DAYS = 30

_NOTIFICATION_INTENT_INSERT = """INSERT INTO pending_notifications
    (topic_id, check_result_id, title, body, url, last_error, created_at,
     retry_count, max_retries, status, kind, next_attempt_at, latch_value)
    VALUES (:topic_id, :check_result_id, :title, :body, :url, :last_error,
     :created_at, :retry_count, :max_retries, :status, :kind, :next_attempt_at,
     :latch_value)"""

# Only rows that are pending, still have attempts left, and are due. All three
# conditions are part of the atomic claim, not a preceding SELECT.
_NOTIFICATION_DUE = (
    "status = 'pending' AND retry_count < max_retries AND (next_attempt_at IS NULL OR next_attempt_at <= ?)"
)


def create_notification_intents(conn: sqlite3.Connection, intents: list[PendingNotification]) -> list[int]:
    """Insert delivery intents and return their ids. NO commit.

    Called from inside the durable check transaction, so the intents land in the
    same commit as the CheckResult that justifies them: either the check happened
    and every target owes a delivery, or neither is true.
    """
    ids: list[int] = []
    for intent in intents:
        cursor = conn.execute(_NOTIFICATION_INTENT_INSERT, intent.to_insert_dict())
        intent.id = cursor.lastrowid
        ids.append(int(cursor.lastrowid or 0))
    return ids


def create_pending_notification(conn: sqlite3.Connection, notification: PendingNotification) -> PendingNotification:
    """Insert a single delivery intent, returning it with its assigned id."""
    create_notification_intents(conn, [notification])
    return notification


def list_pending_notifications(
    conn: sqlite3.Connection,
) -> list[PendingNotification]:
    """Every intent still awaiting delivery, oldest first.

    Rows a drainer is currently sending ('sending') are excluded, as are the
    terminal states — this is the queue, not the ledger.
    """
    rows = conn.execute(
        "SELECT * FROM pending_notifications "
        "WHERE status = 'pending' AND retry_count < max_retries "
        "ORDER BY created_at ASC, id ASC"
    ).fetchall()
    return [PendingNotification.from_row(row) for row in rows]


def list_due_notification_intents(
    conn: sqlite3.Connection,
    now_iso: str,
    limit: int,
) -> list[PendingNotification]:
    """Intents eligible to be sent right now, oldest first, at most ``limit``.

    The limit is what keeps a backlog of queued messages from consuming a whole
    scheduler cycle before any due topic is checked (AUG-027). The rows are only a
    candidate list: eligibility is re-tested atomically by the claim.
    """
    rows = conn.execute(
        f"SELECT * FROM pending_notifications WHERE {_NOTIFICATION_DUE} ORDER BY created_at ASC, id ASC LIMIT ?",
        (now_iso, limit),
    ).fetchall()
    return [PendingNotification.from_row(row) for row in rows]


def claim_notification_intent(
    conn: sqlite3.Connection,
    intent_id: int,
    claim_token: str,
    now_iso: str,
) -> bool:
    """Atomically take ownership of one intent. True only for the winner.

    Rejects, in one statement, every row this caller must not send: already
    claimed or terminal, out of attempts, or not yet due.
    """
    cursor = conn.execute(
        "UPDATE pending_notifications SET status = 'sending', claimed_at = ?, claim_token = ? "
        f"WHERE id = ? AND {_NOTIFICATION_DUE}",
        (now_iso, claim_token, intent_id, now_iso),
    )
    return cursor.rowcount == 1


def apply_notification_outcome(
    conn: sqlite3.Connection,
    intent_id: int,
    claim_token: str,
    *,
    sent: bool,
    error: str | None = None,
    next_attempt_at: str | None = None,
    delivered_at: str | None = None,
    terminal: bool = False,
) -> bool:
    """Record the outcome of one send, fenced by the winning claim token.

    ``sent`` retains the row as the delivery ledger with a ``delivered_at``
    stamp. A failure re-arms the row at ``next_attempt_at`` unless it was the last
    attempt or ``terminal`` says the target will never accept this payload — both
    of which land on 'abandoned' rather than burning three identical retries.

    Returns False when the fence rejected the write, which is the whole point: a
    late apply from a superseded owner must be a no-op, not a silent mutation of
    someone else's row (AUG-277).
    """
    if sent:
        cursor = conn.execute(
            "UPDATE pending_notifications SET status = 'sent', delivered_at = ?, last_error = NULL, "
            "claimed_at = NULL, claim_token = NULL "
            "WHERE id = ? AND claim_token = ? AND status = 'sending'",
            (delivered_at or to_db_utc(datetime.now(UTC)), intent_id, claim_token),
        )
        return cursor.rowcount == 1

    if terminal:
        cursor = conn.execute(
            "UPDATE pending_notifications SET status = 'abandoned', retry_count = retry_count + 1, "
            "last_error = ?, claimed_at = NULL, claim_token = NULL "
            "WHERE id = ? AND claim_token = ? AND status = 'sending'",
            (error, intent_id, claim_token),
        )
        return cursor.rowcount == 1

    cursor = conn.execute(
        "UPDATE pending_notifications SET retry_count = retry_count + 1, last_error = ?, "
        "next_attempt_at = ?, claimed_at = NULL, claim_token = NULL, "
        "status = CASE WHEN retry_count + 1 >= max_retries THEN 'abandoned' ELSE 'pending' END "
        "WHERE id = ? AND claim_token = ? AND status = 'sending'",
        (error, next_attempt_at, intent_id, claim_token),
    )
    return cursor.rowcount == 1


# Both heartbeat message kinds: what a latch transition (or the feature being
# switched off) supersedes.
HEARTBEAT_INTENT_KINDS: tuple[str, ...] = (
    NotificationKind.HEARTBEAT_ALERT.value,
    NotificationKind.HEARTBEAT_RECOVERY.value,
)


def list_sent_heartbeat_alert_targets(conn: sqlite3.Connection, topic_id: int, latch_value: str | None) -> set[str]:
    """Targets that actually received the outage alert stamped ``latch_value``.

    "Sources recovered" is only meaningful to somebody who was told they were
    failing: a target whose alert failed, was revoked, or is still queued would
    otherwise get an unexplained all-clear for an outage it never heard about
    (AUG-019). The latch value is the outage's identity, so an alert from an
    earlier outage never addresses this recovery.
    """
    if latch_value is None:
        return set()
    rows = conn.execute(
        "SELECT DISTINCT url FROM pending_notifications "
        "WHERE topic_id = ? AND kind = ? AND latch_value = ? AND status = 'sent' AND url IS NOT NULL",
        (topic_id, NotificationKind.HEARTBEAT_ALERT.value, latch_value),
    ).fetchall()
    return {row["url"] for row in rows}


def reset_all_heartbeat_state(conn: sqlite3.Connection) -> int:
    """Clear every latch and revoke every queued heartbeat message. NO commit.

    ``silence_heartbeat_checks = 0`` is the off switch, but the per-check reset
    only reaches a topic when that topic next runs. Disabling and re-enabling
    inside one long interval would otherwise preserve the old latch — which then
    either suppresses the newly enabled outage alert or fires a recovery for an
    outage nobody was told about (AUG-260). Returns the number of latches cleared.
    """
    cursor = conn.execute("UPDATE topics SET heartbeat_alerted_at = NULL WHERE heartbeat_alerted_at IS NOT NULL")
    cleared = cursor.rowcount
    placeholders = ",".join("?" for _ in HEARTBEAT_INTENT_KINDS)
    conn.execute(
        "UPDATE pending_notifications SET status = 'revoked', claimed_at = NULL, claim_token = NULL "
        f"WHERE status = 'pending' AND kind IN ({placeholders})",
        HEARTBEAT_INTENT_KINDS,
    )
    return cleared


def revoke_heartbeat_intents(conn: sqlite3.Connection, topic_id: int, kinds: tuple[str, ...]) -> int:
    """Revoke a topic's undelivered heartbeat intents of the given kinds.

    A queued outage alert that has not left yet must not be delivered after the
    recovery that contradicts it, and neither survives the feature being switched
    off (AUG-019/AUG-132). Only 'pending' rows are revoked: one already claimed
    may be mid-flight, and pretending otherwise would be a lie in the ledger.
    """
    if not kinds:
        return 0
    placeholders = ",".join("?" for _ in kinds)
    cursor = conn.execute(
        "UPDATE pending_notifications SET status = 'revoked', claimed_at = NULL, claim_token = NULL "
        f"WHERE topic_id = ? AND status = 'pending' AND kind IN ({placeholders})",
        (topic_id, *kinds),
    )
    return cursor.rowcount


def release_stale_notification_claims(conn: sqlite3.Connection, cutoff: str) -> int:
    """Re-arm intents claimed at or before ``cutoff`` (ISO string).

    A drainer that claims a row then dies — or whose send timed out with an
    unknown outcome — would otherwise leave it 'sending' forever. Re-arming makes
    the queue self-healing; the claim token makes the original owner's late apply
    harmless if it ever comes back.
    """
    cursor = conn.execute(
        "UPDATE pending_notifications SET status = 'pending', claimed_at = NULL, claim_token = NULL "
        "WHERE status = 'sending' AND claimed_at IS NOT NULL AND claimed_at <= ?",
        (cutoff,),
    )
    return cursor.rowcount


def abandon_expired_notifications(conn: sqlite3.Connection) -> list[PendingNotification]:
    """Mark out-of-attempts intents 'abandoned', returning what was abandoned.

    Rows normally reach 'abandoned' at apply time; this sweeps the ones that
    cannot — a row that was already at its retry ceiling before intents existed
    would otherwise sit 'pending' forever, invisible to both the drain (the claim
    refuses it) and retention (which only prunes terminal rows). Returning the
    rows lets the caller log exactly what was dropped (OVH-040).
    """
    rows = conn.execute(
        "SELECT * FROM pending_notifications WHERE status = 'pending' AND retry_count >= max_retries"
    ).fetchall()
    abandoned = [PendingNotification.from_row(row) for row in rows]
    conn.execute(
        "UPDATE pending_notifications SET status = 'abandoned' WHERE status = 'pending' AND retry_count >= max_retries"
    )
    return abandoned


def delete_old_delivery_intents(conn: sqlite3.Connection, days: int) -> int:
    """Prune terminal delivery intents older than ``days``. Returns rows removed.

    Only terminal rows are eligible: anything still 'pending' or 'sending' owes a
    delivery no matter how old it is.
    """
    cutoff = to_db_utc(datetime.now(UTC) - timedelta(days=days))
    removed = conn.execute(
        "DELETE FROM pending_notifications WHERE status IN ('sent', 'abandoned', 'revoked') AND created_at < ?",
        (cutoff,),
    ).rowcount
    removed += conn.execute(
        "DELETE FROM pending_webhooks WHERE status IN ('sent', 'abandoned', 'revoked') AND created_at < ?",
        (cutoff,),
    ).rowcount
    return removed


# --- Webhook delivery intents ---
#
# An exact mirror of the notification intent helpers above, over pending_webhooks.
# Same two invariants: eligibility inside the claim, apply fenced by claim_token.

_WEBHOOK_INTENT_INSERT = """INSERT INTO pending_webhooks
    (topic_id, check_result_id, url, payload, created_at, retry_count,
     max_retries, status, next_attempt_at, last_error)
    VALUES (:topic_id, :check_result_id, :url, :payload, :created_at,
     :retry_count, :max_retries, :status, :next_attempt_at, :last_error)"""

_WEBHOOK_DUE = "status = 'pending' AND retry_count < max_retries AND (next_attempt_at IS NULL OR next_attempt_at <= ?)"


def create_webhook_intents(conn: sqlite3.Connection, intents: list[PendingWebhook]) -> list[int]:
    """Insert webhook delivery intents and return their ids. NO commit."""
    ids: list[int] = []
    for intent in intents:
        cursor = conn.execute(_WEBHOOK_INTENT_INSERT, intent.to_insert_dict())
        intent.id = cursor.lastrowid
        ids.append(int(cursor.lastrowid or 0))
    return ids


def create_pending_webhook(
    conn: sqlite3.Connection,
    topic_id: int,
    url: str,
    payload: dict,
    check_result_id: int | None = None,
    max_retries: int = 3,
) -> int:
    """Insert a single webhook delivery intent. Returns the new row id."""
    intent = PendingWebhook(
        topic_id=topic_id,
        check_result_id=check_result_id,
        url=url,
        payload=payload,
        max_retries=max_retries,
    )
    return create_webhook_intents(conn, [intent])[0]


def list_pending_webhooks(conn: sqlite3.Connection) -> list[PendingWebhook]:
    """Every webhook intent still awaiting delivery, oldest first."""
    rows = conn.execute(
        "SELECT * FROM pending_webhooks "
        "WHERE status = 'pending' AND retry_count < max_retries "
        "ORDER BY created_at ASC, id ASC"
    ).fetchall()
    return [PendingWebhook.from_row(row) for row in rows]


def list_due_webhook_intents(
    conn: sqlite3.Connection,
    now_iso: str,
    limit: int,
) -> list[PendingWebhook]:
    """Webhook intents eligible to be sent right now, oldest first."""
    rows = conn.execute(
        f"SELECT * FROM pending_webhooks WHERE {_WEBHOOK_DUE} ORDER BY created_at ASC, id ASC LIMIT ?",
        (now_iso, limit),
    ).fetchall()
    return [PendingWebhook.from_row(row) for row in rows]


def claim_webhook_intent(
    conn: sqlite3.Connection,
    intent_id: int,
    claim_token: str,
    now_iso: str,
) -> bool:
    """Atomically take ownership of one webhook intent. True only for the winner."""
    cursor = conn.execute(
        "UPDATE pending_webhooks SET status = 'sending', claimed_at = ?, claim_token = ? "
        f"WHERE id = ? AND {_WEBHOOK_DUE}",
        (now_iso, claim_token, intent_id, now_iso),
    )
    return cursor.rowcount == 1


def apply_webhook_outcome(
    conn: sqlite3.Connection,
    intent_id: int,
    claim_token: str,
    *,
    sent: bool,
    error: str | None = None,
    next_attempt_at: str | None = None,
    delivered_at: str | None = None,
    terminal: bool = False,
) -> bool:
    """Record one webhook send's outcome, fenced by the winning claim token.

    ``terminal`` is what makes a permanent rejection cheap: a 400 or a 422 will
    still be a 400 on the third identical POST, so the intent is abandoned on the
    spot instead of consuming the whole retry budget (AUG-324).
    """
    if sent:
        cursor = conn.execute(
            "UPDATE pending_webhooks SET status = 'sent', delivered_at = ?, last_error = NULL, "
            "claimed_at = NULL, claim_token = NULL "
            "WHERE id = ? AND claim_token = ? AND status = 'sending'",
            (delivered_at or to_db_utc(datetime.now(UTC)), intent_id, claim_token),
        )
        return cursor.rowcount == 1

    if terminal:
        cursor = conn.execute(
            "UPDATE pending_webhooks SET status = 'abandoned', retry_count = retry_count + 1, "
            "last_error = ?, claimed_at = NULL, claim_token = NULL "
            "WHERE id = ? AND claim_token = ? AND status = 'sending'",
            (error, intent_id, claim_token),
        )
        return cursor.rowcount == 1

    cursor = conn.execute(
        "UPDATE pending_webhooks SET retry_count = retry_count + 1, last_error = ?, "
        "next_attempt_at = ?, claimed_at = NULL, claim_token = NULL, "
        "status = CASE WHEN retry_count + 1 >= max_retries THEN 'abandoned' ELSE 'pending' END "
        "WHERE id = ? AND claim_token = ? AND status = 'sending'",
        (error, next_attempt_at, intent_id, claim_token),
    )
    return cursor.rowcount == 1


def release_stale_webhook_claims(conn: sqlite3.Connection, cutoff: str) -> int:
    """Re-arm webhook intents claimed at or before ``cutoff`` (ISO string)."""
    cursor = conn.execute(
        "UPDATE pending_webhooks SET status = 'pending', claimed_at = NULL, claim_token = NULL "
        "WHERE status = 'sending' AND claimed_at IS NOT NULL AND claimed_at <= ?",
        (cutoff,),
    )
    return cursor.rowcount


def abandon_expired_webhooks(conn: sqlite3.Connection) -> list[PendingWebhook]:
    """Mark out-of-attempts webhook intents 'abandoned', returning what was abandoned.

    The URL is redacted by the caller before it reaches a log.
    """
    rows = conn.execute(
        "SELECT * FROM pending_webhooks WHERE status = 'pending' AND retry_count >= max_retries"
    ).fetchall()
    abandoned = [PendingWebhook.from_row(row) for row in rows]
    conn.execute(
        "UPDATE pending_webhooks SET status = 'abandoned' WHERE status = 'pending' AND retry_count >= max_retries"
    )
    return abandoned


# --- FeedHealth CRUD ---


def upsert_feed_health_success(
    conn: sqlite3.Connection,
    feed_url: str,
    etag: str | None = None,
    last_modified: str | None = None,
    *,
    replace_validators: bool = False,
) -> None:
    """Record a successful feed fetch.

    ``etag`` / ``last_modified`` are the response's conditional-GET validators.
    ``replace_validators`` says whether this response is entitled to speak for
    them: a 200 body is the feed's current statement of which validators it
    issues, so an absent header must CLEAR the stored value rather than leave an
    obsolete one to be sent forever (AUG-152). A 304 says only "unchanged", so it
    preserves what is stored and merely refreshes a validator it does supply.
    """
    now = datetime.now(UTC).isoformat()
    if replace_validators:
        validator_update = "etag = ?, last_modified = ?"
    else:
        validator_update = "etag = COALESCE(?, etag), last_modified = COALESCE(?, last_modified)"
    conn.execute(
        f"""INSERT INTO feed_health
               (feed_url, last_success_at, consecutive_failures, total_fetches, etag, last_modified)
           VALUES (?, ?, 0, 1, ?, ?)
           ON CONFLICT(feed_url) DO UPDATE SET
               last_success_at = ?,
               consecutive_failures = 0,
               total_fetches = total_fetches + 1,
               {validator_update}""",  # noqa: S608 - fixed SQL fragment, no interpolated values
        (feed_url, now, etag, last_modified, now, etag, last_modified),
    )


def upsert_feed_health_failure(conn: sqlite3.Connection, feed_url: str, error_msg: str) -> None:
    """Record a failed feed fetch."""
    now = datetime.now(UTC).isoformat()
    conn.execute(
        """INSERT INTO feed_health (feed_url, last_error_at, last_error_message,
               consecutive_failures, total_fetches, total_failures)
           VALUES (?, ?, ?, 1, 1, 1)
           ON CONFLICT(feed_url) DO UPDATE SET
               last_error_at = ?,
               last_error_message = ?,
               consecutive_failures = consecutive_failures + 1,
               total_fetches = total_fetches + 1,
               total_failures = total_failures + 1""",
        (feed_url, now, error_msg, now, error_msg),
    )


def upsert_feed_health_aborted(conn: sqlite3.Connection, feed_url: str, error_msg: str) -> None:
    """Record a fetch this process abandoned, without blaming the feed.

    The topic's whole source budget running out is usually some *other* feed being
    slow, so charging it to this feed's consecutive-failure count would put a
    healthy feed into exponential backoff for something it did not do. The attempt
    and its reason are still recorded, because the Feed Health page has to be able
    to explain why a feed shows no recent success.
    """
    now = datetime.now(UTC).isoformat()
    conn.execute(
        """INSERT INTO feed_health (feed_url, last_error_at, last_error_message,
               consecutive_failures, total_fetches, total_failures)
           VALUES (?, ?, ?, 0, 1, 0)
           ON CONFLICT(feed_url) DO UPDATE SET
               last_error_at = ?,
               last_error_message = ?,
               total_fetches = total_fetches + 1""",
        (feed_url, now, error_msg, now, error_msg),
    )


def get_feed_health(conn: sqlite3.Connection, feed_url: str) -> FeedHealth | None:
    """Get health info for a specific feed URL."""
    row = conn.execute("SELECT * FROM feed_health WHERE feed_url = ?", (feed_url,)).fetchone()
    return FeedHealth.from_row(row) if row else None


def list_all_feed_health(conn: sqlite3.Connection) -> list[FeedHealth]:
    """List health info for all tracked feeds."""
    rows = conn.execute("SELECT * FROM feed_health ORDER BY consecutive_failures DESC, feed_url").fetchall()
    return [FeedHealth.from_row(row) for row in rows]


# --- NEW topic CRUD ---


def get_new_topics(conn: sqlite3.Connection, limit: int = 1) -> list[Topic]:
    """Get topics in NEW status, oldest first (for gradual scheduler init).

    Paused topics are excluded: automatic initialization fetches sources and
    spends LLM/Exa credit, which is exactly what disabling a topic is supposed to
    stop (AUG-140). The claim below repeats the filter to close the window between
    this SELECT and the claim.
    """
    rows = conn.execute(
        "SELECT * FROM topics WHERE status = ? AND is_active = 1 ORDER BY created_at ASC LIMIT ?",
        (TopicStatus.NEW.value, limit),
    ).fetchall()
    return [Topic.from_row(row) for row in rows]


def claim_topic_for_init(conn: sqlite3.Connection, topic_id: int, expected_status: TopicStatus) -> bool:
    """Atomically claim a topic for initialization (expected_status -> RESEARCHING).

    Returns True only if this caller won the claim (rowcount == 1). Every
    initializer — scheduler tick, web Retry, CLI ``init`` — goes through this one
    conditional UPDATE, so "read the status, then write RESEARCHING" can no longer
    let two of them both proceed (AUG-288). ``is_active = 1`` is part of the
    predicate: a topic paused between the caller's read and this claim is not
    initialized (AUG-140).

    Commits so the claim is durable and immediately visible to concurrent WAL
    connections — a claim only another process can see is not a claim.
    """
    cursor = conn.execute(
        """UPDATE topics SET status = ?, status_changed_at = ?, error_message = ?
           WHERE id = ? AND status = ? AND is_active = 1""",
        (
            TopicStatus.RESEARCHING.value,
            to_db_utc(datetime.now(UTC)),
            None,
            topic_id,
            expected_status.value,
        ),
    )
    conn.commit()
    return cursor.rowcount == 1


def claim_new_topic_for_init(conn: sqlite3.Connection, topic_id: int) -> bool:
    """Atomically claim a NEW topic for initialization (NEW -> RESEARCHING).

    The scheduler's gradual-init spelling of :func:`claim_topic_for_init`
    (OVH-032).
    """
    return claim_topic_for_init(conn, topic_id, TopicStatus.NEW)


# Fences a heartbeat latch transition to the exact topic incarnation and the exact
# check the decision was computed from. ``topics.id`` is a recyclable rowid, so a
# check still holding a deleted topic could otherwise latch its replacement — an
# alert naming the deleted topic, and a replacement whose real outage is then
# suppressed (AUG-020/TW-AUD-007). The head-check conjunct is the same idea in time:
# a decision computed from check N must not land once check N+1 exists. It selects
# the head with the canonical ``checked_at DESC, id DESC`` the heartbeat's own
# streak query uses (AUG-258), not ``MAX(id)`` — a row carrying a newer id with an
# older timestamp (a clock step, a restored row) is not the latest check, and
# treating it as one would refuse every later transition for that topic forever.
_HEARTBEAT_GENERATION_FENCE = " AND generation = ?"
_HEARTBEAT_HEAD_FENCE = (
    " AND (SELECT id FROM check_results WHERE topic_id = topics.id ORDER BY checked_at DESC, id DESC LIMIT 1) = ?"
)


def _heartbeat_fence(sql: str, params: list, generation: str | None, head_check_id: int | None) -> tuple[str, list]:
    """Append the optional generation / head-check conjuncts to a latch UPDATE."""
    if generation is not None:
        sql += _HEARTBEAT_GENERATION_FENCE
        params.append(generation)
    if head_check_id is not None:
        sql += _HEARTBEAT_HEAD_FENCE
        params.append(head_check_id)
    return sql, params


def claim_heartbeat_alert(
    conn: sqlite3.Connection,
    topic_id: int,
    alerted_at: datetime,
    *,
    generation: str | None = None,
    head_check_id: int | None = None,
) -> bool:
    """Atomically claim the right to announce a source outage for a topic.

    Returns True only for the caller that won (rowcount == 1). The guard is
    ``heartbeat_alerted_at IS NULL``, so an outage already announced — by this
    process, or by a CLI ``check-all`` running against a live server — yields
    False and no second alert. Mirrors ``claim_new_topic_for_init``; the caller
    commits. ``generation`` and ``head_check_id`` add the lifecycle fences
    described above.
    """
    sql = "UPDATE topics SET heartbeat_alerted_at = ? WHERE id = ? AND heartbeat_alerted_at IS NULL"
    params: list = [to_db_utc(alerted_at), topic_id]
    sql, params = _heartbeat_fence(sql, params, generation, head_check_id)
    cursor = conn.execute(sql, params)
    return cursor.rowcount == 1


def get_heartbeat_latch_raw(conn: sqlite3.Connection, topic_id: int) -> str | None:
    """Return the stored latch cell verbatim: NULL is None, anything else is text.

    The model hydrates ``heartbeat_alerted_at`` through the permissive optional
    datetime coercer, which turns corrupt or forward-incompatible text into
    ``None`` — while the latch SQL compares against the raw cell with ``IS NULL``.
    A decision made on the hydrated value therefore claims a latch that is already
    set and never clears the bad one, wedging the topic's heartbeat until someone
    edits the database by hand (AUG-144). The heartbeat reads the cell itself, so
    "set" means exactly what the UPDATE guards mean by it.
    """
    row = conn.execute("SELECT heartbeat_alerted_at FROM topics WHERE id = ?", (topic_id,)).fetchone()
    if row is None or row["heartbeat_alerted_at"] is None:
        return None
    return str(row["heartbeat_alerted_at"])


def clear_heartbeat_alert(
    conn: sqlite3.Connection,
    topic_id: int,
    *,
    generation: str | None = None,
    head_check_id: int | None = None,
) -> bool:
    """Clear an outstanding source-outage alert. True only if one was set.

    The ``IS NOT NULL`` guard makes the recovery notice exactly-once for the same
    reason the claim makes the alert exactly-once. The caller commits. Fenced by
    the same optional lifecycle conjuncts as :func:`claim_heartbeat_alert`.
    """
    sql = "UPDATE topics SET heartbeat_alerted_at = NULL WHERE id = ? AND heartbeat_alerted_at IS NOT NULL"
    params: list = [topic_id]
    sql, params = _heartbeat_fence(sql, params, generation, head_check_id)
    cursor = conn.execute(sql, params)
    return cursor.rowcount == 1


def get_all_feed_urls(conn: sqlite3.Connection) -> set[str]:
    """Get all feed URLs across all topics for OPML dedup."""
    rows = conn.execute("SELECT DISTINCT json_each.value FROM topics, json_each(topics.feed_urls)").fetchall()
    return {row[0] for row in rows}


def get_all_topic_names(conn: sqlite3.Connection) -> set[str]:
    """Get all topic names for OPML name-collision dedup."""
    rows = conn.execute("SELECT name FROM topics").fetchall()
    return {row[0] for row in rows}


# --- Dashboard Stats ---


def get_dashboard_stats(conn: sqlite3.Connection) -> DashboardStats:
    """Get aggregate statistics for the dashboard."""
    row = conn.execute(
        """
        SELECT
            (SELECT COUNT(*) FROM topics) AS total_topics,
            (SELECT COUNT(*) FROM topics WHERE is_active = 1) AS active_topics,
            (SELECT COUNT(*) FROM check_results
             WHERE datetime(checked_at) >= datetime('now', '-1 day')) AS checks_24h,
            (SELECT COUNT(*) FROM check_results) AS checks_total,
            (SELECT COUNT(*) FROM check_results
             WHERE has_new_info = 1 AND datetime(checked_at) >= datetime('now', '-1 day')) AS new_info_24h,
            (SELECT COUNT(*) FROM check_results WHERE has_new_info = 1) AS new_info_total,
            -- "Last notified" means the most recent alert of ANY kind that actually
            -- left. The delivery ledger is the truthful source: heartbeat outage and
            -- recovery messages create no check_results row at all, so deriving this
            -- from notification_sent alone showed an older timestamp (or 'never')
            -- right after a heartbeat alert (AUG-153). The check_results half is kept
            -- so history predating the ledger still reads correctly.
            (SELECT MAX(stamp) FROM (
                SELECT MAX(checked_at) AS stamp FROM check_results WHERE notification_sent = 1
                UNION ALL
                SELECT MAX(delivered_at) FROM pending_notifications WHERE status = 'sent'
            )) AS last_notification_at
        """
    ).fetchone()
    assert row is not None
    last_notif = None
    if row["last_notification_at"]:
        import contextlib

        with contextlib.suppress(ValueError, TypeError):
            # Aware UTC like every other hydrated timestamp, so the template's
            # relative-time arithmetic cannot meet a naive value (TW-AUD-013).
            last_notif = to_utc(datetime.fromisoformat(row["last_notification_at"]))
    return DashboardStats(
        total_topics=row["total_topics"],
        active_topics=row["active_topics"],
        checks_24h=row["checks_24h"],
        checks_total=row["checks_total"],
        new_info_24h=row["new_info_24h"],
        new_info_total=row["new_info_total"],
        last_notification_at=last_notif,
    )
