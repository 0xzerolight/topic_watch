"""CRUD operations for all data models.

All functions accept a sqlite3.Connection as their first parameter
for explicit dependency injection and testability.
"""

import logging
import sqlite3
from datetime import UTC, datetime, timedelta

from app.models import (
    Article,
    CheckResult,
    DashboardStats,
    FeedHealth,
    KnowledgeState,
    PendingNotification,
    PendingWebhook,
    Topic,
    TopicStatus,
)

logger = logging.getLogger(__name__)


# --- Topic CRUD ---


def create_topic(conn: sqlite3.Connection, topic: Topic) -> Topic:
    """Insert a new topic and return it with the generated ID."""
    data = topic.to_insert_dict()
    cursor = conn.execute(
        """INSERT INTO topics (name, description, feed_urls, feed_mode,
           created_at, status_changed_at, is_active, status, error_message, check_interval_minutes, tags,
           confidence_threshold, relevance_threshold, novelty_instruction, init_attempts)
           VALUES (:name, :description, :feed_urls, :feed_mode,
           :created_at, :status_changed_at, :is_active, :status, :error_message, :check_interval_minutes, :tags,
           :confidence_threshold, :relevance_threshold, :novelty_instruction, :init_attempts)""",
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
           init_attempts=:init_attempts
           WHERE id=:id""",
        data,
    )
    logger.info("Updated topic: %s (id=%d)", topic.name, topic.id)
    return topic


def update_topic_init_status(
    conn: sqlite3.Connection,
    topic_id: int,
    *,
    status: TopicStatus,
    status_changed_at: datetime,
    error_message: str | None,
    init_attempts: int,
) -> None:
    """Targeted UPDATE of only the init-lifecycle columns a topic init owns.

    Unlike ``update_topic`` (which rewrites the whole row from a possibly-stale
    in-memory ``Topic``), this writes only status/error/init_attempts so a
    concurrent UI edit to feeds/thresholds during the long init await is not
    clobbered (OVH-100). Does not commit; the caller owns the transaction.
    """
    conn.execute(
        """UPDATE topics
           SET status = ?, status_changed_at = ?, error_message = ?, init_attempts = ?
           WHERE id = ?""",
        (status.value, status_changed_at.isoformat(), error_message, init_attempts, topic_id),
    )


def recover_stuck_topics(conn: sqlite3.Connection) -> int:
    """Mark all RESEARCHING topics as ERROR.

    Called at startup — any topic still in RESEARCHING status when the
    server starts is definitively stuck (the background task is dead).
    """
    cursor = conn.execute(
        "UPDATE topics SET status = ?, error_message = ? WHERE status = ?",
        (
            TopicStatus.ERROR.value,
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
    """
    cursor = conn.execute(
        """UPDATE topics SET status = ?, error_message = ?
           WHERE status = ?
             AND status_changed_at IS NOT NULL
             AND datetime(status_changed_at, '+' || ? || ' minutes') <= datetime('now')""",
        (
            TopicStatus.ERROR.value,
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


def search_dashboard_data(
    conn: sqlite3.Connection,
    query: str | None = None,
    status: str | None = None,
) -> list[dict]:
    """Get topics with last check and article count, with optional name/status filters."""
    where_clauses = []
    params: list = []

    if query:
        where_clauses.append("t.name LIKE ?")
        params.append(f"%{query}%")

    if status:
        where_clauses.append("t.status = ?")
        params.append(status)

    where_sql = ""
    if where_clauses:
        where_sql = " WHERE " + " AND ".join(where_clauses)

    return _query_dashboard_rows(conn, where_sql, params)


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
        """INSERT OR REPLACE INTO knowledge_states (topic_id, summary_text, token_count, updated_at)
           VALUES (:topic_id, :summary_text, :token_count, :updated_at)""",
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
           notification_error, prompt_tokens, completion_tokens, stage_error)
           VALUES (:topic_id, :checked_at, :articles_found, :articles_new,
           :has_new_info, :llm_response, :notification_sent,
           :notification_error, :prompt_tokens, :completion_tokens, :stage_error)""",
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


def get_check_result(conn: sqlite3.Connection, check_id: int) -> CheckResult | None:
    """Get a check result by ID, or None if not found."""
    row = conn.execute("SELECT * FROM check_results WHERE id = ?", (check_id,)).fetchone()
    return CheckResult.from_row(row) if row else None


def list_check_results(
    conn: sqlite3.Connection,
    topic_id: int,
    limit: int = 20,
    offset: int = 0,
) -> list[CheckResult]:
    """Get recent check results for a topic, newest first."""
    rows = conn.execute(
        "SELECT * FROM check_results WHERE topic_id = ? ORDER BY checked_at DESC LIMIT ? OFFSET ?",
        (topic_id, limit, offset),
    ).fetchall()
    return [CheckResult.from_row(row) for row in rows]


def count_check_results(conn: sqlite3.Connection, topic_id: int) -> int:
    """Count total check results for a topic."""
    row = conn.execute("SELECT COUNT(*) FROM check_results WHERE topic_id = ?", (topic_id,)).fetchone()
    return int(row[0])


def sum_check_tokens(conn: sqlite3.Connection, topic_id: int) -> tuple[int, int]:
    """Return (total_prompt_tokens, total_completion_tokens) across all checks."""
    row = conn.execute(
        """SELECT COALESCE(SUM(prompt_tokens), 0), COALESCE(SUM(completion_tokens), 0)
           FROM check_results WHERE topic_id = ?""",
        (topic_id,),
    ).fetchone()
    return int(row[0]), int(row[1])


# --- PendingNotification CRUD ---


def create_pending_notification(conn: sqlite3.Connection, notification: PendingNotification) -> PendingNotification:
    """Store a failed notification for later retry.

    ``url`` scopes the row to a single failed target so retry never re-hits the
    targets that already delivered (OVH-039); ``last_error`` records why it
    failed for operator diagnostics.
    """
    data = notification.to_insert_dict()
    cursor = conn.execute(
        """INSERT INTO pending_notifications (topic_id, check_result_id,
           title, body, url, last_error, created_at, retry_count, max_retries)
           VALUES (:topic_id, :check_result_id, :title, :body, :url, :last_error,
           :created_at, :retry_count, :max_retries)""",
        data,
    )
    notification.id = cursor.lastrowid
    return notification


def list_pending_notifications(
    conn: sqlite3.Connection,
) -> list[PendingNotification]:
    """Get pending notifications that haven't exceeded max retries.

    Already-claimed rows (``claimed_at`` set) are excluded so a concurrent
    drainer never re-snapshots an item another drainer is in the middle of
    sending (see :func:`claim_pending_notification` / OVH-017).
    """
    rows = conn.execute(
        "SELECT * FROM pending_notifications "
        "WHERE retry_count < max_retries AND claimed_at IS NULL "
        "ORDER BY created_at ASC"
    ).fetchall()
    return [PendingNotification.from_row(row) for row in rows]


def claim_pending_notification(conn: sqlite3.Connection, notification_id: int, claimed_at: str) -> bool:
    """Atomically claim a pending notification for sending.

    Returns True only if this caller won the claim (the row was unclaimed and
    is now stamped). A concurrent drainer that lost the race gets False and
    must skip the row, preventing double-delivery across processes.
    """
    cursor = conn.execute(
        "UPDATE pending_notifications SET claimed_at = ? WHERE id = ? AND claimed_at IS NULL",
        (claimed_at, notification_id),
    )
    return cursor.rowcount == 1


def increment_notification_retry(conn: sqlite3.Connection, notification_id: int, last_error: str | None = None) -> None:
    """Increment the retry count and release the claim for a pending notification.

    Clearing ``claimed_at`` re-arms the row so the next cycle can re-claim and
    retry it. ``last_error`` (when given) records the most recent failure reason
    so a permanently-broken channel is distinguishable from a transient blip.
    """
    if last_error is not None:
        conn.execute(
            "UPDATE pending_notifications "
            "SET retry_count = retry_count + 1, claimed_at = NULL, last_error = ? WHERE id = ?",
            (last_error, notification_id),
        )
    else:
        conn.execute(
            "UPDATE pending_notifications SET retry_count = retry_count + 1, claimed_at = NULL WHERE id = ?",
            (notification_id,),
        )


def delete_pending_notification(conn: sqlite3.Connection, notification_id: int) -> None:
    """Delete a pending notification (after successful send or max retries)."""
    conn.execute("DELETE FROM pending_notifications WHERE id = ?", (notification_id,))


def release_stale_notification_claims(conn: sqlite3.Connection, cutoff: str) -> int:
    """Release notification claims stamped at or before ``cutoff`` (ISO string).

    A drainer that claims a row then crashes before applying its result would
    otherwise leave the row claimed forever (and so never re-sent). Clearing
    stale claims at snapshot time makes the queue self-healing.
    """
    cursor = conn.execute(
        "UPDATE pending_notifications SET claimed_at = NULL WHERE claimed_at IS NOT NULL AND claimed_at <= ?",
        (cutoff,),
    )
    return cursor.rowcount


def delete_expired_notifications(conn: sqlite3.Connection) -> list[PendingNotification]:
    """Delete notifications that have exceeded their max retries.

    Returns the rows that were permanently abandoned (selected before the
    DELETE) so the caller can log exactly what was dropped instead of only a
    count — a silently-pruned notification is otherwise unobservable (OVH-040).
    """
    rows = conn.execute("SELECT * FROM pending_notifications WHERE retry_count >= max_retries").fetchall()
    abandoned = [PendingNotification.from_row(row) for row in rows]
    conn.execute("DELETE FROM pending_notifications WHERE retry_count >= max_retries")
    return abandoned


# --- PendingWebhook CRUD ---
#
# The webhook retry queue mirrors pending_notifications. Rows map to the
# PendingWebhook model (see app/models.py); the payload is stored as a JSON TEXT
# column. list_pending_webhooks returns PendingWebhook models, symmetric with
# list_pending_notifications (OVH-152).


def create_pending_webhook(
    conn: sqlite3.Connection,
    topic_id: int,
    url: str,
    payload: dict,
    check_result_id: int | None = None,
    max_retries: int = 3,
) -> int:
    """Store a failed webhook delivery for later retry. Returns the new row id."""
    webhook = PendingWebhook(
        topic_id=topic_id,
        check_result_id=check_result_id,
        url=url,
        payload=payload,
        max_retries=max_retries,
    )
    data = webhook.to_insert_dict()
    cursor = conn.execute(
        """INSERT INTO pending_webhooks (topic_id, check_result_id, url, payload,
           created_at, retry_count, max_retries)
           VALUES (:topic_id, :check_result_id, :url, :payload,
           :created_at, :retry_count, :max_retries)""",
        data,
    )
    return int(cursor.lastrowid or 0)


def list_pending_webhooks(conn: sqlite3.Connection) -> list[PendingWebhook]:
    """Get pending webhooks that haven't exceeded max retries.

    Returns ``PendingWebhook`` models (payload decoded from JSON), symmetric with
    :func:`list_pending_notifications` (OVH-152). Already-claimed rows
    (``claimed_at`` set) are excluded so a concurrent drainer never re-snapshots
    an item another drainer is sending (see :func:`claim_pending_webhook` /
    OVH-017).
    """
    rows = conn.execute(
        "SELECT * FROM pending_webhooks WHERE retry_count < max_retries AND claimed_at IS NULL ORDER BY created_at ASC"
    ).fetchall()
    return [PendingWebhook.from_row(row) for row in rows]


def claim_pending_webhook(conn: sqlite3.Connection, webhook_id: int, claimed_at: str) -> bool:
    """Atomically claim a pending webhook for sending.

    Returns True only if this caller won the claim (the row was unclaimed and
    is now stamped). A concurrent drainer that lost the race gets False and
    must skip the row, preventing double-delivery across processes.
    """
    cursor = conn.execute(
        "UPDATE pending_webhooks SET claimed_at = ? WHERE id = ? AND claimed_at IS NULL",
        (claimed_at, webhook_id),
    )
    return cursor.rowcount == 1


def increment_webhook_retry(conn: sqlite3.Connection, webhook_id: int) -> None:
    """Increment the retry count and release the claim for a pending webhook.

    Clearing ``claimed_at`` re-arms the row so the next cycle can re-claim and
    retry it.
    """
    conn.execute(
        "UPDATE pending_webhooks SET retry_count = retry_count + 1, claimed_at = NULL WHERE id = ?",
        (webhook_id,),
    )


def delete_pending_webhook(conn: sqlite3.Connection, webhook_id: int) -> None:
    """Delete a pending webhook (after successful send or max retries)."""
    conn.execute("DELETE FROM pending_webhooks WHERE id = ?", (webhook_id,))


def release_stale_webhook_claims(conn: sqlite3.Connection, cutoff: str) -> int:
    """Release webhook claims stamped at or before ``cutoff`` (ISO string).

    Mirrors :func:`release_stale_notification_claims`: a drainer that claims a
    row then crashes before applying would otherwise strand it claimed forever.
    """
    cursor = conn.execute(
        "UPDATE pending_webhooks SET claimed_at = NULL WHERE claimed_at IS NOT NULL AND claimed_at <= ?",
        (cutoff,),
    )
    return cursor.rowcount


def delete_expired_webhooks(conn: sqlite3.Connection) -> list[PendingWebhook]:
    """Delete webhooks that have exceeded their max retries.

    Returns the rows that were permanently abandoned (selected before the
    DELETE) so the caller can log exactly what was dropped — including the
    topic_id/check_result_id for traceability — instead of only a count
    (OVH-040). The URL is redacted by the caller before it reaches a log.
    """
    rows = conn.execute("SELECT * FROM pending_webhooks WHERE retry_count >= max_retries").fetchall()
    abandoned = [PendingWebhook.from_row(row) for row in rows]
    conn.execute("DELETE FROM pending_webhooks WHERE retry_count >= max_retries")
    return abandoned


# --- FeedHealth CRUD ---


def upsert_feed_health_success(
    conn: sqlite3.Connection,
    feed_url: str,
    etag: str | None = None,
    last_modified: str | None = None,
) -> None:
    """Record a successful feed fetch.

    ``etag`` / ``last_modified`` are the response's conditional-GET validators.
    A 304 (unchanged) passes ``None`` for both; ``COALESCE`` then preserves the
    previously stored validators instead of wiping them.
    """
    now = datetime.now(UTC).isoformat()
    conn.execute(
        """INSERT INTO feed_health
               (feed_url, last_success_at, consecutive_failures, total_fetches, etag, last_modified)
           VALUES (?, ?, 0, 1, ?, ?)
           ON CONFLICT(feed_url) DO UPDATE SET
               last_success_at = ?,
               consecutive_failures = 0,
               total_fetches = total_fetches + 1,
               etag = COALESCE(?, etag),
               last_modified = COALESCE(?, last_modified)""",
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
    """Get topics in NEW status, oldest first (for gradual scheduler init)."""
    rows = conn.execute(
        "SELECT * FROM topics WHERE status = ? ORDER BY created_at ASC LIMIT ?",
        (TopicStatus.NEW.value, limit),
    ).fetchall()
    return [Topic.from_row(row) for row in rows]


def claim_new_topic_for_init(conn: sqlite3.Connection, topic_id: int) -> bool:
    """Atomically claim a NEW topic for initialization (NEW -> RESEARCHING).

    Returns True only if this caller won the claim (rowcount == 1). A concurrent
    init (a same-minute web Retry click, a second scheduler tick on another
    process, or a CLI init) that already transitioned the row out of NEW loses
    here, so only one initializer proceeds (OVH-032). Mirrors the conditional-UPDATE
    pattern in ``recover_stuck_researching``. Commits so the claim is durable and
    immediately visible to concurrent WAL connections.
    """
    now = datetime.now(UTC).isoformat()
    cursor = conn.execute(
        """UPDATE topics SET status = ?, status_changed_at = ?, error_message = ?
           WHERE id = ? AND status = ?""",
        (TopicStatus.RESEARCHING.value, now, None, topic_id, TopicStatus.NEW.value),
    )
    conn.commit()
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
            (SELECT MAX(checked_at) FROM check_results WHERE notification_sent = 1) AS last_notification_at
        """
    ).fetchone()
    assert row is not None
    last_notif = None
    if row["last_notification_at"]:
        import contextlib

        with contextlib.suppress(ValueError, TypeError):
            last_notif = datetime.fromisoformat(row["last_notification_at"])
    return DashboardStats(
        total_topics=row["total_topics"],
        active_topics=row["active_topics"],
        checks_24h=row["checks_24h"],
        checks_total=row["checks_total"],
        new_info_24h=row["new_info_24h"],
        new_info_total=row["new_info_total"],
        last_notification_at=last_notif,
    )
