"""CRUD operations for all data models.

All functions accept a sqlite3.Connection as their first parameter
for explicit dependency injection and testability.
"""

import logging
import sqlite3
from datetime import UTC, datetime

from app.models import (
    Article,
    CheckResult,
    DashboardStats,
    FeedHealth,
    KnowledgeState,
    PendingNotification,
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
           created_at, status_changed_at, is_active, status, error_message, check_interval_minutes, tags)
           VALUES (:name, :description, :feed_urls, :feed_mode,
           :created_at, :status_changed_at, :is_active, :status, :error_message, :check_interval_minutes, :tags)""",
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
) -> list[Topic]:
    """List all topics, optionally filtering to active ones and/or by tag."""
    where_clauses = []
    params: list = []

    if active_only:
        where_clauses.append("t.is_active = 1")

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
           tags=:tags
           WHERE id=:id""",
        data,
    )
    logger.info("Updated topic: %s (id=%d)", topic.name, topic.id)
    return topic


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
        conn.commit()
        logger.warning("Recovered %d stuck researching topic(s)", count)
    return count


def get_dashboard_data(conn: sqlite3.Connection) -> list[dict]:
    """Get all topics with last check and article count in a single query."""
    rows = conn.execute(
        """
        SELECT t.*,
               cr.id AS cr_id,
               cr.checked_at AS cr_checked_at,
               cr.articles_found AS cr_articles_found,
               cr.articles_new AS cr_articles_new,
               cr.has_new_info AS cr_has_new_info,
               cr.llm_response AS cr_llm_response,
               cr.notification_sent AS cr_notification_sent,
               cr.notification_error AS cr_notification_error,
               (SELECT COUNT(*) FROM articles WHERE articles.topic_id = t.id) AS article_count
        FROM topics t
        LEFT JOIN check_results cr ON cr.id = (
            SELECT id FROM check_results
            WHERE topic_id = t.id
            ORDER BY checked_at DESC LIMIT 1
        )
        ORDER BY t.name
        """
    ).fetchall()

    result = []
    for row in rows:
        topic = Topic.from_row(row)
        last_check = None
        if row["cr_id"] is not None and topic.id is not None:
            last_check = CheckResult(
                id=row["cr_id"],
                topic_id=topic.id,
                checked_at=row["cr_checked_at"],
                articles_found=row["cr_articles_found"],
                articles_new=row["cr_articles_new"],
                has_new_info=bool(row["cr_has_new_info"]),
                llm_response=row["cr_llm_response"],
                notification_sent=bool(row["cr_notification_sent"]),
                notification_error=row["cr_notification_error"],
            )
        result.append(
            {
                "topic": topic,
                "last_check": last_check,
                "article_count": row["article_count"],
            }
        )
    return result


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
        where_sql = "WHERE " + " AND ".join(where_clauses)

    rows = conn.execute(
        f"""
        SELECT t.*,
               cr.id AS cr_id,
               cr.checked_at AS cr_checked_at,
               cr.articles_found AS cr_articles_found,
               cr.articles_new AS cr_articles_new,
               cr.has_new_info AS cr_has_new_info,
               cr.llm_response AS cr_llm_response,
               cr.notification_sent AS cr_notification_sent,
               cr.notification_error AS cr_notification_error,
               (SELECT COUNT(*) FROM articles WHERE articles.topic_id = t.id) AS article_count
        FROM topics t
        LEFT JOIN check_results cr ON cr.id = (
            SELECT id FROM check_results
            WHERE topic_id = t.id
            ORDER BY checked_at DESC LIMIT 1
        )
        {where_sql}
        ORDER BY t.name
        """,
        params,
    ).fetchall()

    result = []
    for row in rows:
        topic = Topic.from_row(row)
        last_check = None
        if row["cr_id"] is not None and topic.id is not None:
            last_check = CheckResult(
                id=row["cr_id"],
                topic_id=topic.id,
                checked_at=row["cr_checked_at"],
                articles_found=row["cr_articles_found"],
                articles_new=row["cr_articles_new"],
                has_new_info=bool(row["cr_has_new_info"]),
                llm_response=row["cr_llm_response"],
                notification_sent=bool(row["cr_notification_sent"]),
                notification_error=row["cr_notification_error"],
            )
        result.append(
            {
                "topic": topic,
                "last_check": last_check,
                "article_count": row["article_count"],
            }
        )
    return result


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
           raw_content, source_feed, source_provider, fetched_at, processed)
           VALUES (:topic_id, :title, :url, :content_hash,
           :raw_content, :source_feed, :source_provider, :fetched_at, :processed)""",
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


def delete_old_articles(conn: sqlite3.Connection, retention_days: int) -> int:
    """Delete articles older than retention_days. Returns count of deleted rows."""
    cursor = conn.execute(
        "DELETE FROM articles WHERE fetched_at < datetime('now', ? || ' days')",
        (f"-{retention_days}",),
    )
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
    cursor = conn.execute(
        """INSERT INTO check_results (topic_id, checked_at, articles_found,
           articles_new, has_new_info, llm_response, notification_sent,
           notification_error)
           VALUES (:topic_id, :checked_at, :articles_found, :articles_new,
           :has_new_info, :llm_response, :notification_sent,
           :notification_error)""",
        data,
    )
    result.id = cursor.lastrowid
    return result


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


# --- PendingNotification CRUD ---


def create_pending_notification(conn: sqlite3.Connection, notification: PendingNotification) -> PendingNotification:
    """Store a failed notification for later retry."""
    data = notification.to_insert_dict()
    cursor = conn.execute(
        """INSERT INTO pending_notifications (topic_id, check_result_id,
           title, body, created_at, retry_count, max_retries)
           VALUES (:topic_id, :check_result_id, :title, :body,
           :created_at, :retry_count, :max_retries)""",
        data,
    )
    notification.id = cursor.lastrowid
    return notification


def list_pending_notifications(
    conn: sqlite3.Connection,
) -> list[PendingNotification]:
    """Get all pending notifications that haven't exceeded max retries."""
    rows = conn.execute(
        "SELECT * FROM pending_notifications WHERE retry_count < max_retries ORDER BY created_at ASC"
    ).fetchall()
    return [PendingNotification.from_row(row) for row in rows]


def increment_notification_retry(conn: sqlite3.Connection, notification_id: int) -> None:
    """Increment the retry count for a pending notification."""
    conn.execute(
        "UPDATE pending_notifications SET retry_count = retry_count + 1 WHERE id = ?",
        (notification_id,),
    )


def delete_pending_notification(conn: sqlite3.Connection, notification_id: int) -> None:
    """Delete a pending notification (after successful send or max retries)."""
    conn.execute("DELETE FROM pending_notifications WHERE id = ?", (notification_id,))


def delete_expired_notifications(conn: sqlite3.Connection) -> int:
    """Delete notifications that have exceeded their max retries."""
    cursor = conn.execute("DELETE FROM pending_notifications WHERE retry_count >= max_retries")
    return cursor.rowcount


# --- FeedHealth CRUD ---


def upsert_feed_health_success(conn: sqlite3.Connection, feed_url: str) -> None:
    """Record a successful feed fetch."""
    now = datetime.now(UTC).isoformat()
    conn.execute(
        """INSERT INTO feed_health (feed_url, last_success_at, consecutive_failures, total_fetches)
           VALUES (?, ?, 0, 1)
           ON CONFLICT(feed_url) DO UPDATE SET
               last_success_at = ?,
               consecutive_failures = 0,
               total_fetches = total_fetches + 1""",
        (feed_url, now, now),
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


def get_all_feed_urls(conn: sqlite3.Connection) -> set[str]:
    """Get all feed URLs across all topics for OPML dedup."""
    rows = conn.execute("SELECT DISTINCT json_each.value FROM topics, json_each(topics.feed_urls)").fetchall()
    return {row[0] for row in rows}


# --- Dashboard Stats ---


def get_dashboard_stats(conn: sqlite3.Connection) -> DashboardStats:
    """Get aggregate statistics for the dashboard."""
    row = conn.execute(
        """
        SELECT
            (SELECT COUNT(*) FROM topics) AS total_topics,
            (SELECT COUNT(*) FROM topics WHERE is_active = 1) AS active_topics,
            (SELECT COUNT(*) FROM check_results WHERE checked_at >= datetime('now', '-1 day')) AS checks_24h,
            (SELECT COUNT(*) FROM check_results) AS checks_total,
            (SELECT COUNT(*) FROM check_results WHERE has_new_info = 1 AND checked_at >= datetime('now', '-1 day')) AS new_info_24h,
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
