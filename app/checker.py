"""Core check loop: scrape, analyze, notify, record.

Orchestrates the full pipeline for checking topics for new information.
Each check cycle fetches articles, analyzes them against the knowledge
state, sends notifications for genuine updates, and records the outcome.
"""

import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from app.analysis.knowledge import initialize_knowledge, update_knowledge
from app.analysis.llm import analyze_articles
from app.check_context import check_id_var, generate_check_id
from app.config import Settings
from app.crud import (
    create_check_result,
    create_pending_notification,
    delete_expired_notifications,
    delete_pending_notification,
    get_knowledge_state,
    get_topics_due_for_check,
    increment_notification_retry,
    list_pending_notifications,
    mark_articles_processed,
    update_topic,
)
from app.database import short_conn
from app.models import CheckResult, PendingNotification, Topic, TopicStatus
from app.notifications import format_notification, send_notification
from app.scraping import fetch_new_articles_for_topic
from app.webhooks import send_webhooks

logger = logging.getLogger(__name__)

# Maximum number of initialization passes before a thin topic is forced READY
# with whatever (insufficient) knowledge exists, to avoid looping forever.
MAX_INIT_ATTEMPTS = 3


async def check_topic(
    topic: Topic,
    conn: sqlite3.Connection,
    settings: Settings,
) -> CheckResult:
    """Run the full check pipeline for a single topic.

    Steps:
        1. Validate topic is in READY status
        2. Fetch new articles (scraping + dedup)
        3. If no new articles, return early
        4. Analyze articles against knowledge state (LLM)
        5. If new info found: update knowledge state, send notification
        6. Mark articles as processed
        7. Record and return CheckResult

    Args:
        topic: The topic to check. Must have an id and be in READY status.
        conn: Database connection for reads and writes.
        settings: Application settings.

    Returns:
        CheckResult recording the outcome of this check.
    """
    if topic.id is None:
        raise ValueError("Topic must have an ID")

    cid = generate_check_id()
    check_id_var.set(cid)

    try:
        return await _check_topic_inner(topic, conn, settings, cid)
    finally:
        check_id_var.set(None)


async def _check_topic_inner(
    topic: Topic,
    conn: sqlite3.Connection,
    settings: Settings,
    cid: str,
) -> CheckResult:
    """Inner implementation of check_topic with check_id already set."""
    if topic.id is None:
        raise ValueError("Topic must have an ID to be checked")
    topic_id: int = topic.id
    result = CheckResult(topic_id=topic_id)

    logger.info("Starting check for topic '%s' [check_id=%s]", topic.name, cid)

    # Only check READY topics
    if topic.status != TopicStatus.READY:
        logger.warning(
            "Skipping topic '%s' — status is '%s', not 'ready'",
            topic.name,
            topic.status,
        )
        return _record_result(conn, result)

    # Step 1: Fetch new articles
    try:
        fetch_result = await fetch_new_articles_for_topic(
            topic,
            conn,
            max_articles=settings.max_articles_per_check,
            feed_fetch_timeout=settings.feed_fetch_timeout,
            article_fetch_timeout=settings.article_fetch_timeout,
            feed_max_retries=settings.feed_max_retries,
            concurrency=settings.content_fetch_concurrency,
        )
    except Exception:
        logger.warning("Scraping failed for topic '%s'", topic.name, exc_info=True)
        return _record_result(conn, result)

    new_articles = fetch_result.articles
    result.articles_found = fetch_result.total_feed_entries
    result.articles_new = len(new_articles)

    if not new_articles:
        logger.info("Topic '%s': no new articles found", topic.name)
        return _record_result(conn, result)

    # Step 2: Get current knowledge state
    knowledge = get_knowledge_state(conn, topic_id)
    knowledge_summary = knowledge.summary_text if knowledge else ""

    # Step 3: Analyze articles for novelty (returns safe default on LLM error)
    novelty = await analyze_articles(new_articles, knowledge_summary, topic, settings)
    result.has_new_info = novelty.has_new_info
    result.llm_response = novelty.model_dump_json()
    result.prompt_tokens += novelty.prompt_tokens
    result.completion_tokens += novelty.completion_tokens

    # Effective thresholds: per-topic override (NULL = inherit global).
    confidence_threshold = (
        topic.confidence_threshold if topic.confidence_threshold is not None else settings.min_confidence_threshold
    )
    relevance_threshold = (
        topic.relevance_threshold if topic.relevance_threshold is not None else settings.min_relevance_threshold
    )

    # Step 4: If new info, update knowledge and notify
    if novelty.has_new_info:
        if novelty.confidence < confidence_threshold:
            logger.info(
                "Topic '%s': new info detected but confidence %.2f below threshold %.2f, skipping notification",
                topic.name,
                novelty.confidence,
                confidence_threshold,
            )
        elif novelty.relevance < relevance_threshold:
            logger.info(
                "Topic '%s': new info detected but relevance %.2f below threshold %.2f, skipping notification",
                topic.name,
                novelty.relevance,
                relevance_threshold,
            )
        else:
            try:
                write_result = await update_knowledge(topic, novelty, conn, settings)
                result.prompt_tokens += write_result.usage.prompt_tokens
                result.completion_tokens += write_result.usage.completion_tokens
            except Exception:
                logger.warning(
                    "Knowledge update failed for topic '%s'",
                    topic.name,
                    exc_info=True,
                )

            title, body = format_notification(topic.name, novelty)
            try:
                sent = await send_notification(title, body, settings)
                result.notification_sent = sent
                if not sent:
                    result.notification_error = "Delivery failed"
                    _queue_notification(conn, topic_id, title, body)
            except Exception as exc:
                logger.warning(
                    "Notification failed for topic '%s'",
                    topic.name,
                    exc_info=True,
                )
                result.notification_error = str(exc)
                _queue_notification(conn, topic_id, title, body)

            # Send webhooks (independent of Apprise success/failure). Pass the
            # connection + topic_id so failed deliveries are enqueued to
            # pending_webhooks for retry instead of being dropped.
            try:
                await send_webhooks(topic.name, novelty, settings, conn=conn, topic_id=topic_id)
            except Exception:
                logger.warning(
                    "Webhook delivery failed for topic '%s'",
                    topic.name,
                    exc_info=True,
                )

    # Step 5: Mark articles as processed. "processed" means "we've evaluated
    # this article" — set even for below-threshold (new-but-not-notified) and
    # not-new articles, so they are never re-analyzed. Leaving them unprocessed
    # would re-fetch + re-analyze them every cycle after retention deletion +
    # feed reappearance, wasting LLM quota.
    article_ids = [a.id for a in new_articles if a.id is not None]
    if article_ids:
        mark_articles_processed(conn, article_ids)

    logger.info(
        "Topic '%s': %d articles, new_info=%s, notified=%s",
        topic.name,
        len(new_articles),
        novelty.has_new_info,
        result.notification_sent,
    )

    return _record_result(conn, result)


def _record_result(conn: sqlite3.Connection, result: CheckResult) -> CheckResult:
    """Persist a CheckResult and commit."""
    created = create_check_result(conn, result)
    conn.commit()
    return created


def _queue_notification(conn: sqlite3.Connection, topic_id: int, title: str, body: str) -> None:
    """Queue a failed notification for retry."""
    try:
        create_pending_notification(
            conn,
            PendingNotification(topic_id=topic_id, title=title, body=body),
        )
        logger.info("Queued notification for retry (topic_id=%d)", topic_id)
    except Exception:
        logger.warning("Failed to queue notification for retry", exc_info=True)


async def retry_pending_notifications(
    conn: sqlite3.Connection | None = None,
    settings: Settings | None = None,
    *,
    db_path: Path | None = None,
) -> None:
    """Retry any pending notifications from previous check cycles.

    Successful notifications are deleted. Failed ones get their retry
    count incremented. Notifications exceeding max_retries are deleted.

    Connection handling mirrors retry_pending_webhooks: no sqlite connection
    is held across the (potentially slow) notification sends. Pending rows are
    snapshotted under a short connection, sends run with no open connection,
    and each result is applied with a commit *per item* so a mid-loop crash
    can't roll back already-applied delete/increment operations.

    Args:
        conn: Optional existing connection (back-compat). Reused if given but
            committed per item and never held across a send.
        settings: Application settings (required).
        db_path: Path used to open short-lived connections when ``conn`` is None.
    """
    if settings is None:
        raise ValueError("settings is required")

    # --- Phase 1: snapshot pending rows under a short-lived connection. ---
    with short_conn(conn, db_path) as snapshot:
        expired = delete_expired_notifications(snapshot)
        if expired:
            logger.warning("Deleted %d expired pending notification(s)", expired)
        snapshot.commit()
        pending = list_pending_notifications(snapshot)

    if not pending:
        return

    logger.info("Retrying %d pending notification(s)", len(pending))

    # --- Phase 2: send with NO connection held, then apply per item. ---
    for notification in pending:
        assert notification.id is not None
        try:
            sent = await send_notification(notification.title, notification.body, settings)
        except Exception:
            sent = False
            logger.warning("Retry error for notification id=%d", notification.id, exc_info=True)

        with short_conn(conn, db_path) as apply_conn:
            if sent:
                delete_pending_notification(apply_conn, notification.id)
                logger.info("Retry succeeded for notification id=%d", notification.id)
            else:
                increment_notification_retry(apply_conn, notification.id)
                logger.warning("Retry failed for notification id=%d", notification.id)
            apply_conn.commit()


async def check_all_topics(
    conn: sqlite3.Connection,
    settings: Settings,
) -> list[CheckResult]:
    """Check all active, ready topics for new information.

    Each topic is checked independently. Errors in one topic do not
    affect others.

    Args:
        conn: Database connection.
        settings: Application settings.

    Returns:
        List of CheckResults, one per topic checked.
    """
    # Retry any failed notifications from previous cycles
    await retry_pending_notifications(conn, settings)

    due_topics = get_topics_due_for_check(conn, settings.check_interval_minutes)

    if not due_topics:
        return []

    logger.info("Starting check cycle for %d due topics", len(due_topics))

    results: list[CheckResult] = []
    for topic in due_topics:
        try:
            result = await check_topic(topic, conn, settings)
            results.append(result)
        except Exception:
            logger.error(
                "Unexpected error checking topic '%s'",
                topic.name,
                exc_info=True,
            )

    logger.info(
        "Check cycle complete: %d topics checked, %d with new info",
        len(results),
        sum(1 for r in results if r.has_new_info),
    )
    return results


async def initialize_new_topic(
    topic: Topic,
    conn: sqlite3.Connection,
    settings: Settings,
) -> None:
    """Initialize a topic's knowledge state from its first batch of articles.

    Transitions: NEW/RESEARCHING → RESEARCHING → READY (or ERROR on failure).
    Called by both the web layer (background task) and the scheduler (gradual init).
    """
    if topic.id is None:
        raise ValueError("Topic must have an ID")

    # Immediately mark as RESEARCHING (concurrency guard: UI shows spinner, prevents re-trigger)
    topic.status = TopicStatus.RESEARCHING
    topic.status_changed_at = datetime.now(UTC)
    topic.error_message = None
    update_topic(conn, topic)
    conn.commit()

    logger.info("Initializing knowledge for topic '%s' (id=%d)", topic.name, topic.id)

    try:
        fetch_result = await fetch_new_articles_for_topic(
            topic,
            conn,
            max_articles=settings.max_articles_per_check,
            feed_fetch_timeout=settings.feed_fetch_timeout,
            article_fetch_timeout=settings.article_fetch_timeout,
            feed_max_retries=settings.feed_max_retries,
            concurrency=settings.content_fetch_concurrency,
        )
        articles = fetch_result.articles

        if not articles:
            topic.status = TopicStatus.ERROR
            topic.status_changed_at = datetime.now(UTC)
            topic.error_message = "No articles found during initialization"
            update_topic(conn, topic)
            conn.commit()
            return

        # create_knowledge_state uses INSERT OR REPLACE, so re-init works atomically
        write_result = await initialize_knowledge(topic, articles, conn, settings)

        article_ids = [a.id for a in articles if a.id is not None]
        if article_ids:
            mark_articles_processed(conn, article_ids)

        if not write_result.sufficient_data and topic.init_attempts < MAX_INIT_ATTEMPTS:
            # Thin data: retry on a later cycle. Bump attempts and send the topic
            # back to NEW so the scheduler's gradual init re-runs it. Do NOT mark
            # READY yet.
            topic.init_attempts += 1
            topic.status = TopicStatus.NEW
            topic.status_changed_at = datetime.now(UTC)
            topic.error_message = None
            update_topic(conn, topic)
            conn.commit()
            logger.info(
                "Knowledge for topic '%s' insufficient — retry %d/%d, back to NEW",
                topic.name,
                topic.init_attempts,
                MAX_INIT_ATTEMPTS,
            )
            return

        # Either knowledge is sufficient, or attempts are exhausted: go READY.
        topic.status = TopicStatus.READY
        topic.status_changed_at = datetime.now(UTC)
        topic.error_message = None
        topic.init_attempts = 0
        update_topic(conn, topic)
        conn.commit()

        if write_result.sufficient_data:
            logger.info("Knowledge initialized for topic '%s' — now READY", topic.name)
        else:
            logger.warning(
                "Topic '%s' READY with insufficient knowledge after %d attempts",
                topic.name,
                MAX_INIT_ATTEMPTS,
            )

    except Exception as exc:
        logger.error("Knowledge init failed for topic '%s'", topic.name, exc_info=True)
        topic.status = TopicStatus.ERROR
        topic.status_changed_at = datetime.now(UTC)
        topic.error_message = str(exc)
        update_topic(conn, topic)
        conn.commit()
