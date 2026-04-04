"""Core check loop: scrape, analyze, notify, record.

Orchestrates the full pipeline for checking topics for new information.
Each check cycle fetches articles, analyzes them against the knowledge
state, sends notifications for genuine updates, and records the outcome.
"""

import logging
import sqlite3

from app.analysis.knowledge import update_knowledge
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
)
from app.models import CheckResult, PendingNotification, Topic, TopicStatus
from app.notifications import format_notification, send_notification
from app.scraping import fetch_new_articles_for_topic
from app.webhooks import send_webhooks

logger = logging.getLogger(__name__)


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

    # Step 4: If new info, update knowledge and notify
    below_threshold = False
    if novelty.has_new_info:
        if novelty.confidence < settings.min_confidence_threshold:
            logger.info(
                "Topic '%s': new info detected but confidence %.2f below threshold %.2f, skipping notification",
                topic.name,
                novelty.confidence,
                settings.min_confidence_threshold,
            )
            below_threshold = True
        elif novelty.relevance < settings.min_relevance_threshold:
            logger.info(
                "Topic '%s': new info detected but relevance %.2f below threshold %.2f, skipping notification",
                topic.name,
                novelty.relevance,
                settings.min_relevance_threshold,
            )
            below_threshold = True
        else:
            try:
                await update_knowledge(topic, novelty, conn, settings)
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

            # Send webhooks (independent of Apprise success/failure)
            try:
                await send_webhooks(topic.name, novelty, settings)
            except Exception:
                logger.warning(
                    "Webhook delivery failed for topic '%s'",
                    topic.name,
                    exc_info=True,
                )

    # Step 5: Mark articles as processed (skip if below threshold — re-examine next cycle)
    if not below_threshold:
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


async def retry_pending_notifications(conn: sqlite3.Connection, settings: Settings) -> None:
    """Retry any pending notifications from previous check cycles.

    Successful notifications are deleted. Failed ones get their retry
    count incremented. Notifications exceeding max_retries are deleted.
    """
    expired = delete_expired_notifications(conn)
    if expired:
        logger.warning("Deleted %d expired pending notification(s)", expired)

    pending = list_pending_notifications(conn)
    if not pending:
        return

    logger.info("Retrying %d pending notification(s)", len(pending))
    for notification in pending:
        assert notification.id is not None
        try:
            sent = await send_notification(notification.title, notification.body, settings)
            if sent:
                delete_pending_notification(conn, notification.id)
                logger.info("Retry succeeded for notification id=%d", notification.id)
            else:
                increment_notification_retry(conn, notification.id)
                logger.warning("Retry failed for notification id=%d", notification.id)
        except Exception:
            increment_notification_retry(conn, notification.id)
            logger.warning("Retry error for notification id=%d", notification.id, exc_info=True)
    conn.commit()


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
