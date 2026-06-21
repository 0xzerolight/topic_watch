"""Core check loop: scrape, analyze, notify, record.

Orchestrates the full pipeline for checking topics for new information.
Each check cycle fetches articles, analyzes them against the knowledge
state, sends notifications for genuine updates, and records the outcome.
"""

import asyncio
import logging
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.analysis.knowledge import initialize_knowledge, update_knowledge
from app.analysis.llm import analyze_articles
from app.check_context import check_id_var, generate_check_id
from app.config import Settings
from app.crud import (
    claim_pending_notification,
    create_check_result,
    create_pending_notification,
    delete_expired_notifications,
    delete_pending_notification,
    get_knowledge_state,
    get_topics_due_for_check,
    increment_notification_retry,
    list_pending_notifications,
    mark_articles_processed,
    release_stale_notification_claims,
    update_check_result_delivery,
    update_topic_init_status,
)
from app.database import get_db, short_conn
from app.models import CheckResult, PendingNotification, Topic, TopicStatus
from app.notifications import format_notification, send_notification
from app.scraping import fetch_new_articles_for_topic
from app.webhooks import retry_pending_webhooks, send_webhooks

logger = logging.getLogger(__name__)

# Maximum number of initialization passes before a thin topic is forced READY
# with whatever (insufficient) knowledge exists, to avoid looping forever.
MAX_INIT_ATTEMPTS = 3

# Single-flight guard: serializes notification drains within this process so
# two overlapping drains (scheduler tick vs. a UI/CLI check-all) cannot both
# walk the queue at once. The cross-process case is covered by the atomic
# per-row claim (claimed_at) below. (OVH-017)
_notification_retry_lock = asyncio.Lock()

# Claims older than this are treated as stale (a drainer crashed mid-send) and
# released so the row can be re-claimed.
_CLAIM_STALE_AFTER = timedelta(minutes=10)


def _summarize_exc(exc: BaseException, *, limit: int = 200) -> str:
    """One-line, length-bounded exception summary for the stored stage_error."""
    summary = f"{type(exc).__name__}: {exc}".replace("\n", " ").strip()
    return summary[:limit]


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
    except Exception as exc:
        logger.warning("Scraping failed for topic '%s'", topic.name, exc_info=True)
        result.stage_error = f"scrape_failed: {_summarize_exc(exc)}"
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

    # analyze_articles stays fail-safe (never raises), so an LLM failure surfaces
    # as the safe default plus a populated ``error``. Record it distinctly so a
    # broken analysis is not byte-identical to a clean "nothing new" run.
    if novelty.error:
        result.stage_error = f"analysis_failed: {novelty.error}"

    # Effective thresholds: per-topic override (NULL = inherit global).
    confidence_threshold = (
        topic.confidence_threshold if topic.confidence_threshold is not None else settings.min_confidence_threshold
    )
    relevance_threshold = (
        topic.relevance_threshold if topic.relevance_threshold is not None else settings.min_relevance_threshold
    )

    # Step 4: If new info above thresholds, update knowledge. The notification
    # /webhook SENDS are deferred to Step 6 — AFTER the durable state is
    # committed — so an irreversible alert is never dispatched inside the same
    # transaction window that could later roll back (OVH-066).
    knowledge_update_failed = False
    should_notify = False
    notification: tuple[str, str] | None = None
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
            should_notify = True
            try:
                write_result = await update_knowledge(topic, novelty, conn, settings)
                result.prompt_tokens += write_result.usage.prompt_tokens
                result.completion_tokens += write_result.usage.completion_tokens
            except Exception as exc:
                logger.warning(
                    "Knowledge update failed for topic '%s'",
                    topic.name,
                    exc_info=True,
                )
                # OVH-009: the alert still fires, but record the failure distinctly
                # and do NOT mark these new-info-bearing articles processed, so the
                # next cycle re-attempts the knowledge update (no silent drift).
                knowledge_update_failed = True
                result.stage_error = f"knowledge_update_failed: {_summarize_exc(exc)}"
            notification = format_notification(topic.name, novelty)

    # Step 5: Mark articles as processed. "processed" means "we've evaluated
    # this article" — set even for below-threshold (new-but-not-notified) and
    # not-new articles, so they are never re-analyzed. Leaving them unprocessed
    # would re-fetch + re-analyze them every cycle after retention deletion +
    # feed reappearance, wasting LLM quota.
    #
    # Exception (OVH-009): when the knowledge update failed, the recorded
    # knowledge state is now stale. Leave these articles unprocessed so the next
    # cycle re-fetches and re-attempts the update instead of silently diverging.
    if not knowledge_update_failed:
        article_ids = [a.id for a in new_articles if a.id is not None]
        if article_ids:
            mark_articles_processed(conn, article_ids)

    # Step 6: Durable-state commit boundary (OVH-066). Persist the knowledge
    # update + processed flags + CheckResult in one explicit write transaction
    # BEFORE any irreversible network send. If this commit fails, no alert has
    # gone out and the next cycle re-runs cleanly. Creating the CheckResult here
    # also gives the webhook queue a real check_result_id (OVH-101).
    result = create_check_result(conn, result)
    conn.commit()

    # Step 7: Irreversible network sends, now that durable state is committed.
    if should_notify and notification is not None:
        title, body = notification
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
        # connection + topic_id + check_result_id so failed deliveries are
        # enqueued to pending_webhooks (correlated to this check) for retry
        # instead of being dropped.
        try:
            await send_webhooks(
                topic.name,
                novelty,
                settings,
                conn=conn,
                topic_id=topic_id,
                check_result_id=result.id,
            )
        except Exception:
            logger.warning(
                "Webhook delivery failed for topic '%s'",
                topic.name,
                exc_info=True,
            )

        # Step 8: Record the post-send delivery outcome onto the committed row.
        if result.id is not None:
            update_check_result_delivery(
                conn,
                result.id,
                notification_sent=result.notification_sent,
                notification_error=result.notification_error,
            )
            conn.commit()

    logger.info(
        "Topic '%s': %d articles, new_info=%s, notified=%s",
        topic.name,
        len(new_articles),
        novelty.has_new_info,
        result.notification_sent,
    )

    return result


def _record_result(conn: sqlite3.Connection, result: CheckResult) -> CheckResult:
    """Persist a CheckResult and commit (used by the no-send early-return paths)."""
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

    # Single-flight: only one drain runs at a time in this process. A second
    # caller skips rather than walking the same queue concurrently (OVH-017).
    if _notification_retry_lock.locked():
        logger.debug("Notification retry already in progress; skipping overlapping drain")
        return

    async with _notification_retry_lock:
        await _drain_pending_notifications(conn, settings, db_path)


async def _drain_pending_notifications(
    conn: sqlite3.Connection | None,
    settings: Settings,
    db_path: Path | None,
) -> None:
    """Drain the notification retry queue once (caller holds the retry lock)."""
    # --- Phase 1: snapshot pending rows under a short-lived connection. ---
    stale_cutoff = (datetime.now(UTC) - _CLAIM_STALE_AFTER).isoformat()
    with short_conn(conn, db_path) as snapshot:
        released = release_stale_notification_claims(snapshot, stale_cutoff)
        if released:
            logger.warning("Released %d stale notification claim(s)", released)
        expired = delete_expired_notifications(snapshot)
        if expired:
            logger.warning("Deleted %d expired pending notification(s)", expired)
        snapshot.commit()
        pending = list_pending_notifications(snapshot)

    if not pending:
        return

    logger.info("Retrying %d pending notification(s)", len(pending))

    # --- Phase 2: claim, send with NO connection held, then apply per item. ---
    for notification in pending:
        assert notification.id is not None

        # Atomically claim this row. A concurrent (cross-process) drainer that
        # already claimed it returns rowcount 0 here, so we skip — only the
        # winner sends, preventing double-delivery (OVH-017).
        claimed_at = datetime.now(UTC).isoformat()
        with short_conn(conn, db_path) as claim_conn:
            won = claim_pending_notification(claim_conn, notification.id, claimed_at)
            claim_conn.commit()
        if not won:
            logger.debug("Notification id=%d already claimed by another drain; skipping", notification.id)
            continue

        try:
            sent = await send_notification(notification.title, notification.body, settings)
        except Exception:
            sent = False
            logger.warning("Retry error for notification id=%d", notification.id, exc_info=True)

        # Apply this single result and commit immediately. On failure,
        # increment_notification_retry also clears the claim so the next cycle
        # can re-claim and retry.
        with short_conn(conn, db_path) as apply_conn:
            if sent:
                delete_pending_notification(apply_conn, notification.id)
                logger.info("Retry succeeded for notification id=%d", notification.id)
            else:
                increment_notification_retry(apply_conn, notification.id)
                logger.warning("Retry failed for notification id=%d", notification.id)
            apply_conn.commit()


async def check_all_topics(
    settings: Settings,
    db_path: Path | None = None,
) -> list[CheckResult]:
    """Check all active, ready topics for new information.

    Uses per-topic connection granularity: a single connection held for the
    whole cycle would stay open across every topic's HTTP + LLM awaits,
    blocking concurrent web requests. Instead each phase uses its own
    short-lived connection that is committed and closed promptly:

      * retry passes (notifications + webhooks) — each manages its own
        short-lived connections internally (snapshot, send with none held,
        commit per item)
      * the due-topics query — one connection
      * each topic check — a fresh connection per topic

    Each topic is checked independently. Errors in one topic do not
    affect others.

    Args:
        settings: Application settings.
        db_path: Optional database path override for testing.

    Returns:
        List of CheckResults, one per topic checked.
    """
    # Retry any failed deliveries from previous cycles. Each retry function
    # manages its own short-lived connections: it snapshots pending rows,
    # sends with NO connection held, and commits per item.
    await retry_pending_notifications(settings=settings, db_path=db_path)
    await retry_pending_webhooks(settings=settings, db_path=db_path)

    # Snapshot the due topics, then release the connection before the long
    # per-topic HTTP/LLM work begins.
    with get_db(db_path) as conn:
        due_topics = get_topics_due_for_check(conn, settings.check_interval_minutes)

    if not due_topics:
        return []

    logger.info("Starting check cycle for %d due topics", len(due_topics))

    results: list[CheckResult] = []
    for topic in due_topics:
        # Fresh, short-lived connection per topic so no connection is held
        # across that topic's HTTP/LLM awaits.
        try:
            with get_db(db_path) as conn:
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

    Connection invariant (OVH-099): every status write below commits eagerly, and
    the fetch (OVH-007) + LLM (``initialize_knowledge`` writes only after its
    await) phases hold no write transaction across their awaits. So while the
    caller's connection is passed in, no write lock spans the fetch/LLM awaits —
    a concurrent WAL writer is never starved during initialization.
    """
    if topic.id is None:
        raise ValueError("Topic must have an ID")
    topic_id: int = topic.id

    # Status transitions here use ``update_topic_init_status`` (a targeted UPDATE of
    # only status/error/init_attempts) rather than ``update_topic`` so a concurrent
    # UI edit to this topic's feeds/thresholds during the long fetch/LLM await is
    # never clobbered by a stale in-memory snapshot (OVH-100).

    def _set_init_status(status: TopicStatus, *, error_message: str | None, init_attempts: int) -> None:
        now = datetime.now(UTC)
        update_topic_init_status(
            conn,
            topic_id,
            status=status,
            status_changed_at=now,
            error_message=error_message,
            init_attempts=init_attempts,
        )
        conn.commit()
        topic.status = status
        topic.status_changed_at = now
        topic.error_message = error_message
        topic.init_attempts = init_attempts

    # Immediately mark as RESEARCHING (concurrency guard: UI shows spinner, prevents re-trigger).
    _set_init_status(TopicStatus.RESEARCHING, error_message=None, init_attempts=topic.init_attempts)

    logger.info("Initializing knowledge for topic '%s' (id=%d)", topic.name, topic_id)

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
            # OVH-001: during a NEW-topic re-init (init_attempts>0) every prior
            # article is already stored, so a feed with no fresh entries yields an
            # empty fetch. That is not a failure — keep waiting in NEW for a later
            # cycle. Only the very first attempt (init_attempts==0) with no articles
            # at all is a genuine initialization error.
            if topic.init_attempts > 0:
                _set_init_status(TopicStatus.NEW, error_message=None, init_attempts=topic.init_attempts)
                logger.info(
                    "Topic '%s': no new articles on re-init (attempt %d) — staying NEW",
                    topic.name,
                    topic.init_attempts,
                )
                return
            _set_init_status(
                TopicStatus.ERROR,
                error_message="No articles found during initialization",
                init_attempts=topic.init_attempts,
            )
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
            next_attempts = topic.init_attempts + 1
            _set_init_status(TopicStatus.NEW, error_message=None, init_attempts=next_attempts)
            logger.info(
                "Knowledge for topic '%s' insufficient — retry %d/%d, back to NEW",
                topic.name,
                next_attempts,
                MAX_INIT_ATTEMPTS,
            )
            return

        # Either knowledge is sufficient, or attempts are exhausted: go READY.
        _set_init_status(TopicStatus.READY, error_message=None, init_attempts=0)

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
        _set_init_status(TopicStatus.ERROR, error_message=str(exc), init_attempts=topic.init_attempts)
