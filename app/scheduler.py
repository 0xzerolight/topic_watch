"""APScheduler integration for periodic topic checking.

Uses APScheduler 3.x AsyncIOScheduler to run check cycles within
an asyncio event loop. Designed to integrate with FastAPI's event
loop in Session 5.
"""

import asyncio
import logging
from pathlib import Path

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.checker import (
    check_topic,
    initialize_new_topic,
    retry_pending_notifications,
)
from app.config import Settings
from app.crud import (
    delete_old_articles,
    get_new_topics,
    get_topics_due_for_check,
    recover_stuck_researching,
)
from app.database import get_db
from app.webhooks import retry_pending_webhooks

logger = logging.getLogger(__name__)

_scheduler: AsyncIOScheduler | None = None


async def _cleanup_old_articles(settings: Settings, db_path: Path | None = None) -> None:
    """Delete articles older than the configured retention period."""
    try:
        with get_db(db_path) as conn:
            deleted = delete_old_articles(conn, settings.article_retention_days)
            if deleted:
                logger.info("Article cleanup: deleted %d old article(s)", deleted)
    except Exception:
        logger.warning("Article cleanup failed", exc_info=True)


def _vacuum_db_sync(db_path: Path | None = None) -> None:
    """Run VACUUM synchronously with its own short-lived connection.

    VACUUM rewrites the whole database file and can take a long time on a
    large DB; run it off the event loop (see _vacuum_db).
    """
    with get_db(db_path) as conn:
        conn.execute("VACUUM")
        logger.info("Database VACUUM completed")


async def _vacuum_db(db_path: Path | None = None) -> None:
    """Run VACUUM in a worker thread so it can't block the event loop."""
    try:
        await asyncio.to_thread(_vacuum_db_sync, db_path)
    except Exception:
        logger.warning("Database VACUUM failed", exc_info=True)


async def _recover_stuck(timeout_minutes: int = 15, db_path: Path | None = None) -> None:
    """Recover topics stuck in RESEARCHING status during runtime."""
    try:
        with get_db(db_path) as conn:
            count = recover_stuck_researching(conn, timeout_minutes)
            if count:
                logger.warning("Recovered %d stuck researching topic(s)", count)
    except Exception:
        logger.warning("Stuck topic recovery failed", exc_info=True)


async def _init_new_topics(settings: Settings, db_path: Path | None = None) -> None:
    """Initialize one NEW topic per tick for gradual knowledge building.

    OPML imports create topics with NEW status. This processes them
    one at a time (~1 per minute) to avoid hammering the LLM API.
    """
    try:
        with get_db(db_path) as conn:
            new_topics = get_new_topics(conn, limit=1)
            if new_topics:
                await initialize_new_topic(new_topics[0], conn, settings)
    except Exception:
        logger.error("NEW topic initialization failed", exc_info=True)


async def _run_check_cycle(settings: Settings, db_path: Path | None = None) -> None:
    """Run one check cycle with per-topic connection granularity.

    A single connection held for the whole cycle would stay open across every
    topic's HTTP + LLM awaits, blocking concurrent web requests (only
    busy_timeout would save them). Instead each phase uses its own short-lived
    connection that is committed and closed promptly:

      * retry passes (notifications + webhooks) — each manages its own
        short-lived connections internally (snapshot, send with none held,
        commit per item)
      * the due-topics query — one connection
      * each topic check — a fresh connection per topic
    """
    # Retry any failed deliveries from previous cycles. Each retry function
    # manages its own short-lived connections: it snapshots pending rows,
    # sends with NO connection held, and commits per item. Holding one shared
    # connection here would keep it open across every retry's HTTP POST.
    await retry_pending_notifications(settings=settings, db_path=db_path)
    await retry_pending_webhooks(settings=settings, db_path=db_path)

    # Snapshot the due topics, then release the connection before the long
    # per-topic HTTP/LLM work begins.
    with get_db(db_path) as conn:
        due_topics = get_topics_due_for_check(conn, settings.check_interval_minutes)

    if not due_topics:
        return

    logger.info("Starting check cycle for %d due topics", len(due_topics))

    new_info_count = 0
    for topic in due_topics:
        # Fresh, short-lived connection per topic so no connection is held
        # across that topic's HTTP/LLM awaits.
        try:
            with get_db(db_path) as conn:
                result = await check_topic(topic, conn, settings)
            if result.has_new_info:
                new_info_count += 1
        except Exception:
            logger.error("Unexpected error checking topic '%s'", topic.name, exc_info=True)

    logger.info(
        "Check cycle complete: %d topics checked, %d with new info",
        len(due_topics),
        new_info_count,
    )


async def _scheduled_check(settings: Settings, db_path: Path | None = None) -> None:
    """Callback invoked by APScheduler on each interval.

    Uses per-topic short-lived connections (see _run_check_cycle) so no
    connection is held across the long HTTP/LLM awaits of a full cycle.
    Also initializes one NEW topic per tick for gradual OPML import processing.
    """
    logger.debug("Scheduled check tick")
    try:
        await _run_check_cycle(settings, db_path)
    except Exception:
        logger.error("Scheduled check cycle failed", exc_info=True)

    # Process one NEW topic per tick (separate connection for isolation)
    await _init_new_topics(settings, db_path)


def start_scheduler(
    settings: Settings,
    db_path: Path | None = None,
) -> AsyncIOScheduler:
    """Create and start the background scheduler.

    Schedules check_all_topics to run at the configured interval.

    Args:
        settings: Application settings (provides check_interval).
        db_path: Optional database path override for testing.

    Returns:
        The running AsyncIOScheduler instance.
    """
    global _scheduler

    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        _scheduled_check,
        "interval",
        minutes=1,
        args=[settings, db_path],
        id="check_all_topics",
        name="Check due topics for updates",
        replace_existing=True,
        coalesce=True,
        max_instances=1,
        misfire_grace_time=settings.scheduler_misfire_grace_time,
        jitter=settings.scheduler_jitter_seconds,
    )
    scheduler.add_job(
        _recover_stuck,
        "interval",
        minutes=5,
        kwargs={"timeout_minutes": 15, "db_path": db_path},
        id="recover_stuck_researching",
        name="Recover stuck researching topics",
        replace_existing=True,
        coalesce=True,
        max_instances=1,
        misfire_grace_time=settings.scheduler_misfire_grace_time,
    )
    scheduler.add_job(
        _vacuum_db,
        "cron",
        day_of_week="sun",
        hour=3,
        args=[db_path],
        id="vacuum_db",
        name="Weekly database VACUUM",
        coalesce=True,
        max_instances=1,
    )
    scheduler.add_job(
        _cleanup_old_articles,
        "cron",
        hour=4,
        args=[settings, db_path],
        id="cleanup_old_articles",
        name="Daily article cleanup",
        coalesce=True,
        max_instances=1,
    )
    scheduler.start()
    _scheduler = scheduler

    logger.info(
        "Scheduler started: ticking every minute (jitter=%ds), default interval %s",
        settings.scheduler_jitter_seconds,
        settings.check_interval,
    )
    return scheduler


def stop_scheduler() -> None:
    """Stop the background scheduler, waiting for running jobs to finish."""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=True)
        logger.info("Scheduler stopped")
    _scheduler = None
