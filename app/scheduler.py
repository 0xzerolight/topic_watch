"""APScheduler integration for periodic topic checking.

Uses APScheduler 3.x AsyncIOScheduler to run check cycles within
an asyncio event loop. Designed to integrate with FastAPI's event
loop in Session 5.
"""

import logging
from pathlib import Path

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.checker import check_all_topics
from app.config import Settings
from app.crud import delete_old_articles, recover_stuck_researching
from app.database import get_db

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


async def _vacuum_db(db_path: Path | None = None) -> None:
    """Run VACUUM on the database to reclaim disk space."""
    try:
        with get_db(db_path) as conn:
            conn.execute("VACUUM")
            logger.info("Database VACUUM completed")
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


async def _scheduled_check(settings: Settings, db_path: Path | None = None) -> None:
    """Callback invoked by APScheduler on each interval.

    Creates a fresh database connection for each check cycle.
    """
    logger.debug("Scheduled check tick")
    try:
        with get_db(db_path) as conn:
            await check_all_topics(conn, settings)
    except Exception:
        logger.error("Scheduled check cycle failed", exc_info=True)


def start_scheduler(
    settings: Settings,
    db_path: Path | None = None,
) -> AsyncIOScheduler:
    """Create and start the background scheduler.

    Schedules check_all_topics to run at the configured interval.

    Args:
        settings: Application settings (provides check_interval_hours).
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
        "Scheduler started: ticking every minute (jitter=%ds), default interval %d hour(s)",
        settings.scheduler_jitter_seconds,
        settings.check_interval_hours,
    )
    return scheduler


def stop_scheduler() -> None:
    """Stop the background scheduler, waiting for running jobs to finish."""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=True)
        logger.info("Scheduler stopped")
    _scheduler = None
