"""APScheduler integration for periodic topic checking.

Uses APScheduler 3.x AsyncIOScheduler to run check cycles within
an asyncio event loop. Designed to integrate with FastAPI's event
loop in Session 5.
"""

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.checker import initialize_new_topic
from app.config import Settings
from app.crud import (
    claim_new_topic_for_init,
    delete_old_articles,
    get_new_topics,
    recover_stuck_researching,
)
from app.database import get_db
from app.models import TopicStatus
from app.web.state import _checking_state

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

_scheduler: AsyncIOScheduler | None = None

# Cron maintenance (VACUUM, article cleanup) tolerates running hours late so a
# slept/woken host still catches up missed runs instead of skipping them (OVH-029).
_MAINTENANCE_MISFIRE_GRACE_SECONDS = 6 * 60 * 60


def _resolve_settings(app: "FastAPI | None", fallback: Settings) -> Settings:
    """Return the live settings for this tick.

    When the scheduler is wired to the running app, read ``app.state.settings`` so
    in-place ``/settings`` edits take effect on the next tick without a restart
    (OVH-015/036). When no app is wired (e.g. unit tests calling ``start_scheduler``
    with settings directly), fall back to the settings bound at start.
    """
    if app is not None:
        live = getattr(app.state, "settings", None)
        if isinstance(live, Settings):
            return live
    return fallback


def _resolve_db_path(app: "FastAPI | None", fallback: Path | None) -> Path | None:
    """Return the live db_path for this tick (mirrors ``_resolve_settings``)."""
    if app is not None:
        live = getattr(app.state, "db_path", None)
        if isinstance(live, Path):
            return live
    return fallback


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

    Atomically claims the topic (NEW -> RESEARCHING) before the long fetch+LLM
    work so a same-minute web Retry click — or a second initializer — can never
    double-initialize the same topic: only the caller whose conditional UPDATE
    matched a still-NEW row proceeds (OVH-032).
    """
    try:
        with get_db(db_path) as conn:
            new_topics = get_new_topics(conn, limit=1)
            if not new_topics:
                return
            topic = new_topics[0]
            if topic.id is None:
                return
            topic_id = topic.id

            # In-process guard: shares the same slot the web background init
            # (_run_init) holds, so a same-process Retry click and this tick
            # can't both run init. Skip rather than queue behind it.
            if not await _checking_state.start_check(topic_id):
                logger.debug("NEW topic init: topic '%s' already being initialized; skipping", topic.name)
                return
            try:
                # Cross-process atomic claim (NEW -> RESEARCHING): only the caller
                # whose conditional UPDATE matched a still-NEW row proceeds (OVH-032).
                if not claim_new_topic_for_init(conn, topic_id):
                    logger.debug(
                        "NEW topic init: topic '%s' no longer NEW (claimed elsewhere); skipping",
                        topic.name,
                    )
                    return
                # Reflect the won claim in the in-memory snapshot before init runs.
                topic.status = TopicStatus.RESEARCHING
                await initialize_new_topic(topic, conn, settings)
            finally:
                await _checking_state.finish_check(topic_id)
    except Exception:
        logger.error("NEW topic initialization failed", exc_info=True)


async def _run_check_cycle(settings: Settings, db_path: Path | None = None) -> None:
    """Run one check cycle, delegating to the unified ``check_all_topics``.

    The check loop (per-topic short-lived connections, both retry queues) lives
    in ``app.checker.check_all_topics`` so the CLI, web layer, and scheduler all
    share one implementation. This wrapper keeps the scheduler's stable name and
    signature for existing tests/imports.
    """
    from app.checker import check_all_topics

    await check_all_topics(settings, db_path)


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


async def _tick_check(settings: Settings, db_path: Path | None, app: "FastAPI | None") -> None:
    """Minute-tick job: run a check cycle using settings live from app.state (OVH-015/036)."""
    await _scheduled_check(_resolve_settings(app, settings), _resolve_db_path(app, db_path))


async def _tick_recover(timeout_minutes: int, db_path: Path | None, app: "FastAPI | None") -> None:
    """Recovery job: read the live db_path from app.state at tick time."""
    await _recover_stuck(timeout_minutes=timeout_minutes, db_path=_resolve_db_path(app, db_path))


async def _tick_vacuum(db_path: Path | None, app: "FastAPI | None") -> None:
    """VACUUM job: read the live db_path from app.state at tick time."""
    await _vacuum_db(_resolve_db_path(app, db_path))


async def _tick_cleanup(settings: Settings, db_path: Path | None, app: "FastAPI | None") -> None:
    """Cleanup job: read live settings (retention) and db_path from app.state at tick time."""
    await _cleanup_old_articles(_resolve_settings(app, settings), _resolve_db_path(app, db_path))


def start_scheduler(
    settings: Settings,
    db_path: Path | None = None,
    app: "FastAPI | None" = None,
) -> AsyncIOScheduler:
    """Create and start the background scheduler (idempotent, single owner).

    Schedules check_all_topics to run every minute. If a scheduler is already
    running it is shut down first, so a second call (lifespan, setup, or a future
    reschedule) never orphans a live scheduler (OVH-067/125). All start/stop goes
    through this guarded entry point and ``stop_scheduler``.

    Args:
        settings: Application settings, used as the fallback when no app is wired.
        db_path: Optional database path override for testing.
        app: Optional FastAPI app; when given, jobs read settings/db_path from
            ``app.state`` at tick time so ``/settings`` edits take effect without a
            restart (OVH-015/036).

    Returns:
        The running AsyncIOScheduler instance.
    """
    global _scheduler

    # Idempotent: never overwrite a live scheduler — shut it down first (OVH-067/125).
    if _scheduler is not None and _scheduler.running:
        logger.warning("start_scheduler called while a scheduler is running; restarting cleanly")
        stop_scheduler()

    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        _tick_check,
        "interval",
        minutes=1,
        args=[settings, db_path, app],
        id="check_all_topics",
        name="Check due topics for updates",
        replace_existing=True,
        coalesce=True,
        max_instances=1,
        misfire_grace_time=settings.scheduler_misfire_grace_time,
        jitter=settings.scheduler_jitter_seconds,
    )
    scheduler.add_job(
        _tick_recover,
        "interval",
        minutes=5,
        kwargs={"timeout_minutes": 15, "db_path": db_path, "app": app},
        id="recover_stuck_researching",
        name="Recover stuck researching topics",
        replace_existing=True,
        coalesce=True,
        max_instances=1,
        misfire_grace_time=settings.scheduler_misfire_grace_time,
    )
    scheduler.add_job(
        _tick_vacuum,
        "cron",
        day_of_week="sun",
        hour=3,
        args=[db_path, app],
        id="vacuum_db",
        name="Weekly database VACUUM",
        coalesce=True,
        max_instances=1,
        misfire_grace_time=_MAINTENANCE_MISFIRE_GRACE_SECONDS,
    )
    scheduler.add_job(
        _tick_cleanup,
        "cron",
        hour=4,
        args=[settings, db_path, app],
        id="cleanup_old_articles",
        name="Daily article cleanup",
        coalesce=True,
        max_instances=1,
        misfire_grace_time=_MAINTENANCE_MISFIRE_GRACE_SECONDS,
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
    """Stop the background scheduler; in-flight coroutine jobs are CANCELLED, not drained.

    AsyncIOScheduler's executor cancels any running coroutine job mid-await on shutdown
    (``wait=True`` waits for the executor's threadpool, not for cancelled coroutines to
    finish naturally). A topic mid-initialization has already committed status=RESEARCHING
    before its first long await, so cancellation can leave it stuck in RESEARCHING. The
    stuck-RESEARCHING recovery — at startup (main.py) and via the periodic recover job — is
    the safety net for that, not a graceful drain here.
    """
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=True)
        logger.info("Scheduler stopped")
    _scheduler = None
