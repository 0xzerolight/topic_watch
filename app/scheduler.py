"""APScheduler integration for periodic topic checking.

Uses APScheduler 3.x AsyncIOScheduler to run check cycles within
an asyncio event loop. Designed to integrate with FastAPI's event
loop in Session 5.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

from apscheduler.events import EVENT_JOB_MAX_INSTANCES, EVENT_JOB_MISSED
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app import clock
from app.check_context import check_id_var, request_id_var
from app.checker import initialize_new_topic
from app.config import Settings
from app.crud import (
    DELIVERY_INTENT_RETENTION_DAYS,
    claim_new_topic_for_init,
    delete_old_articles,
    delete_old_delivery_intents,
    get_new_topics,
    recover_stuck_researching,
    update_topic_init_status,
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

# End-to-end bound on one gradual-init tick, matching the web background task's
# own bound (``app.web.routers.background._INIT_TIMEOUT_SECONDS``). Without it a
# scheduler init could outlive the 15-minute stuck-recovery window (AUG-139).
_INIT_TIMEOUT_SECONDS = 600

# How long a completed check cycle stays "fresh" for the health surface. The job
# ticks every minute, so five is several missed ticks — long enough not to flap
# during one slow cycle, short enough that a wedged scheduler is visible while it
# still matters.
_CYCLE_STALE_AFTER_SECONDS = 300

# The lock file naming the one process that owns the scheduler, beside the
# database it schedules work against.
_LEASE_FILENAME = "scheduler.lock"


@dataclass
class _JobHealth:
    """The outcome record for one scheduled job.

    Every scheduled callback catches its own failures and returns normally, so a
    process whose checks, recovery, cleanup and maintenance had all been failing
    for days still answered ``/health`` with ``ok`` and nothing but log lines to
    say otherwise (TW-AUD-008). The outcome is recorded here instead of only
    logged, and the health endpoint reports it.

    ``last_success_monotonic`` is what freshness is computed from: elapsed time,
    never a difference of wall-clock readings (wave-A clock policy). The two
    ``datetime`` fields exist to be displayed.
    """

    last_success_at: datetime | None = None
    last_success_monotonic: float | None = None
    last_error_at: datetime | None = None
    last_error: str | None = None
    consecutive_failures: int = 0
    missed_runs: int = 0
    """Ticks APScheduler dropped: past their misfire grace, or already running."""

    def record_success(self) -> None:
        self.last_success_at = datetime.now(UTC)
        self.last_success_monotonic = clock.monotonic_now()
        self.consecutive_failures = 0

    def record_failure(self, error: str) -> None:
        self.last_error_at = datetime.now(UTC)
        self.last_error = error
        self.consecutive_failures += 1


@dataclass
class _SchedulerHealth:
    """Per-job outcomes for the running scheduler; reset when it restarts."""

    jobs: dict[str, _JobHealth] = field(default_factory=dict)

    def job(self, job_id: str) -> _JobHealth:
        return self.jobs.setdefault(job_id, _JobHealth())

    def reset(self) -> None:
        self.jobs.clear()


_health = _SchedulerHealth()

CHECK_JOB_ID = "check_all_topics"
RECOVER_JOB_ID = "recover_stuck_researching"
VACUUM_JOB_ID = "vacuum_db"
CLEANUP_JOB_ID = "cleanup_old_articles"
INIT_JOB_ID = "init_new_topics"


def _summarize(exc: BaseException) -> str:
    """One short line naming a failure, safe to put in a health payload."""
    detail = str(exc).strip().splitlines()
    return f"{type(exc).__name__}: {detail[0][:200]}" if detail else type(exc).__name__


def scheduler_health() -> dict:
    """The background-monitoring half of the health endpoint (TW-AUD-008).

    ``monitoring`` is the one-word verdict a human or a probe reads:

    * ``stopped`` — no scheduler in this process (setup mode, or another worker
      owns the lease).
    * ``starting`` — running, but no cycle has completed yet.
    * ``failing`` — the last check cycle raised.
    * ``stale`` — the last successful cycle is older than the freshness window.
    * ``ok``.

    Deliberately does NOT change the endpoint's status code: liveness ("this
    process answers") and monitoring freshness are different questions, and a
    container healthcheck must not restart a perfectly serviceable web process
    because a feed provider is down.
    """
    running = _scheduler is not None and _scheduler.running
    cycle = _health.jobs.get(CHECK_JOB_ID)
    if not running:
        monitoring = "stopped"
    elif cycle is None or cycle.last_success_monotonic is None:
        monitoring = "failing" if cycle and cycle.consecutive_failures else "starting"
    elif cycle.consecutive_failures:
        monitoring = "failing"
    elif clock.monotonic_now() - cycle.last_success_monotonic > _CYCLE_STALE_AFTER_SECONDS:
        monitoring = "stale"
    else:
        monitoring = "ok"

    return {
        "running": running,
        "monitoring": monitoring,
        "last_cycle_at": cycle.last_success_at.isoformat() if cycle and cycle.last_success_at else None,
        "jobs": {
            job_id: {
                "last_success_at": job.last_success_at.isoformat() if job.last_success_at else None,
                "last_error_at": job.last_error_at.isoformat() if job.last_error_at else None,
                "last_error": job.last_error,
                "consecutive_failures": job.consecutive_failures,
                "missed_runs": job.missed_runs,
            }
            for job_id, job in sorted(_health.jobs.items())
        },
    }


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
        _health.job(CLEANUP_JOB_ID).record_success()
    except Exception as exc:
        logger.warning("Article cleanup failed", exc_info=True)
        _health.job(CLEANUP_JOB_ID).record_failure(_summarize(exc))


async def _cleanup_delivery_intents(db_path: Path | None = None) -> None:
    """Prune finished delivery intents older than the retention constant.

    Delivered intents are kept as the ledger the dashboard's "last notified"
    reads (AUG-153), so something has to age them out. It rides the existing
    daily maintenance tick rather than adding a job, and the window is a
    constant rather than a setting — nobody tunes how long a delivery receipt
    lives.
    """
    try:
        with get_db(db_path) as conn:
            removed = delete_old_delivery_intents(conn, DELIVERY_INTENT_RETENTION_DAYS)
            if removed:
                logger.info("Delivery ledger cleanup: removed %d finished intent(s)", removed)
    except Exception as exc:
        logger.warning("Delivery ledger cleanup failed", exc_info=True)
        _health.job(CLEANUP_JOB_ID).record_failure(_summarize(exc))


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
        _health.job(VACUUM_JOB_ID).record_success()
    except Exception as exc:
        logger.warning("Database VACUUM failed", exc_info=True)
        _health.job(VACUUM_JOB_ID).record_failure(_summarize(exc))


async def _recover_stuck(timeout_minutes: int = 15, db_path: Path | None = None) -> None:
    """Recover topics stuck in RESEARCHING status during runtime."""
    try:
        with get_db(db_path) as conn:
            count = recover_stuck_researching(conn, timeout_minutes)
            if count:
                logger.warning("Recovered %d stuck researching topic(s)", count)
        _health.job(RECOVER_JOB_ID).record_success()
    except Exception as exc:
        logger.warning("Stuck topic recovery failed", exc_info=True)
        _health.job(RECOVER_JOB_ID).record_failure(_summarize(exc))


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
        owner = await _checking_state.start_check(topic_id)
        if owner is None:
            logger.debug("NEW topic init: topic '%s' already being initialized; skipping", topic.name)
            return
        try:
            # Cross-process atomic claim (NEW -> RESEARCHING): only the caller
            # whose conditional UPDATE matched a still-NEW row proceeds (OVH-032).
            with get_db(db_path) as conn:
                claimed = claim_new_topic_for_init(conn, topic_id)
                conn.commit()
            if not claimed:
                logger.debug(
                    "NEW topic init: topic '%s' no longer NEW (claimed elsewhere); skipping",
                    topic.name,
                )
                return
            # Reflect the won claim in the in-memory snapshot before init runs.
            topic.status = TopicStatus.RESEARCHING
            # No connection is held across the init's fetch + LLM awaits: it opens
            # its own short-lived one per phase (AUG-136).
            #
            # Bounded like the web path (AUG-139): an unbounded init could still be
            # running when stuck recovery declares it timed out at 15 minutes, which
            # is how a live initializer and recovery ended up racing for the terminal
            # status. The claim itself is fenced too, so the loser writes nothing.
            try:
                await asyncio.wait_for(
                    initialize_new_topic(topic, settings, db_path=db_path, claimed=True),
                    timeout=_INIT_TIMEOUT_SECONDS,
                )
            except TimeoutError:
                logger.error(
                    "NEW topic init timed out for '%s' after %d seconds",
                    topic.name,
                    _INIT_TIMEOUT_SECONDS,
                )
                # The cancelled initializer already handed its claim back (ERROR);
                # this only names the reason, fenced so it cannot land on a topic
                # somebody re-claimed in between.
                with get_db(db_path) as conn:
                    landed = update_topic_init_status(
                        conn,
                        topic_id,
                        status=TopicStatus.ERROR,
                        status_changed_at=datetime.now(UTC),
                        error_message="Research timed out. Click Retry.",
                        init_attempts=topic.init_attempts,
                        expected_status=TopicStatus.ERROR,
                        generation=topic.generation,
                    )
                    conn.commit()
                if not landed:
                    logger.warning(
                        "Timeout reason for topic %d not recorded: it left ERROR or was replaced",
                        topic_id,
                    )
        finally:
            await _checking_state.finish_check(topic_id, owner)
        _health.job(INIT_JOB_ID).record_success()
    except Exception as exc:
        logger.error("NEW topic initialization failed", exc_info=True)
        _health.job(INIT_JOB_ID).record_failure(_summarize(exc))


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
        _health.job(CHECK_JOB_ID).record_success()
    except Exception as exc:
        logger.error("Scheduled check cycle failed", exc_info=True)
        # Recorded, not just logged: a cycle that fails every minute used to be
        # invisible to anything but the log stream (TW-AUD-008).
        _health.job(CHECK_JOB_ID).record_failure(_summarize(exc))

    # Process one NEW topic per tick (separate connection for isolation)
    await _init_new_topics(settings, db_path)


def reconfigure_scheduler(settings: Settings) -> bool:
    """Apply live edits of the two APScheduler-owned settings. True if anything moved.

    ``scheduler_jitter_seconds`` lives in the check job's trigger and
    ``scheduler_misfire_grace_time`` in the job's own properties; both are copied
    at ``add_job`` time. Saving them only replaced ``app.state.settings``, so the
    Settings page reported values as active that would not take effect until a
    restart (AUG-023). Called from the minute tick — the one place that runs
    after every save no matter who made it (web, API, CLI) — so a new value is
    live within a tick without the settings routes needing to know the scheduler
    exists.
    """
    scheduler = _scheduler
    if scheduler is None or not scheduler.running:
        return False

    changed = False
    check_job = scheduler.get_job(CHECK_JOB_ID)
    if check_job is not None and getattr(check_job.trigger, "jitter", None) != settings.scheduler_jitter_seconds:
        # A trigger's jitter cannot be edited in place; rebuild the same
        # one-minute interval trigger carrying the new value.
        scheduler.reschedule_job(
            CHECK_JOB_ID,
            trigger="interval",
            minutes=1,
            jitter=settings.scheduler_jitter_seconds,
        )
        changed = True

    # The maintenance jobs keep their own generous grace (OVH-029) and are not
    # what this setting names.
    for job_id in (CHECK_JOB_ID, RECOVER_JOB_ID):
        job = scheduler.get_job(job_id)
        if job is not None and job.misfire_grace_time != settings.scheduler_misfire_grace_time:
            scheduler.modify_job(job_id, misfire_grace_time=settings.scheduler_misfire_grace_time)
            changed = True

    if changed:
        logger.info(
            "Scheduler reconfigured live: jitter=%ss misfire_grace=%ss",
            settings.scheduler_jitter_seconds,
            settings.scheduler_misfire_grace_time,
        )
    return changed


async def _tick_check(settings: Settings, db_path: Path | None, app: "FastAPI | None") -> None:
    """Minute-tick job: run a check cycle using settings live from app.state (OVH-015/036)."""
    live = _resolve_settings(app, settings)
    try:
        reconfigure_scheduler(live)
    except Exception:
        # Never let a reschedule failure cost a check cycle.
        logger.warning("Live scheduler reconfiguration failed", exc_info=True)
    await _scheduled_check(live, _resolve_db_path(app, db_path))


async def _tick_recover(timeout_minutes: int, db_path: Path | None, app: "FastAPI | None") -> None:
    """Recovery job: read the live db_path from app.state at tick time."""
    await _recover_stuck(timeout_minutes=timeout_minutes, db_path=_resolve_db_path(app, db_path))


async def _tick_vacuum(db_path: Path | None, app: "FastAPI | None") -> None:
    """VACUUM job: read the live db_path from app.state at tick time."""
    await _vacuum_db(_resolve_db_path(app, db_path))


async def _tick_cleanup(settings: Settings, db_path: Path | None, app: "FastAPI | None") -> None:
    """Cleanup job: read live settings (retention) and db_path from app.state at tick time."""
    live_db_path = _resolve_db_path(app, db_path)
    await _cleanup_old_articles(_resolve_settings(app, settings), live_db_path)
    await _cleanup_delivery_intents(live_db_path)


def _record_missed_run(event) -> None:
    """Count a tick APScheduler dropped (misfire grace exceeded, or still running)."""
    _health.job(event.job_id).missed_runs += 1
    logger.warning("Scheduled job %s missed a run", event.job_id)


_lease: BinaryIO | None = None


def _acquire_scheduler_lease(db_path: Path | None) -> bool:
    """Claim the right to be this deployment's ONE scheduler process.

    Every worker's lifespan starts a scheduler, while ``max_instances``, the
    in-process check guards and due-topic selection are all process-local — so
    running the app with several web workers gave each of them its own minute
    tick. Three workers record three failure rows for one logical minute, and the
    heartbeat counts rows, so a threshold of three fires an outage alert after a
    single interval (AUG-024). One worker holds an advisory lock on a file beside
    the database; the others serve requests without a scheduler.

    The lock is held by the kernel against the open file, so a killed process
    releases it with no stale state to clean up. Platforms without ``fcntl``
    (Windows) keep the previously documented one-worker rule: nothing is
    enforced, and nothing changes for them.
    """
    global _lease
    if db_path is None:
        # No state directory to coordinate through: unit tests and one-shot CLI
        # paths, neither of which runs a second worker.
        return True
    try:
        import fcntl
    except ImportError:  # pragma: no cover - POSIX-only enforcement
        return True

    handle: BinaryIO | None = None
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        handle = (db_path.parent / _LEASE_FILENAME).open("a+b")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        if handle is not None:
            handle.close()
        return False
    _lease = handle
    return True


def _release_scheduler_lease() -> None:
    """Drop the lease so another worker (or a restart in this one) can take it."""
    global _lease
    if _lease is not None:
        _lease.close()  # closing the file releases the flock
        _lease = None


def start_scheduler(
    settings: Settings,
    db_path: Path | None = None,
    app: "FastAPI | None" = None,
) -> AsyncIOScheduler | None:
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
        The running AsyncIOScheduler, or None when another process holds the
        scheduler lease (see :func:`_acquire_scheduler_lease`).
    """
    global _scheduler

    # Idempotent: never overwrite a live scheduler — shut it down first (OVH-067/125).
    if _scheduler is not None and _scheduler.running:
        logger.warning("start_scheduler called while a scheduler is running; restarting cleanly")
        stop_scheduler()

    if not _acquire_scheduler_lease(db_path):
        logger.warning(
            "Another process already owns the scheduler for this database; "
            "this worker will serve requests without one. Topic Watch expects a single application worker."
        )
        return None

    _health.reset()
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        _tick_check,
        "interval",
        minutes=1,
        args=[settings, db_path, app],
        id=CHECK_JOB_ID,
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
        id=RECOVER_JOB_ID,
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
        id=VACUUM_JOB_ID,
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
        id=CLEANUP_JOB_ID,
        name="Daily article cleanup",
        coalesce=True,
        max_instances=1,
        misfire_grace_time=_MAINTENANCE_MISFIRE_GRACE_SECONDS,
    )
    # AsyncIOScheduler.start() schedules its wakeup via the event loop, which
    # copies whatever contextvars.Context is active right now — and every later
    # timer/job it chains from that wakeup keeps copying forward from there.
    # Called synchronously from the first-run setup POST, that context still
    # carries the request's request_id_var, so every scheduler/maintenance log
    # line was falsely attributed to that one setup request until the next
    # restart (AUG-272). Clear both correlation vars for just this call.
    # A tick APScheduler drops — past its misfire grace, or skipped because the
    # previous one is still running — is otherwise only an APScheduler log line
    # (TW-AUD-008).
    scheduler.add_listener(_record_missed_run, EVENT_JOB_MISSED | EVENT_JOB_MAX_INSTANCES)
    check_token = check_id_var.set(None)
    request_token = request_id_var.set(None)
    try:
        scheduler.start()
    finally:
        check_id_var.reset(check_token)
        request_id_var.reset(request_token)
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
    _release_scheduler_lease()
