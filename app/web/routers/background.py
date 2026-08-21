"""Background-task helpers for topic initialization and checks.

These run after the request connection is closed, so each opens its own
database connection. State is tracked via the shared ``_checking_state``.
"""

import asyncio
import logging
import math
from datetime import UTC, datetime
from pathlib import Path

from app.checker import CHECK_TIMEOUT_SECONDS, check_all_topics, check_topic
from app.config import Settings
from app.crud import get_topic, get_topics_due_for_check, update_topic_init_status
from app.models import TopicStatus
from app.web.state import _checking_state

logger = logging.getLogger(__name__)

_INIT_TIMEOUT_SECONDS = 600  # 10 minutes

# Fixed overhead of a cycle, on top of what the backlog itself needs: the due
# query and both retry drains.
_CHECK_ALL_TIMEOUT_SECONDS = 1800  # 30 minutes


def _check_all_deadline_seconds(due_count: int, concurrency: int) -> float:
    """The bound for one check-all: overhead plus what the backlog can legitimately take.

    One fixed 1800-second deadline was count-blind, so a perfectly healthy large
    backlog was cancelled partway through — at the default concurrency of 3, 500
    topics need only ~11 seconds each to exceed it — and cancellation propagates
    through the gather to topics that had not started (AUG-211). Each topic is
    itself bounded now (``CHECK_TIMEOUT_SECONDS``), so the worst legitimate cycle
    is one bounded wave per batch of ``concurrency`` topics; the deadline is that
    plus the fixed overhead, and stays a safety net for a wedged cycle rather
    than a cap on how much work is allowed.
    """
    waves = math.ceil(max(due_count, 0) / max(concurrency, 1))
    return _CHECK_ALL_TIMEOUT_SECONDS + waves * CHECK_TIMEOUT_SECONDS


def _due_topic_count(settings: Settings, db_path: Path | None) -> int:
    """How many topics this cycle has to get through, for the deadline above."""
    try:
        from app.database import get_db

        with get_db(db_path) as conn:
            return len(get_topics_due_for_check(conn, settings.check_interval_minutes))
    except Exception:
        # The cycle itself will report the real problem; fall back to the
        # overhead-only bound rather than refusing to run.
        logger.warning("Could not size the check-all backlog; using the base deadline", exc_info=True)
        return 0


async def _run_init(
    topic_id: int,
    settings: Settings,
    db_path: Path | None = None,
    owner: str | None = None,
    *,
    claimed: bool = False,
) -> None:
    """Background task: fetch articles and build initial knowledge state.

    Creates its own database connection since the request connection
    is closed by the time this runs. Delegates to initialize_new_topic()
    for the actual init logic.

    ``owner`` is the in-flight guard token the enqueueing handler already holds:
    the Retry handler takes ownership *before* it commits RESEARCHING, so a live
    check of the same topic makes it refuse visibly instead of leaving this task
    to discover the busy slot and exit silently, stranding the topic in
    RESEARCHING (AUG-137). Without a token this task takes the guard itself.
    ``claimed`` says the caller already made the durable RESEARCHING claim.
    """
    from app.checker import TopicInitRefused, initialize_new_topic
    from app.database import get_db

    held = owner
    if held is None:
        held = await _checking_state.start_check(topic_id)
        if held is None:
            logger.info("Init background task: topic %d already being initialized, skipping", topic_id)
            return

    try:
        with get_db(db_path) as conn:
            topic = get_topic(conn, topic_id)
        if topic is None:
            logger.error("Init background task: topic %d not found", topic_id)
            return

        try:
            await asyncio.wait_for(
                initialize_new_topic(topic, settings, db_path=db_path, claimed=claimed),
                timeout=_INIT_TIMEOUT_SECONDS,
            )
        except TopicInitRefused as exc:
            logger.info("Init background task: %s", exc)
        except TimeoutError:
            logger.error(
                "Init timed out for topic '%s' after %d seconds",
                topic.name,
                _INIT_TIMEOUT_SECONDS,
            )
            # ``wait_for`` cancelled the initializer, which hands its claim back on
            # the way out — so the row is already ERROR here and this write only
            # replaces the generic interruption message with the specific reason.
            # Fencing it to ERROR is what keeps it off a topic somebody has since
            # re-claimed (AUG-139), and a targeted write means it cannot restore
            # name/feeds/thresholds the user edited while the init ran (AUG-022).
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
                # A refused fence is the normal outcome of a race, not a bug — but
                # silently dropping a terminal write leaves nothing to read when a
                # topic ends up with a status nobody can account for.
                logger.warning(
                    "Timeout reason for topic %d not recorded: it left ERROR or was replaced",
                    topic_id,
                )
    finally:
        await _checking_state.finish_check(topic_id, held)


async def _run_single_check(topic_id: int, settings: Settings, db_path: Path | None = None) -> None:
    """Background task: check a single topic by ID.

    Authoritatively owns the per-topic ``_checking_state`` guard for every caller
    that enqueues it — the manual ``/check`` handler and bulk-check alike
    (OVH-033). It acquires ``start_check`` at entry and skips (no fetch/LLM/notify)
    when another check of the same topic is already in flight, then releases the
    guard in ``finally``. ``check_topic`` is called with ``guard=False`` because
    this task already holds the guard.

    Bounded like the other two background tasks: the guard it holds is evictable
    by ``clear_stale`` once the entry passes the handlers' staleness threshold, and
    eviction admits a second checker of the same topic.
    """
    from app.database import get_db

    owner = await _checking_state.start_check(topic_id)
    if owner is None:
        logger.info("Single check: topic %d already being checked; skipping", topic_id)
        return

    try:
        with get_db(db_path) as conn:
            topic = get_topic(conn, topic_id)
        if topic:
            await asyncio.wait_for(
                check_topic(topic, settings, db_path=db_path, guard=False),
                timeout=CHECK_TIMEOUT_SECONDS,
            )
    except TimeoutError:
        logger.error(
            "Check timed out for topic %d after %d seconds",
            topic_id,
            CHECK_TIMEOUT_SECONDS,
        )
    except Exception:
        logger.error("Background check failed for topic %d", topic_id, exc_info=True)
    finally:
        await _checking_state.finish_check(topic_id, owner)


async def _run_check_all(settings: Settings, db_path: Path | None = None, owner: str | None = None) -> None:
    """Background task: check all topics for new information.

    The web ``/check-all`` handler acquires the whole-cycle ``start_check_all``
    gate synchronously to decide whether to enqueue, so this task runs the cycle
    with ``guard=False`` and releases the gate — with the handler's owner token —
    in ``finally`` (OVH-034/AUG-264).
    """
    try:
        deadline = _check_all_deadline_seconds(_due_topic_count(settings, db_path), settings.topic_check_concurrency)
        try:
            await asyncio.wait_for(check_all_topics(settings, db_path, guard=False), timeout=deadline)
        except TimeoutError:
            logger.error("Check all timed out after %d seconds", deadline)
    except Exception:
        logger.error("Check all background task failed", exc_info=True)
    finally:
        if owner is not None:
            _checking_state.finish_check_all(owner)
