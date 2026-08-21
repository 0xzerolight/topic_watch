"""Background-task helpers for topic initialization and checks.

These run after the request connection is closed, so each opens its own
database connection. State is tracked via the shared ``_checking_state``.
"""

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path

from app.checker import check_all_topics, check_topic
from app.config import Settings
from app.crud import get_topic, update_topic_init_status
from app.models import TopicStatus
from app.web.state import _checking_state

logger = logging.getLogger(__name__)

_INIT_TIMEOUT_SECONDS = 600  # 10 minutes
_CHECK_ALL_TIMEOUT_SECONDS = 1800  # 30 minutes
# Bounds one single-topic check. Deliberately under the 600-second staleness
# threshold the check handlers pass to ``clear_stale``: eviction frees the slot
# for a second checker, so an entry old enough to be evicted must already have
# been released by its owner. Unbounded, this task outlived its own guard and one
# finding was committed and delivered twice (AUG-264).
_CHECK_TIMEOUT_SECONDS = 540  # 9 minutes


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
                timeout=_CHECK_TIMEOUT_SECONDS,
            )
    except TimeoutError:
        logger.error(
            "Check timed out for topic %d after %d seconds",
            topic_id,
            _CHECK_TIMEOUT_SECONDS,
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
        try:
            await asyncio.wait_for(check_all_topics(settings, db_path, guard=False), timeout=_CHECK_ALL_TIMEOUT_SECONDS)
        except TimeoutError:
            logger.error("Check all timed out after %d seconds", _CHECK_ALL_TIMEOUT_SECONDS)
    except Exception:
        logger.error("Check all background task failed", exc_info=True)
    finally:
        if owner is not None:
            _checking_state.finish_check_all(owner)
