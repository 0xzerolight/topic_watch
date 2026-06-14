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
from app.crud import get_topic, update_topic
from app.models import TopicStatus
from app.web.state import _checking_state

logger = logging.getLogger(__name__)

_INIT_TIMEOUT_SECONDS = 600  # 10 minutes
_CHECK_ALL_TIMEOUT_SECONDS = 1800  # 30 minutes


async def _run_init(topic_id: int, settings: Settings, db_path: Path | None = None) -> None:
    """Background task: fetch articles and build initial knowledge state.

    Creates its own database connection since the request connection
    is closed by the time this runs. Delegates to initialize_new_topic()
    for the actual init logic.
    """
    from app.checker import initialize_new_topic
    from app.database import get_db

    if not await _checking_state.start_check(topic_id):
        logger.info("Init background task: topic %d already being initialized, skipping", topic_id)
        return

    try:
        with get_db(db_path) as conn:
            topic = get_topic(conn, topic_id)
            if topic is None:
                logger.error("Init background task: topic %d not found", topic_id)
                return

            try:
                await asyncio.wait_for(initialize_new_topic(topic, conn, settings), timeout=_INIT_TIMEOUT_SECONDS)
            except TimeoutError:
                logger.error(
                    "Init timed out for topic '%s' after %d seconds",
                    topic.name,
                    _INIT_TIMEOUT_SECONDS,
                )
                topic.status = TopicStatus.ERROR
                topic.status_changed_at = datetime.now(UTC)
                topic.error_message = "Research timed out. Click Retry."
                update_topic(conn, topic)
    finally:
        await _checking_state.finish_check(topic_id)


async def _run_single_check(topic_id: int, settings: Settings, db_path: Path | None = None) -> None:
    """Background task: check a single topic by ID."""
    from app.database import get_db

    try:
        with get_db(db_path) as conn:
            topic = get_topic(conn, topic_id)
            if topic:
                await check_topic(topic, conn, settings)
    except Exception:
        logger.error("Background check failed for topic %d", topic_id, exc_info=True)


async def _run_check_all(settings: Settings, db_path: Path | None = None) -> None:
    """Background task: check all topics for new information."""
    try:
        try:
            await asyncio.wait_for(check_all_topics(settings, db_path), timeout=_CHECK_ALL_TIMEOUT_SECONDS)
        except TimeoutError:
            logger.error("Check all timed out after %d seconds", _CHECK_ALL_TIMEOUT_SECONDS)
    except Exception:
        logger.error("Check all background task failed", exc_info=True)
    finally:
        await _checking_state.finish_check_all()
