"""Silence Heartbeat: decide when a topic's sources have gone dark.

topic_watch's promise is "silence means nothing new". That only holds while the
sources actually work, so a run of checks in which no source produced usable
results must be announced once — and its recovery announced once — instead of
being indistinguishable from a genuinely quiet topic.

Pure decision layer: this module reads the ``stage_error`` values the pipeline
already records and returns a message to send, or nothing. The checker owns the
latch write (``claim_heartbeat_alert`` / ``clear_heartbeat_alert``) and the
irreversible send, so the "announce once per outage" guarantee is a conditional
UPDATE rather than a property of this arithmetic.
"""

import logging
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from app.crud import list_recent_check_stage_errors
from app.models import Topic, is_source_failure

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HeartbeatAction:
    """A heartbeat message to dispatch."""

    kind: Literal["alert", "recovered"]
    title: str
    body: str


def _leading_failures(stage_errors: Sequence[str | None]) -> int:
    """Count leading source-failing entries in a newest-first sequence."""
    count = 0
    for stage_error in stage_errors:
        if not is_source_failure(stage_error):
            break
        count += 1
    return count


def _format_alert(topic: Topic, streak: int, last_error: str | None) -> tuple[str, str]:
    title = f"Topic Watch: {topic.name} (sources failing)"
    # "at least": the streak is counted over a bounded window, so a very long
    # outage is reported as the window size rather than its true length.
    parts = [f'No source returned results for "{topic.name}" in at least {streak} consecutive checks.']
    if last_error:
        parts.append(f"Last error: {last_error}")
    parts.extend(
        [
            "",
            "If several topics report this at once, check the shared cause first "
            "(API key, network) on the Feed Health page.",
        ]
    )
    return title, "\n".join(parts)


def _format_recovery(topic: Topic) -> tuple[str, str]:
    title = f"Topic Watch: {topic.name} (sources recovered)"
    return title, f'Sources for "{topic.name}" are returning results again.'


def evaluate_heartbeat(
    conn: sqlite3.Connection,
    topic: Topic,
    threshold: int,
) -> HeartbeatAction | None:
    """Decide whether this topic should raise or clear a Silence Heartbeat.

    Call this only after the current check has been recorded and committed: the
    streak is read back from the stored rows, so the just-recorded check must be
    the head of the run.

    Returns ``None`` whenever nothing should be sent — not yet at the threshold,
    an outage already announced (latch set), a healthy check with no announced
    outage behind it, or ``threshold <= 0`` (disabled).
    """
    if threshold <= 0 or topic.id is None:
        return None

    # One row past the threshold is all the DECISION needs; the wider floor exists
    # only so the message can report a realistic outage length. The count is
    # reported as "at least N" because it saturates at this window.
    recent = list_recent_check_stage_errors(conn, topic.id, limit=max(threshold + 1, 50))
    if not recent:
        return None

    streak = _leading_failures(recent)

    if streak >= threshold and topic.heartbeat_alerted_at is None:
        logger.warning(
            "Silence Heartbeat: topic '%s' has had %d consecutive source-failing check(s)",
            topic.name,
            streak,
        )
        title, body = _format_alert(topic, streak, recent[0])
        return HeartbeatAction(kind="alert", title=title, body=body)

    if streak == 0 and topic.heartbeat_alerted_at is not None:
        logger.info("Silence Heartbeat: topic '%s' sources recovered", topic.name)
        title, body = _format_recovery(topic)
        return HeartbeatAction(kind="recovered", title=title, body=body)

    return None
