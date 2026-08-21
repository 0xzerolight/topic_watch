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

from app.crud import get_heartbeat_latch_raw, list_recent_check_stage_errors
from app.models import Topic, is_internal_failure, is_source_failure

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HeartbeatDecision:
    """A heartbeat message to dispatch, tied to the state it was decided from.

    ``head_check_id`` is the newest check this decision saw: the latch UPDATE is
    fenced to it, so a decision computed from check N never lands once check N+1
    exists (AUG-131). ``latch_value`` is the raw latch a recovery clears, which is
    what the alert intents for that outage were stamped with — it is how the
    recovery finds the targets that actually received the alert.
    """

    kind: Literal["alert", "recovered"]
    title: str
    body: str
    head_check_id: int
    latch_value: str | None = None


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
) -> HeartbeatDecision | None:
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
    # The head is the newest row under the canonical order, internal failures
    # included: it is what the latch write is fenced against, not what the streak
    # is counted from.
    head_check_id = recent[0][0]
    # Checks that broke inside the pipeline never reached the sources, so they are
    # dropped rather than counted either way: keeping them would let a run of
    # locked-database failures either announce a source outage or claim a recovery
    # nobody observed (AUG-133).
    observed = [stage_error for _, stage_error in recent if not is_internal_failure(stage_error)]
    if not observed:
        return None

    streak = _leading_failures(observed)
    # Read raw, not through the hydrated model: a corrupt cell must read as "set"
    # here exactly as it does in the latch SQL's IS NULL guard (AUG-144).
    latch = get_heartbeat_latch_raw(conn, topic.id)

    if streak >= threshold and latch is None:
        logger.warning(
            "Silence Heartbeat: topic '%s' has had %d consecutive source-failing check(s)",
            topic.name,
            streak,
        )
        title, body = _format_alert(topic, streak, observed[0])
        return HeartbeatDecision(kind="alert", title=title, body=body, head_check_id=head_check_id)

    if streak == 0 and latch is not None:
        logger.info("Silence Heartbeat: topic '%s' sources recovered", topic.name)
        title, body = _format_recovery(topic)
        return HeartbeatDecision(
            kind="recovered",
            title=title,
            body=body,
            head_check_id=head_check_id,
            latch_value=latch,
        )

    return None
