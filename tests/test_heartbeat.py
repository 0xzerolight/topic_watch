"""Silence Heartbeat: streak evaluation and message copy."""

import sqlite3
from datetime import UTC, datetime, timedelta

from app.crud import claim_heartbeat_alert, create_check_result, create_topic, get_topic
from app.heartbeat import evaluate_heartbeat
from app.models import CheckResult, Topic, TopicStatus


def _topic(conn: sqlite3.Connection, **overrides) -> Topic:
    base = {"name": "Fusion", "description": "desc", "status": TopicStatus.READY}
    base.update(overrides)
    topic = create_topic(conn, Topic(**base))
    conn.commit()
    return topic


def _record(conn: sqlite3.Connection, topic_id: int, stage_error: str | None, minutes_ago: int) -> None:
    create_check_result(
        conn,
        CheckResult(
            topic_id=topic_id,
            checked_at=datetime.now(UTC) - timedelta(minutes=minutes_ago),
            stage_error=stage_error,
        ),
    )
    conn.commit()


def _fail_run(conn: sqlite3.Connection, topic_id: int, count: int, *, oldest_minutes: int = 120) -> None:
    for i in range(count):
        _record(conn, topic_id, "sources_failed: all feed source(s) failed (see logs)", oldest_minutes - i * 5)


def _latched(conn: sqlite3.Connection, topic: Topic) -> Topic:
    claim_heartbeat_alert(conn, topic.id, datetime.now(UTC))
    conn.commit()
    return get_topic(conn, topic.id)


class TestEvaluateHeartbeat:
    def test_below_threshold_is_silent(self, db_conn: sqlite3.Connection) -> None:
        topic = _topic(db_conn)
        _fail_run(db_conn, topic.id, 2)
        assert evaluate_heartbeat(db_conn, topic, threshold=3) is None

    def test_alert_at_the_threshold(self, db_conn: sqlite3.Connection) -> None:
        topic = _topic(db_conn)
        _fail_run(db_conn, topic.id, 3)
        action = evaluate_heartbeat(db_conn, topic, threshold=3)
        assert action is not None
        assert action.kind == "alert"
        assert "Fusion" in action.title
        assert "Fusion" in action.body
        assert "at least 3 consecutive checks" in action.body
        assert "sources_failed" in action.body

    def test_alerts_on_an_outage_already_in_progress(self, db_conn: sqlite3.Connection) -> None:
        """Upgrading into, or enabling mid-outage, must still announce it once."""
        topic = _topic(db_conn)
        _fail_run(db_conn, topic.id, 12, oldest_minutes=300)
        action = evaluate_heartbeat(db_conn, topic, threshold=3)
        assert action is not None and action.kind == "alert"
        assert "at least 12 consecutive checks" in action.body

    def test_latched_outage_does_not_re_alert(self, db_conn: sqlite3.Connection) -> None:
        topic = _topic(db_conn)
        _fail_run(db_conn, topic.id, 8)
        assert evaluate_heartbeat(db_conn, _latched(db_conn, topic), threshold=3) is None

    def test_mixed_source_failure_kinds_count_together(self, db_conn: sqlite3.Connection) -> None:
        topic = _topic(db_conn)
        _record(db_conn, topic.id, "sources_failed: x", 30)
        _record(db_conn, topic.id, "sources_unavailable: no source attempted (2 feed(s) in backoff)", 20)
        _record(db_conn, topic.id, "scrape_failed: TimeoutError: boom", 10)
        action = evaluate_heartbeat(db_conn, topic, threshold=3)
        assert action is not None and action.kind == "alert"

    def test_healthy_check_breaks_the_streak(self, db_conn: sqlite3.Connection) -> None:
        topic = _topic(db_conn)
        _fail_run(db_conn, topic.id, 3, oldest_minutes=60)
        _record(db_conn, topic.id, None, 30)
        _fail_run(db_conn, topic.id, 2, oldest_minutes=20)
        assert evaluate_heartbeat(db_conn, topic, threshold=3) is None

    def test_analysis_failure_is_not_a_source_failure(self, db_conn: sqlite3.Connection) -> None:
        """Articles arrived, so the sources are fine — an LLM failure is not source silence."""
        topic = _topic(db_conn)
        _fail_run(db_conn, topic.id, 3, oldest_minutes=60)
        _record(db_conn, topic.id, "analysis_failed: timeout", 10)
        assert evaluate_heartbeat(db_conn, topic, threshold=3) is None

    def test_recovery_after_an_announced_outage(self, db_conn: sqlite3.Connection) -> None:
        topic = _topic(db_conn)
        _fail_run(db_conn, topic.id, 4)
        latched = _latched(db_conn, topic)
        _record(db_conn, topic.id, None, 1)
        action = evaluate_heartbeat(db_conn, latched, threshold=3)
        assert action is not None
        assert action.kind == "recovered"
        assert "Fusion" in action.title
        assert "recovered" in action.title.lower()

    def test_no_recovery_without_an_announced_outage(self, db_conn: sqlite3.Connection) -> None:
        topic = _topic(db_conn)
        _fail_run(db_conn, topic.id, 2)
        _record(db_conn, topic.id, None, 1)
        assert evaluate_heartbeat(db_conn, topic, threshold=3) is None

    def test_no_recovery_while_the_newest_check_still_fails(self, db_conn: sqlite3.Connection) -> None:
        topic = _topic(db_conn)
        _fail_run(db_conn, topic.id, 4)
        assert evaluate_heartbeat(db_conn, _latched(db_conn, topic), threshold=3) is None

    def test_no_checks_yet_is_silent(self, db_conn: sqlite3.Connection) -> None:
        topic = _topic(db_conn)
        assert evaluate_heartbeat(db_conn, topic, threshold=3) is None

    def test_threshold_zero_disables_everything(self, db_conn: sqlite3.Connection) -> None:
        topic = _topic(db_conn)
        _fail_run(db_conn, topic.id, 5)
        assert evaluate_heartbeat(db_conn, topic, threshold=0) is None
        latched = _latched(db_conn, topic)
        _record(db_conn, topic.id, None, 1)
        assert evaluate_heartbeat(db_conn, latched, threshold=0) is None

    def test_threshold_one_alerts_on_the_first_failure(self, db_conn: sqlite3.Connection) -> None:
        topic = _topic(db_conn)
        _record(db_conn, topic.id, "sources_failed: x", 1)
        action = evaluate_heartbeat(db_conn, topic, threshold=1)
        assert action is not None and action.kind == "alert"

    def test_alert_body_points_at_a_shared_cause(self, db_conn: sqlite3.Connection) -> None:
        topic = _topic(db_conn)
        _fail_run(db_conn, topic.id, 3)
        action = evaluate_heartbeat(db_conn, topic, threshold=3)
        assert action is not None
        assert "several topics" in action.body
