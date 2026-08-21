"""Clock-semantics and ordering tests (AUG-277..AUG-284, AUG-029).

One home for the wave-A clock policy's remaining half: durable timestamps carry
one canonical UTC spelling, causal order breaks ties by ``check_results.id``,
liveness is measured monotonically, and an impossible future anchor never parks
scheduling or renders as "just now".
"""

import sqlite3
from datetime import UTC, datetime, timedelta, timezone

import pytest

from app.crud import (
    create_check_result,
    create_topic,
    get_dashboard_data,
    get_topics_due_for_check,
    list_check_results,
    list_recent_check_stage_errors,
    mark_check_seen,
    mark_latest_check_seen,
)
from app.models import CheckResult, FeedMode, Topic, TopicStatus


def _topic(name: str = "Topic", **overrides) -> Topic:
    data = {
        "name": name,
        "description": "d",
        "feed_mode": FeedMode.AUTO,
        "status": TopicStatus.READY,
        "is_active": True,
    }
    data.update(overrides)
    return Topic(**data)


def _check(topic_id: int, checked_at: datetime, **overrides) -> CheckResult:
    data = {"topic_id": topic_id, "checked_at": checked_at, "has_new_info": True}
    data.update(overrides)
    return CheckResult(**data)


class TestRateLimitWindow:
    """AUG-283 — the feed-validation window is elapsed time, not wall time."""

    def _fresh(self, monkeypatch: pytest.MonkeyPatch, ip: str) -> list[float]:
        from app.web.state import _rate_limit_store

        _rate_limit_store.pop(ip, None)
        return _rate_limit_store.setdefault(ip, [])

    def test_forward_wall_clock_step_does_not_refund_the_quota(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import time as time_mod

        from app import clock
        from app.web.state import _check_rate_limit

        ip = "10.0.0.101"
        self._fresh(monkeypatch, ip)
        monkeypatch.setattr(clock, "monotonic_now", lambda: 1000.0)
        for _ in range(10):
            assert _check_rate_limit(ip) is True

        # The host clock jumps an hour forward; elapsed time has not moved.
        wall = time_mod.time()
        monkeypatch.setattr(time_mod, "time", lambda: wall + 3600)
        assert _check_rate_limit(ip) is False

    def test_backward_wall_clock_step_does_not_extend_the_window(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import time as time_mod

        from app import clock
        from app.web.state import _check_rate_limit

        ip = "10.0.0.102"
        self._fresh(monkeypatch, ip)
        monkeypatch.setattr(clock, "monotonic_now", lambda: 1000.0)
        for _ in range(10):
            _check_rate_limit(ip)

        # The host clock is corrected an hour backwards, but 61 seconds really passed.
        wall = time_mod.time()
        monkeypatch.setattr(time_mod, "time", lambda: wall - 3600)
        monkeypatch.setattr(clock, "monotonic_now", lambda: 1061.0)
        assert _check_rate_limit(ip) is True


class TestLatestCheckOrdering:
    """AUG-279 — every "latest check" query uses the heartbeat's own head order."""

    def test_tied_timestamps_resolve_to_the_newest_id_everywhere(self, db_conn: sqlite3.Connection) -> None:
        topic = create_topic(db_conn, _topic())
        db_conn.commit()
        assert topic.id is not None
        moment = datetime(2026, 8, 20, 12, 0, tzinfo=UTC)
        older = create_check_result(db_conn, _check(topic.id, moment, stage_error="first"))
        newer = create_check_result(db_conn, _check(topic.id, moment, stage_error="second"))
        db_conn.commit()
        assert older.id is not None and newer.id is not None and newer.id > older.id

        # The heartbeat's definition, which the others must agree with.
        assert list_recent_check_stage_errors(db_conn, topic.id, 1)[0][0] == newer.id
        assert list_check_results(db_conn, topic.id, limit=1)[0].id == newer.id
        dashboard = get_dashboard_data(db_conn)[0]
        assert dashboard["last_check"].id == newer.id

    def test_ack_marks_the_newest_id_of_a_tied_pair(self, db_conn: sqlite3.Connection) -> None:
        topic = create_topic(db_conn, _topic())
        db_conn.commit()
        assert topic.id is not None
        moment = datetime(2026, 8, 20, 12, 0, tzinfo=UTC)
        older = create_check_result(db_conn, _check(topic.id, moment))
        newer = create_check_result(db_conn, _check(topic.id, moment))
        db_conn.commit()
        assert older.id is not None and newer.id is not None

        mark_latest_check_seen(db_conn, topic.id)
        db_conn.commit()
        seen = {row["id"]: row["seen_at"] for row in db_conn.execute("SELECT id, seen_at FROM check_results")}
        assert seen[newer.id] is not None
        assert seen[older.id] is None

    def test_keyed_ack_refuses_the_older_row_of_a_tied_pair(self, db_conn: sqlite3.Connection) -> None:
        topic = create_topic(db_conn, _topic())
        db_conn.commit()
        assert topic.id is not None
        moment = datetime(2026, 8, 20, 12, 0, tzinfo=UTC)
        older = create_check_result(db_conn, _check(topic.id, moment))
        create_check_result(db_conn, _check(topic.id, moment))
        db_conn.commit()
        assert older.id is not None

        mark_check_seen(db_conn, topic.id, older.id)
        db_conn.commit()
        row = db_conn.execute("SELECT seen_at FROM check_results WHERE id = ?", (older.id,)).fetchone()
        assert row["seen_at"] is None


class TestCanonicalUtcSpelling:
    """AUG-280 — an offset-carrying datetime persists as canonical UTC TEXT."""

    def test_offset_timestamp_is_stored_as_utc(self, db_conn: sqlite3.Connection) -> None:
        topic = create_topic(db_conn, _topic())
        db_conn.commit()
        assert topic.id is not None
        east = timezone(timedelta(hours=2))
        create_check_result(db_conn, _check(topic.id, datetime(2026, 8, 20, 12, 0, tzinfo=east)))
        db_conn.commit()
        stored = db_conn.execute("SELECT checked_at FROM check_results").fetchone()[0]
        assert stored == "2026-08-20T10:00:00+00:00"

    def test_offset_timestamp_does_not_outrank_a_later_utc_sibling(self, db_conn: sqlite3.Connection) -> None:
        """12:00+02:00 is 10:00 UTC — it must sort BELOW an 11:00+00:00 sibling."""
        topic = create_topic(db_conn, _topic())
        db_conn.commit()
        assert topic.id is not None
        east = timezone(timedelta(hours=2))
        latest = create_check_result(db_conn, _check(topic.id, datetime(2026, 8, 20, 11, 0, tzinfo=UTC)))
        create_check_result(db_conn, _check(topic.id, datetime(2026, 8, 20, 12, 0, tzinfo=east)))
        db_conn.commit()
        assert latest.id is not None

        assert list_check_results(db_conn, topic.id, limit=1)[0].id == latest.id
        assert get_dashboard_data(db_conn)[0]["last_check"].id == latest.id


class TestDueSelection:
    """AUG-278 / AUG-029 — scheduling anchors on the causally latest check."""

    def test_recent_check_is_not_due(self, db_conn: sqlite3.Connection) -> None:
        topic = create_topic(db_conn, _topic(check_interval_minutes=60))
        db_conn.commit()
        assert topic.id is not None
        create_check_result(db_conn, _check(topic.id, datetime.now(UTC) - timedelta(minutes=5)))
        db_conn.commit()
        assert get_topics_due_for_check(db_conn, 60) == []

    def test_elapsed_interval_is_due(self, db_conn: sqlite3.Connection) -> None:
        topic = create_topic(db_conn, _topic(check_interval_minutes=60))
        db_conn.commit()
        assert topic.id is not None
        create_check_result(db_conn, _check(topic.id, datetime.now(UTC) - timedelta(minutes=90)))
        db_conn.commit()
        assert [t.id for t in get_topics_due_for_check(db_conn, 60)] == [topic.id]

    def test_future_anchor_does_not_park_the_topic(
        self,
        db_conn: sqlite3.Connection,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A check stamped during a forward clock jump must not suppress checking."""
        topic = create_topic(db_conn, _topic(check_interval_minutes=60))
        db_conn.commit()
        assert topic.id is not None
        create_check_result(db_conn, _check(topic.id, datetime.now(UTC) + timedelta(hours=6)))
        db_conn.commit()

        with caplog.at_level("WARNING"):
            due = get_topics_due_for_check(db_conn, 60)
        assert [t.id for t in due] == [topic.id]
        assert any("future" in record.message.lower() for record in caplog.records)

    def test_a_correcting_check_heals_the_schedule(self, db_conn: sqlite3.Connection) -> None:
        """Once a current-time row lands, the stranded future row stops deciding."""
        topic = create_topic(db_conn, _topic(check_interval_minutes=60))
        db_conn.commit()
        assert topic.id is not None
        create_check_result(db_conn, _check(topic.id, datetime.now(UTC) + timedelta(hours=6)))
        db_conn.commit()
        assert get_topics_due_for_check(db_conn, 60)

        create_check_result(db_conn, _check(topic.id, datetime.now(UTC)))
        db_conn.commit()
        assert get_topics_due_for_check(db_conn, 60) == []

    def test_due_query_does_not_scan_the_check_history(self, db_conn: sqlite3.Connection) -> None:
        """AUG-029 — one index seek per active topic, never a full-history scan."""
        topic = create_topic(db_conn, _topic(check_interval_minutes=60))
        db_conn.commit()
        assert topic.id is not None
        for minutes in range(1, 20):
            create_check_result(db_conn, _check(topic.id, datetime.now(UTC) - timedelta(minutes=minutes)))
        db_conn.commit()

        from app.crud import _DUE_TOPICS_SQL

        get_topics_due_for_check(db_conn, 60)
        rows = db_conn.execute(
            f"EXPLAIN QUERY PLAN {_DUE_TOPICS_SQL}",
            {"horizon": "2026-08-20T00:00:00+00:00", "default_interval": 60},
        ).fetchall()
        plan = " ".join(str(row["detail"]) for row in rows)
        assert "check_results" in plan
        assert "SCAN check_results" not in plan
        assert "idx_check_results_topic_time (topic_id=? AND checked_at<?)" in plan
