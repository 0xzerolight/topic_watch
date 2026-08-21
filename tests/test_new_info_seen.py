"""Tests for the "new info" seen/acknowledged mechanism.

The dashboard "Ready · new info" badge is gated on ``has_new_info AND seen_at IS
NULL``. Acknowledging a topic's detail page stamps ``seen_at`` on the check it
displayed so the badge clears, without mutating ``has_new_info`` (which still
drives the detail-page history column and the Notify button). TW-AUD-024: the
GET route itself is query-only — ``mark_check_seen`` is called by a POST the
rendered page fires after content has loaded, keyed to the exact check_id shown,
not "whatever is latest right now" (``mark_latest_check_seen``, still used to
seed/verify state directly and by the dashboard-badge render path).

Covered here: the m020 migration column, and the ``mark_latest_check_seen`` /
``mark_check_seen`` CRUD guards. Render-path regressions live in test_web.py /
test_toggle_active.py; the dashboard-alias mapping lives in
test_models_from_row.py.
"""

import sqlite3
from datetime import UTC, datetime, timedelta

from app.crud import create_check_result, create_topic, get_check_result, mark_check_seen, mark_latest_check_seen
from app.migrations.m020_check_result_seen_at import up as m020_up
from app.models import CheckResult, Topic, TopicStatus


def _topic(conn: sqlite3.Connection, name: str = "Topic") -> Topic:
    topic = create_topic(conn, Topic(name=name, description="d", status=TopicStatus.READY))
    conn.commit()
    return topic


def _check(
    conn: sqlite3.Connection,
    topic_id: int,
    *,
    has_new_info: bool,
    checked_at: datetime,
) -> CheckResult:
    result = create_check_result(
        conn,
        CheckResult(topic_id=topic_id, has_new_info=has_new_info, checked_at=checked_at, articles_found=1),
    )
    conn.commit()
    return result


class TestSeenAtMigration:
    def test_column_present_after_init(self, db_conn: sqlite3.Connection) -> None:
        cols = {row[1] for row in db_conn.execute("PRAGMA table_info(check_results)").fetchall()}
        assert "seen_at" in cols

    def test_up_is_idempotent(self, db_conn: sqlite3.Connection) -> None:
        # Re-running on an already-migrated DB must be a no-op, not an error.
        m020_up(db_conn)
        m020_up(db_conn)
        cols = {row[1] for row in db_conn.execute("PRAGMA table_info(check_results)").fetchall()}
        assert "seen_at" in cols


class TestMarkLatestCheckSeen:
    def test_marks_latest_unseen_new_info(self, db_conn: sqlite3.Connection) -> None:
        topic = _topic(db_conn)
        now = datetime.now(UTC)
        check = _check(db_conn, topic.id, has_new_info=True, checked_at=now)

        mark_latest_check_seen(db_conn, topic.id)
        db_conn.commit()

        reloaded = get_check_result(db_conn, check.id)
        assert reloaded is not None
        assert reloaded.seen_at is not None
        # has_new_info is never mutated — history/Notify stay intact.
        assert reloaded.has_new_info is True

    def test_noop_when_latest_has_no_new_info(self, db_conn: sqlite3.Connection) -> None:
        topic = _topic(db_conn)
        now = datetime.now(UTC)
        check = _check(db_conn, topic.id, has_new_info=False, checked_at=now)

        mark_latest_check_seen(db_conn, topic.id)
        db_conn.commit()

        reloaded = get_check_result(db_conn, check.id)
        assert reloaded is not None
        assert reloaded.seen_at is None

    def test_idempotent_timestamp_unchanged_on_recall(self, db_conn: sqlite3.Connection) -> None:
        topic = _topic(db_conn)
        now = datetime.now(UTC)
        check = _check(db_conn, topic.id, has_new_info=True, checked_at=now)

        mark_latest_check_seen(db_conn, topic.id)
        db_conn.commit()
        first = get_check_result(db_conn, check.id)
        assert first is not None and first.seen_at is not None

        # A second view must NOT rewrite the timestamp (proves the seen_at IS NULL guard).
        mark_latest_check_seen(db_conn, topic.id)
        db_conn.commit()
        second = get_check_result(db_conn, check.id)
        assert second is not None
        assert second.seen_at == first.seen_at

    def test_only_latest_row_marked(self, db_conn: sqlite3.Connection) -> None:
        topic = _topic(db_conn)
        now = datetime.now(UTC)
        older = _check(db_conn, topic.id, has_new_info=True, checked_at=now - timedelta(hours=2))
        newer = _check(db_conn, topic.id, has_new_info=True, checked_at=now)

        mark_latest_check_seen(db_conn, topic.id)
        db_conn.commit()

        older_reloaded = get_check_result(db_conn, older.id)
        newer_reloaded = get_check_result(db_conn, newer.id)
        assert older_reloaded is not None and older_reloaded.seen_at is None
        assert newer_reloaded is not None and newer_reloaded.seen_at is not None

    def test_noop_on_topic_without_checks(self, db_conn: sqlite3.Connection) -> None:
        topic = _topic(db_conn)
        # No check rows: must not raise.
        mark_latest_check_seen(db_conn, topic.id)
        db_conn.commit()


class TestMarkCheckSeen:
    """TW-AUD-024: keyed to an explicit check_id, not "whatever is latest now"."""

    def test_marks_the_given_check_when_it_is_latest(self, db_conn: sqlite3.Connection) -> None:
        topic = _topic(db_conn)
        now = datetime.now(UTC)
        check = _check(db_conn, topic.id, has_new_info=True, checked_at=now)

        mark_check_seen(db_conn, topic.id, check.id)
        db_conn.commit()

        reloaded = get_check_result(db_conn, check.id)
        assert reloaded is not None
        assert reloaded.seen_at is not None
        assert reloaded.has_new_info is True

    def test_stale_check_id_is_a_noop(self, db_conn: sqlite3.Connection) -> None:
        """A check that is no longer the topic's latest is never acked, even if
        explicitly named — the page displaying it is stale."""
        topic = _topic(db_conn)
        now = datetime.now(UTC)
        older = _check(db_conn, topic.id, has_new_info=True, checked_at=now - timedelta(hours=2))
        newer = _check(db_conn, topic.id, has_new_info=True, checked_at=now)

        mark_check_seen(db_conn, topic.id, older.id)
        db_conn.commit()

        older_reloaded = get_check_result(db_conn, older.id)
        newer_reloaded = get_check_result(db_conn, newer.id)
        assert older_reloaded is not None and older_reloaded.seen_at is None
        assert newer_reloaded is not None and newer_reloaded.seen_at is None

    def test_wrong_topic_id_is_a_noop(self, db_conn: sqlite3.Connection) -> None:
        topic_a = _topic(db_conn, "A")
        topic_b = _topic(db_conn, "B")
        check = _check(db_conn, topic_a.id, has_new_info=True, checked_at=datetime.now(UTC))

        mark_check_seen(db_conn, topic_b.id, check.id)
        db_conn.commit()

        reloaded = get_check_result(db_conn, check.id)
        assert reloaded is not None
        assert reloaded.seen_at is None

    def test_idempotent_timestamp_unchanged_on_recall(self, db_conn: sqlite3.Connection) -> None:
        topic = _topic(db_conn)
        check = _check(db_conn, topic.id, has_new_info=True, checked_at=datetime.now(UTC))

        mark_check_seen(db_conn, topic.id, check.id)
        db_conn.commit()
        first = get_check_result(db_conn, check.id)
        assert first is not None and first.seen_at is not None

        mark_check_seen(db_conn, topic.id, check.id)
        db_conn.commit()
        second = get_check_result(db_conn, check.id)
        assert second is not None
        assert second.seen_at == first.seen_at

    def test_noop_when_no_new_info(self, db_conn: sqlite3.Connection) -> None:
        topic = _topic(db_conn)
        check = _check(db_conn, topic.id, has_new_info=False, checked_at=datetime.now(UTC))

        mark_check_seen(db_conn, topic.id, check.id)
        db_conn.commit()

        reloaded = get_check_result(db_conn, check.id)
        assert reloaded is not None
        assert reloaded.seen_at is None
