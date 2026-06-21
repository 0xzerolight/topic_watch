"""Tests for CheckingState async-safe check tracking and retry-drain claims."""

import asyncio
import sqlite3
import time

import pytest

from app.crud import (
    claim_pending_notification,
    claim_pending_webhook,
    create_pending_notification,
    create_pending_webhook,
    create_topic,
    release_stale_notification_claims,
    release_stale_webhook_claims,
)
from app.models import PendingNotification, Topic, TopicStatus
from app.web.state import CheckingState


@pytest.fixture
def state() -> CheckingState:
    return CheckingState()


# --- start_check / finish_check ---


async def test_start_check_returns_true_first_time(state: CheckingState) -> None:
    result = await state.start_check(1)
    assert result is True


async def test_start_check_returns_false_when_already_checking(state: CheckingState) -> None:
    await state.start_check(1)
    result = await state.start_check(1)
    assert result is False


async def test_start_check_allows_different_topics(state: CheckingState) -> None:
    assert await state.start_check(1) is True
    assert await state.start_check(2) is True


async def test_is_checking_true_after_start(state: CheckingState) -> None:
    await state.start_check(42)
    assert await state.is_checking(42) is True


async def test_is_checking_false_before_start(state: CheckingState) -> None:
    assert await state.is_checking(99) is False


async def test_finish_check_releases(state: CheckingState) -> None:
    await state.start_check(1)
    await state.finish_check(1)
    assert await state.is_checking(1) is False


async def test_finish_check_allows_restart(state: CheckingState) -> None:
    await state.start_check(1)
    await state.finish_check(1)
    result = await state.start_check(1)
    assert result is True


async def test_finish_check_nonexistent_is_noop(state: CheckingState) -> None:
    # Should not raise
    await state.finish_check(999)


# --- start_check_all / finish_check_all / is_checking_all ---


async def test_start_check_all_returns_true_first_time(state: CheckingState) -> None:
    result = await state.start_check_all()
    assert result is True


async def test_start_check_all_returns_false_when_running(state: CheckingState) -> None:
    await state.start_check_all()
    result = await state.start_check_all()
    assert result is False


async def test_is_checking_all_false_initially(state: CheckingState) -> None:
    assert await state.is_checking_all() is False


async def test_is_checking_all_true_after_start(state: CheckingState) -> None:
    await state.start_check_all()
    assert await state.is_checking_all() is True


async def test_finish_check_all_resets_flag(state: CheckingState) -> None:
    await state.start_check_all()
    await state.finish_check_all()
    assert await state.is_checking_all() is False


async def test_finish_check_all_allows_restart(state: CheckingState) -> None:
    await state.start_check_all()
    await state.finish_check_all()
    result = await state.start_check_all()
    assert result is True


# --- clear_stale ---


async def test_clear_stale_removes_old_entries(state: CheckingState) -> None:
    await state.start_check(10)
    # Backdate the start time so the entry appears stale
    state._start_times[10] = time.monotonic() - 700
    cleared = await state.clear_stale(600)
    assert 10 in cleared
    assert await state.is_checking(10) is False


async def test_clear_stale_keeps_fresh_entries(state: CheckingState) -> None:
    await state.start_check(20)
    cleared = await state.clear_stale(600)
    assert 20 not in cleared
    assert await state.is_checking(20) is True


async def test_clear_stale_returns_empty_when_nothing_stale(state: CheckingState) -> None:
    await state.start_check(30)
    cleared = await state.clear_stale(600)
    assert cleared == []


async def test_clear_stale_multiple_topics(state: CheckingState) -> None:
    await state.start_check(1)
    await state.start_check(2)
    await state.start_check(3)
    # Make topics 1 and 3 stale
    state._start_times[1] = time.monotonic() - 700
    state._start_times[3] = time.monotonic() - 700
    cleared = await state.clear_stale(600)
    assert set(cleared) == {1, 3}
    assert await state.is_checking(2) is True
    assert await state.is_checking(1) is False
    assert await state.is_checking(3) is False


# --- Concurrent access ---


async def test_concurrent_start_check_only_one_wins(state: CheckingState) -> None:
    """Only one coroutine should win the start_check race."""
    results = await asyncio.gather(*[state.start_check(5) for _ in range(10)])
    assert results.count(True) == 1
    assert results.count(False) == 9


async def test_concurrent_start_check_all_only_one_wins(state: CheckingState) -> None:
    """Only one coroutine should win the start_check_all race."""
    results = await asyncio.gather(*[state.start_check_all() for _ in range(10)])
    assert results.count(True) == 1
    assert results.count(False) == 9


async def test_concurrent_different_topics_all_win(state: CheckingState) -> None:
    """Each different topic_id should be able to start independently."""
    topic_ids = list(range(100, 110))
    results = await asyncio.gather(*[state.start_check(tid) for tid in topic_ids])
    assert all(results)


async def test_state_not_corrupted_after_concurrent_finish(state: CheckingState) -> None:
    """Concurrent finish_check calls should not corrupt internal state."""
    for tid in range(5):
        await state.start_check(tid)
    await asyncio.gather(*[state.finish_check(tid) for tid in range(5)])
    for tid in range(5):
        assert await state.is_checking(tid) is False
    # All entries should be startable again
    for tid in range(5):
        assert await state.start_check(tid) is True


# --- Retry-drain atomic claim (cross-process double-delivery guard, OVH-017) ---


def _make_topic(conn: sqlite3.Connection) -> Topic:
    topic = create_topic(conn, Topic(name="Claimed", description="d", status=TopicStatus.READY))
    conn.commit()
    return topic


def test_notification_claim_succeeds_once_then_fails(db_conn: sqlite3.Connection) -> None:
    """Only the first claim of an unclaimed notification wins; a second loses."""
    topic = _make_topic(db_conn)
    n = create_pending_notification(db_conn, PendingNotification(topic_id=topic.id, title="T", body="B"))
    db_conn.commit()

    assert claim_pending_notification(db_conn, n.id, "2025-01-01T00:00:00+00:00") is True
    # Second claim of the now-claimed row loses (would have caused a double-send).
    assert claim_pending_notification(db_conn, n.id, "2025-01-01T00:00:01+00:00") is False


def test_webhook_claim_succeeds_once_then_fails(db_conn: sqlite3.Connection) -> None:
    """Only the first claim of an unclaimed webhook wins; a second loses."""
    topic = _make_topic(db_conn)
    webhook_id = create_pending_webhook(db_conn, topic.id, "https://a.com/hook", {"k": "v"})
    db_conn.commit()

    assert claim_pending_webhook(db_conn, webhook_id, "2025-01-01T00:00:00+00:00") is True
    assert claim_pending_webhook(db_conn, webhook_id, "2025-01-01T00:00:01+00:00") is False


def test_release_stale_notification_claim_rearms_row(db_conn: sqlite3.Connection) -> None:
    """A claim older than the cutoff is released so the row can be re-claimed."""
    topic = _make_topic(db_conn)
    n = create_pending_notification(db_conn, PendingNotification(topic_id=topic.id, title="T", body="B"))
    db_conn.commit()
    assert claim_pending_notification(db_conn, n.id, "2020-01-01T00:00:00+00:00") is True
    db_conn.commit()

    # A fresh claim (newer) is NOT released; an old one is.
    released = release_stale_notification_claims(db_conn, "2020-06-01T00:00:00+00:00")
    db_conn.commit()
    assert released == 1
    # Re-claimable now.
    assert claim_pending_notification(db_conn, n.id, "2025-01-01T00:00:00+00:00") is True


def test_release_stale_webhook_claim_rearms_row(db_conn: sqlite3.Connection) -> None:
    """Stale webhook claims are released; recent ones are left alone."""
    topic = _make_topic(db_conn)
    webhook_id = create_pending_webhook(db_conn, topic.id, "https://a.com/hook", {"k": "v"})
    db_conn.commit()
    assert claim_pending_webhook(db_conn, webhook_id, "2020-01-01T00:00:00+00:00") is True
    db_conn.commit()

    # Cutoff before the claim time: nothing released.
    assert release_stale_webhook_claims(db_conn, "2019-01-01T00:00:00+00:00") == 0
    # Cutoff after the claim time: released and re-claimable.
    assert release_stale_webhook_claims(db_conn, "2020-06-01T00:00:00+00:00") == 1
    db_conn.commit()
    assert claim_pending_webhook(db_conn, webhook_id, "2025-01-01T00:00:00+00:00") is True
