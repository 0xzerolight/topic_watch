"""Tests for CheckingState async-safe check tracking."""

import asyncio
import time

import pytest

from app.web.routes import CheckingState


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
