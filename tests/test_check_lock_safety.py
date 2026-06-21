"""Tests for CheckingState async-safe check tracking and retry-drain claims."""

import asyncio
import sqlite3
import time
from unittest.mock import AsyncMock

import pytest

from app.config import LLMSettings, Settings
from app.crud import (
    claim_new_topic_for_init,
    claim_pending_notification,
    claim_pending_webhook,
    create_pending_notification,
    create_pending_webhook,
    create_topic,
    get_topic,
    release_stale_notification_claims,
    release_stale_webhook_claims,
)
from app.models import PendingNotification, Topic, TopicStatus
from app.web.state import CheckingState, _checking_state


def _make_settings(**overrides) -> Settings:
    defaults = {"llm": LLMSettings(model="openai/gpt-4o-mini", api_key="test-key-12345678")}
    defaults.update(overrides)
    return Settings(**defaults)


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


# --- check_topic per-topic guard authoritative across entry points (OVH-096) ---


@pytest.fixture
def clean_state():
    """Ensure the process-global _checking_state singleton is empty around a test."""
    _checking_state._topics.clear()
    _checking_state._start_times.clear()
    _checking_state._checking_all = False
    yield _checking_state
    _checking_state._topics.clear()
    _checking_state._start_times.clear()
    _checking_state._checking_all = False


def _patch_empty_fetch(monkeypatch):
    """Patch the scraper so check_topic returns early at 'no new articles' (no LLM)."""
    from app.scraping import FetchResult

    async def _no_articles(*args, **kwargs):
        return FetchResult(articles=[], total_feed_entries=0)

    monkeypatch.setattr("app.checker.fetch_new_articles_for_topic", _no_articles)


async def test_check_topic_skips_when_already_in_flight(db_conn: sqlite3.Connection, monkeypatch, clean_state) -> None:
    """A second check_topic on an already-claimed topic skips the pipeline (OVH-096)."""
    from app.checker import check_topic

    topic = create_topic(db_conn, Topic(name="Guarded", description="d", status=TopicStatus.READY))
    db_conn.commit()
    settings = _make_settings()

    fetch_spy = AsyncMock(side_effect=AssertionError("pipeline must not run while in flight"))
    monkeypatch.setattr("app.checker.fetch_new_articles_for_topic", fetch_spy)

    # Simulate an in-flight check by pre-claiming the per-topic slot.
    assert await clean_state.start_check(topic.id) is True
    result = await check_topic(topic, db_conn, settings)

    assert result.stage_error == "skipped: already in flight"
    fetch_spy.assert_not_awaited()


async def test_check_topic_acquires_and_releases_guard(db_conn: sqlite3.Connection, monkeypatch, clean_state) -> None:
    """check_topic claims the per-topic guard for the run and releases it after (OVH-096)."""
    from app.checker import check_topic

    topic = create_topic(db_conn, Topic(name="Solo", description="d", status=TopicStatus.READY))
    db_conn.commit()
    settings = _make_settings()
    _patch_empty_fetch(monkeypatch)

    assert await clean_state.is_checking(topic.id) is False
    await check_topic(topic, db_conn, settings)
    # Released in finally -> startable again.
    assert await clean_state.is_checking(topic.id) is False
    assert await clean_state.start_check(topic.id) is True


async def test_check_topic_guard_false_does_not_touch_state(
    db_conn: sqlite3.Connection, monkeypatch, clean_state
) -> None:
    """guard=False runs even when the slot is taken (caller owns the guard)."""
    from app.checker import check_topic

    topic = create_topic(db_conn, Topic(name="Owned", description="d", status=TopicStatus.READY))
    db_conn.commit()
    settings = _make_settings()
    _patch_empty_fetch(monkeypatch)

    # Caller already holds the slot; guard=False must still execute the pipeline.
    assert await clean_state.start_check(topic.id) is True
    result = await check_topic(topic, db_conn, settings, guard=False)
    assert result.stage_error != "skipped: already in flight"
    # The slot the caller owns is untouched by the inner run.
    assert await clean_state.is_checking(topic.id) is True


async def test_concurrent_check_topic_only_one_runs_pipeline(
    db_conn: sqlite3.Connection, monkeypatch, tmp_path, clean_state
) -> None:
    """Two concurrent check_topic calls on the same topic: only one runs the pipeline."""
    from app.checker import check_topic
    from app.database import get_db, init_db
    from app.scraping import FetchResult

    db_path = tmp_path / "race.db"
    init_db(db_path)
    with get_db(db_path) as seed:
        topic = create_topic(seed, Topic(name="Racer", description="d", status=TopicStatus.READY))
        seed.commit()
    settings = _make_settings()

    runs = 0

    async def _slow_fetch(*args, **kwargs):
        nonlocal runs
        runs += 1
        await asyncio.sleep(0.05)
        return FetchResult(articles=[], total_feed_entries=0)

    monkeypatch.setattr("app.checker.fetch_new_articles_for_topic", _slow_fetch)

    async def _do_check():
        with get_db(db_path) as conn:
            return await check_topic(topic, conn, settings)

    results = await asyncio.gather(_do_check(), _do_check())
    skipped = [r for r in results if r.stage_error == "skipped: already in flight"]
    assert runs == 1
    assert len(skipped) == 1


# --- check_all_topics whole-cycle gate shared with web check-all (OVH-034) ---


async def test_check_all_skips_when_cycle_already_in_flight(monkeypatch, clean_state) -> None:
    """A check_all_topics run while a cycle is in flight skips (OVH-034)."""
    from app.checker import check_all_topics

    settings = _make_settings()
    inner = AsyncMock(side_effect=AssertionError("cycle must not run twice"))
    monkeypatch.setattr("app.checker._check_all_topics_inner", inner)

    # Simulate a web check-all already holding the whole-cycle gate.
    assert await clean_state.start_check_all() is True
    result = await check_all_topics(settings)
    assert result == []
    inner.assert_not_awaited()


async def test_check_all_releases_gate_after_run(monkeypatch, clean_state) -> None:
    """check_all_topics releases the whole-cycle gate so the next caller proceeds."""
    from app.checker import check_all_topics

    settings = _make_settings()
    monkeypatch.setattr("app.checker._check_all_topics_inner", AsyncMock(return_value=[]))

    await check_all_topics(settings)
    assert await clean_state.is_checking_all() is False
    # Gate is free again.
    assert await clean_state.start_check_all() is True


async def test_check_all_guard_false_skips_gate(monkeypatch, clean_state) -> None:
    """guard=False runs the inner cycle even if the gate is held (caller owns it)."""
    from app.checker import check_all_topics

    settings = _make_settings()
    inner = AsyncMock(return_value=[])
    monkeypatch.setattr("app.checker._check_all_topics_inner", inner)

    assert await clean_state.start_check_all() is True
    await check_all_topics(settings, guard=False)
    inner.assert_awaited_once()


# --- Atomic NEW -> RESEARCHING init claim (OVH-032) ---


def test_claim_new_topic_for_init_wins_once(db_conn: sqlite3.Connection) -> None:
    """Only the first claim transitions NEW -> RESEARCHING; a second loses."""
    topic = create_topic(db_conn, Topic(name="ToInit", description="d", status=TopicStatus.NEW))
    db_conn.commit()

    assert claim_new_topic_for_init(db_conn, topic.id) is True
    # Row is now RESEARCHING, not NEW -> second claim fails.
    assert claim_new_topic_for_init(db_conn, topic.id) is False
    refreshed = get_topic(db_conn, topic.id)
    assert refreshed.status == TopicStatus.RESEARCHING


def test_claim_new_topic_for_init_noop_when_not_new(db_conn: sqlite3.Connection) -> None:
    """A topic that is not NEW (already READY) cannot be claimed for init."""
    topic = create_topic(db_conn, Topic(name="Ready", description="d", status=TopicStatus.READY))
    db_conn.commit()
    assert claim_new_topic_for_init(db_conn, topic.id) is False


async def test_concurrent_init_claim_only_one_wins(tmp_path) -> None:
    """Two connections racing claim_new_topic_for_init: exactly one wins."""
    from app.database import get_db, init_db

    db_path = tmp_path / "init.db"
    init_db(db_path)
    with get_db(db_path) as seed:
        topic = create_topic(seed, Topic(name="Race", description="d", status=TopicStatus.NEW))
        seed.commit()

    def _claim() -> bool:
        with get_db(db_path) as conn:
            return claim_new_topic_for_init(conn, topic.id)

    results = await asyncio.gather(asyncio.to_thread(_claim), asyncio.to_thread(_claim))
    assert results.count(True) == 1
    assert results.count(False) == 1


# --- Cross-entry-point race: pipeline check_topic vs JSON API (OVH-086) ---


async def test_cross_entry_point_check_topic_blocks_api_trigger(
    db_conn: sqlite3.Connection, monkeypatch, clean_state
) -> None:
    """The real shared ``_checking_state`` singleton dedups across entry points.

    Earlier lock tests exercised either the mutex object alone or two calls
    through the SAME entry point. This drives the genuine hazard: a slow
    ``check_topic`` (pipeline entry, e.g. a scheduler minute-tick) is in flight,
    and the JSON ``api_trigger_check`` entry point (a manual API call) hits the
    SAME topic mid-flight. Both reach ``app.web.state._checking_state``, so the
    second entry point must be deduped — here a 409 — instead of launching a
    duplicate fetch+analyze+notify that double-spends the LLM and double-notifies.
    """
    from fastapi import HTTPException

    from app.checker import check_topic
    from app.scraping import FetchResult
    from app.web.api import api_trigger_check

    topic = create_topic(db_conn, Topic(name="CrossEntry", description="d", status=TopicStatus.READY))
    db_conn.commit()
    settings = _make_settings()

    in_flight = asyncio.Event()
    release = asyncio.Event()

    async def _slow_fetch(*args, **kwargs):
        # Signal that check_topic now HOLDS the per-topic guard, then block so
        # the API entry point races it mid-flight.
        in_flight.set()
        await release.wait()
        return FetchResult(articles=[], total_feed_entries=0)

    monkeypatch.setattr("app.checker.fetch_new_articles_for_topic", _slow_fetch)

    pipeline = asyncio.create_task(check_topic(topic, db_conn, settings))
    try:
        await asyncio.wait_for(in_flight.wait(), timeout=1.0)
        # check_topic is mid-flight and owns the slot. The API entry point shares
        # the same singleton, so it must reject with 409 (not run the pipeline).
        with pytest.raises(HTTPException) as exc_info:
            await api_trigger_check(topic.id, conn=db_conn, settings=settings)
        assert exc_info.value.status_code == 409
    finally:
        release.set()
        await pipeline

    # The pipeline run completed normally (its slot released in finally).
    result = pipeline.result()
    assert result.stage_error != "skipped: already in flight"
    assert await clean_state.is_checking(topic.id) is False
