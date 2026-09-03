"""Tests for timeout protection in background task functions."""

import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from app.analysis.knowledge import KnowledgeUpdatePlan
from app.analysis.llm import TokenUsage
from app.config import LLMSettings, NotificationSettings, Settings
from app.crud import create_check_intents, create_topic, delete_topic, get_topic
from app.database import get_connection, init_db
from app.models import Article, CheckIntent, Topic, TopicStatus
from app.scraping import FetchResult
from app.web.routers.background import _run_check_all, _run_init


def _make_settings(**overrides) -> Settings:
    defaults = {
        "llm": LLMSettings(model="openai/gpt-4o-mini", api_key="test-key"),
        "notifications": NotificationSettings(urls=["json://localhost"]),
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _make_topic(conn: sqlite3.Connection, **overrides) -> Topic:
    defaults = {
        "name": "Test Topic",
        "description": "A test topic",
        "feed_urls": ["https://example.com/feed.xml"],
        "status": TopicStatus.RESEARCHING,
    }
    defaults.update(overrides)
    topic = create_topic(conn, Topic(**defaults))
    conn.commit()
    return topic


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "test.db"
    init_db(path)
    return path


class TestRunInitTimeout:
    """Tests for _run_init() timeout behaviour."""

    async def test_timeout_sets_topic_to_error(self, db_path: Path) -> None:
        """When fetch hangs beyond the timeout, topic is set to ERROR."""
        settings = _make_settings()

        conn = get_connection(db_path)
        try:
            topic = _make_topic(conn)
            topic_id = topic.id
        finally:
            conn.close()

        async def _hang(*args, **kwargs):
            await asyncio.sleep(9999)

        with (
            patch("app.web.routers.background._INIT_TIMEOUT_SECONDS", 0.05),
            patch(
                "app.checker.fetch_new_articles_for_topic",
                side_effect=_hang,
            ),
        ):
            await _run_init(topic_id, settings, db_path, claimed=True)

        conn = get_connection(db_path)
        try:
            refreshed = get_topic(conn, topic_id)
        finally:
            conn.close()

        assert refreshed is not None
        assert refreshed.status == TopicStatus.ERROR
        assert refreshed.error_message == "Research timed out. Click Retry."

    async def test_timeout_stamps_status_changed_at(self, db_path: Path) -> None:
        """Timeout error transition re-stamps status_changed_at for stuck-recovery timing.

        The RESEARCHING stamp is written by whoever claimed the topic (here, the
        seeded row), so the timeout path must re-stamp it on the ERROR transition
        rather than leaving the stale RESEARCHING timestamp.
        """
        settings = _make_settings()

        researching_stamp = {"at": datetime.now(UTC) - timedelta(minutes=5)}

        conn = get_connection(db_path)
        try:
            topic = _make_topic(conn, status_changed_at=researching_stamp["at"])
            topic_id = topic.id
        finally:
            conn.close()

        async def _hang(*args, **kwargs):
            await asyncio.sleep(9999)

        with (
            patch("app.web.routers.background._INIT_TIMEOUT_SECONDS", 0.05),
            patch(
                "app.checker.fetch_new_articles_for_topic",
                side_effect=_hang,
            ),
        ):
            await _run_init(topic_id, settings, db_path, claimed=True)

        conn = get_connection(db_path)
        try:
            refreshed = get_topic(conn, topic_id)
        finally:
            conn.close()

        assert refreshed is not None
        assert refreshed.status == TopicStatus.ERROR
        assert refreshed.status_changed_at is not None
        # Stamp must reflect the ERROR transition, not the stale RESEARCHING stamp.
        assert refreshed.status_changed_at > researching_stamp["at"]

    async def test_timeout_is_logged(self, db_path: Path, caplog) -> None:
        """Timeout event is logged at ERROR level."""
        import logging

        settings = _make_settings()

        conn = get_connection(db_path)
        try:
            topic = _make_topic(conn)
            topic_id = topic.id
        finally:
            conn.close()

        async def _hang(*args, **kwargs):
            await asyncio.sleep(9999)

        with (
            patch("app.web.routers.background._INIT_TIMEOUT_SECONDS", 0.05),
            patch(
                "app.checker.fetch_new_articles_for_topic",
                side_effect=_hang,
            ),
            caplog.at_level(logging.ERROR, logger="app.web.routers.background"),
        ):
            await _run_init(topic_id, settings, db_path, claimed=True)

        assert any("timed out" in record.message.lower() for record in caplog.records)

    async def test_normal_completion_sets_topic_to_ready(self, db_path: Path) -> None:
        """When everything succeeds, topic is set to READY."""

        settings = _make_settings()

        conn = get_connection(db_path)
        try:
            topic = _make_topic(conn)
            topic_id = topic.id
        finally:
            conn.close()

        fake_article = Article(
            id=1,
            topic_id=topic_id,
            title="Test Article",
            url="https://example.com/article-1",
            content_hash="abc123",
            raw_content="Some content.",
            source_feed="https://example.com/feed.xml",
        )

        with (
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=[fake_article], total_feed_entries=1),
            ),
            patch(
                "app.checker.prepare_initial_knowledge",
                new_callable=AsyncMock,
                return_value=KnowledgeUpdatePlan(
                    summary_text="s",
                    token_count=0,
                    usage=TokenUsage(),
                    sufficient_data=True,
                ),
            ),
            patch(
                "app.checker.mark_articles_processed",
                return_value=None,
            ),
        ):
            await _run_init(topic_id, settings, db_path, claimed=True)

        conn = get_connection(db_path)
        try:
            refreshed = get_topic(conn, topic_id)
        finally:
            conn.close()

        assert refreshed is not None
        assert refreshed.status == TopicStatus.READY
        assert refreshed.error_message is None

    async def test_missing_topic_returns_gracefully(self, db_path: Path) -> None:
        """If the topic has been deleted, _run_init returns without crashing."""
        settings = _make_settings()
        await _run_init(999_999, settings, db_path)  # non-existent topic id

    async def test_timeout_write_spares_a_rowid_reused_replacement(self, db_path: Path, caplog) -> None:
        """The timeout message never lands on the topic that recycled the rowid.

        ``topics.id`` is a plain rowid, so a topic deleted mid-init hands its id to
        the next INSERT. Fenced on status alone, this write replaced the
        replacement's own error with a timeout it never had. The refusal is logged:
        a dropped terminal write used to leave nothing to read.
        """
        import logging

        settings = _make_settings()

        conn = get_connection(db_path)
        try:
            topic = _make_topic(conn)
            topic_id = topic.id
        finally:
            conn.close()

        async def _replace_then_hang(*args, **kwargs):
            conn = get_connection(db_path)
            try:
                delete_topic(conn, topic_id)
                conn.commit()
                replacement = create_topic(
                    conn,
                    Topic(
                        name="Replacement",
                        description="d",
                        status=TopicStatus.ERROR,
                        error_message="Its own failure.",
                    ),
                )
                conn.commit()
                assert replacement.id == topic_id
            finally:
                conn.close()
            await asyncio.sleep(9999)

        with (
            patch("app.web.routers.background._INIT_TIMEOUT_SECONDS", 0.05),
            patch("app.checker.fetch_new_articles_for_topic", side_effect=_replace_then_hang),
            caplog.at_level(logging.WARNING, logger="app.web.routers.background"),
        ):
            await _run_init(topic_id, settings, db_path, claimed=True)

        conn = get_connection(db_path)
        try:
            survivor = get_topic(conn, topic_id)
        finally:
            conn.close()

        assert survivor is not None
        assert survivor.name == "Replacement"
        assert survivor.error_message == "Its own failure."
        assert any("not recorded" in record.message for record in caplog.records)


class TestRunSingleCheckTimeout:
    """AUG-264: a single check cannot outlive the slot-eviction threshold.

    ``clear_stale`` hands an over-age entry's slot to the next caller, so an
    unbounded check stayed live with no guard and a second checker committed and
    delivered the same finding a second time.
    """

    async def test_bound_is_below_the_handlers_eviction_threshold(self) -> None:
        from app.checker import CHECK_TIMEOUT_SECONDS

        assert CHECK_TIMEOUT_SECONDS < 600

    def _seed_intent(self, db_path: Path) -> tuple[int, CheckIntent]:
        conn = get_connection(db_path)
        try:
            topic = _make_topic(conn, status=TopicStatus.READY)
            intent = CheckIntent(request_id="req-1", topic_id=topic.id)
            create_check_intents(conn, [intent])
            conn.commit()
            return topic.id, intent
        finally:
            conn.close()

    async def test_hanging_check_is_cancelled_and_releases_the_guard(self, db_path: Path, caplog) -> None:
        import logging

        from app.web.routers.background import _run_single_check
        from app.web.state import _checking_state

        settings = _make_settings()
        topic_id, intent = self._seed_intent(db_path)

        # Patched BELOW the guard so the real check_topic takes it: the guard
        # moved out of _run_single_check into check_topic (AUG-286), so mocking
        # check_topic itself would leave nothing holding it and the release
        # assertion below could not fail.
        held: list[bool] = []

        async def _hang(*args, **kwargs):
            held.append(await _checking_state.is_checking(topic_id))
            await asyncio.sleep(9999)

        _checking_state._topics.clear()
        try:
            with (
                patch("app.checker.CHECK_TIMEOUT_SECONDS", 0.05),
                patch("app.checker._check_topic_guarded", side_effect=_hang),
                caplog.at_level(logging.ERROR, logger="app.checker"),
            ):
                # Outer bound so an unbounded task fails the test instead of hanging it.
                await asyncio.wait_for(_run_single_check(intent, settings, db_path), timeout=5)
            # Read before the cleanup clear below, which answers False either way.
            still_held = await _checking_state.is_checking(topic_id)
        finally:
            _checking_state._topics.clear()

        assert any("timed out" in record.message.lower() for record in caplog.records)
        assert held == [True]  # the guard was really held while the check ran
        assert still_held is False  # and released when the timeout cancelled it

    async def test_normal_completion_runs_the_check(self, db_path: Path) -> None:
        from app.models import CheckResult
        from app.web.routers.background import _run_single_check
        from app.web.state import _checking_state

        settings = _make_settings()
        topic_id, intent = self._seed_intent(db_path)

        _checking_state._topics.clear()
        try:
            with patch(
                "app.checker.check_topic", new_callable=AsyncMock, return_value=CheckResult(topic_id=topic_id)
            ) as mock_check:
                await _run_single_check(intent, settings, db_path)
        finally:
            _checking_state._topics.clear()

        mock_check.assert_awaited_once()


class TestRunCheckAllTimeout:
    """Tests for _run_check_all() timeout behaviour."""

    async def test_timeout_logs_and_does_not_crash(self, db_path: Path, caplog) -> None:
        """When check_all_topics hangs beyond the timeout, a warning is logged and the task returns cleanly."""
        import logging

        settings = _make_settings()

        async def _hang(*args, **kwargs):
            await asyncio.sleep(9999)

        with (
            patch("app.web.routers.background._CHECK_ALL_TIMEOUT_SECONDS", 0.05),
            patch("app.web.routers.background.check_all_topics", side_effect=_hang),
            caplog.at_level(logging.ERROR, logger="app.web.routers.background"),
        ):
            # Should complete without raising
            await _run_check_all(settings, db_path)

        assert any("timed out" in record.message.lower() for record in caplog.records)

    async def test_timeout_clears_checking_state(self, db_path: Path) -> None:
        """After a timeout, the checking-all flag is cleared so the next run can proceed."""
        from app.web.state import _checking_state

        settings = _make_settings()

        async def _hang(*args, **kwargs):
            await asyncio.sleep(9999)

        owner = _checking_state.start_check_all()
        with (
            patch("app.web.routers.background._CHECK_ALL_TIMEOUT_SECONDS", 0.05),
            patch("app.web.routers.background.check_all_topics", side_effect=_hang),
        ):
            await _run_check_all(settings, db_path, owner)

        assert not await _checking_state.is_checking_all()

    async def test_normal_completion_returns_cleanly(self, db_path: Path) -> None:
        """When check_all_topics completes normally, the task finishes without error."""
        from app.web.state import _checking_state

        settings = _make_settings()

        owner = _checking_state.start_check_all()
        with patch(
            "app.web.routers.background.check_all_topics",
            new_callable=AsyncMock,
            return_value=[],
        ):
            await _run_check_all(settings, db_path, owner)

        assert not await _checking_state.is_checking_all()

    async def test_normal_completion_does_not_log_error(self, db_path: Path, caplog) -> None:
        """Successful run produces no error-level log entries."""
        import logging

        settings = _make_settings()

        with (
            patch(
                "app.web.routers.background.check_all_topics",
                new_callable=AsyncMock,
                return_value=[],
            ),
            caplog.at_level(logging.ERROR, logger="app.web.routers.background"),
        ):
            await _run_check_all(settings, db_path)

        assert not caplog.records


class TestCheckAllDeadlineScalesWithBacklog:
    """AUG-211: a count-blind deadline cancelled healthy large backlogs mid-cycle."""

    def test_deadline_grows_one_bounded_wave_at_a_time(self) -> None:
        from app.checker import CHECK_TIMEOUT_SECONDS
        from app.web.routers.background import _CHECK_ALL_TIMEOUT_SECONDS, _check_all_deadline_seconds

        assert _check_all_deadline_seconds(0, 3) == _CHECK_ALL_TIMEOUT_SECONDS
        assert _check_all_deadline_seconds(3, 3) == _CHECK_ALL_TIMEOUT_SECONDS + CHECK_TIMEOUT_SECONDS
        assert _check_all_deadline_seconds(4, 3) == _CHECK_ALL_TIMEOUT_SECONDS + 2 * CHECK_TIMEOUT_SECONDS
        # The case the finding is filed against: 500 topics at concurrency 3 no
        # longer share one 30-minute budget.
        assert _check_all_deadline_seconds(500, 3) > 10 * _CHECK_ALL_TIMEOUT_SECONDS

    def test_zero_concurrency_does_not_divide_by_zero(self) -> None:
        from app.web.routers.background import _check_all_deadline_seconds

        assert _check_all_deadline_seconds(5, 0) > 0

    async def test_the_deadline_used_reflects_the_due_backlog(self, db_path: Path) -> None:
        from app.checker import CHECK_TIMEOUT_SECONDS
        from app.web.routers.background import _CHECK_ALL_TIMEOUT_SECONDS, _run_check_all

        settings = _make_settings()
        conn = get_connection(db_path)
        try:
            for i in range(4):
                _make_topic(conn, name=f"Due {i}", status=TopicStatus.READY)
        finally:
            conn.close()

        captured: list[float] = []

        async def _capture(coro, timeout=None):
            captured.append(timeout)
            coro.close()

        with (
            patch("app.web.routers.background.asyncio.wait_for", side_effect=_capture),
            patch("app.web.routers.background.check_all_topics", new_callable=AsyncMock, return_value=[]),
        ):
            await _run_check_all(settings, db_path)

        waves = -(-4 // settings.topic_check_concurrency)
        assert captured == [_CHECK_ALL_TIMEOUT_SECONDS + waves * CHECK_TIMEOUT_SECONDS]

    async def test_a_backlog_query_failure_falls_back_to_the_base(self, db_path: Path) -> None:
        from app.web.routers.background import _CHECK_ALL_TIMEOUT_SECONDS, _due_topic_count

        settings = _make_settings()
        with patch("app.web.routers.background.get_topics_due_for_check", side_effect=RuntimeError("boom")):
            assert _due_topic_count(settings, db_path) == 0
        from app.web.routers.background import _check_all_deadline_seconds

        assert _check_all_deadline_seconds(0, settings.topic_check_concurrency) == _CHECK_ALL_TIMEOUT_SECONDS
