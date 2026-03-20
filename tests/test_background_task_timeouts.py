"""Tests for timeout protection in background task functions."""

import asyncio
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from app.config import LLMSettings, NotificationSettings, Settings
from app.crud import create_topic, get_topic
from app.database import get_connection, init_db
from app.models import Topic, TopicStatus
from app.web.routes import _run_check_all, _run_init


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
            patch("app.web.routes._INIT_TIMEOUT_SECONDS", 0.05),
            patch(
                "app.scraping.fetch_new_articles_for_topic",
                side_effect=_hang,
            ),
        ):
            await _run_init(topic_id, settings, db_path)

        conn = get_connection(db_path)
        try:
            refreshed = get_topic(conn, topic_id)
        finally:
            conn.close()

        assert refreshed is not None
        assert refreshed.status == TopicStatus.ERROR
        assert refreshed.error_message == "Research timed out. Click Retry."

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
            patch("app.web.routes._INIT_TIMEOUT_SECONDS", 0.05),
            patch(
                "app.scraping.fetch_new_articles_for_topic",
                side_effect=_hang,
            ),
            caplog.at_level(logging.ERROR, logger="app.web.routes"),
        ):
            await _run_init(topic_id, settings, db_path)

        assert any("timed out" in record.message.lower() for record in caplog.records)

    async def test_normal_completion_sets_topic_to_ready(self, db_path: Path) -> None:
        """When everything succeeds, topic is set to READY."""
        from app.models import Article

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
                "app.scraping.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=[fake_article],
            ),
            patch(
                "app.analysis.knowledge.initialize_knowledge",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "app.crud.mark_articles_processed",
                return_value=None,
            ),
        ):
            await _run_init(topic_id, settings, db_path)

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


class TestRunCheckAllTimeout:
    """Tests for _run_check_all() timeout behaviour."""

    async def test_timeout_logs_and_does_not_crash(self, db_path: Path, caplog) -> None:
        """When check_all_topics hangs beyond the timeout, a warning is logged and the task returns cleanly."""
        import logging

        settings = _make_settings()

        async def _hang(*args, **kwargs):
            await asyncio.sleep(9999)

        with (
            patch("app.web.routes._CHECK_ALL_TIMEOUT_SECONDS", 0.05),
            patch("app.web.routes.check_all_topics", side_effect=_hang),
            caplog.at_level(logging.ERROR, logger="app.web.routes"),
        ):
            # Should complete without raising
            await _run_check_all(settings, db_path)

        assert any("timed out" in record.message.lower() for record in caplog.records)

    async def test_timeout_clears_checking_state(self, db_path: Path) -> None:
        """After a timeout, the checking-all flag is cleared so the next run can proceed."""
        from app.web.routes import _checking_state

        settings = _make_settings()

        async def _hang(*args, **kwargs):
            await asyncio.sleep(9999)

        with (
            patch("app.web.routes._CHECK_ALL_TIMEOUT_SECONDS", 0.05),
            patch("app.web.routes.check_all_topics", side_effect=_hang),
        ):
            await _run_check_all(settings, db_path)

        assert not await _checking_state.is_checking_all()

    async def test_normal_completion_returns_cleanly(self, db_path: Path) -> None:
        """When check_all_topics completes normally, the task finishes without error."""
        from app.web.routes import _checking_state

        settings = _make_settings()

        with patch(
            "app.web.routes.check_all_topics",
            new_callable=AsyncMock,
            return_value=[],
        ):
            await _run_check_all(settings, db_path)

        assert not await _checking_state.is_checking_all()

    async def test_normal_completion_does_not_log_error(self, db_path: Path, caplog) -> None:
        """Successful run produces no error-level log entries."""
        import logging

        settings = _make_settings()

        with (
            patch(
                "app.web.routes.check_all_topics",
                new_callable=AsyncMock,
                return_value=[],
            ),
            caplog.at_level(logging.ERROR, logger="app.web.routes"),
        ):
            await _run_check_all(settings, db_path)

        assert not caplog.records
