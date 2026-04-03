"""Tests for the CLI module: commands and error handling."""

import sqlite3
from unittest.mock import AsyncMock, patch

import pytest

from app.cli import _cmd_check, _cmd_init, _cmd_list
from app.config import LLMSettings, Settings
from app.crud import create_topic, get_topic
from app.models import Article, Topic, TopicStatus
from app.scraping import FetchResult
from app.scraping.rss import FeedResponse


def _make_settings(**overrides) -> Settings:
    defaults = {
        "llm": LLMSettings(model="openai/gpt-4o-mini", api_key="test-key"),
    }
    defaults.update(overrides)
    return Settings(**defaults)


class TestCmdCheck:
    """Tests for the 'check' CLI command."""

    async def test_check_existing_topic(self, db_conn: sqlite3.Connection) -> None:
        create_topic(
            db_conn,
            Topic(name="CLI Topic", description="d", status=TopicStatus.READY),
        )
        db_conn.commit()
        settings = _make_settings()

        with (
            patch("app.cli.load_settings", return_value=settings),
            patch("app.cli.init_db"),
            patch("app.cli.get_db") as mock_get_db,
            patch(
                "app.checker.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=[], total_feed_entries=0),
            ),
        ):
            mock_get_db.return_value.__enter__ = lambda s: db_conn
            mock_get_db.return_value.__exit__ = lambda s, *a: None
            await _cmd_check("CLI Topic")

    async def test_check_nonexistent_topic_exits(self, db_conn: sqlite3.Connection) -> None:
        settings = _make_settings()

        with (
            patch("app.cli.load_settings", return_value=settings),
            patch("app.cli.init_db"),
            patch("app.cli.get_db") as mock_get_db,
        ):
            mock_get_db.return_value.__enter__ = lambda s: db_conn
            mock_get_db.return_value.__exit__ = lambda s, *a: None
            with pytest.raises(SystemExit, match="1"):
                await _cmd_check("Nonexistent")


class TestCmdInit:
    """Tests for the 'init' CLI command — the most complex CLI path."""

    async def test_init_nonexistent_topic_exits(self, db_conn: sqlite3.Connection) -> None:
        settings = _make_settings()

        with (
            patch("app.cli.load_settings", return_value=settings),
            patch("app.cli.init_db"),
            patch("app.cli.get_db") as mock_get_db,
        ):
            mock_get_db.return_value.__enter__ = lambda s: db_conn
            mock_get_db.return_value.__exit__ = lambda s, *a: None
            with pytest.raises(SystemExit, match="1"):
                await _cmd_init("Nonexistent")

    async def test_init_ready_topic_reinitializes(self, db_conn: sqlite3.Connection) -> None:
        """READY topics should be re-initialized, not rejected."""
        from app.analysis.llm import KnowledgeStateUpdate
        from app.crud import create_knowledge_state, get_knowledge_state
        from app.models import KnowledgeState

        topic = create_topic(
            db_conn,
            Topic(name="Ready", description="d", status=TopicStatus.READY),
        )
        # Pre-existing knowledge state
        create_knowledge_state(
            db_conn,
            KnowledgeState(topic_id=topic.id, summary_text="Old knowledge.", token_count=10),
        )
        db_conn.commit()
        settings = _make_settings()

        mock_article = Article(
            topic_id=topic.id,
            title="Art",
            url="https://example.com/1",
            content_hash="h",
            source_feed="f",
        )
        llm_result = KnowledgeStateUpdate(
            sufficient_data=True, confidence=0.9, updated_summary="New knowledge.", token_count=15
        )

        with (
            patch("app.cli.load_settings", return_value=settings),
            patch("app.cli.init_db"),
            patch("app.cli.get_db") as mock_get_db,
            patch(
                "app.scraping.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=[mock_article], total_feed_entries=1),
            ),
            patch(
                "app.analysis.knowledge.generate_initial_knowledge",
                new_callable=AsyncMock,
                return_value=llm_result,
            ),
        ):
            mock_get_db.return_value.__enter__ = lambda s: db_conn
            mock_get_db.return_value.__exit__ = lambda s, *a: None
            await _cmd_init("Ready")

        updated = get_topic(db_conn, topic.id)
        assert updated.status == TopicStatus.READY
        state = get_knowledge_state(db_conn, topic.id)
        assert state is not None
        assert state.summary_text == "New knowledge."

    async def test_init_scraping_failure_sets_error(self, db_conn: sqlite3.Connection) -> None:
        topic = create_topic(
            db_conn,
            Topic(name="ScrapeErr", description="d", status=TopicStatus.RESEARCHING),
        )
        db_conn.commit()
        settings = _make_settings()

        with (
            patch("app.cli.load_settings", return_value=settings),
            patch("app.cli.init_db"),
            patch("app.cli.get_db") as mock_get_db,
            patch(
                "app.scraping.fetch_feeds_for_topic",
                new_callable=AsyncMock,
                side_effect=Exception("Network error"),
            ),
        ):
            mock_get_db.return_value.__enter__ = lambda s: db_conn
            mock_get_db.return_value.__exit__ = lambda s, *a: None
            with pytest.raises(SystemExit, match="1"):
                await _cmd_init("ScrapeErr")

        updated = get_topic(db_conn, topic.id)
        assert updated.status == TopicStatus.ERROR
        assert "fetch articles" in updated.error_message.lower()

    async def test_init_no_articles_sets_error(self, db_conn: sqlite3.Connection) -> None:
        topic = create_topic(
            db_conn,
            Topic(name="NoArticles", description="d", status=TopicStatus.RESEARCHING),
        )
        db_conn.commit()
        settings = _make_settings()

        with (
            patch("app.cli.load_settings", return_value=settings),
            patch("app.cli.init_db"),
            patch("app.cli.get_db") as mock_get_db,
            patch(
                "app.scraping.fetch_feeds_for_topic",
                new_callable=AsyncMock,
                return_value=FeedResponse(),
            ),
        ):
            mock_get_db.return_value.__enter__ = lambda s: db_conn
            mock_get_db.return_value.__exit__ = lambda s, *a: None
            with pytest.raises(SystemExit, match="1"):
                await _cmd_init("NoArticles")

        updated = get_topic(db_conn, topic.id)
        assert updated.status == TopicStatus.ERROR
        assert "no articles" in updated.error_message.lower()

    async def test_init_knowledge_failure_sets_error(self, db_conn: sqlite3.Connection) -> None:
        topic = create_topic(
            db_conn,
            Topic(
                name="KnowledgeFail",
                description="d",
                status=TopicStatus.RESEARCHING,
                feed_urls=["https://example.com/feed.xml"],
            ),
        )
        db_conn.commit()
        settings = _make_settings()

        mock_article = Article(
            id=1,
            topic_id=topic.id,
            title="Art",
            url="https://example.com/1",
            content_hash="h",
            source_feed="f",
        )

        with (
            patch("app.cli.load_settings", return_value=settings),
            patch("app.cli.init_db"),
            patch("app.cli.get_db") as mock_get_db,
            patch(
                "app.scraping.fetch_new_articles_for_topic",
                new_callable=AsyncMock,
                return_value=FetchResult(articles=[mock_article], total_feed_entries=1),
            ),
            patch(
                "app.analysis.knowledge.generate_initial_knowledge",
                new_callable=AsyncMock,
                side_effect=Exception("LLM error"),
            ),
        ):
            mock_get_db.return_value.__enter__ = lambda s: db_conn
            mock_get_db.return_value.__exit__ = lambda s, *a: None
            with pytest.raises(SystemExit, match="1"):
                await _cmd_init("KnowledgeFail")

        updated = get_topic(db_conn, topic.id)
        assert updated.status == TopicStatus.ERROR
        assert "llm" in updated.error_message.lower()


class TestCmdList:
    """Tests for the 'list' CLI command."""

    def test_list_empty(self, db_conn: sqlite3.Connection, capsys) -> None:
        with (
            patch("app.cli.init_db"),
            patch("app.cli.get_db") as mock_get_db,
        ):
            mock_get_db.return_value.__enter__ = lambda s: db_conn
            mock_get_db.return_value.__exit__ = lambda s, *a: None
            _cmd_list()

        captured = capsys.readouterr()
        assert "No topics configured" in captured.out

    def test_list_shows_topics(self, db_conn: sqlite3.Connection, capsys) -> None:
        create_topic(
            db_conn,
            Topic(name="Topic Alpha", description="d", status=TopicStatus.READY),
        )
        create_topic(
            db_conn,
            Topic(
                name="Topic Beta",
                description="d",
                status=TopicStatus.ERROR,
                is_active=False,
            ),
        )
        db_conn.commit()

        with (
            patch("app.cli.init_db"),
            patch("app.cli.get_db") as mock_get_db,
        ):
            mock_get_db.return_value.__enter__ = lambda s: db_conn
            mock_get_db.return_value.__exit__ = lambda s, *a: None
            _cmd_list()

        captured = capsys.readouterr()
        assert "Topic Alpha" in captured.out
        assert "ready" in captured.out
        assert "Topic Beta" in captured.out
        assert "error" in captured.out
