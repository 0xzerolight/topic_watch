"""Tests for the CLI module: commands and error handling."""

import sqlite3
from functools import partial
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from app.cli import _cmd_check, _cmd_init, _cmd_list
from app.config import LLMSettings, Settings
from app.crud import create_topic, get_topic, get_topic_by_name
from app.database import get_connection, get_db, init_db
from app.models import Article, Topic, TopicStatus
from app.scraping import FetchResult
from app.scraping.rss import FeedResponse


def _make_settings(**overrides) -> Settings:
    defaults = {
        "llm": LLMSettings(model="openai/gpt-4o-mini", api_key="test-key"),
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _real_db(tmp_path: Path) -> Path:
    """Initialize a real on-disk DB and return its path.

    Tests using this exercise the *real* ``get_db`` commit/rollback path on a
    tmp DB (never the shared in-memory ``db_conn`` whose ``__exit__`` is faked),
    so they catch writes lost to a mid-context ``sys.exit`` (OVH-002).
    """
    db_path = tmp_path / "cli.db"
    init_db(db_path)
    return db_path


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
        # status_changed_at must be refreshed on the READY transition.
        assert updated.status_changed_at is not None
        state = get_knowledge_state(db_conn, topic.id)
        assert state is not None
        assert state.summary_text == "New knowledge."

    async def test_init_scraping_failure_sets_error(self, db_conn: sqlite3.Connection) -> None:
        topic = create_topic(
            db_conn,
            Topic(name="ScrapeErr", description="d", status=TopicStatus.NEW),
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
            Topic(name="NoArticles", description="d", status=TopicStatus.NEW),
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
        # status_changed_at must be refreshed on the ERROR transition too.
        assert updated.status_changed_at is not None

    async def test_init_knowledge_failure_sets_error(self, db_conn: sqlite3.Connection) -> None:
        topic = create_topic(
            db_conn,
            Topic(
                name="KnowledgeFail",
                description="d",
                status=TopicStatus.NEW,
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


class TestCmdInitPersistence:
    """OVH-002: init error-status writes must actually COMMIT on a real DB.

    These tests run the real ``get_db`` against a tmp file (not the faked
    ``db_conn`` __exit__), then re-open a *fresh* connection to assert the
    ERROR status survived — proving ``sys.exit`` no longer fires inside the
    ``get_db`` context and rolls the write back.
    """

    async def test_scrape_failure_commits_error_status(self, tmp_path: Path) -> None:
        db_path = _real_db(tmp_path)
        with get_db(db_path) as setup:
            topic = create_topic(
                setup,
                Topic(name="ScrapeErr", description="d", status=TopicStatus.NEW),
            )

        settings = _make_settings()
        with (
            patch("app.cli.load_settings", return_value=settings),
            patch("app.cli.init_db"),
            patch("app.cli.get_db", partial(get_db, db_path=db_path)),
            patch(
                "app.scraping.fetch_feeds_for_topic",
                new_callable=AsyncMock,
                side_effect=Exception("Network error"),
            ),
            pytest.raises(SystemExit, match="1"),
        ):
            await _cmd_init("ScrapeErr")

        # Fresh connection: only a committed write is visible here.
        verify = get_connection(db_path)
        try:
            updated = get_topic(verify, topic.id)
        finally:
            verify.close()
        assert updated.status == TopicStatus.ERROR
        assert updated.error_message is not None
        assert "fetch articles" in updated.error_message.lower()

    async def test_no_articles_commits_error_status(self, tmp_path: Path) -> None:
        db_path = _real_db(tmp_path)
        with get_db(db_path) as setup:
            topic = create_topic(
                setup,
                Topic(name="NoArticles", description="d", status=TopicStatus.NEW),
            )

        settings = _make_settings()
        with (
            patch("app.cli.load_settings", return_value=settings),
            patch("app.cli.init_db"),
            patch("app.cli.get_db", partial(get_db, db_path=db_path)),
            patch(
                "app.scraping.fetch_feeds_for_topic",
                new_callable=AsyncMock,
                return_value=FeedResponse(),
            ),
            pytest.raises(SystemExit, match="1"),
        ):
            await _cmd_init("NoArticles")

        verify = get_connection(db_path)
        try:
            updated = get_topic(verify, topic.id)
        finally:
            verify.close()
        assert updated.status == TopicStatus.ERROR
        assert updated.error_message is not None
        assert "no articles" in updated.error_message.lower()

    async def test_knowledge_failure_commits_error_status(self, tmp_path: Path) -> None:
        db_path = _real_db(tmp_path)
        with get_db(db_path) as setup:
            topic = create_topic(
                setup,
                Topic(
                    name="KnowledgeFail",
                    description="d",
                    status=TopicStatus.NEW,
                    feed_urls=["https://example.com/feed.xml"],
                ),
            )

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
            patch("app.cli.get_db", partial(get_db, db_path=db_path)),
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
            pytest.raises(SystemExit, match="1"),
        ):
            await _cmd_init("KnowledgeFail")

        verify = get_connection(db_path)
        try:
            updated = get_topic(verify, topic.id)
        finally:
            verify.close()
        assert updated.status == TopicStatus.ERROR
        assert updated.error_message is not None
        assert "llm" in updated.error_message.lower()


class TestCmdInitResearchingClaim:
    """OVH-018: CLI init claims RESEARCHING and excludes a concurrent init."""

    async def test_claims_researching_before_long_work(self, tmp_path: Path) -> None:
        """The RESEARCHING claim must be committed *before* fetch/LLM run, so a
        concurrent scheduler/CLI init observing the topic sees it as taken."""
        db_path = _real_db(tmp_path)
        with get_db(db_path) as setup:
            create_topic(
                setup,
                Topic(name="Claim", description="d", status=TopicStatus.NEW),
            )

        observed: dict[str, TopicStatus] = {}

        async def _fetch_spy(*args, **kwargs):
            # Open an independent connection mid-fetch and read the status a
            # concurrent initializer would see — proving the claim committed.
            other = get_connection(db_path)
            try:
                seen = get_topic_by_name(other, "Claim")
                observed["status"] = seen.status
            finally:
                other.close()
            return FetchResult(articles=[], total_feed_entries=0)

        settings = _make_settings()
        with (
            patch("app.cli.load_settings", return_value=settings),
            patch("app.cli.init_db"),
            patch("app.cli.get_db", partial(get_db, db_path=db_path)),
            patch("app.scraping.fetch_new_articles_for_topic", new=_fetch_spy),
            # No articles -> ERROR exit, but the claim must already be visible.
            pytest.raises(SystemExit, match="1"),
        ):
            await _cmd_init("Claim")

        assert observed.get("status") == TopicStatus.RESEARCHING

    async def test_already_researching_is_excluded(self, tmp_path: Path) -> None:
        """A second init on an already-RESEARCHING topic bails without fetching."""
        db_path = _real_db(tmp_path)
        with get_db(db_path) as setup:
            create_topic(
                setup,
                Topic(name="Busy", description="d", status=TopicStatus.RESEARCHING),
            )

        settings = _make_settings()
        fetch_mock = AsyncMock(return_value=FetchResult(articles=[], total_feed_entries=0))
        with (
            patch("app.cli.load_settings", return_value=settings),
            patch("app.cli.init_db"),
            patch("app.cli.get_db", partial(get_db, db_path=db_path)),
            patch("app.scraping.fetch_new_articles_for_topic", fetch_mock),
            pytest.raises(SystemExit, match="1"),
        ):
            await _cmd_init("Busy")

        # Bailed before any fetch/LLM work was attempted.
        fetch_mock.assert_not_called()


class TestCmdLogging:
    """OVH-042: CLI logs carry the check_id correlation id."""

    def test_cli_consolidates_onto_shared_setup_logging(self) -> None:
        """The duplicate _setup_logging is gone; main uses the shared config."""
        import app.cli as cli_mod
        from app.logging_config import setup_logging as shared_setup_logging

        assert not hasattr(cli_mod, "_setup_logging")
        # main() must call the shared setup_logging (imported into cli's namespace).
        assert cli_mod.setup_logging is shared_setup_logging

    def test_cli_logs_render_check_id(self, monkeypatch) -> None:
        """A check_id set in context renders in the configured CLI log output.

        Mirrors a real CLI run: clear root handlers (fresh process), run the
        shared setup_logging in text mode, then format a record through the
        configured handler and assert the correlation id is present.
        """
        import io
        import logging

        from app.check_context import check_id_var
        from app.cli import setup_logging

        monkeypatch.delenv("TOPIC_WATCH_LOG_FORMAT", raising=False)

        root = logging.root
        saved_handlers = root.handlers[:]
        saved_level = root.level
        root.handlers.clear()
        try:
            setup_logging()
            handler = root.handlers[0]
            # Redirect the configured handler to a buffer and emit through it.
            buf = io.StringIO()
            handler.setStream(buf)  # type: ignore[attr-defined]
            token = check_id_var.set("abc12345")
            try:
                record = logging.LogRecord("app.cli", logging.INFO, __file__, 0, "hello cli", None, None)
                for filt in handler.filters:
                    filt.filter(record)
                handler.emit(record)
            finally:
                check_id_var.reset(token)
            output = buf.getvalue()
        finally:
            root.handlers.clear()
            root.handlers.extend(saved_handlers)
            root.setLevel(saved_level)

        assert "abc12345" in output
        assert "hello cli" in output


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
