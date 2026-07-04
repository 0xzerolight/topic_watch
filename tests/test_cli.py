"""Tests for the CLI module: commands and error handling."""

import sqlite3
from functools import partial
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from app.cli import _cmd_check, _cmd_doctor, _cmd_init, _cmd_list
from app.config import LLMSettings, NotificationSettings, Settings
from app.crud import (
    create_topic,
    get_topic,
    get_topic_by_name,
    upsert_feed_health_failure,
    upsert_feed_health_success,
)
from app.database import get_connection, get_db, get_schema_version, init_db
from app.migrations import MIGRATIONS
from app.models import Article, FeedMode, Topic, TopicStatus
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


class TestCmdDoctor:
    """Tests for the 'doctor' diagnostic command.

    doctor must be secret-safe (no api_key/token ever printed) and read-only
    against the database (never creating or migrating it).
    """

    def _run(
        self,
        settings: Settings,
        capsys: pytest.CaptureFixture[str],
        *,
        in_docker: bool = False,
        env_sourced: bool = False,
    ) -> str:
        with (
            patch("app.cli.load_settings", return_value=settings),
            patch("app.cli._in_docker", return_value=in_docker),
            patch("app.cli.is_api_key_env_sourced", return_value=env_sourced),
        ):
            _cmd_doctor()
        return capsys.readouterr().out

    def test_version_line_present(self, capsys: pytest.CaptureFixture[str]) -> None:
        from app import __version__

        out = self._run(_make_settings(db_path="/nonexistent/x.db"), capsys)
        assert f"version: {__version__}" in out
        assert "python:" in out and "os:" in out

    def test_api_key_rendered_as_boolean_not_value(self, capsys: pytest.CaptureFixture[str]) -> None:
        s = _make_settings(
            llm=LLMSettings(model="openai/gpt-4o-mini", api_key="SECRETVALUE"),
            db_path="/nonexistent/x.db",
        )
        out = self._run(s, capsys)
        assert "llm.api_key: set" in out
        assert "SECRETVALUE" not in out

    def test_env_sourced_key_marked_and_not_leaked(self, capsys: pytest.CaptureFixture[str]) -> None:
        s = _make_settings(
            llm=LLMSettings(model="openai/gpt-4o-mini", api_key="SECRETVALUE"),
            db_path="/nonexistent/x.db",
        )
        out = self._run(s, capsys, env_sourced=True)
        assert "(from env)" in out
        assert "SECRETVALUE" not in out

    def test_notification_urls_scheme_count_only_no_leak(self, capsys: pytest.CaptureFixture[str]) -> None:
        urls = ["slack://TokenA/TokenB/TokenC", "pover://USERKEY@APPTOKEN", "https://ntfy.sh/shh"]
        s = _make_settings(notifications=NotificationSettings(urls=urls), db_path="/nonexistent/x.db")
        out = self._run(s, capsys)
        for secret in ("TokenA", "TokenB", "TokenC", "USERKEY", "APPTOKEN", "/shh"):
            assert secret not in out
        assert "notifications.urls: 3" in out
        assert "slack x1" in out and "pover x1" in out and "https x1" in out

    def test_base_url_redacted_keeps_host_drops_creds(self, capsys: pytest.CaptureFixture[str]) -> None:
        s = _make_settings(
            llm=LLMSettings(
                model="ollama/llama3",
                api_key="k",
                base_url="https://user:k3y@ollama.internal/v1?api-key=SEKRET",
            ),
            db_path="/nonexistent/x.db",
        )
        out = self._run(s, capsys)
        assert "k3y" not in out
        assert "SEKRET" not in out
        assert "user:" not in out
        assert "ollama.internal" in out

    def test_failing_feed_url_redacted(self, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        db_path = _real_db(tmp_path)
        with get_db(db_path) as conn:
            upsert_feed_health_failure(conn, "https://example.com/feed?key=SECRETTOKEN", "boom")
            upsert_feed_health_failure(conn, "https://example.com/feed?key=SECRETTOKEN", "boom")
        out = self._run(_make_settings(db_path=str(db_path)), capsys)
        assert "SECRETTOKEN" not in out
        assert "feeds: 0 OK / 1 failing" in out
        assert "(x2)" in out

    def test_exa_block_rendered_secret_safe(self, capsys: pytest.CaptureFixture[str]) -> None:
        from app.config import ExaSettings

        s = _make_settings(
            exa=ExaSettings(enabled=True, api_key="EXASECRET"),
            db_path="/nonexistent/x.db",
        )
        out = self._run(s, capsys)
        assert "exa.enabled: True" in out
        assert "exa.api_key: set" in out
        assert "EXASECRET" not in out

    def test_exa_key_not_set_and_env_marked(self, capsys: pytest.CaptureFixture[str]) -> None:
        s = _make_settings(db_path="/nonexistent/x.db")
        out = self._run(s, capsys)
        assert "exa.enabled: False" in out
        assert "exa.api_key: not set" in out
        # env-sourced marker uses the exa-specific helper
        with (
            patch("app.cli.load_settings", return_value=s),
            patch("app.cli._in_docker", return_value=False),
            patch("app.cli.is_exa_key_env_sourced", return_value=True),
        ):
            from app.cli import _cmd_doctor

            _cmd_doctor()
        out2 = capsys.readouterr().out
        assert "exa.api_key: not set (from env)" in out2

    def test_topic_and_feed_counts(self, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        db_path = _real_db(tmp_path)
        with get_db(db_path) as conn:
            create_topic(conn, Topic(name="A", description="d", status=TopicStatus.READY))
            create_topic(conn, Topic(name="B", description="d", status=TopicStatus.READY))
            create_topic(conn, Topic(name="C", description="d", status=TopicStatus.ERROR))
            upsert_feed_health_success(conn, "https://ok.example/feed")
            upsert_feed_health_failure(conn, "https://bad.example/feed", "boom")
        out = self._run(_make_settings(db_path=str(db_path)), capsys)
        assert "ready 2" in out
        assert "error 1" in out
        assert "feeds: 1 OK / 1 failing" in out

    def test_not_configured_renders_no(self, capsys: pytest.CaptureFixture[str]) -> None:
        s = _make_settings(llm=LLMSettings(model="", api_key=""), db_path="/nonexistent/x.db")
        out = self._run(s, capsys)
        assert "is_configured: no" in out

    def test_config_load_failure_still_prints_version(self, capsys: pytest.CaptureFixture[str]) -> None:
        from app import __version__

        with (
            patch("app.cli.load_settings", side_effect=RuntimeError("bad yaml")),
            patch("app.cli._in_docker", return_value=False),
        ):
            _cmd_doctor()
        out = capsys.readouterr().out
        assert "configuration: unavailable" in out
        assert f"version: {__version__}" in out

    def test_database_path_and_schema_version(self, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        db_path = _real_db(tmp_path)
        out = self._run(_make_settings(db_path=str(db_path)), capsys)
        assert f"database: {db_path}" in out
        latest = max(v for v, _, _ in MIGRATIONS)
        assert f"schema: {latest}" in out

    def test_missing_db_creates_nothing(self, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        absent_dir = tmp_path / "absent"
        db_path = absent_dir / "x.db"
        out = self._run(_make_settings(db_path=str(db_path)), capsys)
        assert "unavailable (file not found)" in out
        assert not absent_dir.exists()
        assert not db_path.exists()

    def test_deployment_docker(self, capsys: pytest.CaptureFixture[str]) -> None:
        out = self._run(_make_settings(db_path="/nonexistent/x.db"), capsys, in_docker=True)
        assert "deployment: docker" in out

    def test_deployment_local(self, capsys: pytest.CaptureFixture[str]) -> None:
        out = self._run(_make_settings(db_path="/nonexistent/x.db"), capsys, in_docker=False)
        assert "deployment: local" in out

    def test_corrupt_db_degrades_without_raising(self, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        db_path = tmp_path / "corrupt.db"
        db_path.write_bytes(b"not a database")
        out = self._run(_make_settings(db_path=str(db_path)), capsys)
        assert "unavailable" in out  # never raises, never exits

    def test_get_schema_version_tableless_returns_zero(self, tmp_path: Path) -> None:
        db_path = tmp_path / "empty.db"
        conn = sqlite3.connect(str(db_path))
        try:
            assert get_schema_version(conn) == 0
        finally:
            conn.close()

    def test_get_schema_version_populated(self, tmp_path: Path) -> None:
        db_path = _real_db(tmp_path)
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            assert get_schema_version(conn) == max(v for v, _, _ in MIGRATIONS)
        finally:
            conn.close()


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

    async def test_init_all_sources_failed_message(self, db_conn: sqlite3.Connection) -> None:
        """A total source failure (e.g. bad Exa key) reports credentials, not 'no articles'."""
        topic = create_topic(
            db_conn,
            Topic(name="ExaKeyBad", description="d", feed_mode=FeedMode.EXA, feed_urls=[], status=TopicStatus.NEW),
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
                return_value=FeedResponse(provider_name="exa", feeds_total=1, feeds_failed=1),
            ),
        ):
            mock_get_db.return_value.__enter__ = lambda s: db_conn
            mock_get_db.return_value.__exit__ = lambda s, *a: None
            with pytest.raises(SystemExit, match="1"):
                await _cmd_init("ExaKeyBad")

        updated = get_topic(db_conn, topic.id)
        assert updated.status == TopicStatus.ERROR
        assert updated.error_message.startswith("All feed source(s) failed")

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
