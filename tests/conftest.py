"""Shared test fixtures for Topic Watch tests."""

import os
import socket
import sqlite3
from collections.abc import Generator
from pathlib import Path

import pytest

# Make the app self-configured for tests (mirrors the CI env). Without these the
# lifespan marks the app setup-required and the SetupRedirectMiddleware 307s every
# /api request to /setup, so tests relying on a configured app fail when run in
# isolation or in an order that does not happen to leak a configured state.
os.environ.setdefault("TOPIC_WATCH_LLM__MODEL", "openai/gpt-4o-mini")
os.environ.setdefault("TOPIC_WATCH_LLM__API_KEY", "test-key-not-real")

from app.database import get_connection, init_db
from app.main import app


@pytest.fixture(autouse=True)
def _isolate_app_state():
    """Snapshot/restore shared FastAPI app state between tests.

    The app is a module-global imported across test files; a test that mutates
    app.dependency_overrides or app.state and fails to clean up otherwise bleeds
    into later tests, producing order-dependent failures. Reset around every test.
    """
    overrides = dict(app.dependency_overrides)
    app_state = dict(app.state._state)
    yield
    app.dependency_overrides.clear()
    app.dependency_overrides.update(overrides)
    app.state._state.clear()
    app.state._state.update(app_state)


@pytest.fixture(autouse=True)
def _stub_dns_resolution(monkeypatch: pytest.MonkeyPatch):
    """Resolve any hostname to a public IP by default.

    SSRF validation (app.url_validation) resolves DNS at check time and now
    fails closed on resolution failure. The test sandbox has no network, so
    without this stub every public test host would be blocked. Tests that need
    a private IP or a resolution failure override socket.getaddrinfo themselves.
    """

    def _public(*_args, **_kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]

    monkeypatch.setattr(socket, "getaddrinfo", _public)


@pytest.fixture(autouse=True)
def _safe_config_path(tmp_path: Path):
    """Ensure app.state.config_path always points to a temp directory.

    Prevents tests from accidentally writing to the real data/config.yml.
    This runs for every test automatically.
    """
    app.state.config_path = tmp_path / "config.yml"


@pytest.fixture(autouse=True)
def _safe_lifespan_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point the app lifespan's init_db at a temp DB.

    The lifespan resolves its DB via ``app.main.resolve_db_path(settings)`` and
    calls ``init_db`` on it. Under ``with TestClient(app)`` (test_api, test_setup)
    that otherwise writes the real ``data/topic_watch.db`` and creates
    ``data/backups``. Mirrors ``_safe_config_path`` for the DB path so no test
    touches the real data/ directory.
    """
    monkeypatch.setattr("app.main.resolve_db_path", lambda settings: tmp_path / "lifespan.db")


@pytest.fixture(autouse=True)
def _reset_checking_state():
    """Reset the in-progress check tracker between tests to prevent bleed.

    ``_checking_state`` is a process-global guard. A web test that enqueues a
    check via BackgroundTasks can leave the per-topic or whole-cycle flag set if
    the task hasn't drained by the time the test ends; that leaks into a later
    test, where ``check_all_topics`` then short-circuits (returns ``[]``) and
    ``check_topic`` dedupes. Clear it around every test.
    """
    from app.web.state import _checking_state

    def _clear() -> None:
        _checking_state._topics.clear()
        _checking_state._start_times.clear()
        _checking_state._checking_all = False

    _clear()
    yield
    _clear()


@pytest.fixture
def db_conn(tmp_path: Path) -> Generator[sqlite3.Connection, None, None]:
    """Provide a fresh database with schema initialized."""
    db_path = tmp_path / "test.db"
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def sample_config_yaml(tmp_path: Path) -> Path:
    """Create a valid config YAML file and return its path."""
    config_file = tmp_path / "config.yml"
    config_file.write_text(
        """
llm:
  model: "openai/gpt-4o-mini"
  api_key: "test-api-key-12345"

notifications:
  urls:
    - "json://localhost"

check_interval: "6h"
max_articles_per_check: 10
knowledge_state_max_tokens: 2000
"""
    )
    return config_file


@pytest.fixture
def minimal_config_yaml(tmp_path: Path) -> Path:
    """Create a minimal config YAML with only required fields."""
    config_file = tmp_path / "config.yml"
    config_file.write_text(
        """
llm:
  model: "openai/gpt-4o-mini"
  api_key: "test-key"
"""
    )
    return config_file
