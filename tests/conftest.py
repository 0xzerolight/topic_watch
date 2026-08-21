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
_TEST_ENV_DEFAULTS = {
    "TOPIC_WATCH_LLM__MODEL": "openai/gpt-4o-mini",
    "TOPIC_WATCH_LLM__API_KEY": "test-key-not-real",
    # Test clients address the app as "test"/"testserver", which the Host
    # allowlist (AUG-002) rejects. Disable the check for the module-global app;
    # the allowlist's own behavior is covered against purpose-built apps in
    # tests/test_host_allowlist.py.
    "TOPIC_WATCH_ALLOWED_HOSTS": "*",
}


def _scrub_ambient_env(environ: dict) -> None:
    """Remove every ambient ``TOPIC_WATCH_*`` var, then seed exact test values.

    ``Settings`` gives every ``TOPIC_WATCH_*`` env var precedence over YAML and
    defaults (env > YAML > defaults), so a developer's configured shell — a real
    LLM key, a non-default check interval, Exa settings — otherwise leaks into
    every bare ``Settings()`` built in-process, running the suite against
    different config than CI (AUG-039). ``os.environ.setdefault`` does not fix
    this: it only supplies a value when one is absent, so an ambient value for
    one of these three keys still wins. Tests that want a specific env value
    opt it back in with ``monkeypatch.setenv``.
    """
    for key in [k for k in environ if k.startswith("TOPIC_WATCH_")]:
        del environ[key]
    environ.update(_TEST_ENV_DEFAULTS)


_scrub_ambient_env(os.environ)

from app.database import get_connection, init_db  # noqa: E402 -- must follow the env scrub above
from app.main import app  # noqa: E402 -- must follow the env scrub above


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
def _safe_config_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Keep the YAML settings source off the developer's real data/config.yml.

    ``Settings()`` runs ``settings_customise_sources``, which falls back to
    ``DEFAULT_CONFIG_PATH`` whenever no explicit override is set — so a bare
    ``Settings(...)`` in a test silently inherits whatever the developer has
    configured locally (an Exa key, a non-default interval). CI has no such file,
    so those tests pass there and fail on a real machine. Point the fallback at a
    nonexistent temp path: unset fields resolve to their declared defaults.

    ``_safe_config_path`` is the write-side counterpart; this is the read side.
    """
    monkeypatch.setattr("app.config.DEFAULT_CONFIG_PATH", tmp_path / "absent-config.yml")


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
def _safe_default_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point ``get_db(None)`` / ``init_db(None)`` at a temp DB, never the real one.

    The pipeline now opens its own short-lived connections from a path instead of
    borrowing the caller's connection, so a code path that forgets to thread
    ``db_path`` falls back to ``DEFAULT_DB_PATH`` — the developer's real
    ``data/topic_watch.db``. Redirect the fallback so such a test fails loudly on
    an empty database instead of quietly mutating live data. Mirrors
    ``_safe_config_path`` and ``_safe_lifespan_db``.
    """
    monkeypatch.setattr("app.database.DEFAULT_DB_PATH", tmp_path / "default-fallback.db")


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
        _checking_state._checking_all = None

    _clear()
    yield
    _clear()


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Path of this test's database file.

    The pipeline no longer accepts a caller's connection — it opens a short-lived
    one per phase from a path — so tests that drive ``check_topic`` /
    ``initialize_new_topic`` / ``fetch_new_articles_for_topic`` pass this instead
    of ``db_conn``. Request ``db_conn`` too when the test also wants to read or
    seed rows directly; both name the same file.
    """
    return tmp_path / "test.db"


@pytest.fixture
def db_conn(db_path: Path) -> Generator[sqlite3.Connection, None, None]:
    """Provide a fresh database with schema initialized."""
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
