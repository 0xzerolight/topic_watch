"""Shared test fixtures for Topic Watch tests."""

import sqlite3
from collections.abc import Generator
from pathlib import Path

import pytest

from app.database import get_connection, init_db
from app.main import app


@pytest.fixture(autouse=True)
def _safe_config_path(tmp_path: Path):
    """Ensure app.state.config_path always points to a temp directory.

    Prevents tests from accidentally writing to the real data/config.yml.
    This runs for every test automatically.
    """
    app.state.config_path = tmp_path / "config.yml"


@pytest.fixture(autouse=True)
def _reset_stats_cache():
    """Reset the dashboard stats cache between tests to prevent bleed."""
    from app.web import routes

    routes._stats_cache["data"] = None
    routes._stats_cache["expires"] = 0.0
    yield
    routes._stats_cache["data"] = None
    routes._stats_cache["expires"] = 0.0


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
