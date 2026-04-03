"""Tests for the first-run setup wizard."""

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app


@pytest.fixture
def unconfigured_app(tmp_path: Path):
    """Create a test client with an unconfigured app state."""
    from app.database import init_db

    db_path = tmp_path / "test.db"
    init_db(db_path)

    with TestClient(app, raise_server_exceptions=False) as client:
        # Set state after lifespan runs to avoid it being overwritten
        app.state.settings = Settings()  # type: ignore[call-arg]
        app.state.db_path = db_path
        app.state.setup_required = True
        yield client

    # Reset state
    app.state.setup_required = False


@pytest.fixture
def configured_app(tmp_path: Path, sample_config_yaml: Path, monkeypatch: pytest.MonkeyPatch):
    """Create a test client with a configured app state."""
    from app.config import load_settings
    from app.database import init_db

    monkeypatch.delenv("TOPIC_WATCH_LLM__API_KEY", raising=False)
    monkeypatch.delenv("TOPIC_WATCH_LLM__MODEL", raising=False)

    db_path = tmp_path / "test.db"
    init_db(db_path)
    settings = load_settings(config_path=sample_config_yaml)

    with TestClient(app, raise_server_exceptions=False) as client:
        # Set state after lifespan runs to avoid it being overwritten
        app.state.settings = settings
        app.state.db_path = db_path
        app.state.setup_required = False
        yield client


class TestSetupRedirect:
    """Test that unconfigured apps redirect to /setup."""

    def test_root_redirects_to_setup(self, unconfigured_app: TestClient) -> None:
        response = unconfigured_app.get("/", follow_redirects=False)
        assert response.status_code == 307
        assert response.headers["location"] == "/setup"

    def test_settings_redirects_to_setup(self, unconfigured_app: TestClient) -> None:
        response = unconfigured_app.get("/settings", follow_redirects=False)
        assert response.status_code == 307
        assert response.headers["location"] == "/setup"

    def test_health_not_redirected(self, unconfigured_app: TestClient) -> None:
        response = unconfigured_app.get("/health")
        assert response.status_code == 200

    def test_static_not_redirected(self, unconfigured_app: TestClient) -> None:
        response = unconfigured_app.get("/static/vendor/pico.min.css")
        # Either 200 or 404 is fine — point is it's not a redirect
        assert response.status_code != 307

    def test_setup_page_not_redirected(self, unconfigured_app: TestClient) -> None:
        response = unconfigured_app.get("/setup")
        assert response.status_code == 200


class TestSetupWizard:
    """Test the setup wizard GET and POST routes."""

    def test_get_setup_shows_form(self, unconfigured_app: TestClient) -> None:
        response = unconfigured_app.get("/setup")
        assert response.status_code == 200
        assert "Welcome to Topic Watch" in response.text
        assert 'name="llm_model"' in response.text
        assert 'name="llm_api_key"' in response.text

    def test_get_setup_when_configured_redirects(self, configured_app: TestClient) -> None:
        response = configured_app.get("/setup", follow_redirects=False)
        assert response.status_code == 303
        assert response.headers["location"] == "/"

    def test_post_setup_success(self, unconfigured_app: TestClient) -> None:
        # Get CSRF token first
        get_response = unconfigured_app.get("/setup")
        csrf_token = get_response.cookies.get("csrf_token")
        assert csrf_token

        with patch("app.scheduler.start_scheduler") as mock_sched:
            mock_sched.return_value = None

            response = unconfigured_app.post(
                "/setup",
                data={
                    "llm_model": "openai/gpt-4o-mini",
                    "llm_api_key": "sk-test-key-123",
                    "llm_base_url": "",
                    "csrf_token": csrf_token,
                },
                follow_redirects=False,
            )

        assert response.status_code == 303
        assert response.headers["location"] == "/"
        assert app.state.setup_required is False
        assert app.state.settings.llm.model == "openai/gpt-4o-mini"
        assert app.state.settings.llm.api_key == "sk-test-key-123"

    def test_post_setup_with_base_url(self, unconfigured_app: TestClient) -> None:
        get_response = unconfigured_app.get("/setup")
        csrf_token = get_response.cookies.get("csrf_token")

        with patch("app.scheduler.start_scheduler"):
            response = unconfigured_app.post(
                "/setup",
                data={
                    "llm_model": "ollama/llama3",
                    "llm_api_key": "ollama",
                    "llm_base_url": "http://localhost:11434",
                    "csrf_token": csrf_token,
                },
                follow_redirects=False,
            )

        assert response.status_code == 303
        assert app.state.settings.llm.base_url == "http://localhost:11434"

    def test_post_setup_cloud_provider_strips_base_url(self, unconfigured_app: TestClient) -> None:
        """POST /setup with a cloud provider model and stale base_url strips base_url."""
        get_response = unconfigured_app.get("/setup")
        csrf_token = get_response.cookies.get("csrf_token")

        with patch("app.scheduler.start_scheduler"):
            response = unconfigured_app.post(
                "/setup",
                data={
                    "llm_model": "anthropic/claude-haiku-4-5",
                    "llm_api_key": "sk-ant-test",
                    "llm_base_url": "http://localhost:11434",
                    "csrf_token": csrf_token,
                },
                follow_redirects=False,
            )

        assert response.status_code == 303
        assert app.state.settings.llm.base_url is None

    def test_post_setup_nav_hidden(self, unconfigured_app: TestClient) -> None:
        response = unconfigured_app.get("/setup")
        assert response.status_code == 200
        # Nav links should not be present in setup mode
        assert 'href="/settings"' not in response.text
        assert 'href="/feeds"' not in response.text
