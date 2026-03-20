"""Tests for settings editing (POST /settings and save_settings_to_yaml)."""

import sqlite3
from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
import yaml

from app.config import LLMSettings, NotificationSettings, Settings
from app.main import app
from app.web.dependencies import get_db_conn, get_settings


def _make_settings(**overrides) -> Settings:
    defaults = {
        "llm": LLMSettings(model="openai/gpt-4o-mini", api_key="test-key-12345678"),
        "notifications": NotificationSettings(urls=["json://localhost"]),
    }
    defaults.update(overrides)
    return Settings(**defaults)


CSRF_TEST_TOKEN = "test-csrf-token-for-settings-tests"


@pytest.fixture
async def client(
    db_conn: sqlite3.Connection,
) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Test client with DB and settings overrides and CSRF token pre-set."""
    settings = _make_settings()

    # POST /settings reads request.app.state.settings directly, so set it here.
    app.state.settings = settings

    def override_db():
        yield db_conn

    def override_settings():
        return settings

    app.dependency_overrides[get_db_conn] = override_db
    app.dependency_overrides[get_settings] = override_settings

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        cookies={"csrf_token": CSRF_TEST_TOKEN},
        headers={"X-CSRF-Token": CSRF_TEST_TOKEN},
    ) as ac:
        yield ac

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Unit tests for save_settings_to_yaml
# ---------------------------------------------------------------------------


class TestSaveSettingsToYaml:
    """Direct unit tests for the save_settings_to_yaml function."""

    async def test_writes_valid_yaml(self, tmp_path: Path) -> None:
        """save_settings_to_yaml writes a readable YAML file."""
        from app.config import save_settings_to_yaml

        settings = _make_settings()
        config_file = tmp_path / "config.yml"
        save_settings_to_yaml(settings, config_file)

        assert config_file.exists()
        data = yaml.safe_load(config_file.read_text())
        assert isinstance(data, dict)
        assert data["llm"]["model"] == "openai/gpt-4o-mini"
        assert data["llm"]["api_key"] == "test-key-12345678"

    async def test_omits_none_base_url(self, tmp_path: Path) -> None:
        """base_url key should be absent from YAML when it is None."""
        from app.config import save_settings_to_yaml

        settings = _make_settings()
        assert settings.llm.base_url is None
        config_file = tmp_path / "config.yml"
        save_settings_to_yaml(settings, config_file)

        data = yaml.safe_load(config_file.read_text())
        assert "base_url" not in data.get("llm", {})

    async def test_includes_base_url_when_set(self, tmp_path: Path) -> None:
        """base_url should appear in YAML when it is set."""
        from app.config import save_settings_to_yaml

        settings = _make_settings(
            llm=LLMSettings(
                model="openai/gpt-4o-mini",
                api_key="test-key-12345678",
                base_url="http://localhost:11434",
            )
        )
        config_file = tmp_path / "config.yml"
        save_settings_to_yaml(settings, config_file)

        data = yaml.safe_load(config_file.read_text())
        assert data["llm"]["base_url"] == "http://localhost:11434"

    async def test_writes_notification_urls_as_list(self, tmp_path: Path) -> None:
        """Notification URLs are written as a YAML list."""
        from app.config import save_settings_to_yaml

        settings = _make_settings(notifications=NotificationSettings(urls=["ntfy://alerts", "discord://webhook/123"]))
        config_file = tmp_path / "config.yml"
        save_settings_to_yaml(settings, config_file)

        data = yaml.safe_load(config_file.read_text())
        assert data["notifications"]["urls"] == ["ntfy://alerts", "discord://webhook/123"]

    async def test_empty_notification_urls_omitted(self, tmp_path: Path) -> None:
        """Empty notification URLs list is omitted from the YAML."""
        from app.config import save_settings_to_yaml

        settings = _make_settings(notifications=NotificationSettings(urls=[]))
        config_file = tmp_path / "config.yml"
        save_settings_to_yaml(settings, config_file)

        data = yaml.safe_load(config_file.read_text())
        assert "urls" not in data.get("notifications", {})

    async def test_writes_scalar_settings(self, tmp_path: Path) -> None:
        """Scalar settings (interval, max articles, etc.) are written correctly."""
        from app.config import save_settings_to_yaml

        settings = _make_settings(check_interval_hours=12, max_articles_per_check=25)
        config_file = tmp_path / "config.yml"
        save_settings_to_yaml(settings, config_file)

        data = yaml.safe_load(config_file.read_text())
        assert data["check_interval_hours"] == 12
        assert data["max_articles_per_check"] == 25


# ---------------------------------------------------------------------------
# Route tests for GET /settings
# ---------------------------------------------------------------------------


class TestSettingsGet:
    """Tests for GET /settings."""

    async def test_settings_page_loads(self, client: httpx.AsyncClient) -> None:
        """GET /settings returns 200 with form elements."""
        response = await client.get("/settings")
        assert response.status_code == 200
        assert "<form" in response.text

    async def test_settings_page_shows_model(self, client: httpx.AsyncClient) -> None:
        """Settings page pre-populates the LLM model field."""
        response = await client.get("/settings")
        assert "openai/gpt-4o-mini" in response.text

    async def test_settings_page_has_interval_field(self, client: httpx.AsyncClient) -> None:
        """Settings page includes check_interval_hours field."""
        response = await client.get("/settings")
        assert "check_interval_hours" in response.text


# ---------------------------------------------------------------------------
# Route tests for POST /settings
# ---------------------------------------------------------------------------


class TestSettingsPost:
    """Tests for POST /settings."""

    def _valid_form_data(self, **overrides) -> dict:
        data = {
            "llm_model": "openai/gpt-4o-mini",
            "llm_api_key": "",
            "llm_base_url": "",
            "notification_urls": "",
            "webhook_urls": "",
            "check_interval_hours": "6",
            "max_articles_per_check": "10",
            "knowledge_state_max_tokens": "2000",
            "article_retention_days": "90",
            "feed_fetch_timeout": "15.0",
            "article_fetch_timeout": "20.0",
            "llm_analysis_timeout": "60",
            "llm_knowledge_timeout": "120",
            "web_page_size": "20",
        }
        data.update(overrides)
        return data

    async def test_valid_post_redirects(self, client: httpx.AsyncClient) -> None:
        """POST /settings with valid data redirects to /settings?saved=1."""
        with patch("app.web.routes.save_settings_to_yaml"):
            response = await client.post(
                "/settings",
                data=self._valid_form_data(),
                follow_redirects=False,
            )

        assert response.status_code == 303
        assert response.headers["location"] == "/settings?saved=1"

    async def test_valid_post_updates_app_state(self, client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
        """POST /settings updates app.state.settings with the new values."""
        with patch("app.web.routes.save_settings_to_yaml"):
            await client.post(
                "/settings",
                data=self._valid_form_data(
                    llm_model="anthropic/claude-3-haiku-20240307",
                    check_interval_hours="12",
                ),
                follow_redirects=False,
            )

        assert app.state.settings.llm.model == "anthropic/claude-3-haiku-20240307"
        assert app.state.settings.check_interval_hours == 12

    async def test_valid_post_calls_save(self, client: httpx.AsyncClient) -> None:
        """POST /settings calls save_settings_to_yaml."""
        with patch("app.web.routes.save_settings_to_yaml") as mock_save:
            await client.post(
                "/settings",
                data=self._valid_form_data(),
                follow_redirects=False,
            )
            mock_save.assert_called_once()

    async def test_invalid_interval_returns_error(self, client: httpx.AsyncClient) -> None:
        """POST with check_interval_hours=0 returns 422 with an error."""
        response = await client.post(
            "/settings",
            data=self._valid_form_data(check_interval_hours="0"),
            follow_redirects=False,
        )
        assert response.status_code == 422
        assert "check_interval_hours" in response.text.lower() or "error" in response.text.lower()

    async def test_empty_llm_model_returns_error(self, client: httpx.AsyncClient) -> None:
        """POST with empty llm_model returns 422."""
        response = await client.post(
            "/settings",
            data=self._valid_form_data(llm_model=""),
            follow_redirects=False,
        )
        assert response.status_code == 422

    async def test_requires_csrf(self, db_conn: sqlite3.Connection) -> None:
        """POST /settings without CSRF token returns 403."""
        settings = _make_settings()
        app.state.settings = settings

        def override_db():
            yield db_conn

        def override_settings():
            return settings

        app.dependency_overrides[get_db_conn] = override_db
        app.dependency_overrides[get_settings] = override_settings

        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
            ) as ac:
                response = await ac.post(
                    "/settings",
                    data={
                        "llm_model": "openai/gpt-4o-mini",
                        "check_interval_hours": "6",
                        "max_articles_per_check": "10",
                    },
                    follow_redirects=False,
                )
            assert response.status_code == 403
        finally:
            app.dependency_overrides.clear()

    async def test_parses_notification_urls_from_textarea(self, client: httpx.AsyncClient) -> None:
        """Newline-separated notification URLs are parsed into a list."""
        with patch("app.web.routes.save_settings_to_yaml"):
            await client.post(
                "/settings",
                data=self._valid_form_data(
                    notification_urls="ntfy://topic1\ndiscord://webhook/123\n",
                ),
                follow_redirects=False,
            )

        assert app.state.settings.notifications.urls == [
            "ntfy://topic1",
            "discord://webhook/123",
        ]

    async def test_empty_notification_urls_results_in_empty_list(self, client: httpx.AsyncClient) -> None:
        """Empty notification_urls textarea results in an empty list."""
        with patch("app.web.routes.save_settings_to_yaml"):
            await client.post(
                "/settings",
                data=self._valid_form_data(notification_urls=""),
                follow_redirects=False,
            )

        assert app.state.settings.notifications.urls == []

    async def test_saves_to_yaml_with_correct_content(self, client: httpx.AsyncClient, tmp_path: Path) -> None:
        """POST /settings writes the correct content to the YAML file."""
        config_file = tmp_path / "config.yml"

        def save_to_tmp(settings, config_path=None):
            from app.config import save_settings_to_yaml as _real_save

            _real_save(settings, config_file)

        with patch("app.web.routes.save_settings_to_yaml", side_effect=save_to_tmp):
            await client.post(
                "/settings",
                data=self._valid_form_data(
                    llm_model="openai/gpt-4o-mini",
                    check_interval_hours="8",
                ),
                follow_redirects=False,
            )

        assert config_file.exists()
        data = yaml.safe_load(config_file.read_text())
        assert data["llm"]["model"] == "openai/gpt-4o-mini"
        assert data["check_interval_hours"] == 8
