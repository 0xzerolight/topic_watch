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


def valid_form_data(**overrides) -> dict:
    """A complete, valid POST /settings form payload (override individual fields)."""
    data = {
        "llm_model": "openai/gpt-4o-mini",
        "llm_api_key": "",
        "llm_base_url": "",
        "notification_urls": "",
        "webhook_urls": "",
        "check_interval": "6h",
        "max_articles_per_check": "10",
        "knowledge_state_max_tokens": "2000",
        "article_retention_days": "90",
        "feed_fetch_timeout": "15.0",
        "article_fetch_timeout": "20.0",
        "llm_analysis_timeout": "60",
        "llm_knowledge_timeout": "120",
        "web_page_size": "20",
        "min_confidence_threshold": "0.7",
        "min_relevance_threshold": "0.5",
        "secure_cookies": "true",
        "feed_max_retries": "2",
        "content_fetch_concurrency": "3",
        "scheduler_misfire_grace_time": "300",
        "scheduler_jitter_seconds": "30",
        "llm_max_retries": "2",
        "llm_temperature": "0.2",
        "apprise_timeout_seconds": "30",
    }
    data.update(overrides)
    return data


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

    # GET /settings calls load_settings() directly instead of using Depends
    with patch("app.web.routers.settings.load_settings", return_value=settings):
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
        """base_url should appear in YAML when it is set for a local provider."""
        from app.config import save_settings_to_yaml

        settings = _make_settings(
            llm=LLMSettings(
                model="ollama/llama3",
                api_key="dummy",
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

        settings = _make_settings(check_interval="12h", max_articles_per_check=25)
        config_file = tmp_path / "config.yml"
        save_settings_to_yaml(settings, config_file)

        data = yaml.safe_load(config_file.read_text())
        assert data["check_interval"] == "12h"
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
        """Settings page includes check_interval field."""
        response = await client.get("/settings")
        assert "check_interval" in response.text


# ---------------------------------------------------------------------------
# Route tests for POST /settings
# ---------------------------------------------------------------------------


class TestSettingsPost:
    """Tests for POST /settings."""

    def _valid_form_data(self, **overrides) -> dict:
        return valid_form_data(**overrides)

    async def test_valid_post_redirects(self, client: httpx.AsyncClient) -> None:
        """POST /settings with valid data redirects to /settings?saved=1."""
        with patch("app.web.routers.settings.save_settings_to_yaml"):
            response = await client.post(
                "/settings",
                data=self._valid_form_data(),
                follow_redirects=False,
            )

        assert response.status_code == 303
        assert response.headers["location"] == "/settings?saved=1"

    async def test_valid_post_updates_app_state(self, client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
        """POST /settings updates app.state.settings with the new values."""
        with patch("app.web.routers.settings.save_settings_to_yaml"):
            await client.post(
                "/settings",
                data=self._valid_form_data(
                    llm_model="anthropic/claude-3-haiku-20240307",
                    check_interval="12h",
                ),
                follow_redirects=False,
            )

        assert app.state.settings.llm.model == "anthropic/claude-3-haiku-20240307"
        assert app.state.settings.check_interval == "12h"

    async def test_valid_post_calls_save(self, client: httpx.AsyncClient) -> None:
        """POST /settings calls save_settings_to_yaml."""
        with patch("app.web.routers.settings.save_settings_to_yaml") as mock_save:
            await client.post(
                "/settings",
                data=self._valid_form_data(),
                follow_redirects=False,
            )
            mock_save.assert_called_once()

    async def test_invalid_interval_returns_error(self, client: httpx.AsyncClient) -> None:
        """POST with invalid check_interval returns 422 with an error."""
        response = await client.post(
            "/settings",
            data=self._valid_form_data(check_interval="bogus"),
            follow_redirects=False,
        )
        assert response.status_code == 422
        assert "error" in response.text.lower()

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
                        "check_interval": "6h",
                        "max_articles_per_check": "10",
                    },
                    follow_redirects=False,
                )
            assert response.status_code == 403
        finally:
            app.dependency_overrides.clear()

    async def test_parses_notification_urls_from_textarea(self, client: httpx.AsyncClient) -> None:
        """Newline-separated notification URLs are parsed into a list."""
        with patch("app.web.routers.settings.save_settings_to_yaml"):
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
        with patch("app.web.routers.settings.save_settings_to_yaml"):
            await client.post(
                "/settings",
                data=self._valid_form_data(notification_urls=""),
                follow_redirects=False,
            )

        assert app.state.settings.notifications.urls == []

    async def test_saves_to_yaml_with_correct_content(self, client: httpx.AsyncClient, tmp_path: Path) -> None:
        """POST /settings writes the correct content to the YAML file."""
        config_file = tmp_path / "config.yml"

        def save_to_tmp(settings, config_path=None, **kwargs):
            from app.config import save_settings_to_yaml as _real_save

            _real_save(settings, config_file)

        with patch("app.web.routers.settings.save_settings_to_yaml", side_effect=save_to_tmp):
            await client.post(
                "/settings",
                data=self._valid_form_data(
                    llm_model="openai/gpt-4o-mini",
                    check_interval="8h",
                ),
                follow_redirects=False,
            )

        assert config_file.exists()
        data = yaml.safe_load(config_file.read_text())
        assert data["llm"]["model"] == "openai/gpt-4o-mini"
        assert data["check_interval"] == "8h"

    async def test_yaml_parse_error_returns_422(self, client: httpx.AsyncClient) -> None:
        """Non-ValidationError (e.g. YAML ScannerError) returns 422, not 500."""
        import yaml as _yaml

        error = _yaml.scanner.ScannerError("while scanning", None, "could not find expected ':'", None)
        with patch("app.web.routers.settings.Settings", side_effect=error):
            response = await client.post(
                "/settings",
                data=self._valid_form_data(),
                follow_redirects=False,
            )
        assert response.status_code == 422
        assert "failed to save settings" in response.text.lower()

    async def test_save_io_error_returns_422(self, client: httpx.AsyncClient) -> None:
        """I/O error during save_settings_to_yaml returns 422, not 500."""
        with patch(
            "app.web.routers.settings.save_settings_to_yaml",
            side_effect=PermissionError("[Errno 13] Permission denied"),
        ):
            response = await client.post(
                "/settings",
                data=self._valid_form_data(),
                follow_redirects=False,
            )
        assert response.status_code == 422
        assert "failed to save settings" in response.text.lower()

    async def test_settings_page_calls_load_settings(self, db_conn: sqlite3.Connection) -> None:
        """GET /settings calls load_settings() to show fresh values from disk."""
        fresh_settings = _make_settings(check_interval="1d")
        app.state.settings = _make_settings(check_interval="6h")

        def override_db():
            yield db_conn

        app.dependency_overrides[get_db_conn] = override_db

        try:
            with patch("app.web.routers.settings.load_settings", return_value=fresh_settings) as mock_load:
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app),
                    base_url="http://test",
                    cookies={"csrf_token": CSRF_TEST_TOKEN},
                ) as ac:
                    response = await ac.get("/settings")
                mock_load.assert_called_once()
                assert response.status_code == 200
                # Fresh value (1d) should appear, not stale value (6h)
                assert "1d" in response.text
        finally:
            app.dependency_overrides.clear()

    async def test_min_relevance_threshold_persisted(self, client: httpx.AsyncClient) -> None:
        """min_relevance_threshold from the form is saved, not reset to default."""
        with patch("app.web.routers.settings.save_settings_to_yaml"):
            await client.post(
                "/settings",
                data=self._valid_form_data(min_relevance_threshold="0.85"),
                follow_redirects=False,
            )
        assert app.state.settings.min_relevance_threshold == 0.85

    async def test_secure_cookies_persisted(self, client: httpx.AsyncClient) -> None:
        """secure_cookies checkbox from the form is saved, not silently reset to False."""
        with patch("app.web.routers.settings.save_settings_to_yaml"):
            await client.post(
                "/settings",
                data=self._valid_form_data(secure_cookies="true"),
                follow_redirects=False,
            )
        assert app.state.settings.secure_cookies is True

    async def test_unchecked_secure_cookies_is_false(self, client: httpx.AsyncClient) -> None:
        """An absent secure_cookies checkbox results in False."""
        data = self._valid_form_data()
        data.pop("secure_cookies")
        with patch("app.web.routers.settings.save_settings_to_yaml"):
            await client.post("/settings", data=data, follow_redirects=False)
        assert app.state.settings.secure_cookies is False

    async def test_advanced_fields_round_trip(self, client: httpx.AsyncClient) -> None:
        """Advanced fields editable in the full UI persist their submitted values."""
        with patch("app.web.routers.settings.save_settings_to_yaml"):
            await client.post(
                "/settings",
                data=self._valid_form_data(
                    feed_max_retries="5",
                    content_fetch_concurrency="7",
                    scheduler_misfire_grace_time="600",
                    scheduler_jitter_seconds="10",
                    llm_max_retries="4",
                    llm_temperature="0.9",
                    apprise_timeout_seconds="45",
                ),
                follow_redirects=False,
            )
        s = app.state.settings
        assert s.feed_max_retries == 5
        assert s.content_fetch_concurrency == 7
        assert s.scheduler_misfire_grace_time == 600
        assert s.scheduler_jitter_seconds == 10
        assert s.llm_max_retries == 4
        assert s.llm_temperature == 0.9
        assert s.apprise_timeout_seconds == 45

    async def test_unspecified_field_not_silently_changed(self, client: httpx.AsyncClient) -> None:
        """Saving a form preserves current values for fields the form leaves at defaults."""
        app.state.settings = _make_settings(min_relevance_threshold=0.42, secure_cookies=True)
        data = self._valid_form_data(min_relevance_threshold="0.42", secure_cookies="true")
        with patch("app.web.routers.settings.save_settings_to_yaml"):
            await client.post("/settings", data=data, follow_redirects=False)
        assert app.state.settings.min_relevance_threshold == 0.42
        assert app.state.settings.secure_cookies is True

    async def test_cloud_provider_base_url_stripped(self, client: httpx.AsyncClient) -> None:
        """POST /settings with a cloud provider model strips any stale base_url."""
        with patch("app.web.routers.settings.save_settings_to_yaml"):
            await client.post(
                "/settings",
                data=self._valid_form_data(
                    llm_model="anthropic/claude-haiku-4-5",
                    llm_base_url="http://localhost:11434",
                ),
                follow_redirects=False,
            )
        assert app.state.settings.llm.base_url is None


class TestApiKeyRetention:
    """OVH-081: a blank api-key field retains the existing key; a non-blank one overwrites."""

    async def test_blank_key_retains_existing(self, client: httpx.AsyncClient) -> None:
        """Submitting an empty llm_api_key keeps the current (sentinel) key."""
        sentinel = "sk-sentinel-do-not-wipe"
        app.state.settings = _make_settings(llm=LLMSettings(model="openai/gpt-4o-mini", api_key=sentinel))
        with patch("app.web.routers.settings.save_settings_to_yaml"):
            await client.post(
                "/settings",
                data=valid_form_data(llm_api_key=""),
                follow_redirects=False,
            )
        assert app.state.settings.llm.api_key == sentinel

    async def test_nonblank_key_overwrites(self, client: httpx.AsyncClient) -> None:
        """Submitting a non-blank llm_api_key replaces the existing key."""
        app.state.settings = _make_settings(llm=LLMSettings(model="openai/gpt-4o-mini", api_key="sk-old"))
        with patch("app.web.routers.settings.save_settings_to_yaml"):
            await client.post(
                "/settings",
                data=valid_form_data(llm_api_key="sk-new-explicit"),
                follow_redirects=False,
            )
        assert app.state.settings.llm.api_key == "sk-new-explicit"


class TestEnvSourcedSecretSafety:
    """OVH-003: an env-supplied API key must not be materialized into plaintext YAML."""

    async def test_env_key_not_written_when_blank_submitted(
        self, client: httpx.AsyncClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the key is env-sourced and the field is blank, save preserves the on-disk key."""
        # On-disk YAML has an explicit (different) key; env supplies the live value.
        config_file = tmp_path / "config.yml"
        config_file.write_text('llm:\n  model: "openai/gpt-4o-mini"\n  api_key: "sk-on-disk-original"\n')
        monkeypatch.setenv("TOPIC_WATCH_LLM__API_KEY", "sk-env-secret-keep-out")
        # The running app's in-memory settings reflect the env value (as load_settings would).
        app.state.settings = _make_settings(
            llm=LLMSettings(model="openai/gpt-4o-mini", api_key="sk-env-secret-keep-out")
        )
        app.state.config_path = config_file

        def save_real(settings, config_path=None, **kwargs):
            from app.config import save_settings_to_yaml as _real_save

            _real_save(settings, config_path or config_file, **kwargs)

        with patch("app.web.routers.settings.save_settings_to_yaml", side_effect=save_real):
            await client.post(
                "/settings",
                data=valid_form_data(llm_api_key=""),
                follow_redirects=False,
            )

        data = yaml.safe_load(config_file.read_text())
        # The env secret must NOT have leaked into the plaintext file.
        assert data["llm"]["api_key"] != "sk-env-secret-keep-out"
        # The prior on-disk value is preserved.
        assert data["llm"]["api_key"] == "sk-on-disk-original"

    async def test_env_key_edit_is_guarded_noop(
        self, client: httpx.AsyncClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Editing the env-overridden key field does not overwrite the on-disk value."""
        config_file = tmp_path / "config.yml"
        config_file.write_text('llm:\n  model: "openai/gpt-4o-mini"\n  api_key: "sk-on-disk-original"\n')
        monkeypatch.setenv("TOPIC_WATCH_LLM__API_KEY", "sk-env-secret")
        app.state.settings = _make_settings(llm=LLMSettings(model="openai/gpt-4o-mini", api_key="sk-env-secret"))
        app.state.config_path = config_file

        def save_real(settings, config_path=None, **kwargs):
            from app.config import save_settings_to_yaml as _real_save

            _real_save(settings, config_path or config_file, **kwargs)

        with patch("app.web.routers.settings.save_settings_to_yaml", side_effect=save_real):
            # Operator types a new key in the UI; env override means this is a no-op on disk.
            await client.post(
                "/settings",
                data=valid_form_data(llm_api_key="sk-typed-in-ui"),
                follow_redirects=False,
            )

        data = yaml.safe_load(config_file.read_text())
        assert data["llm"]["api_key"] == "sk-on-disk-original"

    async def test_settings_page_marks_env_key_readonly(
        self, client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the key is env-sourced, the API-key field is rendered read-only with a note."""
        monkeypatch.setenv("TOPIC_WATCH_LLM__API_KEY", "sk-env-secret")
        response = await client.get("/settings")
        assert response.status_code == 200
        assert "set via environment" in response.text.lower()
