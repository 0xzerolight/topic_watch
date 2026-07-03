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
        app.state.config_path = tmp_path / "config.yml"
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
        app.state.config_path = tmp_path / "config.yml"
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

    def test_setupx_is_redirected(self, unconfigured_app: TestClient) -> None:
        """OVH-144: a path that merely starts with /setup (no segment boundary) is NOT exempt."""
        response = unconfigured_app.get("/setupx", follow_redirects=False)
        assert response.status_code == 307
        assert response.headers["location"] == "/setup"

    def test_healthz_is_redirected(self, unconfigured_app: TestClient) -> None:
        """OVH-144: /healthz is not the /health segment and must not be exempt."""
        response = unconfigured_app.get("/healthz", follow_redirects=False)
        assert response.status_code == 307
        assert response.headers["location"] == "/setup"

    def test_static_leak_is_redirected(self, unconfigured_app: TestClient) -> None:
        """OVH-144: /static-leak is not the /static segment and must not be exempt."""
        response = unconfigured_app.get("/static-leak", follow_redirects=False)
        assert response.status_code == 307
        assert response.headers["location"] == "/setup"

    def test_setup_subpath_not_redirected(self, unconfigured_app: TestClient) -> None:
        """OVH-144: a true /setup/* subpath stays exempt (prefix + boundary)."""
        response = unconfigured_app.get("/setup/anything", follow_redirects=False)
        assert response.status_code != 307


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

        with (
            patch("app.scheduler.start_scheduler") as mock_sched,
            patch("app.web.routers.settings.verify_llm_credentials", return_value=None),
        ):
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

        with (
            patch("app.scheduler.start_scheduler"),
            patch("app.web.routers.settings.verify_llm_credentials", return_value=None),
        ):
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

    def test_post_setup_cloud_provider_keeps_base_url(self, unconfigured_app: TestClient) -> None:
        """POST /setup with a cloud provider model keeps base_url (OpenAI-compatible gateway)."""
        get_response = unconfigured_app.get("/setup")
        csrf_token = get_response.cookies.get("csrf_token")

        with (
            patch("app.scheduler.start_scheduler"),
            patch("app.web.routers.settings.verify_llm_credentials", return_value=None),
        ):
            response = unconfigured_app.post(
                "/setup",
                data={
                    "llm_model": "openai/glm-5.2",
                    "llm_api_key": "sk-opencode-test",
                    "llm_base_url": "https://opencode.ai/zen/go/v1",
                    "csrf_token": csrf_token,
                },
                follow_redirects=False,
            )

        assert response.status_code == 303
        assert app.state.settings.llm.base_url == "https://opencode.ai/zen/go/v1"

    def test_post_setup_when_configured_is_guarded(self, configured_app: TestClient) -> None:
        """OVH-059/082: re-POSTing /setup once configured redirects without re-running setup.

        A double-submit / replay / stale bookmark must not clobber credentials or start a
        second scheduler (which would orphan the running one).
        """
        # Seed the CSRF cookie (the GET redirects but still sets it via middleware).
        configured_app.get("/setup", follow_redirects=False)
        csrf_token = configured_app.cookies.get("csrf_token")
        assert csrf_token

        original_model = app.state.settings.llm.model
        with (
            patch("app.scheduler.start_scheduler") as mock_sched,
            patch("app.web.routers.settings.save_settings_to_yaml") as mock_save,
            patch("app.web.routers.settings.verify_llm_credentials", return_value=None),
        ):
            response = configured_app.post(
                "/setup",
                data={
                    "llm_model": "openai/gpt-clobber",
                    "llm_api_key": "sk-attacker-key",
                    "llm_base_url": "",
                    "csrf_token": csrf_token,
                },
                follow_redirects=False,
            )

        assert response.status_code == 303
        assert response.headers["location"] == "/"
        # Single-shot setup: no scheduler restart, no config rewrite.
        mock_sched.assert_not_called()
        mock_save.assert_not_called()
        # Live credentials untouched.
        assert app.state.settings.llm.model == original_model

    def test_post_setup_nav_hidden(self, unconfigured_app: TestClient) -> None:
        response = unconfigured_app.get("/setup")
        assert response.status_code == 200
        # Nav links should not be present in setup mode
        assert 'href="/settings"' not in response.text
        assert 'href="/feeds"' not in response.text


class TestSetupPreflight:
    """Pre-flight LLM credential validation before completing setup."""

    def _post(self, client: TestClient, **data):
        get_response = client.get("/setup")
        csrf_token = get_response.cookies.get("csrf_token")
        payload = {
            "llm_model": "openai/gpt-4o-mini",
            "llm_api_key": "sk-bad-key-xyz",
            "llm_base_url": "",
            "csrf_token": csrf_token,
        }
        payload.update(data)
        return client.post("/setup", data=payload, follow_redirects=False)

    def test_valid_key_completes_setup(self, unconfigured_app: TestClient) -> None:
        """A passing preflight check lets setup complete normally."""
        with (
            patch("app.scheduler.start_scheduler"),
            patch("app.web.routers.settings.verify_llm_credentials", return_value=None) as mock_check,
        ):
            response = self._post(unconfigured_app, llm_api_key="sk-good-key")

        assert response.status_code == 303
        assert response.headers["location"] == "/"
        assert app.state.setup_required is False
        mock_check.assert_awaited_once()

    def test_invalid_key_does_not_complete_setup(self, unconfigured_app: TestClient) -> None:
        """A failing preflight check re-renders setup with an error and does NOT complete."""
        from app.web.routers.settings import LLMValidationError

        with (
            patch("app.scheduler.start_scheduler") as mock_sched,
            patch(
                "app.web.routers.settings.verify_llm_credentials",
                side_effect=LLMValidationError("Authentication failed: the API key was rejected by the provider."),
            ),
        ):
            response = self._post(unconfigured_app, llm_api_key="sk-bad-key-xyz")

        assert response.status_code == 422
        assert app.state.setup_required is True
        # Setup must not have started the scheduler on a failed preflight.
        mock_sched.assert_not_called()
        # Friendly error surfaced.
        assert "Authentication failed" in response.text

    def test_error_does_not_echo_api_key(self, unconfigured_app: TestClient) -> None:
        """The rendered error page must never contain the submitted API key."""
        from app.web.routers.settings import LLMValidationError

        secret = "sk-super-secret-9999"
        with (
            patch("app.scheduler.start_scheduler"),
            patch(
                "app.web.routers.settings.verify_llm_credentials",
                side_effect=LLMValidationError("The model could not be reached. Check the base URL."),
            ),
        ):
            response = self._post(unconfigured_app, llm_api_key=secret)

        assert response.status_code == 422
        assert secret not in response.text
        assert "could not be reached" in response.text

    def test_skip_validation_completes_despite_failing_preflight(self, unconfigured_app: TestClient) -> None:
        """The 'Save anyway' escape hatch bypasses the pre-flight so a transient
        provider error or stale default model can't dead-end a brand-new user at /setup."""
        from app.web.routers.settings import LLMValidationError

        with (
            patch("app.scheduler.start_scheduler") as mock_sched,
            patch(
                "app.web.routers.settings.verify_llm_credentials",
                side_effect=LLMValidationError("would have failed"),
            ) as mock_check,
        ):
            response = self._post(unconfigured_app, llm_api_key="sk-unverified", skip_validation="true")

        assert response.status_code == 303
        assert response.headers["location"] == "/"
        assert app.state.setup_required is False
        # The pre-flight was skipped, and setup still completed and started the scheduler.
        mock_check.assert_not_awaited()
        mock_sched.assert_called_once()

    def test_preflight_called_with_submitted_values(self, unconfigured_app: TestClient) -> None:
        """The preflight receives the submitted model / key / base_url."""
        with (
            patch("app.scheduler.start_scheduler"),
            patch("app.web.routers.settings.verify_llm_credentials", return_value=None) as mock_check,
        ):
            self._post(
                unconfigured_app,
                llm_model="ollama/llama3",
                llm_api_key="unused",
                llm_base_url="http://localhost:11434",
            )

        kwargs = mock_check.await_args.kwargs
        # Accept positional or keyword binding.
        bound = mock_check.await_args
        all_args = list(bound.args) + list(kwargs.values())
        assert "ollama/llama3" in all_args
        assert "http://localhost:11434" in all_args


class TestVerifyLLMCredentials:
    """Unit tests for the verify_llm_credentials preflight helper."""

    async def test_success_returns_none(self) -> None:
        """A successful acompletion ping returns without raising."""
        from app.web.routers.settings import verify_llm_credentials

        with patch("app.web.routers.settings.litellm.acompletion", return_value=object()) as mock_call:
            await verify_llm_credentials(model="openai/gpt-4o-mini", api_key="sk-test", base_url=None)
        mock_call.assert_awaited_once()

    async def test_auth_error_friendly_message(self) -> None:
        """An authentication error maps to a friendly, key-free message."""
        import litellm

        from app.web.routers.settings import LLMValidationError, verify_llm_credentials

        exc = litellm.AuthenticationError(message="bad key sk-secret", llm_provider="openai", model="gpt-4o-mini")
        with (
            patch("app.web.routers.settings.litellm.acompletion", side_effect=exc),
            pytest.raises(LLMValidationError) as ei,
        ):
            await verify_llm_credentials(model="openai/gpt-4o-mini", api_key="sk-secret", base_url=None)
        msg = str(ei.value)
        assert "sk-secret" not in msg
        assert "key" in msg.lower()

    async def test_connection_error_friendly_message(self) -> None:
        """A connection error mentions the base URL / reachability, not the key."""
        import litellm

        from app.web.routers.settings import LLMValidationError, verify_llm_credentials

        exc = litellm.APIConnectionError(message="conn refused", llm_provider="ollama", model="llama3")
        with (
            patch("app.web.routers.settings.litellm.acompletion", side_effect=exc),
            pytest.raises(LLMValidationError) as ei,
        ):
            await verify_llm_credentials(model="ollama/llama3", api_key="unused", base_url="http://localhost:11434")
        assert "reach" in str(ei.value).lower() or "url" in str(ei.value).lower()

    async def test_not_found_error_mentions_model(self) -> None:
        """A not-found error suggests checking the model string."""
        import litellm

        from app.web.routers.settings import LLMValidationError, verify_llm_credentials

        exc = litellm.NotFoundError(message="model not found", llm_provider="openai", model="gpt-nope")
        with (
            patch("app.web.routers.settings.litellm.acompletion", side_effect=exc),
            pytest.raises(LLMValidationError) as ei,
        ):
            await verify_llm_credentials(model="openai/gpt-nope", api_key="sk-test", base_url=None)
        assert "model" in str(ei.value).lower()

    async def test_generic_error_never_leaks_key(self) -> None:
        """An unexpected error still produces a key-free LLMValidationError."""
        from app.web.routers.settings import LLMValidationError, verify_llm_credentials

        with (
            patch("app.web.routers.settings.litellm.acompletion", side_effect=RuntimeError("boom sk-leak")),
            pytest.raises(LLMValidationError) as ei,
        ):
            await verify_llm_credentials(model="openai/gpt-4o-mini", api_key="sk-leak", base_url=None)
        assert "sk-leak" not in str(ei.value)
