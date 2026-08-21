"""Tests for the first-run setup wizard."""

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app


@pytest.fixture(autouse=True)
def _yaml_owned_llm_credentials(monkeypatch: pytest.MonkeyPatch):
    """Let the wizard own the LLM block.

    conftest (like CI) exports TOPIC_WATCH_LLM__MODEL / __API_KEY so the app counts as
    configured; while they are set the environment owns those fields and setup leaves
    them alone (AUG-241), which is not what the wizard's own tests are about.
    """
    monkeypatch.delenv("TOPIC_WATCH_LLM__MODEL", raising=False)
    monkeypatch.delenv("TOPIC_WATCH_LLM__API_KEY", raising=False)


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

    def test_unsafe_method_redirects_as_a_get(self, unconfigured_app: TestClient) -> None:
        """AUG-210: 307 replayed the method and body into POST /setup."""
        response = unconfigured_app.post("/topics", data={"name": "x"}, follow_redirects=False)
        assert response.status_code == 303
        assert response.headers["location"] == "/setup"

    def test_delete_redirects_as_a_get(self, unconfigured_app: TestClient) -> None:
        response = unconfigured_app.delete("/topics/1", follow_redirects=False)
        assert response.status_code == 303
        assert response.headers["location"] == "/setup"

    def test_api_requests_get_json_not_a_redirect(self, unconfigured_app: TestClient) -> None:
        """AUG-210: API clients get an actionable status, not an HTML form flow."""
        response = unconfigured_app.get("/api/v1/topics", follow_redirects=False)
        assert response.status_code == 503
        assert "setup" in response.json()["detail"].lower()
        assert response.headers["content-type"].startswith("application/json")

    def test_api_mutation_gets_json_too(self, unconfigured_app: TestClient) -> None:
        response = unconfigured_app.post("/api/v1/topics/1/check", follow_redirects=False)
        assert response.status_code == 503

    def test_api_lookalike_path_still_redirects(self, unconfigured_app: TestClient) -> None:
        """/apix is not the /api segment."""
        response = unconfigured_app.get("/apix", follow_redirects=False)
        assert response.status_code == 307


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
    """Unit tests for the verify_llm_credentials preflight helper (AUG-335)."""

    @staticmethod
    def _patch_probe(**kwargs):
        """Patch the live structured-output call the preflight goes through."""
        return patch("app.analysis.llm._create_structured", **kwargs)

    async def test_success_returns_none(self) -> None:
        from app.web.routers.settings import verify_llm_credentials

        with self._patch_probe(return_value=(object(), object())) as mock_call:
            await verify_llm_credentials(model="openai/gpt-4o-mini", api_key="sk-test", base_url=None)
        mock_call.assert_awaited_once()

    async def test_probe_uses_the_live_call_shape(self) -> None:
        """Same response model, same settings-derived deadline as analysis itself."""
        from app.analysis.llm import NoveltyResponse
        from app.web.routers.settings import verify_llm_credentials

        with self._patch_probe(return_value=(object(), object())) as mock_call:
            await verify_llm_credentials(model="ollama/llama3", api_key="unused", base_url="http://localhost:11434")

        probe_settings = mock_call.await_args.args[0]
        kwargs = mock_call.await_args.kwargs
        assert kwargs["response_model"] is NoveltyResponse
        assert kwargs["timeout"] == probe_settings.llm_analysis_timeout
        assert probe_settings.llm.model == "ollama/llama3"
        assert probe_settings.llm.api_key == "unused"
        assert probe_settings.llm.base_url == "http://localhost:11434"
        # The messages are built by a factory, per the live rebuild-per-attempt contract.
        assert kwargs["build_messages"]()

    async def test_auth_error_friendly_message(self) -> None:
        """An authentication error maps to a friendly, key-free message."""
        import litellm

        from app.web.routers.settings import LLMValidationError, verify_llm_credentials

        exc = litellm.AuthenticationError(message="bad key sk-secret", llm_provider="openai", model="gpt-4o-mini")
        with self._patch_probe(side_effect=exc), pytest.raises(LLMValidationError) as ei:
            await verify_llm_credentials(model="openai/gpt-4o-mini", api_key="sk-secret", base_url=None)
        msg = str(ei.value)
        assert "sk-secret" not in msg
        assert "key" in msg.lower()

    async def test_wrapped_auth_error_is_still_recognized(self) -> None:
        """instructor re-raises its own type with the provider error chained."""
        import litellm

        from app.web.routers.settings import LLMValidationError, verify_llm_credentials

        cause = litellm.AuthenticationError(message="bad key", llm_provider="openai", model="gpt-4o-mini")
        wrapper = RuntimeError("retries exhausted")
        wrapper.__cause__ = cause
        with self._patch_probe(side_effect=wrapper), pytest.raises(LLMValidationError) as ei:
            await verify_llm_credentials(model="openai/gpt-4o-mini", api_key="sk-secret", base_url=None)
        assert "Authentication failed" in str(ei.value)

    async def test_connection_error_friendly_message(self) -> None:
        """A connection error mentions the base URL / reachability, not the key."""
        import litellm

        from app.web.routers.settings import LLMValidationError, verify_llm_credentials

        exc = litellm.APIConnectionError(message="conn refused", llm_provider="ollama", model="llama3")
        with self._patch_probe(side_effect=exc), pytest.raises(LLMValidationError) as ei:
            await verify_llm_credentials(model="ollama/llama3", api_key="unused", base_url="http://localhost:11434")
        assert "reach" in str(ei.value).lower() or "url" in str(ei.value).lower()

    async def test_not_found_error_mentions_model(self) -> None:
        """A not-found error suggests checking the model string."""
        import litellm

        from app.web.routers.settings import LLMValidationError, verify_llm_credentials

        exc = litellm.NotFoundError(message="model not found", llm_provider="openai", model="gpt-nope")
        with self._patch_probe(side_effect=exc), pytest.raises(LLMValidationError) as ei:
            await verify_llm_credentials(model="openai/gpt-nope", api_key="sk-test", base_url=None)
        assert "model" in str(ei.value).lower()

    async def test_unstructured_reply_is_reported_as_a_format_problem(self) -> None:
        """A model that answers but cannot follow the schema fails setup with a reason."""
        from pydantic import ValidationError as PydanticValidationError

        from app.analysis.llm import NoveltyResponse
        from app.web.routers.settings import LLMValidationError, verify_llm_credentials

        try:
            NoveltyResponse.model_validate({})
        except PydanticValidationError as parse_error:
            exc: Exception = parse_error

        with self._patch_probe(side_effect=exc), pytest.raises(LLMValidationError) as ei:
            await verify_llm_credentials(model="ollama/tiny", api_key="", base_url="http://localhost:11434")
        assert "structured format" in str(ei.value)

    async def test_structured_output_rejection_is_reported(self) -> None:
        """An endpoint that refuses the request shape is named as such, not as a bad key."""
        import litellm

        from app.web.routers.settings import LLMValidationError, verify_llm_credentials

        exc = litellm.BadRequestError(message="tools unsupported", llm_provider="openai", model="gw/model")
        with self._patch_probe(side_effect=exc), pytest.raises(LLMValidationError) as ei:
            await verify_llm_credentials(model="openai/gw-model", api_key="sk", base_url="https://gw.example/v1")
        assert "structured output" in str(ei.value)

    async def test_generic_error_never_leaks_key(self) -> None:
        """An unexpected error still produces a key-free LLMValidationError."""
        from app.web.routers.settings import LLMValidationError, verify_llm_credentials

        with self._patch_probe(side_effect=RuntimeError("boom sk-leak")), pytest.raises(LLMValidationError) as ei:
            await verify_llm_credentials(model="openai/gpt-4o-mini", api_key="sk-leak", base_url=None)
        assert "sk-leak" not in str(ei.value)


class TestSetupPublication:
    """AUG-199/292: setup publishes disk, state and scheduler as one ordered transition."""

    def _post(self, client: TestClient, **data):
        csrf_token = client.get("/setup").cookies.get("csrf_token")
        payload = {
            "llm_model": "openai/gpt-4o-mini",
            "llm_api_key": "sk-good-key",
            "llm_base_url": "",
            "csrf_token": csrf_token,
        }
        payload.update(data)
        return client.post("/setup", data=payload, follow_redirects=False)

    def test_scheduler_failure_leaves_setup_retryable(self, unconfigured_app: TestClient) -> None:
        """A scheduler that fails to start must not leave a configured app with no monitoring."""
        previous = app.state.settings
        with (
            patch("app.scheduler.start_scheduler", side_effect=RuntimeError("no event loop")),
            patch("app.scheduler.stop_scheduler") as mock_stop,
            patch("app.web.routers.settings.verify_llm_credentials", return_value=None),
        ):
            response = self._post(unconfigured_app)

        assert response.status_code == 422
        # The gate is still open, so the user can fix and retry.
        assert app.state.setup_required is True
        # The partial scheduler is torn down and the prior live settings restored.
        mock_stop.assert_called_once()
        assert app.state.settings is previous

    def test_gate_closes_only_after_the_scheduler_is_running(self, unconfigured_app: TestClient) -> None:
        """setup_required is still True while start_scheduler runs."""
        seen: list[bool] = []

        def record(*_args, **_kwargs):
            seen.append(app.state.setup_required)

        with (
            patch("app.scheduler.start_scheduler", side_effect=record),
            patch("app.web.routers.settings.verify_llm_credentials", return_value=None),
        ):
            self._post(unconfigured_app)

        assert seen == [True]
        assert app.state.setup_required is False

    def test_preserves_configured_notifications_and_exa(self, unconfigured_app: TestClient, tmp_path: Path) -> None:
        """Completing LLM setup must not erase configuration the user already had."""
        from app.config import ExaSettings, NotificationSettings, Settings

        config_file = tmp_path / "config.yml"
        app.state.config_path = config_file
        app.state.settings = Settings(  # type: ignore[call-arg]
            notifications=NotificationSettings(urls=["ntfy://already-configured"]),
            exa=ExaSettings(enabled=True, api_key="exa-already-set"),
            check_interval="12h",
        )

        with (
            patch("app.scheduler.start_scheduler"),
            patch("app.web.routers.settings.verify_llm_credentials", return_value=None),
        ):
            response = self._post(unconfigured_app)

        assert response.status_code == 303
        assert app.state.settings.notifications.urls == ["ntfy://already-configured"]
        assert app.state.settings.exa.enabled is True
        assert app.state.settings.check_interval == "12h"

        data = yaml.safe_load(config_file.read_text())
        assert data["notifications"]["urls"] == ["ntfy://already-configured"]
        assert data["exa"]["api_key"] == "exa-already-set"


class TestSetupSerialization:
    """AUG-292: two first-run submissions cannot both pass the gate and publish."""

    async def test_concurrent_submissions_publish_once(self, tmp_path: Path) -> None:
        import asyncio

        import httpx

        from app.config import Settings
        from app.database import init_db

        db_path = tmp_path / "test.db"
        init_db(db_path)
        app.state.settings = Settings()  # type: ignore[call-arg]
        app.state.db_path = db_path
        app.state.config_path = tmp_path / "config.yml"
        app.state.setup_required = True

        async def slow_preflight(**_kwargs):
            await asyncio.sleep(0.05)  # both requests are in flight across this await

        token = "csrf-setup-race"
        with (
            patch("app.scheduler.start_scheduler") as mock_sched,
            patch("app.web.routers.settings.save_settings_to_yaml") as mock_save,
            patch("app.web.routers.settings.verify_llm_credentials", side_effect=slow_preflight),
        ):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
                cookies={"csrf_token": token},
                headers={"X-CSRF-Token": token},
            ) as ac:
                payloads = [
                    {
                        "llm_model": f"openai/model-{index}",
                        "llm_api_key": f"sk-key-{index}",
                        "llm_base_url": "",
                        "csrf_token": token,
                    }
                    for index in range(2)
                ]
                await asyncio.gather(*(ac.post("/setup", data=p, follow_redirects=False) for p in payloads))

        assert mock_save.call_count == 1
        assert mock_sched.call_count == 1
        assert app.state.setup_required is False


class TestKeylessLocalProvider:
    """AUG-107: the documented keyless Ollama path completes setup."""

    def _post(self, client: TestClient, **data):
        csrf_token = client.get("/setup").cookies.get("csrf_token")
        payload = {"llm_model": "ollama/llama3.3", "llm_api_key": "", "llm_base_url": "", "csrf_token": csrf_token}
        payload.update(data)
        return client.post("/setup", data=payload, follow_redirects=False)

    def test_blank_key_completes_setup_for_ollama(self, unconfigured_app: TestClient) -> None:
        with (
            patch("app.scheduler.start_scheduler"),
            patch("app.web.routers.settings.verify_llm_credentials", return_value=None),
        ):
            response = self._post(unconfigured_app, llm_base_url="http://localhost:11434")

        assert response.status_code == 303
        assert app.state.setup_required is False
        assert app.state.settings.llm.api_key == ""
        assert app.state.settings.is_configured()

    def test_blank_key_is_refused_for_a_hosted_provider(self, unconfigured_app: TestClient) -> None:
        """The server enforces it; client-side `required` is not a contract."""
        with (
            patch("app.scheduler.start_scheduler") as mock_sched,
            patch("app.web.routers.settings.verify_llm_credentials", return_value=None),
        ):
            response = self._post(unconfigured_app, llm_model="openai/gpt-4o-mini")

        assert response.status_code == 422
        assert "API key is required" in response.text
        assert app.state.setup_required is True
        mock_sched.assert_not_called()

    def test_key_field_is_not_unconditionally_required(self, unconfigured_app: TestClient) -> None:
        page = unconfigured_app.get("/setup")
        assert 'id="llm_api_key" name="llm_api_key"' in page.text
        assert 'name="llm_api_key" required' not in page.text
        assert "needs no key" in page.text


class TestEnvSuppliedCredentials:
    """C5-4: setup must not demand a credential the environment already supplies."""

    def _post(self, client: TestClient, **data):
        csrf_token = client.get("/setup").cookies.get("csrf_token")
        payload = {
            "llm_model": "openai/gpt-4o-mini",
            "llm_api_key": "",
            "llm_base_url": "",
            "csrf_token": csrf_token,
        }
        payload.update(data)
        return client.post("/setup", data=payload, follow_redirects=False)

    def test_env_key_satisfies_the_key_requirement(
        self, unconfigured_app: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A blank key field is correct when TOPIC_WATCH_LLM__API_KEY is set."""
        monkeypatch.setenv("TOPIC_WATCH_LLM__API_KEY", "sk-from-the-environment")
        app.state.settings = Settings()  # type: ignore[call-arg]

        with (
            patch("app.scheduler.start_scheduler"),
            patch("app.web.routers.settings.verify_llm_credentials", return_value=None),
        ):
            response = self._post(unconfigured_app)

        assert response.status_code == 303
        assert app.state.setup_required is False

    def test_preflight_probes_the_key_that_will_actually_be_used(
        self, unconfigured_app: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Validating the typed key while saving the env key tests the wrong credential."""
        monkeypatch.setenv("TOPIC_WATCH_LLM__API_KEY", "sk-from-the-environment")
        app.state.settings = Settings()  # type: ignore[call-arg]

        with (
            patch("app.scheduler.start_scheduler"),
            patch("app.web.routers.settings.verify_llm_credentials", return_value=None) as mock_check,
        ):
            self._post(unconfigured_app, llm_api_key="sk-typed-by-the-user")

        bound = mock_check.await_args
        probed = list(bound.args) + list(bound.kwargs.values())
        assert "sk-from-the-environment" in probed
        assert "sk-typed-by-the-user" not in probed

    def test_env_model_is_probed_and_the_typed_one_ignored(
        self, unconfigured_app: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The typed model is discarded as env-owned, so it must not be validated either."""
        monkeypatch.setenv("TOPIC_WATCH_LLM__MODEL", "openai/from-the-environment")
        app.state.settings = Settings()  # type: ignore[call-arg]

        with (
            patch("app.scheduler.start_scheduler"),
            patch("app.web.routers.settings.verify_llm_credentials", return_value=None) as mock_check,
        ):
            self._post(unconfigured_app, llm_model="openai/typed-by-the-user", llm_api_key="sk-typed")

        bound = mock_check.await_args
        probed = list(bound.args) + list(bound.kwargs.values())
        assert "openai/from-the-environment" in probed
        assert "openai/typed-by-the-user" not in probed

    def test_env_owned_model_does_not_have_to_be_retyped(
        self, unconfigured_app: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The model control renders disabled, so the browser submits nothing for it."""
        monkeypatch.setenv("TOPIC_WATCH_LLM__MODEL", "openai/from-the-environment")
        app.state.settings = Settings()  # type: ignore[call-arg]

        with (
            patch("app.scheduler.start_scheduler"),
            patch("app.web.routers.settings.verify_llm_credentials", return_value=None),
        ):
            response = self._post(unconfigured_app, llm_model="", llm_api_key="sk-typed")

        assert response.status_code == 303
        assert app.state.settings.llm.model == "openai/from-the-environment"

    def test_a_blank_model_is_still_refused_when_the_environment_owns_nothing(
        self, unconfigured_app: TestClient
    ) -> None:
        with patch("app.scheduler.start_scheduler") as mock_sched:
            response = self._post(unconfigured_app, llm_model="")

        assert response.status_code == 422
        assert "model" in response.text.lower()
        mock_sched.assert_not_called()

    def test_a_blank_key_is_still_refused_when_the_environment_owns_nothing(self, unconfigured_app: TestClient) -> None:
        with patch("app.scheduler.start_scheduler") as mock_sched:
            response = self._post(unconfigured_app)

        assert response.status_code == 422
        assert "API key is required" in response.text
        mock_sched.assert_not_called()

    def test_setup_page_marks_an_env_key_read_only(
        self, unconfigured_app: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TOPIC_WATCH_LLM__API_KEY", "sk-from-the-environment")
        page = unconfigured_app.get("/setup")
        assert "set via environment" in page.text.lower()
        assert 'name="llm_api_key"' not in page.text
        assert "sk-from-the-environment" not in page.text

    def test_setup_page_marks_an_env_model_read_only(
        self, unconfigured_app: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TOPIC_WATCH_LLM__MODEL", "openai/from-the-environment")
        app.state.settings = Settings()  # type: ignore[call-arg]
        page = unconfigured_app.get("/setup")
        assert 'id="llm_model" name="llm_model" required disabled' in page.text
        assert "openai/from-the-environment" in page.text


class TestProviderSwitchDropsInjectedBaseUrl:
    """AUG-100: the local default the script injected does not survive a switch."""

    def test_setup_page_tracks_its_own_autofill(self, unconfigured_app: TestClient) -> None:
        page = unconfigured_app.get("/setup")
        assert "var autofilled = null;" in page.text
        assert "function dropAutofilledUrl()" in page.text
