"""Setup wizard, settings editor, and notification-test routes."""

import logging

import litellm
from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app.config import (
    CLOUD_PROVIDERS,
    LOCAL_PROVIDER_DEFAULTS,
    Settings,
    is_api_key_env_sourced,
    is_cloud_provider,
    load_settings,
    save_settings_to_yaml,
)
from app.notifications import send_notification
from app.web.csrf import verify_csrf
from app.web.dependencies import get_settings
from app.web.routers.templates import templates

logger = logging.getLogger(__name__)

router = APIRouter()

# Seconds to wait for the pre-flight credential ping before giving up.
_PREFLIGHT_TIMEOUT = 15.0

# Scalar Settings fields the settings form edits 1:1 (name on form == name on model).
# Nested (llm/notifications), checkbox (secure_cookies) and infra-only (db_path) fields
# are handled explicitly below. Derived from Settings so adding a field is one edit (OVH-069).
_SCALAR_FORM_FIELDS: tuple[str, ...] = (
    "check_interval",
    "max_articles_per_check",
    "knowledge_state_max_tokens",
    "article_retention_days",
    "feed_fetch_timeout",
    "article_fetch_timeout",
    "llm_analysis_timeout",
    "llm_knowledge_timeout",
    "apprise_timeout_seconds",
    "web_page_size",
    "min_confidence_threshold",
    "min_relevance_threshold",
    "feed_max_retries",
    "content_fetch_concurrency",
    "scheduler_misfire_grace_time",
    "scheduler_jitter_seconds",
    "llm_max_retries",
    "llm_temperature",
)


def _settings_template_ctx(request: Request, **extra: object) -> dict:
    """Shared template context for the settings page (provider lists + env-key state)."""
    ctx: dict = {
        "config_path": str(request.app.state.config_path),
        "cloud_providers": sorted(CLOUD_PROVIDERS),
        "local_provider_defaults": LOCAL_PROVIDER_DEFAULTS,
        "api_key_env_sourced": is_api_key_env_sourced(),
    }
    ctx.update(extra)
    return ctx


class LLMValidationError(Exception):
    """Raised when a pre-flight LLM credential check fails.

    The message is always user-safe: it never contains the API key and explains
    what went wrong (bad key vs. unreachable base URL vs. bad model) and how to fix it.
    """


async def verify_llm_credentials(model: str, api_key: str, base_url: str | None) -> None:
    """Make a minimal LLM call to confirm the supplied credentials actually work.

    Sends a tiny ``litellm.acompletion`` ping. Returns ``None`` on success. On any
    failure, raises :class:`LLMValidationError` with a friendly, key-free message.
    The api_key is never included in raised messages.
    """
    try:
        await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            api_key=api_key,
            api_base=base_url,
            max_tokens=1,
            timeout=_PREFLIGHT_TIMEOUT,
        )
    except litellm.AuthenticationError as exc:
        logger.warning("Setup pre-flight: authentication rejected for model %s", model)
        raise LLMValidationError(
            "Authentication failed: the API key was rejected by the provider. "
            "Double-check the key for the correct provider and account."
        ) from exc
    except litellm.NotFoundError as exc:
        logger.warning("Setup pre-flight: model not found for %s", model)
        raise LLMValidationError(
            f"The model '{model}' was not found. Check the model string uses the "
            "LiteLLM 'provider/model-name' format and that the model exists."
        ) from exc
    except litellm.APIConnectionError as exc:
        logger.warning("Setup pre-flight: connection failed for model %s", model)
        target = base_url or "the provider's endpoint"
        raise LLMValidationError(
            f"Could not reach {target}. Check the base URL is correct and the server "
            "is running and reachable from this machine."
        ) from exc
    except Exception as exc:
        # Catch-all: never leak the api_key, never crash the request.
        logger.warning("Setup pre-flight: validation failed for model %s (%s)", model, type(exc).__name__)
        raise LLMValidationError(
            f"The LLM credential check failed ({type(exc).__name__}). Verify the model, "
            "API key, and base URL, then try again."
        ) from exc


@router.get("/setup", response_class=HTMLResponse)
async def setup_view(request: Request):
    """Display the first-run setup wizard, or redirect to dashboard if already configured."""
    if not getattr(request.app.state, "setup_required", False):
        return RedirectResponse(url="/", status_code=303)
    _provider_ctx = {"cloud_providers": sorted(CLOUD_PROVIDERS), "local_provider_defaults": LOCAL_PROVIDER_DEFAULTS}
    return templates.TemplateResponse(
        request,
        "setup.html",
        {"setup_mode": True, **_provider_ctx},
    )


@router.post("/setup", dependencies=[Depends(verify_csrf)])
async def complete_setup(
    request: Request,
    llm_model: str = Form(...),
    llm_api_key: str = Form(...),
    llm_base_url: str = Form(""),
):
    """Process setup form and start the application."""
    from pydantic import ValidationError

    from app.config import LLMSettings, NotificationSettings
    from app.scheduler import start_scheduler

    # Strip base_url for cloud providers (e.g. stale Ollama URL when switching to Anthropic)
    effective_base_url = llm_base_url.strip() or None
    if effective_base_url and is_cloud_provider(llm_model):
        effective_base_url = None

    form_values = {
        "llm_model": llm_model,
        "llm_api_key": llm_api_key,
        "llm_base_url": llm_base_url,
    }
    _provider_ctx = {"cloud_providers": sorted(CLOUD_PROVIDERS), "local_provider_defaults": LOCAL_PROVIDER_DEFAULTS}
    try:
        new_settings = Settings(  # type: ignore[call-arg]
            llm=LLMSettings(
                model=llm_model,
                api_key=llm_api_key,
                base_url=effective_base_url,
            ),
            notifications=NotificationSettings(),
        )
        # Pre-flight: confirm the credentials actually work before completing setup,
        # so a bad key/model/base_url is caught here instead of failing silently later.
        await verify_llm_credentials(model=llm_model, api_key=llm_api_key, base_url=effective_base_url)
        save_settings_to_yaml(new_settings, request.app.state.config_path)
        request.app.state.settings = new_settings
        request.app.state.setup_required = False
        # Wire the app so scheduler jobs read live settings from app.state (OVH-015/036).
        start_scheduler(new_settings, db_path=request.app.state.db_path, app=request.app)
    except LLMValidationError as exc:
        return templates.TemplateResponse(
            request,
            "setup.html",
            {"setup_mode": True, "errors": [str(exc)], "form": form_values, **_provider_ctx},
            status_code=422,
        )
    except ValidationError as exc:
        errors = [f"{' → '.join(str(loc) for loc in e['loc'])}: {e['msg']}" for e in exc.errors()]
        return templates.TemplateResponse(
            request,
            "setup.html",
            {"setup_mode": True, "errors": errors, "form": form_values, **_provider_ctx},
            status_code=422,
        )
    except Exception as exc:
        logger.exception("Setup failed: %s", exc)
        return templates.TemplateResponse(
            request,
            "setup.html",
            {"setup_mode": True, "errors": [f"Setup failed: {exc}"], "form": form_values, **_provider_ctx},
            status_code=422,
        )

    logger.info("Setup completed — application is now configured")
    return RedirectResponse(url="/", status_code=303)


@router.get("/settings", response_class=HTMLResponse)
async def settings_view(request: Request):
    """Display of current configuration as an editable form."""
    settings = load_settings(config_path=request.app.state.config_path)
    return templates.TemplateResponse(
        request,
        "settings.html",
        _settings_template_ctx(request, settings=settings),
    )


@router.post("/settings", dependencies=[Depends(verify_csrf)])
async def update_settings(request: Request):
    """Save updated settings to config file and reload into app state.

    Builds Settings from a single parsed form dict rather than restating each field
    (OVH-069); the same dict is reused as ``form_values`` for the 422 re-render, so a
    missed field can no longer render blank silently.
    """
    from pydantic import ValidationError

    from app.config import LLMSettings, NotificationSettings

    form = await request.form()

    def _get(name: str, default: str = "") -> str:
        value = form.get(name, default)
        return value if isinstance(value, str) else default

    llm_model = _get("llm_model")
    llm_api_key = _get("llm_api_key")
    llm_base_url = _get("llm_base_url")
    notification_urls = _get("notification_urls")
    webhook_urls = _get("webhook_urls")
    # An HTML checkbox is absent when unchecked, present (value "true") when checked.
    secure_cookies = form.get("secure_cookies") is not None

    # form_values drives the 422 re-render; build it from the parsed form (single source).
    form_values: dict = {
        "llm_model": llm_model,
        "llm_api_key": llm_api_key,
        "llm_base_url": llm_base_url,
        "notification_urls": notification_urls,
        "webhook_urls": webhook_urls,
        "secure_cookies": secure_cookies,
    }
    for field in _SCALAR_FORM_FIELDS:
        form_values[field] = _get(field)

    parsed_notification_urls = [u.strip() for u in notification_urls.splitlines() if u.strip()]
    parsed_webhook_urls = [u.strip() for u in webhook_urls.splitlines() if u.strip()]

    # API key special-case: a blank field retains the current key (OVH-081). When the key
    # is env-sourced we must not persist the env secret to plaintext YAML (OVH-003), so the
    # field is read-only in the UI and the on-disk value is preserved on save.
    api_key_env_sourced = is_api_key_env_sourced()
    effective_api_key = llm_api_key.strip() or request.app.state.settings.llm.api_key
    # base_url is stripped for cloud providers once on the Settings model (OVH-104).
    effective_base_url = llm_base_url.strip() or None

    # Scalar fields are passed as strings; Pydantic coerces and validates them.
    scalar_kwargs = {field: form_values[field] for field in _SCALAR_FORM_FIELDS}

    # llm_model is required; an empty value has no Pydantic constraint to trip, so guard it
    # explicitly to keep the previous "blank model → 422" behavior (preserved across OVH-069).
    if not llm_model.strip():
        return templates.TemplateResponse(
            request,
            "settings.html",
            _settings_template_ctx(
                request,
                settings=request.app.state.settings,
                errors=["llm_model: Field required"],
                form=form_values,
            ),
            status_code=422,
        )

    try:
        new_settings = Settings(  # type: ignore[call-arg]
            llm=LLMSettings(
                model=llm_model,
                api_key=effective_api_key,
                base_url=effective_base_url,
            ),
            notifications=NotificationSettings(
                urls=parsed_notification_urls,
                webhook_urls=parsed_webhook_urls,
            ),
            secure_cookies=secure_cookies,
            # db_path is infra-only (read-only in the UI); preserve current value.
            db_path=request.app.state.settings.db_path,
            **scalar_kwargs,
        )
        save_settings_to_yaml(
            new_settings,
            request.app.state.config_path,
            preserve_api_key=api_key_env_sourced,
        )
        request.app.state.settings = new_settings
    except ValidationError as exc:
        errors = [f"{' → '.join(str(loc) for loc in e['loc'])}: {e['msg']}" for e in exc.errors()]
        return templates.TemplateResponse(
            request,
            "settings.html",
            _settings_template_ctx(
                request,
                settings=request.app.state.settings,
                errors=errors,
                form=form_values,
            ),
            status_code=422,
        )
    except Exception as exc:
        logger.exception("Failed to save settings: %s", exc)
        return templates.TemplateResponse(
            request,
            "settings.html",
            _settings_template_ctx(
                request,
                settings=request.app.state.settings,
                errors=[f"Failed to save settings: {exc}"],
                form=form_values,
            ),
            status_code=422,
        )

    return RedirectResponse(url="/settings?saved=1", status_code=303)


@router.post("/notifications/test", dependencies=[Depends(verify_csrf)])
async def test_notification(
    request: Request,
    settings: Settings = Depends(get_settings),
):
    """Send a test notification to verify notification configuration."""
    if not settings.notifications.urls:
        return HTMLResponse(
            '<article style="border-left: 4px solid var(--pico-color-orange-500, #f57c00); padding: 1rem;">'
            "<strong>No notification URLs configured.</strong>"
            "<p>To receive notifications, add one or more Apprise notification URLs to your config file "
            "(<code>data/config.yml</code>) under <code>notifications.urls</code>.</p>"
            "<p><small>Supported services include: Ntfy, Discord, Telegram, Slack, Email, Pushover, Gotify, "
            "and <a href='https://github.com/caronc/apprise/wiki#notification-services' target='_blank'>"
            "90+ more via Apprise</a>.</small></p>"
            "<p><small>Example: <code>ntfy://your-topic-name</code></small></p>"
            "</article>",
            status_code=200,
        )

    try:
        success = await send_notification(
            "Topic Watch Test",
            "This is a test notification from Topic Watch. If you received this, notifications are working correctly.",
            settings,
        )
        if success:
            return HTMLResponse(
                '<article style="border-left: 4px solid var(--pico-ins-color, #2e7d32); padding: 1rem;">'
                "<strong>&#10003; Notification sent successfully!</strong>"
                "<p><small>Check your notification service to confirm delivery.</small></p>"
                "</article>",
                status_code=200,
            )
        else:
            return HTMLResponse(
                '<article style="border-left: 4px solid var(--pico-color-orange-500, #f57c00); padding: 1rem;">'
                "<strong>Notification delivery failed.</strong>"
                "<p><small>The notification service rejected the message. Check that your notification URLs "
                "are correct and the service is reachable.</small></p>"
                "</article>",
                status_code=200,
            )
    except Exception:
        return HTMLResponse(
            '<article style="border-left: 4px solid var(--pico-del-color, #c62828); padding: 1rem;">'
            "<strong>Notification error.</strong>"
            "<p><small>An unexpected error occurred. Check the server logs for details.</small></p>"
            "</article>",
            status_code=200,
        )
