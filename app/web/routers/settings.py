"""Setup wizard, settings editor, and notification-test routes."""

import logging

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app.config import (
    CLOUD_PROVIDERS,
    LOCAL_PROVIDER_DEFAULTS,
    Settings,
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
        save_settings_to_yaml(new_settings, request.app.state.config_path)
        request.app.state.settings = new_settings
        request.app.state.setup_required = False
        start_scheduler(new_settings, db_path=request.app.state.db_path)
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
        {
            "settings": settings,
            "config_path": str(request.app.state.config_path),
            "cloud_providers": sorted(CLOUD_PROVIDERS),
            "local_provider_defaults": LOCAL_PROVIDER_DEFAULTS,
        },
    )


@router.post("/settings", dependencies=[Depends(verify_csrf)])
async def update_settings(
    request: Request,
    llm_model: str = Form(...),
    llm_api_key: str = Form(""),
    llm_base_url: str = Form(""),
    notification_urls: str = Form(""),
    webhook_urls: str = Form(""),
    check_interval: str = Form(...),
    max_articles_per_check: int = Form(...),
    knowledge_state_max_tokens: int = Form(2000),
    article_retention_days: int = Form(90),
    feed_fetch_timeout: float = Form(15.0),
    article_fetch_timeout: float = Form(20.0),
    llm_analysis_timeout: int = Form(60),
    llm_knowledge_timeout: int = Form(120),
    apprise_timeout_seconds: int = Form(30),
    web_page_size: int = Form(20),
    min_confidence_threshold: float = Form(0.7),
    min_relevance_threshold: float = Form(0.5),
    secure_cookies: bool = Form(False),
    feed_max_retries: int = Form(2),
    content_fetch_concurrency: int = Form(3),
    scheduler_misfire_grace_time: int = Form(300),
    scheduler_jitter_seconds: int = Form(30),
    llm_max_retries: int = Form(2),
    llm_temperature: float = Form(0.2),
):
    """Save updated settings to config file and reload into app state."""
    from pydantic import ValidationError

    from app.config import LLMSettings, NotificationSettings

    parsed_notification_urls = [u.strip() for u in notification_urls.splitlines() if u.strip()]
    parsed_webhook_urls = [u.strip() for u in webhook_urls.splitlines() if u.strip()]

    # If API key field is empty, retain existing key
    effective_api_key = llm_api_key.strip() or request.app.state.settings.llm.api_key

    # Strip base_url for cloud providers (e.g. stale Ollama URL when switching to Anthropic)
    effective_base_url = llm_base_url.strip() or None
    if effective_base_url and is_cloud_provider(llm_model):
        effective_base_url = None

    # Build a new Settings object to validate via Pydantic, then save
    form_values = {
        "llm_model": llm_model,
        "llm_api_key": llm_api_key,
        "llm_base_url": llm_base_url,
        "notification_urls": notification_urls,
        "webhook_urls": webhook_urls,
        "check_interval": check_interval,
        "max_articles_per_check": max_articles_per_check,
        "knowledge_state_max_tokens": knowledge_state_max_tokens,
        "article_retention_days": article_retention_days,
        "feed_fetch_timeout": feed_fetch_timeout,
        "article_fetch_timeout": article_fetch_timeout,
        "llm_analysis_timeout": llm_analysis_timeout,
        "llm_knowledge_timeout": llm_knowledge_timeout,
        "apprise_timeout_seconds": apprise_timeout_seconds,
        "web_page_size": web_page_size,
        "min_confidence_threshold": min_confidence_threshold,
        "min_relevance_threshold": min_relevance_threshold,
        "secure_cookies": secure_cookies,
        "feed_max_retries": feed_max_retries,
        "content_fetch_concurrency": content_fetch_concurrency,
        "scheduler_misfire_grace_time": scheduler_misfire_grace_time,
        "scheduler_jitter_seconds": scheduler_jitter_seconds,
        "llm_max_retries": llm_max_retries,
        "llm_temperature": llm_temperature,
    }
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
            check_interval=check_interval,
            max_articles_per_check=max_articles_per_check,
            knowledge_state_max_tokens=knowledge_state_max_tokens,
            article_retention_days=article_retention_days,
            feed_fetch_timeout=feed_fetch_timeout,
            article_fetch_timeout=article_fetch_timeout,
            llm_analysis_timeout=llm_analysis_timeout,
            llm_knowledge_timeout=llm_knowledge_timeout,
            apprise_timeout_seconds=apprise_timeout_seconds,
            web_page_size=web_page_size,
            min_confidence_threshold=min_confidence_threshold,
            min_relevance_threshold=min_relevance_threshold,
            secure_cookies=secure_cookies,
            feed_max_retries=feed_max_retries,
            content_fetch_concurrency=content_fetch_concurrency,
            scheduler_misfire_grace_time=scheduler_misfire_grace_time,
            scheduler_jitter_seconds=scheduler_jitter_seconds,
            llm_max_retries=llm_max_retries,
            llm_temperature=llm_temperature,
            # db_path is infra-only (read-only in the UI); preserve current value.
            db_path=request.app.state.settings.db_path,
        )
        save_settings_to_yaml(new_settings, request.app.state.config_path)
        request.app.state.settings = new_settings
    except ValidationError as exc:
        errors = [f"{' → '.join(str(loc) for loc in e['loc'])}: {e['msg']}" for e in exc.errors()]
        return templates.TemplateResponse(
            request,
            "settings.html",
            {
                "settings": request.app.state.settings,
                "config_path": str(request.app.state.config_path),
                "errors": errors,
                "form": form_values,
                "cloud_providers": sorted(CLOUD_PROVIDERS),
                "local_provider_defaults": LOCAL_PROVIDER_DEFAULTS,
            },
            status_code=422,
        )
    except Exception as exc:
        logger.exception("Failed to save settings: %s", exc)
        return templates.TemplateResponse(
            request,
            "settings.html",
            {
                "settings": request.app.state.settings,
                "config_path": str(request.app.state.config_path),
                "errors": [f"Failed to save settings: {exc}"],
                "form": form_values,
                "cloud_providers": sorted(CLOUD_PROVIDERS),
                "local_provider_defaults": LOCAL_PROVIDER_DEFAULTS,
            },
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
