"""Setup wizard, settings editor, and notification-test routes."""

import asyncio
import logging
import re
import sqlite3
from datetime import UTC, datetime

import litellm
from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app.config import (
    CLOUD_PROVIDERS,
    LOCAL_PROVIDER_DEFAULTS,
    Settings,
    config_revision,
    env_owned_field_paths,
    is_api_key_env_sourced,
    is_exa_key_env_sourced,
    is_keyless_llm_provider,
    load_settings,
    save_settings_to_yaml,
    strip_env_owned,
)
from app.crud import reset_all_heartbeat_state
from app.notifications import send_notification
from app.web.csrf import verify_csrf
from app.web.dependencies import get_db_conn, get_settings
from app.web.routers._validation import format_validation_errors, normalize_base_url
from app.web.routers.templates import _mask_url, templates
from app.webhooks import send_webhook

logger = logging.getLogger(__name__)

router = APIRouter()

# Serializes the one-shot setup transition: gate check, persistence and scheduler
# start are one critical section, so two first-run submissions cannot interleave.
_setup_lock = asyncio.Lock()


# A masked delivery target the settings page rendered: "scheme://**** [3]".
_MASKED_TARGET = re.compile(r"^(?P<mask>\S+) \[(?P<index>\d+)\]$")

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
    "silence_heartbeat_checks",
    "feed_max_retries",
    "content_fetch_concurrency",
    "scheduler_misfire_grace_time",
    "scheduler_jitter_seconds",
    "llm_max_retries",
    "llm_temperature",
)


# Form control -> Settings field path, for every control the environment can own.
# A control whose path is env-owned renders disabled: the value it shows cannot be
# changed here, and a save that looked like it worked never applied (C5-2). The two
# API-key controls are deliberately absent — they have their own read-only branch
# and their value must never reach the page.
_FORM_FIELD_PATHS: dict[str, tuple[str, ...]] = {
    "llm_model": ("llm", "model"),
    "llm_base_url": ("llm", "base_url"),
    "enable_exa": ("exa", "enabled"),
    "notification_urls": ("notifications", "urls"),
    "webhook_urls": ("notifications", "webhook_urls"),
    "secure_cookies": ("secure_cookies",),
    **{name: (name,) for name in _SCALAR_FORM_FIELDS},
}


def env_locked_controls() -> set[str]:
    """Names of the form controls the environment currently owns."""
    owned = env_owned_field_paths()
    return {name for name, path in _FORM_FIELD_PATHS.items() if path in owned}


def _control_value(settings: object, name: str, ctx: dict) -> object:
    """What a locked control should display: the value the environment supplies.

    A disabled control submits nothing, so the re-render after a validation error
    elsewhere on the form would otherwise show it empty.
    """
    if name in ("notification_urls", "webhook_urls"):
        masked = ctx.get(f"masked_{name}", [])
        return "\n".join(masked) if isinstance(masked, list) else ""
    value: object = settings
    for part in _FORM_FIELD_PATHS[name]:
        value = getattr(value, part, None)
    return "" if value is None else value


def _interval_preview(raw: str) -> str | None:
    """Human-readable schedule preview for a default-interval string.

    ``parse_interval`` RAISES on invalid/out-of-range input, so guard it and return
    ``None`` when the interval can't be rendered (the template then omits the preview).
    """
    from app.interval import format_interval, parse_interval

    text = (raw or "").strip()
    if not text:
        return None
    try:
        return format_interval(parse_interval(text))
    except ValueError:
        return None


def _render(request: Request, template: str, ctx: dict, status_code: int = 200) -> HTMLResponse:
    """Render a setup/settings page with ``Cache-Control: no-store``.

    These pages carry key hints and delivery targets, so they must not survive in
    a disk cache, a back-forward restore or a saved-page capture (AUG-017).
    """
    response = templates.TemplateResponse(request, template, ctx, status_code=status_code)
    response.headers["Cache-Control"] = "no-store"
    return response


def mask_targets(urls: list[str]) -> list[str]:
    """Render saved delivery URLs as numbered masked placeholders.

    Apprise and webhook URLs routinely carry a token in the userinfo or the path,
    and the settings page put them on screen verbatim (AUG-127). Each entry
    becomes ``scheme://**** [n]``; the index is what lets a placeholder the user
    left alone be resolved back to the URL it stands for on save.
    """
    return [f"{_mask_url(url)} [{index}]" for index, url in enumerate(urls, start=1)]


def restore_masked_targets(submitted: list[str], stored: list[str], *, field: str) -> list[str]:
    """Turn masked placeholders back into the URLs they stand for.

    A line the user left untouched resolves to its stored URL; a line they
    deleted is gone; anything else is taken as typed.

    Raises ``ValueError`` when a line looks like a placeholder but resolves to
    nothing — never resolve it to a different entry, and never store the
    placeholder text itself, which would look like a normal masked line on the
    next page load while quietly being an unusable delivery target.
    """
    restored: list[str] = []
    for line in submitted:
        match = _MASKED_TARGET.match(line)
        if match:
            index = int(match.group("index")) - 1
            if 0 <= index < len(stored) and match.group("mask") == _mask_url(stored[index]):
                restored.append(stored[index])
                continue
            raise ValueError(
                f"{field}: '{line}' does not match a saved URL. Delete the line to remove that "
                "target, or type the full URL to replace it."
            )
        restored.append(line)
    return restored


_STALE_CONFIG_ERROR = (
    "The configuration file changed on disk after this page was loaded, so nothing was saved. "
    "The form below shows the current values — reapply your change and save again."
)


def _disk_settings(request: Request) -> Settings:
    """Settings as the config file currently reads.

    Falls back to the live object when the file cannot be parsed, so a config
    corrupted outside the app still renders an editable page instead of a 500.
    """
    try:
        return load_settings(config_path=request.app.state.config_path)
    except Exception:
        logger.warning("Could not read %s; using the live settings", request.app.state.config_path, exc_info=True)
        settings: Settings = request.app.state.settings
        return settings


def _settings_template_ctx(request: Request, **extra: object) -> dict:
    """Shared template context for the settings page (provider lists + env-owned state)."""
    env_locked = env_locked_controls()
    ctx: dict = {
        # Controls the environment owns. The template renders each one disabled with
        # the same note the LLM key already carries, so an unchangeable setting is
        # never presented as editable (C5-2).
        "env_locked": env_locked,
        "config_path": str(request.app.state.config_path),
        # The generation this page is editing. Carried in the form so a save against a
        # file that changed underneath it is refused instead of silently winning (AUG-291).
        "config_revision": config_revision(request.app.state.config_path),
        "cloud_providers": sorted(CLOUD_PROVIDERS),
        "local_provider_defaults": LOCAL_PROVIDER_DEFAULTS,
        "api_key_env_sourced": is_api_key_env_sourced(),
        "exa_key_env_sourced": is_exa_key_env_sourced(),
    }
    # Server-side schedule preview for the default interval. Use the submitted form value
    # on a re-render (the 422 path passes form=), else the persisted setting.
    form = extra.get("form")
    settings = extra.get("settings")
    if isinstance(form, dict) and "check_interval" in form:
        raw_interval = str(form.get("check_interval", ""))
    elif settings is not None:
        raw_interval = str(getattr(settings, "check_interval", ""))
    else:
        raw_interval = ""
    ctx["interval_preview"] = _interval_preview(raw_interval)
    if settings is not None:
        notifications = getattr(settings, "notifications", None)
        ctx["masked_notification_urls"] = mask_targets(getattr(notifications, "urls", []) or [])
        ctx["masked_webhook_urls"] = mask_targets(getattr(notifications, "webhook_urls", []) or [])
    ctx.update(extra)
    # A disabled control submits nothing, so a re-render driven by the form dict would
    # show every locked control empty. Show what the environment supplies instead.
    if isinstance(ctx.get("form"), dict) and settings is not None:
        ctx["form"] = {
            key: _control_value(settings, key, ctx) if key in env_locked else value
            for key, value in ctx["form"].items()
        }
    return ctx


class LLMValidationError(Exception):
    """Raised when a pre-flight LLM credential check fails.

    The message is always user-safe: it never contains the API key and explains
    what went wrong (bad key vs. unreachable base URL vs. bad model) and how to fix it.
    """


def _preflight_messages() -> list[dict[str, str]]:
    """The smallest request that still asks for the live response schema."""
    return [
        {
            "role": "system",
            "content": "You are a monitoring assistant. Answer only with the requested structured output.",
        },
        {
            "role": "user",
            "content": (
                "Connectivity check, no articles to review. Answer with has_new_info false, "
                "summary null, empty key_facts and source_urls, confidence 0, relevance 0, importance 1."
            ),
        },
    ]


def _preflight_error(exc: BaseException, model: str, base_url: str | None) -> LLMValidationError:
    """Map a probe failure to a user-safe message; the API key is never in it.

    The live call path wraps provider errors (instructor re-raises its own type with
    the real one chained), so the classification walks the whole chain the same way
    ``app.analysis.llm`` does rather than matching only the outermost type.
    """
    from pydantic import ValidationError as PydanticValidationError

    from app.analysis.llm import _iter_error_chain

    chain = list(_iter_error_chain(exc))

    def found(*types: type[BaseException]) -> bool:
        return any(isinstance(inner, types) for inner in chain)

    if found(litellm.AuthenticationError, litellm.PermissionDeniedError):
        logger.warning("Setup pre-flight: authentication rejected for model %s", model)
        return LLMValidationError(
            "Authentication failed: the API key was rejected by the provider. "
            "Double-check the key for the correct provider and account."
        )
    if found(litellm.NotFoundError):
        logger.warning("Setup pre-flight: model not found for %s", model)
        return LLMValidationError(
            f"The model '{model}' was not found. Check the model string uses the "
            "LiteLLM 'provider/model-name' format and that the model exists."
        )
    if found(litellm.APIConnectionError, litellm.Timeout):
        logger.warning("Setup pre-flight: connection failed for model %s", model)
        target = base_url or "the provider's endpoint"
        return LLMValidationError(
            f"Could not reach {target}. Check the base URL is correct and the server "
            "is running and reachable from this machine."
        )
    if found(PydanticValidationError):
        logger.warning("Setup pre-flight: model answered outside the required schema for %s", model)
        return LLMValidationError(
            f"The model '{model}' replied, but not in the structured format Topic Watch needs. "
            "Pick a model that supports tool calling or JSON output."
        )
    if found(litellm.BadRequestError):
        logger.warning("Setup pre-flight: request shape rejected for model %s", model)
        return LLMValidationError(
            f"The provider rejected the request Topic Watch makes for '{model}'. "
            "The endpoint may not support structured output for this model."
        )
    logger.warning("Setup pre-flight: validation failed for model %s (%s)", model, type(exc).__name__)
    return LLMValidationError(
        f"The LLM credential check failed ({type(exc).__name__}). Verify the model, "
        "API key, and base URL, then try again."
    )


async def verify_llm_credentials(model: str, api_key: str, base_url: str | None) -> None:
    """Confirm the supplied credentials satisfy the call analysis actually makes.

    The probe goes through the live structured-output path — same client, same
    response model, same TOOLS -> JSON -> MD_JSON fallback, same temperature, token
    ceiling and analysis timeout. A raw one-token ping tested a different contract
    on a much shorter deadline (AUG-335): a slow but compatible local Ollama failed
    setup, while an endpoint that answers a plain completion but cannot produce the
    structured response passed and then failed every check.

    Returns ``None`` on success; raises :class:`LLMValidationError` with a friendly,
    key-free message otherwise.
    """
    from app.analysis.llm import NoveltyResponse, _create_structured
    from app.config import LLMSettings

    probe = Settings(llm=LLMSettings(model=model, api_key=api_key, base_url=base_url))  # type: ignore[call-arg]
    try:
        await _create_structured(
            probe,
            response_model=NoveltyResponse,
            build_messages=_preflight_messages,
            timeout=probe.llm_analysis_timeout,
        )
    except Exception as exc:
        raise _preflight_error(exc, model, base_url) from exc


def _setup_settings(request: Request, model: str, api_key: str, base_url: str | None) -> Settings:
    """The live settings with only the setup-owned LLM block replaced.

    Setup runs whenever LLM credentials are incomplete, which says nothing about the
    rest of the configuration: notification targets and Exa can already be set in YAML
    or the environment, and building a fresh ``Settings`` erased them (AUG-200). Any
    environment-owned LLM field is left alone for the same reason the settings form
    leaves it alone — the environment would win again at the next load anyway.
    """
    from app.config import LLMSettings

    current: Settings = request.app.state.settings
    submitted = strip_env_owned({"llm": {"model": model, "api_key": api_key, "base_url": base_url}}).get("llm", {})
    return current.model_copy(update={"llm": LLMSettings(**{**current.llm.model_dump(), **submitted})})


def _publish_setup(request: Request, new_settings: Settings) -> None:
    """Persist, start the scheduler, and only then open the app for business.

    The setup gate is single-shot, so closing it before the scheduler exists left an
    application that looked configured, refused further setup attempts, and monitored
    nothing until a restart (AUG-199/292). Ordering here is the fix: on failure the
    partial scheduler is stopped and the previous live settings restored, so setup
    stays retryable. The written file is deliberately left in place — it holds the
    credentials the user just gave, and a retry rewrites it.
    """
    from app.scheduler import start_scheduler, stop_scheduler

    previous_settings = request.app.state.settings
    save_settings_to_yaml(new_settings, request.app.state.config_path)
    # Wire the app so scheduler jobs read live settings from app.state (OVH-015/036).
    request.app.state.settings = new_settings
    try:
        start_scheduler(new_settings, db_path=request.app.state.db_path, app=request.app)
    except BaseException:
        stop_scheduler()
        request.app.state.settings = previous_settings
        raise
    request.app.state.setup_required = False


def _setup_provider_ctx(request: Request) -> dict:
    """Provider lists plus the LLM fields the environment already supplies.

    The wizard writes only what the environment does not own, so a control for an
    env-owned field is rendered read-only rather than demanding an entry that is
    discarded on save (C5-4).
    """
    owned = env_owned_field_paths()
    settings: Settings = request.app.state.settings
    return {
        "cloud_providers": sorted(CLOUD_PROVIDERS),
        "local_provider_defaults": LOCAL_PROVIDER_DEFAULTS,
        "api_key_env_sourced": ("llm", "api_key") in owned,
        "model_env_sourced": ("llm", "model") in owned,
        "env_model": settings.llm.model,
    }


@router.get("/setup", response_class=HTMLResponse)
async def setup_view(request: Request):
    """Display the first-run setup wizard, or redirect to dashboard if already configured."""
    if not getattr(request.app.state, "setup_required", False):
        return RedirectResponse(url="/", status_code=303)
    return _render(
        request,
        "setup.html",
        {"setup_mode": True, **_setup_provider_ctx(request)},
    )


@router.post("/setup", dependencies=[Depends(verify_csrf)])
async def complete_setup(
    request: Request,
    # Optional at the transport level for the same reason as the key below: an
    # env-owned model renders disabled and is not submitted at all (C5-4).
    llm_model: str = Form(""),
    # Optional at the transport level: an empty form value reads as "missing" and would
    # 422 with a generic error page before the handler could explain anything. Whether a
    # key is actually required is provider-aware and enforced below (AUG-107).
    llm_api_key: str = Form(""),
    llm_base_url: str = Form(""),
    skip_validation: str = Form(""),
):
    """Process setup form and start the application."""
    from pydantic import ValidationError

    # Single-shot setup (OVH-059): once configured, a replay/double-submit/stale-bookmark
    # POST must not re-run setup — that would clobber live credentials and start a second
    # scheduler (orphaning the running one). Ongoing changes go through /settings.
    if not getattr(request.app.state, "setup_required", False):
        return RedirectResponse(url="/", status_code=303)

    # Normalize blank -> None so the pre-flight check sees the value the model will
    # persist. base_url is honored for every provider (OVH-104 reversal); OVH-153.
    effective_base_url = normalize_base_url(llm_base_url)

    # The key is deliberately absent: every use of this dict is an error re-render,
    # and a submitted secret must not be written back into the page (AUG-017).
    form_values = {
        "llm_model": llm_model,
        "llm_api_key": "",
        "llm_base_url": llm_base_url,
    }
    _provider_ctx = _setup_provider_ctx(request)
    try:
        # One setup at a time (AUG-292). The gate above is checked before an awaited
        # credential probe, so two first-run submissions could both pass it and both
        # publish; inside the lock it is re-checked after every await instead.
        async with _setup_lock:
            if not getattr(request.app.state, "setup_required", False):
                return RedirectResponse(url="/", status_code=303)

            # What the environment supplies is not the user's to type: those fields are
            # rendered read-only and dropped from the submission, so demanding them here
            # asked for a value that would be discarded anyway (C5-4).
            owned = env_owned_field_paths()
            if not llm_model.strip() and ("llm", "model") not in owned:
                raise LLMValidationError("A model is required, in LiteLLM 'provider/model-name' format.")
            # Provider-aware key requirement (AUG-107): a hosted provider needs one,
            # a local one never did and the wizard used to demand it anyway.
            if not llm_api_key.strip() and ("llm", "api_key") not in owned and not is_keyless_llm_provider(llm_model):
                raise LLMValidationError(
                    f"An API key is required for '{llm_model}'. Only a local provider "
                    f"({', '.join(sorted(LOCAL_PROVIDER_DEFAULTS))}) can be left blank."
                )

            new_settings = _setup_settings(request, llm_model, llm_api_key, effective_base_url)
            # Pre-flight: confirm the credentials actually work before completing setup,
            # so a bad key/model/base_url is caught here instead of failing silently later.
            # The "Save anyway" escape hatch (skip_validation) bypasses this so a transient
            # provider error or a stale default model string can't trap a brand-new user at
            # /setup. It is safe: is_configured() only needs a non-placeholder key, and a bad
            # key then degrades gracefully — analyze_articles() returns has_new_info=False on
            # any LLM failure (no crash, no spurious notification). The user fixes it later in
            # Settings, and Feed Health / `doctor` surface the failing checks.
            if skip_validation != "true":
                # Probe what will actually be used, not what was typed: an env-owned
                # field never reaches the saved settings, so validating the submitted
                # value tested a different credential than the one setup persists (C5-4).
                await verify_llm_credentials(
                    model=new_settings.llm.model,
                    api_key=new_settings.llm.api_key,
                    base_url=new_settings.llm.base_url,
                )
                if not getattr(request.app.state, "setup_required", False):
                    return RedirectResponse(url="/", status_code=303)
            _publish_setup(request, new_settings)
    except LLMValidationError as exc:
        return _render(
            request,
            "setup.html",
            {"setup_mode": True, "errors": [str(exc)], "form": form_values, **_provider_ctx},
            status_code=422,
        )
    except ValidationError as exc:
        return _render(
            request,
            "setup.html",
            {"setup_mode": True, "errors": format_validation_errors(exc), "form": form_values, **_provider_ctx},
            status_code=422,
        )
    except Exception as exc:
        logger.exception("Setup failed: %s", exc)
        return _render(
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
    return _render(
        request,
        "settings.html",
        _settings_template_ctx(request, settings=settings),
    )


@router.post("/settings", dependencies=[Depends(verify_csrf)])
async def update_settings(
    request: Request,
    conn: sqlite3.Connection = Depends(get_db_conn, scope="function"),
):
    """Save updated settings to config file and reload into app state.

    Builds Settings from a single parsed form dict rather than restating each field
    (OVH-069); the same dict is reused as ``form_values`` for the 422 re-render, so a
    missed field can no longer render blank silently.
    """
    from pydantic import ValidationError

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
    enable_exa = form.get("enable_exa") is not None
    exa_api_key = _get("exa_api_key")

    # form_values drives the 422 re-render; build it from the parsed form (single source).
    # The two key fields are blanked: they are password inputs the user cannot read
    # back, so echoing a submitted secret leaves it in the page source, the browser's
    # form cache and any saved-page capture for no benefit (AUG-017). Both fields
    # already mean "leave blank to keep the current key", so re-entry is the norm.
    form_values: dict = {
        "llm_model": llm_model,
        "llm_api_key": "",
        "llm_base_url": llm_base_url,
        "notification_urls": notification_urls,
        "webhook_urls": webhook_urls,
        "secure_cookies": secure_cookies,
        "enable_exa": enable_exa,
        "exa_api_key": "",
    }
    for field in _SCALAR_FORM_FIELDS:
        form_values[field] = _get(field)

    # Optimistic concurrency (AUG-291): the form carries the generation it rendered.
    # Refuse a save built on anything else — a second tab, an external edit, or a key
    # rotated on disk — rather than overwriting the whole document with stale values.
    if _get("config_revision") != config_revision(request.app.state.config_path):
        return _render(
            request,
            "settings.html",
            _settings_template_ctx(
                request,
                settings=_disk_settings(request),
                errors=[_STALE_CONFIG_ERROR],
            ),
            status_code=409,
        )

    # Everything preserved rather than submitted (blank keys, infra-only fields, the
    # URLs behind the masked placeholders) comes from the generation just verified,
    # not from app.state, which can be older than the file (AUG-291).
    disk_settings = _disk_settings(request)

    # Saved targets reach the form as masked placeholders (AUG-127); resolve the ones
    # the user left untouched back to the URLs they stand for.
    current_notifications = disk_settings.notifications
    try:
        parsed_notification_urls = restore_masked_targets(
            [u.strip() for u in notification_urls.splitlines() if u.strip()],
            current_notifications.urls,
            field="notification_urls",
        )
        parsed_webhook_urls = restore_masked_targets(
            [u.strip() for u in webhook_urls.splitlines() if u.strip()],
            current_notifications.webhook_urls,
            field="webhook_urls",
        )
    except ValueError as exc:
        return _render(
            request,
            "settings.html",
            _settings_template_ctx(
                request,
                settings=request.app.state.settings,
                errors=[str(exc)],
                form=form_values,
            ),
            status_code=422,
        )

    # API key special-case: a blank field retains the current key (OVH-081). An
    # env-sourced key is dropped from the submitted data below, so it is neither
    # persisted nor editable here (OVH-003/AUG-241).
    effective_api_key = llm_api_key.strip() or disk_settings.llm.api_key
    # Exa key mirrors the LLM key: blank retains the current one.
    effective_exa_key = exa_api_key.strip() or disk_settings.exa.api_key
    # exa base_url is infra/proxy-only (not a form field); preserve the current value like db_path.
    exa_base_url = disk_settings.exa.base_url
    # Shared normalization (OVH-153): blank -> None. base_url is honored for every
    # provider (OVH-104 reversal); setup and settings share this seam.
    effective_base_url = normalize_base_url(llm_base_url)

    # Scalar fields are passed as strings; Pydantic coerces and validates them.
    scalar_kwargs = {field: form_values[field] for field in _SCALAR_FORM_FIELDS}

    # llm_model is required; an empty value has no Pydantic constraint to trip, so guard it
    # explicitly to keep the previous "blank model → 422" behavior (preserved across OVH-069).
    # Enabling Exa without a key is refused the same way: the model would quietly disable
    # it again, and a silent revert on a save the user asked for is worse than an error.
    field_errors = []
    # An env-owned model renders disabled, so the browser submits nothing for it —
    # that is not a missing field, it is a field this form does not own (C5-2).
    if not llm_model.strip() and "llm_model" not in env_locked_controls():
        field_errors.append("llm_model: Field required")
    if enable_exa and not effective_exa_key.strip():
        field_errors.append(
            "exa_api_key: Enter an Exa API key to enable the Exa source. Without one, Exa topics cannot fetch."
        )
    if field_errors:
        return _render(
            request,
            "settings.html",
            _settings_template_ctx(
                request,
                settings=request.app.state.settings,
                errors=field_errors,
                form=form_values,
            ),
            status_code=422,
        )

    # Everything the form decides, as plain data so pydantic-settings can merge it
    # per field. Environment-owned paths are then removed: init outranks env, so
    # leaving them in would let an edit override an env-owned value until restart
    # (AUG-241). What is left is exactly the YAML-owned half of the document.
    submitted: dict = {
        "llm": {
            "model": llm_model,
            "api_key": effective_api_key,
            "base_url": effective_base_url,
        },
        "notifications": {
            "urls": parsed_notification_urls,
            "webhook_urls": parsed_webhook_urls,
        },
        "exa": {
            "enabled": enable_exa,
            "api_key": effective_exa_key,
            "base_url": exa_base_url,
        },
        "secure_cookies": secure_cookies,
        # db_path is infra-only (read-only in the UI); preserve current value.
        "db_path": disk_settings.db_path,
        **scalar_kwargs,
    }

    try:
        # The result is the canonical merged object — the form's values for what YAML
        # owns, the environment's for what it owns — so app.state and the next page
        # render agree with what a restart would produce.
        new_settings = Settings(**strip_env_owned(submitted))  # type: ignore[call-arg]
        save_settings_to_yaml(new_settings, request.app.state.config_path)
        previous_heartbeat = getattr(request.app.state.settings, "silence_heartbeat_checks", 0)
        request.app.state.settings = new_settings
        if previous_heartbeat > 0 and new_settings.silence_heartbeat_checks <= 0:
            # Switching the feature off reconciles its state now, not whenever each
            # topic next happens to run: disabling and re-enabling inside one check
            # interval would otherwise leave the old latch in place (AUG-260).
            cleared = reset_all_heartbeat_state(conn)
            conn.commit()
            logger.info("Silence Heartbeat switched off: cleared %d outage latch(es)", cleared)
    except ValidationError as exc:
        return _render(
            request,
            "settings.html",
            _settings_template_ctx(
                request,
                settings=request.app.state.settings,
                errors=format_validation_errors(exc),
                form=form_values,
            ),
            status_code=422,
        )
    except Exception as exc:
        logger.exception("Failed to save settings: %s", exc)
        return _render(
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


def _llm_test_result(ok: bool, message: str) -> HTMLResponse:
    """Render the Test LLM configuration result, styled like the notification test."""
    color = "var(--pico-ins-color, #2e7d32)" if ok else "var(--pico-del-color, #c62828)"
    lead = "&#10003; Connection succeeded." if ok else "Connection failed."
    from markupsafe import escape

    return HTMLResponse(
        f'<article style="border-left: 4px solid {color}; padding: 1rem;">'
        f"<strong>{lead}</strong>"
        f"<p><small>{escape(message)}</small></p>"
        "</article>",
        status_code=200,
    )


@router.post("/settings/test-llm", dependencies=[Depends(verify_csrf)])
async def test_llm_configuration(
    request: Request,
    llm_model: str = Form(""),
    llm_api_key: str = Form(""),
    llm_base_url: str = Form(""),
):
    """Probe the currently EDITED (unsaved) LLM fields via the live analysis call path.

    Ordinary Settings saves model, key, and base URL and reports success with no
    live check (AUG-111) — a typo or dead endpoint becomes active configuration
    until a check fails later. This reuses ``verify_llm_credentials`` (AUG-335)
    rather than a second probe shape, against exactly the values in the form right
    now, not the last-saved ones. Blank/absent fields resolve the same way Save
    does: a blank key keeps the current one, an env-owned field (its control
    renders disabled, so the browser never submits it) keeps the environment's
    value.
    """
    env_locked = env_locked_controls()
    disk_settings = _disk_settings(request)

    if not llm_model.strip() and "llm_model" not in env_locked:
        return _llm_test_result(False, "A model is required, in LiteLLM 'provider/model-name' format.")

    effective_model = llm_model.strip() or disk_settings.llm.model
    effective_api_key = llm_api_key.strip() or disk_settings.llm.api_key
    effective_base_url = (
        disk_settings.llm.base_url if "llm_base_url" in env_locked else normalize_base_url(llm_base_url)
    )

    try:
        await verify_llm_credentials(model=effective_model, api_key=effective_api_key, base_url=effective_base_url)
    except LLMValidationError as exc:
        return _llm_test_result(False, str(exc))
    return _llm_test_result(True, "The credentials and model work with the same call analysis makes.")


@router.post("/notifications/test", dependencies=[Depends(verify_csrf)])
async def test_notification(
    request: Request,
    settings: Settings = Depends(get_settings),
):
    """Exercise every configured delivery channel and report each one's result.

    Apprise and webhooks share one Notifications card and one Test button, but
    the test used to cover only Apprise: a webhook-only setup was told it had
    nothing configured and could not verify its sole delivery path before a real
    event depended on it (AUG-108). Each configured channel is now attempted and
    reported on its own.

    Nothing is queued for retry — a test that failed is information, not a
    delivery the user is owed.
    """
    apprise_urls = settings.notifications.urls
    webhook_urls = settings.notifications.webhook_urls
    if not apprise_urls and not webhook_urls:
        return HTMLResponse(
            '<article style="border-left: 4px solid var(--pico-color-orange-500, #f57c00); padding: 1rem;">'
            "<strong>No notification URLs configured.</strong>"
            "<p>To receive notifications, add one or more Apprise notification URLs to your config file "
            "(<code>data/config.yml</code>) under <code>notifications.urls</code>, or a webhook endpoint "
            "under <code>notifications.webhook_urls</code>.</p>"
            "<p><small>Supported services include: Ntfy, Discord, Telegram, Slack, Email, Pushover, Gotify, "
            "and <a href='https://github.com/caronc/apprise/wiki#notification-services' target='_blank'>"
            "90+ more via Apprise</a>.</small></p>"
            "<p><small>Example: <code>ntfy://YOUR_NTFY_TOPIC</code> (replace with your own topic)</small></p>"
            "</article>",
            status_code=200,
        )

    lines: list[str] = []
    all_ok = True
    try:
        if apprise_urls:
            sent = await send_notification(
                "Topic Watch Test",
                "This is a test notification from Topic Watch. "
                "If you received this, notifications are working correctly.",
                settings,
            )
            all_ok = all_ok and sent
            lines.append(f"<li>Apprise: {'delivered' if sent else 'failed'}</li>")

        if webhook_urls:
            payload = {
                "topic": "Topic Watch Test",
                "summary": "This is a test webhook from Topic Watch.",
                "test": True,
                "timestamp": datetime.now(UTC).isoformat(),
            }
            outcomes = await asyncio.gather(*(send_webhook(url, payload) for url in webhook_urls))
            delivered = sum(1 for outcome in outcomes if outcome.ok)
            all_ok = all_ok and delivered == len(webhook_urls)
            lines.append(f"<li>Webhooks: {delivered}/{len(webhook_urls)} delivered</li>")
    except Exception:
        # The old copy told the user to check the logs without ever writing one.
        logger.warning("Test notification failed unexpectedly", exc_info=True)
        return HTMLResponse(
            '<article style="border-left: 4px solid var(--pico-del-color, #c62828); padding: 1rem;">'
            "<strong>Notification error.</strong>"
            "<p><small>An unexpected error occurred. Check the server logs for details.</small></p>"
            "</article>",
            status_code=200,
        )

    per_channel = f"<ul>{''.join(lines)}</ul>"
    if all_ok:
        return HTMLResponse(
            '<article style="border-left: 4px solid var(--pico-ins-color, #2e7d32); padding: 1rem;">'
            "<strong>&#10003; Notification sent successfully!</strong>"
            f"{per_channel}"
            "<p><small>Check your notification service to confirm delivery.</small></p>"
            "</article>",
            status_code=200,
        )
    return HTMLResponse(
        '<article style="border-left: 4px solid var(--pico-color-orange-500, #f57c00); padding: 1rem;">'
        "<strong>Notification delivery failed.</strong>"
        f"{per_channel}"
        "<p><small>The service rejected the message. Check that your notification and webhook URLs "
        "are correct and the services are reachable.</small></p>"
        "</article>",
        status_code=200,
    )
