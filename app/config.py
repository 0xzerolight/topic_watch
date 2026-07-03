"""Configuration management for Topic Watch.

Loads settings from data/config.yml with environment variable overrides.
Environment variables use the prefix TOPIC_WATCH_ with double underscore
for nested keys (e.g., TOPIC_WATCH_LLM__API_KEY).
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_CONFIG_PATH = DATA_DIR / "config.yml"

# Module-level override for testability
_yaml_file_override: str | None = None

# Known cloud providers — used for the unknown-provider warning and the UI base_url
# hint. base_url is NOT stripped for these: an explicitly-set base_url is honored for
# every provider (OVH-104 reversal) so OpenAI-compatible gateways work.
CLOUD_PROVIDERS: frozenset[str] = frozenset(
    {
        "openai",
        "anthropic",
        "gemini",
        "azure",
        "cohere",
        "replicate",
        "huggingface",
        "together_ai",
        "groq",
        "deepseek",
        "mistral",
        "xai",
        "perplexity",
    }
)

# Default base URLs for self-hosted providers (used as form auto-fill hints).
LOCAL_PROVIDER_DEFAULTS: dict[str, str] = {
    "ollama": "http://localhost:11434",
}


# Env var that supplies the LLM API key (env > YAML; see settings_customise_sources).
_API_KEY_ENV_VAR = "TOPIC_WATCH_LLM__API_KEY"


def is_api_key_env_sourced() -> bool:
    """Return True if the LLM API key is supplied via environment (env > YAML).

    When True, the settings UI must treat the key as read-only and the save path must
    NOT materialize the env-derived secret into plaintext config.yml (OVH-003).
    """
    return bool(os.environ.get(_API_KEY_ENV_VAR))


class LLMSettings(BaseModel):
    """LLM provider configuration."""

    model: str = Field(default="", description="LiteLLM model string, e.g. 'openai/gpt-4o-mini'")
    api_key: str = Field(default="", description="API key for the LLM provider")
    base_url: str | None = Field(
        default=None,
        description="Optional base URL for a self-hosted (Ollama) or OpenAI-compatible gateway endpoint",
    )


class NotificationSettings(BaseModel):
    """Notification configuration."""

    urls: list[str] = Field(
        default_factory=list,
        description="List of Apprise notification URLs",
    )
    webhook_urls: list[str] = Field(
        default_factory=list,
        description="List of webhook URLs for JSON POST notifications",
    )


class Settings(BaseSettings):
    """Application settings loaded from YAML with env var overrides.

    Priority (highest to lowest):
    1. Environment variables (TOPIC_WATCH_LLM__API_KEY=...)
    2. YAML config file (data/config.yml)
    3. Field defaults
    """

    model_config = SettingsConfigDict(
        env_prefix="TOPIC_WATCH_",
        env_nested_delimiter="__",
        # Forward-compat: a stale/renamed top-level YAML key must not crash startup
        # (OVH-004). Unknown keys are dropped; the model_validator below logs a warning.
        extra="ignore",
    )

    llm: LLMSettings = LLMSettings()
    notifications: NotificationSettings = NotificationSettings()
    check_interval: str = Field(default="6h", description="Default check interval, e.g. '6h', '1d', '2w', '1M'")

    @field_validator("check_interval")
    @classmethod
    def validate_check_interval(cls, v: str) -> str:
        from app.interval import parse_interval

        parse_interval(v)  # raises ValueError on bad input
        return v

    @property
    def check_interval_minutes(self) -> int:
        from app.interval import parse_interval

        return parse_interval(self.check_interval)

    max_articles_per_check: int = Field(default=10, ge=1, le=100)
    knowledge_state_max_tokens: int = Field(default=2000, ge=500, le=10000)
    article_retention_days: int = Field(default=90, ge=1, le=3650)
    db_path: str = Field(
        default="data/topic_watch.db",
        description="Path to the SQLite database file (relative to project root or absolute)",
    )
    feed_fetch_timeout: float = Field(default=15.0, gt=0, description="Timeout in seconds for RSS feed fetches")
    article_fetch_timeout: float = Field(
        default=20.0, gt=0, description="Timeout in seconds for article content fetches"
    )
    llm_analysis_timeout: int = Field(default=60, gt=0, description="Timeout in seconds for LLM novelty analysis")
    llm_knowledge_timeout: int = Field(default=120, gt=0, description="Timeout in seconds for LLM knowledge generation")
    apprise_timeout_seconds: int = Field(
        default=30, gt=0, description="Timeout in seconds for a single Apprise notification send"
    )
    web_page_size: int = Field(default=20, ge=5, le=200, description="Number of items per page in web UI")
    feed_max_retries: int = Field(default=2, ge=1, le=10, description="Maximum retry attempts for feed fetching")
    feed_backoff_base_minutes: int = Field(
        default=15, ge=1, le=1440, description="Base delay (minutes) for backing off a persistently-failing feed"
    )
    feed_backoff_cap_hours: int = Field(
        default=24, ge=1, le=168, description="Maximum backoff delay (hours) for a persistently-failing feed"
    )
    content_fetch_concurrency: int = Field(default=3, ge=1, le=20, description="Max concurrent article content fetches")
    topic_check_concurrency: int = Field(
        default=3, ge=1, le=20, description="Max concurrent per-topic checks within one scheduler tick"
    )
    scheduler_misfire_grace_time: int = Field(
        default=300, ge=30, le=3600, description="APScheduler misfire grace time in seconds"
    )
    scheduler_jitter_seconds: int = Field(
        default=30,
        ge=0,
        le=120,
        description="Random jitter in seconds added to each scheduler tick to prevent thundering herd",
    )
    llm_max_retries: int = Field(default=2, ge=0, le=10, description="Maximum retries for LLM API calls")
    llm_temperature: float = Field(
        default=0.2, ge=0.0, le=2.0, description="LLM sampling temperature (lower = more factual)"
    )
    min_confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum LLM confidence to act on novelty results",
    )
    min_relevance_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score to act on novelty results (how related to topic description)",
    )
    secure_cookies: bool = Field(
        default=False,
        description="Set Secure flag on cookies (enable when TLS is terminated at reverse proxy)",
    )

    def is_configured(self) -> bool:
        """Return True if minimal required configuration is present."""
        return bool(self.llm.model and self.llm.api_key and self.llm.api_key != "your-api-key-here")

    @model_validator(mode="before")
    @classmethod
    def migrate_check_interval_hours(cls, data: dict) -> dict:  # type: ignore[override]
        """Backward compat: convert old check_interval_hours to check_interval string.

        Also warns about any remaining unrecognized top-level keys, which extra='ignore'
        silently drops (OVH-004). Migration runs first so renamed-but-handled keys
        (check_interval_hours) do not produce a spurious warning.
        """
        if isinstance(data, dict):
            if "check_interval_hours" in data:
                hours = data.pop("check_interval_hours")
                if "check_interval" not in data and hours is not None:
                    data["check_interval"] = f"{int(hours)}h"
            known = set(cls.model_fields)
            for key in data:
                if key not in known:
                    logger.warning("Unknown config key '%s' ignored (renamed or removed?)", key)
        return data

    @model_validator(mode="after")
    def validate_llm_model_format(self) -> Self:
        """Warn about common model string mistakes."""
        model_str = self.llm.model
        if not model_str:
            return self
        known_providers = CLOUD_PROVIDERS | frozenset(LOCAL_PROVIDER_DEFAULTS)
        if "/" in model_str:
            # LiteLLM is case-insensitive, so a capitalized but valid provider is fine (OVH-105).
            provider = model_str.split("/")[0]
            if provider.lower() not in known_providers:
                close = [p for p in sorted(known_providers) if _is_close(provider, p)]
                if close:
                    logger.warning(
                        "Unknown LLM provider '%s'. Did you mean '%s'? Model string: '%s'",
                        provider,
                        close[0],
                        model_str,
                    )
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Configure settings source priority: init > env > YAML."""
        yaml_file = _yaml_file_override or str(DEFAULT_CONFIG_PATH)
        return (
            init_settings,
            env_settings,
            YamlConfigSettingsSource(settings_cls, yaml_file=yaml_file),
        )


def _is_close(a: str, b: str) -> bool:
    """Check if two strings are likely typos of each other (case-insensitive, OVH-105)."""
    a, b = a.lower(), b.lower()
    if abs(len(a) - len(b)) > 2:
        return False
    return bool(a and b and a[0] == b[0] and abs(len(a) - len(b)) <= 1)


def resolve_db_path(settings: Settings) -> Path:
    """Resolve the database path from settings (relative to PROJECT_ROOT or absolute)."""
    p = Path(settings.db_path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def load_settings(config_path: Path | None = None) -> Settings:
    """Load and validate application settings.

    Args:
        config_path: Optional override for the config file path.
                     If None, uses data/config.yml.

    Returns:
        Validated Settings instance.

    Returns:
        Validated Settings instance (may be unconfigured — check is_configured()).
    """
    global _yaml_file_override

    effective_path = config_path or DEFAULT_CONFIG_PATH

    if not effective_path.exists():
        example_path = PROJECT_ROOT / "config.example.yml"
        if example_path.exists():
            effective_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(example_path, effective_path)
            logger.info("First run detected — created config file: %s", effective_path)
        else:
            effective_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("No config file found — starting with defaults (setup required)")
            # Return unconfigured settings so the setup wizard can handle it
            return Settings()  # type: ignore[call-arg]

    _yaml_file_override = str(effective_path) if config_path else None

    try:
        settings = Settings()  # type: ignore[call-arg]
        logger.info("Configuration loaded successfully from %s", effective_path)
        return settings
    finally:
        _yaml_file_override = None


def save_settings_to_yaml(settings: "Settings", config_path: Path, preserve_api_key: bool = False) -> None:
    """Write current settings back to the YAML config file.

    Args:
        settings: The settings to persist.
        config_path: Destination YAML path.
        preserve_api_key: When True, keep whatever api_key is already on disk instead
            of writing ``settings.llm.api_key``. Used when the key is env-sourced so the
            env secret is not materialized into plaintext config.yml (OVH-003).
    """
    effective_path = config_path

    api_key_to_write = settings.llm.api_key
    if preserve_api_key:
        # Read the existing on-disk value so an env-derived secret never lands here.
        existing_key = ""
        if effective_path.exists():
            try:
                existing = yaml.safe_load(effective_path.read_text()) or {}
                existing_key = (existing.get("llm") or {}).get("api_key", "") or ""
            except (OSError, yaml.YAMLError):
                logger.warning("Could not read existing config to preserve api_key; leaving it blank")
        api_key_to_write = existing_key

    data: dict = {
        "llm": {
            "model": settings.llm.model,
            "api_key": api_key_to_write,
        },
        "notifications": {},
        "check_interval": settings.check_interval,
        "max_articles_per_check": settings.max_articles_per_check,
        "knowledge_state_max_tokens": settings.knowledge_state_max_tokens,
        "article_retention_days": settings.article_retention_days,
        "db_path": settings.db_path,
        "feed_fetch_timeout": settings.feed_fetch_timeout,
        "article_fetch_timeout": settings.article_fetch_timeout,
        "llm_analysis_timeout": settings.llm_analysis_timeout,
        "llm_knowledge_timeout": settings.llm_knowledge_timeout,
        "apprise_timeout_seconds": settings.apprise_timeout_seconds,
        "web_page_size": settings.web_page_size,
        "feed_max_retries": settings.feed_max_retries,
        "content_fetch_concurrency": settings.content_fetch_concurrency,
        "topic_check_concurrency": settings.topic_check_concurrency,
        "scheduler_misfire_grace_time": settings.scheduler_misfire_grace_time,
        "scheduler_jitter_seconds": settings.scheduler_jitter_seconds,
        "llm_max_retries": settings.llm_max_retries,
        "llm_temperature": settings.llm_temperature,
        "min_confidence_threshold": settings.min_confidence_threshold,
        "min_relevance_threshold": settings.min_relevance_threshold,
        "secure_cookies": settings.secure_cookies,
    }

    # base_url is written whenever set (honored for every provider); omitted when unset.
    if settings.llm.base_url:
        data["llm"]["base_url"] = settings.llm.base_url

    if settings.notifications.urls:
        data["notifications"]["urls"] = settings.notifications.urls
    if settings.notifications.webhook_urls:
        data["notifications"]["webhook_urls"] = settings.notifications.webhook_urls

    # Ensure the parent directory exists (first-run / fresh data/ dir).
    effective_path.parent.mkdir(parents=True, exist_ok=True)

    with open(effective_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    logger.info("Settings saved to %s", effective_path)
