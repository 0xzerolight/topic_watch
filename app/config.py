"""Configuration management for Topic Watch.

Loads settings from data/config.yml with environment variable overrides.
Environment variables use the prefix TOPIC_WATCH_ with double underscore
for nested keys (e.g., TOPIC_WATCH_LLM__API_KEY).
"""

import logging
import shutil
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, Field, model_validator
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

# Providers that use their own cloud endpoints — base_url should never be set for these.
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
    }
)

# Default base URLs for self-hosted providers (used as form auto-fill hints).
LOCAL_PROVIDER_DEFAULTS: dict[str, str] = {
    "ollama": "http://localhost:11434",
}


def extract_provider(model_str: str) -> str | None:
    """Extract the provider prefix from a LiteLLM model string (e.g. 'openai/gpt-4' → 'openai')."""
    if "/" in model_str:
        return model_str.split("/", 1)[0].lower().strip()
    return None


def is_cloud_provider(model_str: str) -> bool:
    """Return True if the model string's provider prefix is a known cloud provider."""
    provider = extract_provider(model_str)
    return provider in CLOUD_PROVIDERS if provider else False


class LLMSettings(BaseModel):
    """LLM provider configuration."""

    model: str = Field(default="", description="LiteLLM model string, e.g. 'openai/gpt-4o-mini'")
    api_key: str = Field(default="", description="API key for the LLM provider")
    base_url: str | None = Field(
        default=None,
        description="Optional base URL for self-hosted providers like Ollama",
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
    )

    llm: LLMSettings = LLMSettings()
    notifications: NotificationSettings = NotificationSettings()
    check_interval_hours: int = Field(default=6, ge=1, le=168)
    max_articles_per_check: int = Field(default=10, ge=1, le=100)
    knowledge_state_max_tokens: int = Field(default=2000, ge=500, le=10000)
    article_retention_days: int = Field(default=90, ge=1, le=3650)
    db_path: str = Field(
        default="data/topic_watch.db",
        description="Path to the SQLite database file (relative to project root or absolute)",
    )
    feed_fetch_timeout: float = Field(default=15.0, description="Timeout in seconds for RSS feed fetches")
    article_fetch_timeout: float = Field(default=20.0, description="Timeout in seconds for article content fetches")
    llm_analysis_timeout: int = Field(default=60, description="Timeout in seconds for LLM novelty analysis")
    llm_knowledge_timeout: int = Field(default=120, description="Timeout in seconds for LLM knowledge generation")
    web_page_size: int = Field(default=20, ge=5, le=200, description="Number of items per page in web UI")
    feed_max_retries: int = Field(default=2, ge=1, le=10, description="Maximum retry attempts for feed fetching")
    content_fetch_concurrency: int = Field(default=3, ge=1, le=20, description="Max concurrent article content fetches")
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
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum LLM confidence to act on novelty results",
    )

    def is_configured(self) -> bool:
        """Return True if minimal required configuration is present."""
        return bool(self.llm.model and self.llm.api_key and self.llm.api_key != "your-api-key-here")

    @model_validator(mode="after")
    def validate_llm_model_format(self) -> Self:
        """Warn about common model string mistakes."""
        model_str = self.llm.model
        if not model_str:
            return self
        known_providers = CLOUD_PROVIDERS | frozenset(LOCAL_PROVIDER_DEFAULTS)
        if "/" in model_str:
            provider = model_str.split("/")[0]
            if provider not in known_providers:
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
    """Check if two strings are likely typos of each other."""
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


def save_settings_to_yaml(settings: "Settings", config_path: Path) -> None:
    """Write current settings back to the YAML config file."""
    effective_path = config_path

    data: dict = {
        "llm": {
            "model": settings.llm.model,
            "api_key": settings.llm.api_key,
        },
        "notifications": {},
        "check_interval_hours": settings.check_interval_hours,
        "max_articles_per_check": settings.max_articles_per_check,
        "knowledge_state_max_tokens": settings.knowledge_state_max_tokens,
        "article_retention_days": settings.article_retention_days,
        "db_path": settings.db_path,
        "feed_fetch_timeout": settings.feed_fetch_timeout,
        "article_fetch_timeout": settings.article_fetch_timeout,
        "llm_analysis_timeout": settings.llm_analysis_timeout,
        "llm_knowledge_timeout": settings.llm_knowledge_timeout,
        "web_page_size": settings.web_page_size,
        "feed_max_retries": settings.feed_max_retries,
        "content_fetch_concurrency": settings.content_fetch_concurrency,
        "scheduler_misfire_grace_time": settings.scheduler_misfire_grace_time,
        "scheduler_jitter_seconds": settings.scheduler_jitter_seconds,
        "llm_max_retries": settings.llm_max_retries,
        "llm_temperature": settings.llm_temperature,
        "min_confidence_threshold": settings.min_confidence_threshold,
    }

    if settings.llm.base_url and not is_cloud_provider(settings.llm.model):
        data["llm"]["base_url"] = settings.llm.base_url

    if settings.notifications.urls:
        data["notifications"]["urls"] = settings.notifications.urls
    if settings.notifications.webhook_urls:
        data["notifications"]["webhook_urls"] = settings.notifications.webhook_urls

    with open(effective_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    logger.info("Settings saved to %s", effective_path)
