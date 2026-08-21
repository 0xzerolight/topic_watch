"""Configuration management for Topic Watch.

Loads settings from data/config.yml with environment variable overrides.
Environment variables use the prefix TOPIC_WATCH_ with double underscore
for nested keys (e.g., TOPIC_WATCH_LLM__API_KEY).
"""

import hashlib
import logging
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any, Self

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

# Env var pinning the config file explicitly. Authoritative wherever it is set —
# it is the escape hatch for an installed wheel, a read-only package root, or a
# packaged service that wants its own layout (TW-AUD-029).
CONFIG_PATH_ENV_VAR = "TOPIC_WATCH_CONFIG_PATH"

# Directory name used under the user-level state root.
_APP_STATE_DIR_NAME = "topic-watch"

# Files whose presence marks a directory as a state root this install already uses.
# The config file is written on the first run and the database sits beside it, so
# either one means state already lives here.
_STATE_MARKERS = ("config.yml", "topic_watch.db")

# Directory names an interpreter installs packages into. A wheel install puts the
# package there, and it is nobody's application directory: what is written beside it
# dies with the venv, and the directory is shared with every dependency.
_LIBRARY_DIR_NAMES = frozenset({"site-packages", "dist-packages"})


def _user_state_root(environ: Mapping[str, str]) -> Path:
    """Per-user writable state directory, by platform convention."""
    if os.name == "nt":
        base = environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    else:
        base = environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(base).expanduser() / _APP_STATE_DIR_NAME


def _holds_state(directory: Path) -> bool:
    """True when a directory already holds this application's config or database."""
    return any((directory / marker).exists() for marker in _STATE_MARKERS)


def resolve_state_root(*, package_parent: Path, environ: Mapping[str, str]) -> Path:
    """THE writable-state root: the directory holding config.yml and the database.

    One helper so the config root and the database root can never diverge
    (TW-AUD-029). Precedence, highest first:

    1. ``TOPIC_WATCH_CONFIG_PATH`` — an explicitly pinned config file pins the root
       to its parent directory.
    2. Whichever candidate already holds a ``config.yml`` or a ``topic_watch.db``,
       ``data/`` beside the package first. This is what makes the root sticky: an
       install that has ever written state keeps reading that state, and a ``data/``
       directory appearing later — ``docker compose`` creating the bind-mount source,
       the documented ``mkdir -p data`` — cannot take over from a populated user-level
       directory and hand the user an empty database (C5-1).
    3. An existing ``data/`` beside the package: a repo checkout, a git worktree with
       the config copied in, the container's ``/app/data`` bind mount.
    4. ``data/`` beside the package when that directory is writable and is not an
       interpreter library directory. A source checkout is what the documentation
       describes, so a fresh clone must land there rather than in a hidden per-user
       directory; the first write creates it.
    5. A user-level state directory: an installed wheel, or a package root that
       cannot be written to.

    ``TOPIC_WATCH_DB_PATH`` (the ``db_path`` setting) stays authoritative for the
    database on top of this — see ``resolve_db_path``.
    """
    pinned = environ.get(CONFIG_PATH_ENV_VAR)
    if pinned:
        return Path(pinned).expanduser().parent
    beside_package = package_parent / "data"
    user_root = _user_state_root(environ)
    for candidate in (beside_package, user_root):
        if _holds_state(candidate):
            return candidate
    if package_parent.name in _LIBRARY_DIR_NAMES:
        return user_root
    if beside_package.is_dir() or os.access(package_parent, os.W_OK):
        return beside_package
    return user_root


def resolve_config_file(*, package_parent: Path, environ: Mapping[str, str]) -> Path:
    """Path of the config file, resolved through the one state root."""
    pinned = environ.get(CONFIG_PATH_ENV_VAR)
    if pinned:
        return Path(pinned).expanduser()
    return resolve_state_root(package_parent=package_parent, environ=environ) / "config.yml"


STATE_ROOT = resolve_state_root(package_parent=PROJECT_ROOT, environ=os.environ)
DATA_DIR = STATE_ROOT
DEFAULT_CONFIG_PATH = resolve_config_file(package_parent=PROJECT_ROOT, environ=os.environ)
DEFAULT_DB_PATH = STATE_ROOT / "topic_watch.db"

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


def is_keyless_llm_provider(model: str) -> bool:
    """True for a self-hosted provider that authenticates nothing.

    ``LOCAL_PROVIDER_DEFAULTS`` is the registry of providers running on the user's own
    machine, so there is no credential to supply and none should be demanded (AUG-107).
    """
    provider = model.split("/", 1)[0].strip().lower() if "/" in model else ""
    return provider in LOCAL_PROVIDER_DEFAULTS


# Env var that supplies the LLM API key (env > YAML; see settings_customise_sources).
_API_KEY_ENV_VAR = "TOPIC_WATCH_LLM__API_KEY"
# Env var that supplies the Exa API key (same env > YAML precedence).
_EXA_API_KEY_ENV_VAR = "TOPIC_WATCH_EXA__API_KEY"

_ENV_PREFIX = "TOPIC_WATCH_"
_ENV_NESTED_DELIMITER = "__"


def _env_field_path(variable: str) -> tuple[str, ...] | None:
    """Settings field path a ``TOPIC_WATCH_*`` variable addresses, or None.

    Single underscores are part of a field name; the double underscore is the
    nesting delimiter, matching ``SettingsConfigDict``. A variable that maps to no
    declared field (``TOPIC_WATCH_CONFIG_PATH``, a typo) yields None.
    """
    if not variable.upper().startswith(_ENV_PREFIX):
        return None
    path = tuple(variable[len(_ENV_PREFIX) :].lower().split(_ENV_NESTED_DELIMITER))
    model: type[BaseModel] = Settings
    for index, part in enumerate(path):
        if part not in model.model_fields:
            return None
        if index == len(path) - 1:
            return path
        nested = _nested_model(model, part)
        if nested is None:
            return None
        model = nested
    return None


def env_owned_field_paths(environ: Mapping[str, str] | None = None) -> frozenset[tuple[str, ...]]:
    """Field paths the environment currently owns.

    Ownership is decided by PRESENCE, not truthiness (AUG-241): pydantic-settings
    applies ``TOPIC_WATCH_LLM__API_KEY=`` as an empty string that outranks YAML, so
    the variable being set is what makes the field environment-owned. Every writer
    and every UI control keys off this one function, so no field is handled by a
    bespoke rule that can drift.
    """
    source = os.environ if environ is None else environ
    owned = {path for name in source if (path := _env_field_path(name)) is not None}
    return frozenset(owned)


def strip_env_owned(data: dict[str, Any], environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Drop environment-owned paths from submitted settings data.

    What is left is what the user may actually change. Removing a path from the init
    data is what lets the environment source supply it again (init > env > YAML), so
    an edit to an env-owned control is a visible no-op instead of a value that works
    until the next restart and then silently reverts (AUG-241).
    """
    owned = env_owned_field_paths(environ)

    def prune(section: dict[str, Any], prefix: tuple[str, ...]) -> dict[str, Any]:
        kept: dict[str, Any] = {}
        for key, value in section.items():
            path = (*prefix, key)
            if path in owned:
                continue
            kept[key] = prune(value, path) if isinstance(value, dict) else value
        return kept

    return prune(data, ())


def is_api_key_env_sourced() -> bool:
    """Return True if the LLM API key is supplied via environment (env > YAML).

    When True, the settings UI must treat the key as read-only and the save path must
    NOT materialize the env-derived secret into plaintext config.yml (OVH-003).
    """
    return ("llm", "api_key") in env_owned_field_paths()


def is_exa_key_env_sourced() -> bool:
    """Return True if the Exa API key is supplied via environment (env > YAML).

    Same read-only-UI and no-materialize-secret contract as the LLM key (OVH-003).
    """
    return ("exa", "api_key") in env_owned_field_paths()


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


class ExaSettings(BaseModel):
    """Exa AI search-source configuration (optional, opt-in, user-supplied paid key)."""

    enabled: bool = Field(default=False, description="Enable the Exa AI search source for EXA-mode topics")
    api_key: str = Field(default="", description="Exa API key (https://exa.ai)")
    base_url: str | None = Field(
        default=None,
        description="Optional Exa API base URL override (advanced/proxy; defaults to https://api.exa.ai)",
    )

    @model_validator(mode="after")
    def enabled_requires_a_key(self) -> Self:
        """``enabled`` means usable: without a key the source can never fetch.

        Everything downstream reads ``enabled`` alone as availability, so a keyless
        enabled state produced EXA topics that fail their first initialization with a
        generic "no articles found" (AUG-099). Held here rather than raised, so a
        hand-edited or half-migrated config still starts; the settings form rejects
        the same combination up front with a message the user can act on.
        """
        if self.enabled and not self.api_key.strip():
            logger.warning("Exa is enabled but no API key is configured — disabling the Exa source")
            self.enabled = False
        return self


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
    exa: ExaSettings = ExaSettings()
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
    knowledge_revision_limit: int = Field(
        default=50,
        ge=2,
        le=200,
        description="Knowledge revisions retained per topic for the diff timeline; older ones are pruned",
    )
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
    silence_heartbeat_checks: int = Field(
        default=3,
        ge=0,
        le=50,
        description="Consecutive checks with no usable source before a Silence Heartbeat alert (0 disables)",
    )
    secure_cookies: bool = Field(
        default=False,
        description="Set Secure flag on cookies (enable when TLS is terminated at reverse proxy)",
    )

    def is_configured(self) -> bool:
        """Return True if minimal required configuration is present.

        Key requiredness is provider-aware: a self-hosted provider authenticates
        nothing, and the documented keyless Ollama path was unreachable while an
        empty key counted as unconfigured (AUG-107).
        """
        if not self.llm.model:
            return False
        if is_keyless_llm_provider(self.llm.model):
            return True
        return bool(self.llm.api_key and self.llm.api_key != "your-api-key-here")

    @model_validator(mode="before")
    @classmethod
    def migrate_check_interval_hours(cls, data: dict) -> dict:  # type: ignore[override]
        """Backward compat: convert old check_interval_hours to check_interval string.

        Also warns about any remaining unrecognized key, at every level, which
        extra='ignore' silently drops (OVH-004, TW-AUD-028). Migration runs first so
        renamed-but-handled keys (check_interval_hours) do not produce a spurious
        warning. The keys are only ignored for this object — ``save_settings_to_yaml``
        keeps them in the file, so a forward-compatible config is never erased.
        """
        if isinstance(data, dict):
            if "check_interval_hours" in data:
                hours = data.pop("check_interval_hours")
                if "check_interval" not in data and hours is not None:
                    data["check_interval"] = f"{int(hours)}h"
            for path in _unknown_key_paths(data, cls):
                logger.warning("Unknown config key '%s' ignored (renamed or removed?)", path)
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


def _nested_model(model: type[BaseModel], field: str) -> type[BaseModel] | None:
    """The submodel type behind ``field``, or None when the field is a scalar."""
    info = model.model_fields.get(field)
    annotation = info.annotation if info is not None else None
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    return None


def _unknown_key_paths(data: Mapping[str, Any], model: type[BaseModel], prefix: str = "") -> list[str]:
    """Dotted paths in ``data`` that ``model`` does not declare, at any depth."""
    unknown: list[str] = []
    for key, value in data.items():
        path = f"{prefix}{key}"
        if key not in model.model_fields:
            unknown.append(path)
            continue
        nested = _nested_model(model, key)
        if nested is not None and isinstance(value, Mapping):
            unknown.extend(_unknown_key_paths(value, nested, prefix=f"{path}."))
    return unknown


def _is_close(a: str, b: str) -> bool:
    """Check if two strings are likely typos of each other (case-insensitive, OVH-105)."""
    a, b = a.lower(), b.lower()
    if abs(len(a) - len(b)) > 2:
        return False
    return bool(a and b and a[0] == b[0] and abs(len(a) - len(b)) <= 1)


def resolve_db_path(settings: Settings) -> Path:
    """Resolve the database path from settings, through the one state root.

    An absolute ``db_path`` (``TOPIC_WATCH_DB_PATH=/data/topic_watch.db``, the
    container's own setting) is authoritative and used verbatim. A relative one
    keeps resolving against the package parent while the legacy ``data/``
    directory is the state root, so repo checkouts and every existing install are
    byte-for-byte unchanged. Only when state lives elsewhere is the historical
    leading ``data/`` component re-pointed at that root (TW-AUD-029).
    """
    p = Path(settings.db_path).expanduser()
    if p.is_absolute():
        return p
    if STATE_ROOT == PROJECT_ROOT / "data":
        return PROJECT_ROOT / p
    parts = p.parts
    if len(parts) > 1 and parts[0] == "data":
        p = Path(*parts[1:])
    return STATE_ROOT / p


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


def read_config_document(path: Path) -> dict[str, Any]:
    """Return the config file's current top-level mapping, or {} when there is none.

    A missing file, unreadable file, corrupt YAML or non-mapping document all yield
    an empty document rather than raising: a save must never be blocked by whatever
    was there before, it just cannot preserve anything from it.
    """
    if not path.exists():
        return {}
    try:
        loaded = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError):
        logger.warning("Could not read %s; keys it holds that this version does not know will not be preserved", path)
        return {}
    if not isinstance(loaded, dict):
        logger.warning("%s does not contain a YAML mapping; replacing it with the current settings", path)
        return {}
    return loaded


def config_revision(path: Path) -> str:
    """Opaque marker of the config file's current content.

    Rendered into the settings form so a save can tell whether the file changed
    underneath it — an external edit, another tab, or a key rotated on disk
    (AUG-291). A missing file has its own stable marker.
    """
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return "absent"


# Config files carry provider and delivery credentials, so they are owner-only.
_CONFIG_FILE_MODE = 0o600


def _target_mode(path: Path) -> int:
    """Permissions for the written file: 0600 when new, tightened when it exists.

    First-run copies the tracked 0644 example, so an existing permissive mode is
    corrected rather than carried forward (AUG-198). Owner bits the user widened
    deliberately are kept; group and other are always cleared.
    """
    try:
        return (path.stat().st_mode & 0o777) & ~0o077 or _CONFIG_FILE_MODE
    except OSError:
        return _CONFIG_FILE_MODE


def _fsync_directory(directory: Path) -> None:
    """Persist a rename in the directory entry, where the platform supports it."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return  # Windows and some network filesystems: the rename is all we get.
    try:
        os.fsync(fd)
    except OSError:
        logger.debug("Could not fsync %s after replacing the config file", directory)
    finally:
        os.close(fd)


def _atomic_write(path: Path, render: Callable[[Any], None]) -> None:
    """Write ``path`` via a same-directory temp file and one atomic rename.

    ``open(path, "w")`` truncated the last valid configuration before serialization
    even started, so an error or an interrupt destroyed it (AUG-198). Here the
    destination is only touched by ``os.replace``, which is atomic: a reader sees
    either the old file or the new one, never a half-written one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    temp_path = Path(temp_name)
    try:
        # mkstemp already creates 0600; widen/keep only what the destination needs.
        with suppress(OSError, NotImplementedError):
            os.chmod(temp_path, _target_mode(path))
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            render(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    _fsync_directory(path.parent)


def _merge_config_document(
    existing: dict[str, Any],
    desired: Mapping[str, Any],
    *,
    skip: set[tuple[str, ...]],
    prefix: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Patch ``desired`` onto ``existing``, leaving everything else in place.

    Three rules, and they are the whole codec (TW-AUD-028):

    - a path in ``skip`` is not written at all — whatever the file already had for
      it stays, which is how an environment-owned value never gets materialized;
    - a ``None`` value deletes its key, so a cleared optional setting does not come
      back from the old document on the next load;
    - anything the model does not declare is left untouched, so a forward-compatible
      or hand-added key survives a save instead of being silently dropped.
    """
    merged = dict(existing)
    for key, value in desired.items():
        path = (*prefix, key)
        if path in skip:
            continue
        if isinstance(value, Mapping):
            nested = merged.get(key)
            merged[key] = _merge_config_document(
                nested if isinstance(nested, dict) else {},
                value,
                skip=skip,
                prefix=path,
            )
        elif value is None:
            merged.pop(key, None)
        else:
            merged[key] = value
    return merged


def save_settings_to_yaml(settings: "Settings", config_path: Path) -> None:
    """Write current settings back to the YAML config file.

    Only values the YAML file owns are written. A field the environment supplies is
    left exactly as the file already had it, because a ``Settings`` object cannot
    say where each of its values came from — it carries the merged result, and
    serializing that copies environment-derived credentials, notification URLs and
    scalars into plaintext YAML (AUG-241). Provenance comes from
    ``env_owned_field_paths`` instead, which covers every field rather than the two
    API keys OVH-003 special-cased.

    Args:
        settings: The settings to persist.
        config_path: Destination YAML path.
    """
    skip = set(env_owned_field_paths())

    document = _merge_config_document(
        read_config_document(config_path),
        settings.model_dump(mode="json"),
        skip=skip,
    )

    _atomic_write(
        config_path,
        lambda handle: yaml.dump(document, handle, default_flow_style=False, sort_keys=False, allow_unicode=True),
    )
    logger.info("Settings saved to %s", config_path)
