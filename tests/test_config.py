"""Tests for configuration loading and validation."""

import logging
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from app.config import Settings, load_settings, save_settings_to_yaml


@pytest.fixture(autouse=True)
def _yaml_owned_llm_credentials(monkeypatch: pytest.MonkeyPatch):
    """Make the LLM block YAML-owned unless a test says otherwise.

    conftest (like CI) exports TOPIC_WATCH_LLM__MODEL / __API_KEY so the app counts
    as configured; that makes both fields environment-owned, and an environment-owned
    field is deliberately never written to the file (AUG-241). Tests about env
    precedence set the variables they need themselves.
    """
    monkeypatch.delenv("TOPIC_WATCH_LLM__MODEL", raising=False)
    monkeypatch.delenv("TOPIC_WATCH_LLM__API_KEY", raising=False)


class TestConfigLoading:
    """Test loading configuration from YAML files."""

    def test_load_valid_config(self, sample_config_yaml: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TOPIC_WATCH_LLM__API_KEY", raising=False)
        monkeypatch.delenv("TOPIC_WATCH_LLM__MODEL", raising=False)
        settings = load_settings(config_path=sample_config_yaml)
        assert settings.llm.model == "openai/gpt-4o-mini"
        assert settings.llm.api_key == "test-api-key-12345"
        assert settings.check_interval == "6h"
        assert settings.check_interval_minutes == 360
        assert settings.max_articles_per_check == 10
        assert settings.knowledge_state_max_tokens == 2000
        assert len(settings.notifications.urls) == 1

    def test_load_minimal_config(self, minimal_config_yaml: Path) -> None:
        settings = load_settings(config_path=minimal_config_yaml)
        assert settings.llm.model == "openai/gpt-4o-mini"
        assert settings.check_interval == "6h"
        assert settings.max_articles_per_check == 10
        assert settings.notifications.urls == []

    def test_missing_config_returns_unconfigured(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TOPIC_WATCH_LLM__API_KEY", raising=False)
        monkeypatch.delenv("TOPIC_WATCH_LLM__MODEL", raising=False)
        settings = load_settings(config_path=tmp_path / "nonexistent.yml")
        assert not settings.is_configured()

    def test_missing_llm_section_returns_unconfigured(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TOPIC_WATCH_LLM__API_KEY", raising=False)
        monkeypatch.delenv("TOPIC_WATCH_LLM__MODEL", raising=False)
        config = tmp_path / "config.yml"
        config.write_text('check_interval: "6h"\n')
        settings = load_settings(config_path=config)
        assert not settings.is_configured()

    def test_missing_api_key_returns_unconfigured(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TOPIC_WATCH_LLM__API_KEY", raising=False)
        monkeypatch.delenv("TOPIC_WATCH_LLM__MODEL", raising=False)
        config = tmp_path / "config.yml"
        config.write_text('llm:\n  model: "openai/gpt-4o-mini"\n')
        settings = load_settings(config_path=config)
        assert not settings.is_configured()
        assert settings.llm.model == "openai/gpt-4o-mini"

    def test_invalid_check_interval_too_low(self, tmp_path: Path) -> None:
        config = tmp_path / "config.yml"
        config.write_text('llm:\n  model: "openai/gpt-4o-mini"\n  api_key: "k"\ncheck_interval: "5m"\n')
        with pytest.raises(ValidationError):
            load_settings(config_path=config)

    def test_invalid_check_interval_bad_format(self, tmp_path: Path) -> None:
        config = tmp_path / "config.yml"
        config.write_text('llm:\n  model: "openai/gpt-4o-mini"\n  api_key: "k"\ncheck_interval: "bogus"\n')
        with pytest.raises(ValidationError):
            load_settings(config_path=config)

    def test_backward_compat_check_interval_hours(self, tmp_path: Path) -> None:
        """Old check_interval_hours YAML key is auto-converted to check_interval string."""
        config = tmp_path / "config.yml"
        config.write_text('llm:\n  model: "openai/gpt-4o-mini"\n  api_key: "k"\ncheck_interval_hours: 12\n')
        settings = load_settings(config_path=config)
        assert settings.check_interval == "12h"
        assert settings.check_interval_minutes == 720

    def test_invalid_max_articles(self, tmp_path: Path) -> None:
        config = tmp_path / "config.yml"
        config.write_text('llm:\n  model: "openai/gpt-4o-mini"\n  api_key: "k"\nmax_articles_per_check: 200\n')
        with pytest.raises(ValidationError):
            load_settings(config_path=config)

    def test_invalid_knowledge_tokens_too_low(self, tmp_path: Path) -> None:
        config = tmp_path / "config.yml"
        config.write_text('llm:\n  model: "openai/gpt-4o-mini"\n  api_key: "k"\nknowledge_state_max_tokens: 100\n')
        with pytest.raises(ValidationError):
            load_settings(config_path=config)

    def test_optional_base_url(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Clear the env model override so the YAML's ollama model is used (env > YAML).
        monkeypatch.delenv("TOPIC_WATCH_LLM__MODEL", raising=False)
        monkeypatch.delenv("TOPIC_WATCH_LLM__API_KEY", raising=False)
        config = tmp_path / "config.yml"
        config.write_text('llm:\n  model: "ollama/llama3"\n  api_key: "na"\n  base_url: "http://localhost:11434"\n')
        settings = load_settings(config_path=config)
        assert settings.llm.base_url == "http://localhost:11434"

    def test_base_url_defaults_to_none(self, minimal_config_yaml: Path) -> None:
        settings = load_settings(config_path=minimal_config_yaml)
        assert settings.llm.base_url is None


class TestIsConfigured:
    """Test the Settings.is_configured() method."""

    def test_configured_with_valid_key(self, sample_config_yaml: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TOPIC_WATCH_LLM__API_KEY", raising=False)
        monkeypatch.delenv("TOPIC_WATCH_LLM__MODEL", raising=False)
        settings = load_settings(config_path=sample_config_yaml)
        assert settings.is_configured()

    def test_unconfigured_with_empty_key(self) -> None:
        settings = Settings(llm={"model": "openai/gpt-4o-mini", "api_key": ""})  # type: ignore[call-arg]
        assert not settings.is_configured()

    def test_unconfigured_with_empty_model(self) -> None:
        settings = Settings(llm={"model": "", "api_key": "sk-real"})  # type: ignore[call-arg]
        assert not settings.is_configured()

    def test_unconfigured_with_placeholder_key(self) -> None:
        settings = Settings(llm={"model": "openai/gpt-4o-mini", "api_key": "your-api-key-here"})  # type: ignore[call-arg]
        assert not settings.is_configured()

    def test_unconfigured_with_defaults(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TOPIC_WATCH_LLM__API_KEY", raising=False)
        monkeypatch.delenv("TOPIC_WATCH_LLM__MODEL", raising=False)
        settings = load_settings(config_path=tmp_path / "nonexistent.yml")
        assert not settings.is_configured()


class TestEnvVarOverrides:
    """Test that environment variables override YAML values."""

    def test_env_overrides_api_key(self, sample_config_yaml: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TOPIC_WATCH_LLM__API_KEY", "env-override-key")
        settings = load_settings(config_path=sample_config_yaml)
        assert settings.llm.api_key == "env-override-key"

    def test_env_overrides_check_interval(self, sample_config_yaml: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TOPIC_WATCH_CHECK_INTERVAL", "12h")
        settings = load_settings(config_path=sample_config_yaml)
        assert settings.check_interval == "12h"
        assert settings.check_interval_minutes == 720


class TestSaveSettingsBaseUrl:
    """Test that save_settings_to_yaml persists base_url for every provider (OVH-104 reversal)."""

    def test_cloud_provider_base_url_written(self, tmp_path: Path) -> None:
        """base_url IS written to YAML for a cloud provider (OpenAI-compatible gateway)."""
        import yaml

        from app.config import LLMSettings, save_settings_to_yaml

        settings = Settings(
            llm=LLMSettings(model="openai/glm-5.2", api_key="sk-test", base_url="https://opencode.ai/zen/go/v1"),
        )  # type: ignore[call-arg]
        config_file = tmp_path / "config.yml"
        save_settings_to_yaml(settings, config_file)

        data = yaml.safe_load(config_file.read_text())
        assert data["llm"]["base_url"] == "https://opencode.ai/zen/go/v1"

    def test_local_provider_base_url_preserved(self, tmp_path: Path) -> None:
        """base_url IS written to YAML when model is a local provider."""
        import yaml

        from app.config import LLMSettings, save_settings_to_yaml

        settings = Settings(
            llm=LLMSettings(model="ollama/llama3", api_key="dummy", base_url="http://localhost:11434"),
        )  # type: ignore[call-arg]
        config_file = tmp_path / "config.yml"
        save_settings_to_yaml(settings, config_file)

        data = yaml.safe_load(config_file.read_text())
        assert data["llm"]["base_url"] == "http://localhost:11434"


class TestSaveSettingsCreatesParentDir:
    """save_settings_to_yaml must create a missing parent directory before writing."""

    def test_creates_missing_parent_dir(self, tmp_path: Path) -> None:
        """Saving to a path whose parent does not exist creates it and succeeds."""
        import yaml

        from app.config import LLMSettings, save_settings_to_yaml

        settings = Settings(
            llm=LLMSettings(model="openai/gpt-4o-mini", api_key="sk-test"),
        )  # type: ignore[call-arg]
        # Parent directory "data" does not exist yet.
        config_file = tmp_path / "data" / "config.yml"
        assert not config_file.parent.exists()

        save_settings_to_yaml(settings, config_file)

        assert config_file.exists()
        data = yaml.safe_load(config_file.read_text())
        assert data["llm"]["model"] == "openai/gpt-4o-mini"

    def test_creates_nested_missing_parents(self, tmp_path: Path) -> None:
        """Multiple levels of missing parents are created (parents=True)."""
        from app.config import LLMSettings, save_settings_to_yaml

        settings = Settings(
            llm=LLMSettings(model="openai/gpt-4o-mini", api_key="sk-test"),
        )  # type: ignore[call-arg]
        config_file = tmp_path / "a" / "b" / "c" / "config.yml"
        save_settings_to_yaml(settings, config_file)
        assert config_file.exists()


class TestThresholdDefaults:
    """Test default values and bounds for confidence/relevance thresholds."""

    def test_min_confidence_threshold_default(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TOPIC_WATCH_MIN_CONFIDENCE_THRESHOLD", raising=False)
        monkeypatch.delenv("TOPIC_WATCH_LLM__API_KEY", raising=False)
        monkeypatch.delenv("TOPIC_WATCH_LLM__MODEL", raising=False)
        config = tmp_path / "config.yml"
        config.write_text('llm:\n  model: "openai/gpt-4o-mini"\n  api_key: "sk-test"\n')
        settings = load_settings(config_path=config)
        assert settings.min_confidence_threshold == 0.7

    def test_min_relevance_threshold_default(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TOPIC_WATCH_MIN_RELEVANCE_THRESHOLD", raising=False)
        monkeypatch.delenv("TOPIC_WATCH_LLM__API_KEY", raising=False)
        monkeypatch.delenv("TOPIC_WATCH_LLM__MODEL", raising=False)
        config = tmp_path / "config.yml"
        config.write_text('llm:\n  model: "openai/gpt-4o-mini"\n  api_key: "sk-test"\n')
        settings = load_settings(config_path=config)
        assert settings.min_relevance_threshold == 0.5

    def test_min_relevance_threshold_bounds(self) -> None:
        with pytest.raises(ValidationError):
            Settings(
                llm={"model": "openai/gpt-4o-mini", "api_key": "sk-test"},
                min_relevance_threshold=1.5,
            )  # type: ignore[call-arg]
        with pytest.raises(ValidationError):
            Settings(
                llm={"model": "openai/gpt-4o-mini", "api_key": "sk-test"},
                min_relevance_threshold=-0.1,
            )  # type: ignore[call-arg]

    def test_relevance_threshold_in_saved_yaml(self, tmp_path: Path) -> None:
        import yaml

        from app.config import LLMSettings, save_settings_to_yaml

        settings = Settings(
            llm=LLMSettings(model="openai/gpt-4o-mini", api_key="sk-test"),
            min_relevance_threshold=0.6,
        )  # type: ignore[call-arg]
        config_file = tmp_path / "config.yml"
        save_settings_to_yaml(settings, config_file)

        data = yaml.safe_load(config_file.read_text())
        assert data["min_relevance_threshold"] == 0.6


class TestSilenceHeartbeatSetting:
    """Silence Heartbeat threshold: default, bounds, and YAML persistence."""

    def test_default_is_three(self) -> None:
        settings = Settings(llm={"model": "openai/gpt-4o-mini", "api_key": "sk-test"})  # type: ignore[call-arg]
        assert settings.silence_heartbeat_checks == 3

    def test_zero_allowed_negative_rejected(self) -> None:
        disabled = Settings(
            llm={"model": "openai/gpt-4o-mini", "api_key": "sk-test"},
            silence_heartbeat_checks=0,
        )  # type: ignore[call-arg]
        assert disabled.silence_heartbeat_checks == 0
        with pytest.raises(ValidationError):
            Settings(
                llm={"model": "openai/gpt-4o-mini", "api_key": "sk-test"},
                silence_heartbeat_checks=-1,
            )  # type: ignore[call-arg]

    def test_persisted_to_yaml(self, tmp_path: Path) -> None:
        import yaml

        from app.config import LLMSettings, save_settings_to_yaml

        settings = Settings(
            llm=LLMSettings(model="openai/gpt-4o-mini", api_key="sk-test"),
            silence_heartbeat_checks=5,
        )  # type: ignore[call-arg]
        config_file = tmp_path / "config.yml"
        save_settings_to_yaml(settings, config_file)

        data = yaml.safe_load(config_file.read_text())
        assert data["silence_heartbeat_checks"] == 5


class TestTimeoutValidation:
    """Timeouts must be strictly positive; zero/negative breaks every HTTP/LLM call."""

    @pytest.mark.parametrize(
        "field",
        [
            "feed_fetch_timeout",
            "article_fetch_timeout",
            "llm_analysis_timeout",
            "llm_knowledge_timeout",
            "apprise_timeout_seconds",
        ],
    )
    @pytest.mark.parametrize("value", [0, -1, -15.0])
    def test_non_positive_timeout_rejected(self, field: str, value: float) -> None:
        with pytest.raises(ValidationError):
            Settings(
                llm={"model": "openai/gpt-4o-mini", "api_key": "sk-test"},
                **{field: value},
            )  # type: ignore[call-arg]

    def test_positive_timeouts_accepted(self) -> None:
        settings = Settings(
            llm={"model": "openai/gpt-4o-mini", "api_key": "sk-test"},
            feed_fetch_timeout=5.0,
            article_fetch_timeout=10.0,
            llm_analysis_timeout=30,
            llm_knowledge_timeout=60,
            apprise_timeout_seconds=15,
        )  # type: ignore[call-arg]
        assert settings.feed_fetch_timeout == 5.0
        assert settings.apprise_timeout_seconds == 15

    def test_apprise_timeout_default(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TOPIC_WATCH_APPRISE_TIMEOUT_SECONDS", raising=False)
        config = tmp_path / "config.yml"
        config.write_text('llm:\n  model: "openai/gpt-4o-mini"\n  api_key: "sk-test"\n')
        settings = load_settings(config_path=config)
        assert settings.apprise_timeout_seconds == 30

    def test_apprise_timeout_in_saved_yaml(self, tmp_path: Path) -> None:
        import yaml

        from app.config import LLMSettings, save_settings_to_yaml

        settings = Settings(
            llm=LLMSettings(model="openai/gpt-4o-mini", api_key="sk-test"),
            apprise_timeout_seconds=45,
        )  # type: ignore[call-arg]
        config_file = tmp_path / "config.yml"
        save_settings_to_yaml(settings, config_file)

        data = yaml.safe_load(config_file.read_text())
        assert data["apprise_timeout_seconds"] == 45


class TestUnknownYamlKey:
    """OVH-004: an unknown top-level YAML key must not crash startup."""

    def test_unknown_top_level_key_does_not_crash(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A stale/renamed top-level key loads (is ignored) instead of raising."""
        monkeypatch.delenv("TOPIC_WATCH_LLM__API_KEY", raising=False)
        monkeypatch.delenv("TOPIC_WATCH_LLM__MODEL", raising=False)
        config = tmp_path / "config.yml"
        config.write_text('llm:\n  model: "openai/gpt-4o-mini"\n  api_key: "sk-test"\nremoved_legacy_setting: 42\n')
        # Must not raise ValidationError (extra_forbidden) — startup stays alive.
        settings = load_settings(config_path=config)
        assert settings.llm.model == "openai/gpt-4o-mini"
        assert settings.is_configured()

    def test_unknown_top_level_key_logs_warning(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Dropping an unknown key emits a warning naming the key."""
        monkeypatch.delenv("TOPIC_WATCH_LLM__API_KEY", raising=False)
        monkeypatch.delenv("TOPIC_WATCH_LLM__MODEL", raising=False)
        config = tmp_path / "config.yml"
        config.write_text('llm:\n  model: "openai/gpt-4o-mini"\n  api_key: "sk-test"\nremoved_legacy_setting: 42\n')
        with caplog.at_level(logging.WARNING, logger="app.config"):
            load_settings(config_path=config)
        assert any("removed_legacy_setting" in r.message for r in caplog.records)

    def test_known_keys_do_not_warn(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A config of only known keys produces no unknown-key warning."""
        monkeypatch.delenv("TOPIC_WATCH_LLM__API_KEY", raising=False)
        monkeypatch.delenv("TOPIC_WATCH_LLM__MODEL", raising=False)
        config = tmp_path / "config.yml"
        config.write_text('llm:\n  model: "openai/gpt-4o-mini"\n  api_key: "sk-test"\ncheck_interval: "6h"\n')
        with caplog.at_level(logging.WARNING, logger="app.config"):
            load_settings(config_path=config)
        assert not any("Unknown" in r.message and "config key" in r.message for r in caplog.records)


class TestBaseUrlModelHonored:
    """OVH-104 reversal: an explicitly-set base_url is honored for every provider."""

    def test_model_keeps_base_url_for_cloud_provider(self) -> None:
        """Constructing Settings with a cloud model + base_url keeps it (OpenAI-compatible gateway)."""
        settings = Settings(
            llm={"model": "openai/glm-5.2", "api_key": "sk", "base_url": "https://opencode.ai/zen/go/v1"},
        )  # type: ignore[call-arg]
        assert settings.llm.base_url == "https://opencode.ai/zen/go/v1"

    def test_model_keeps_base_url_for_local_provider(self) -> None:
        """A self-hosted provider keeps its base_url."""
        settings = Settings(
            llm={"model": "ollama/llama3", "api_key": "na", "base_url": "http://localhost:11434"},
        )  # type: ignore[call-arg]
        assert settings.llm.base_url == "http://localhost:11434"

    def test_base_url_round_trips_for_cloud_provider(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """load(save(x)) preserves base_url for a cloud model (symmetric, OpenAI-compatible gateway)."""
        from app.config import save_settings_to_yaml

        # Clear the env model override so the reloaded model is the init/YAML one, not CI's.
        monkeypatch.delenv("TOPIC_WATCH_LLM__MODEL", raising=False)
        monkeypatch.delenv("TOPIC_WATCH_LLM__API_KEY", raising=False)
        settings = Settings(
            llm={"model": "openai/glm-5.2", "api_key": "sk", "base_url": "https://opencode.ai/zen/go/v1"},
        )  # type: ignore[call-arg]
        config_file = tmp_path / "config.yml"
        save_settings_to_yaml(settings, config_file)
        reloaded = load_settings(config_path=config_file)
        assert reloaded.llm.base_url == "https://opencode.ai/zen/go/v1"

    def test_base_url_round_trips_for_local_provider(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """load(save(x)) preserves base_url for a local provider (symmetric)."""
        from app.config import save_settings_to_yaml

        monkeypatch.delenv("TOPIC_WATCH_LLM__MODEL", raising=False)
        monkeypatch.delenv("TOPIC_WATCH_LLM__API_KEY", raising=False)
        settings = Settings(
            llm={"model": "ollama/llama3", "api_key": "na", "base_url": "http://localhost:11434"},
        )  # type: ignore[call-arg]
        config_file = tmp_path / "config.yml"
        save_settings_to_yaml(settings, config_file)
        reloaded = load_settings(config_path=config_file)
        assert reloaded.llm.base_url == "http://localhost:11434"


class TestProviderTypoCaseInsensitive:
    """OVH-105: provider-typo suggestion must match case-insensitively."""

    def test_is_close_case_insensitive(self) -> None:
        """_is_close treats case-mismatched-but-identical strings as close."""
        from app.config import _is_close

        assert _is_close("OpenAI", "openai") is True
        assert _is_close("ANTHROPIC", "anthropic") is True

    def test_is_close_real_typo_still_matches(self) -> None:
        """A genuine typo of differing length still registers as close."""
        from app.config import _is_close

        assert _is_close("opena", "openai") is True

    def test_capitalized_known_provider_recognized_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """A case-mismatched valid provider ('OpenAI') is recognized — no typo warning."""
        with caplog.at_level(logging.WARNING, logger="app.config"):
            Settings(llm={"model": "OpenAI/gpt-4o", "api_key": "sk"})  # type: ignore[call-arg]
        assert not any("Did you mean" in r.message for r in caplog.records)

    def test_typo_provider_still_suggests(self, caplog: pytest.LogCaptureFixture) -> None:
        """A genuine typo ('opena') still produces a suggestion."""
        with caplog.at_level(logging.WARNING, logger="app.config"):
            Settings(llm={"model": "opena/gpt-4o", "api_key": "sk"})  # type: ignore[call-arg]
        assert any("Did you mean" in r.message for r in caplog.records)


class TestExaConfig:
    """Exa settings: defaults, YAML load, env override, env-sourced detection."""

    def test_exa_defaults_disabled_and_empty(self, minimal_config_yaml: Path) -> None:
        """A config with no exa block yields a disabled, keyless ExaSettings."""
        settings = load_settings(config_path=minimal_config_yaml)
        assert settings.exa.enabled is False
        assert settings.exa.api_key == ""
        assert settings.exa.base_url is None

    def test_exa_loaded_from_yaml(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TOPIC_WATCH_EXA__API_KEY", raising=False)
        config = tmp_path / "config.yml"
        config.write_text(
            'llm:\n  model: "openai/gpt-4o-mini"\n  api_key: "sk"\nexa:\n  enabled: true\n  api_key: "exa-yaml-key"\n'
        )
        settings = load_settings(config_path=config)
        assert settings.exa.enabled is True
        assert settings.exa.api_key == "exa-yaml-key"

    def test_env_overrides_exa_api_key(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TOPIC_WATCH_EXA__API_KEY", "exa-env-key")
        config = tmp_path / "config.yml"
        config.write_text(
            'llm:\n  model: "openai/gpt-4o-mini"\n  api_key: "sk"\nexa:\n  enabled: true\n  api_key: "exa-yaml-key"\n'
        )
        settings = load_settings(config_path=config)
        assert settings.exa.api_key == "exa-env-key"

    def test_is_exa_key_env_sourced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from app.config import is_exa_key_env_sourced

        monkeypatch.delenv("TOPIC_WATCH_EXA__API_KEY", raising=False)
        assert is_exa_key_env_sourced() is False
        monkeypatch.setenv("TOPIC_WATCH_EXA__API_KEY", "exa-env-key")
        assert is_exa_key_env_sourced() is True


class TestKnowledgeRevisionLimit:
    def test_default(self) -> None:
        settings = Settings(llm={"model": "openai/gpt-4o-mini", "api_key": "k" * 12})
        assert settings.knowledge_revision_limit == 50

    def test_rejects_below_floor(self) -> None:
        with pytest.raises(ValidationError):
            Settings(llm={"model": "openai/gpt-4o-mini", "api_key": "k" * 12}, knowledge_revision_limit=1)

    def test_rejects_above_ceiling(self) -> None:
        with pytest.raises(ValidationError):
            Settings(llm={"model": "openai/gpt-4o-mini", "api_key": "k" * 12}, knowledge_revision_limit=201)

    def test_written_to_yaml(self, tmp_path: Path) -> None:
        """A config-only field must still be persisted by a settings save."""
        path = tmp_path / "config.yml"
        settings = Settings(
            llm={"model": "openai/gpt-4o-mini", "api_key": "k" * 12},
            knowledge_revision_limit=7,
        )
        save_settings_to_yaml(settings, path)
        assert yaml.safe_load(path.read_text())["knowledge_revision_limit"] == 7


class TestStateRoot:
    """TW-AUD-029: one helper resolves the writable state root for config AND database."""

    def test_existing_data_dir_wins(self, tmp_path: Path) -> None:
        """A repo checkout / worktree / container bind mount keeps its own data/ dir."""
        from app.config import resolve_config_file, resolve_state_root

        (tmp_path / "data").mkdir()
        assert resolve_state_root(package_parent=tmp_path, environ={}) == tmp_path / "data"
        assert resolve_config_file(package_parent=tmp_path, environ={}) == tmp_path / "data" / "config.yml"

    def test_user_state_root_when_no_data_dir(self, tmp_path: Path) -> None:
        """A wheel install with no data/ beside the package uses a user-level root."""
        from app.config import resolve_config_file, resolve_state_root

        environ = {"XDG_STATE_HOME": str(tmp_path / "state")}
        root = resolve_state_root(package_parent=tmp_path, environ=environ)
        assert root == tmp_path / "state" / "topic-watch"
        assert resolve_config_file(package_parent=tmp_path, environ=environ) == root / "config.yml"

    def test_explicit_config_path_is_authoritative(self, tmp_path: Path) -> None:
        """TOPIC_WATCH_CONFIG_PATH outranks an existing data/ directory."""
        from app.config import CONFIG_PATH_ENV_VAR, resolve_config_file, resolve_state_root

        (tmp_path / "data").mkdir()
        pinned = tmp_path / "elsewhere" / "tw.yml"
        environ = {CONFIG_PATH_ENV_VAR: str(pinned)}
        assert resolve_config_file(package_parent=tmp_path, environ=environ) == pinned
        assert resolve_state_root(package_parent=tmp_path, environ=environ) == pinned.parent

    def test_config_and_db_roots_cannot_diverge(self) -> None:
        """The database module takes both its root and its default file from config."""
        import app.config as config_module
        import app.database as database_module

        # DEFAULT_DB_PATH / DEFAULT_CONFIG_PATH are redirected per-test by conftest,
        # so assert on the roots they are both derived from.
        assert database_module.DATA_DIR == config_module.STATE_ROOT
        assert config_module.DEFAULT_DB_PATH.parent == config_module.STATE_ROOT

    def test_absolute_db_path_is_used_verbatim(self) -> None:
        """TOPIC_WATCH_DB_PATH (the container's setting) stays authoritative."""
        from app.config import resolve_db_path

        settings = Settings(llm={"model": "openai/gpt-4o-mini", "api_key": "k"}, db_path="/srv/tw/topic_watch.db")
        assert resolve_db_path(settings) == Path("/srv/tw/topic_watch.db")

    def test_default_db_path_lands_in_the_state_root(self) -> None:
        """The default relative db_path resolves beside the config file, wherever that is."""
        from app.config import STATE_ROOT, resolve_db_path

        settings = Settings(llm={"model": "openai/gpt-4o-mini", "api_key": "k"})
        assert resolve_db_path(settings) == STATE_ROOT / "topic_watch.db"


class TestLosslessYamlCodec:
    """TW-AUD-028: one schema-owned codec; unknown keys warn on load and survive a save."""

    def test_unknown_nested_key_logs_warning(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A nested key nobody recognizes is reported, not silently dropped."""
        monkeypatch.delenv("TOPIC_WATCH_LLM__API_KEY", raising=False)
        config = tmp_path / "config.yml"
        config.write_text('llm:\n  model: "openai/gpt-4o-mini"\n  api_key: "sk"\n  temperture: 0.4\n')
        with caplog.at_level(logging.WARNING, logger="app.config"):
            load_settings(config_path=config)
        assert any("llm.temperture" in r.message for r in caplog.records)

    def test_unknown_keys_survive_a_save(self, tmp_path: Path) -> None:
        """A forward-compatible key a save does not understand is preserved, not erased."""
        config = tmp_path / "config.yml"
        config.write_text(
            'llm:\n  model: "openai/gpt-4o-mini"\n  api_key: "sk"\n  future_llm_option: "keep"\nfuture_top_level: 42\n'
        )
        settings = Settings(llm={"model": "openai/gpt-4o-mini", "api_key": "sk"})
        save_settings_to_yaml(settings, config)

        data = yaml.safe_load(config.read_text())
        assert data["future_top_level"] == 42
        assert data["llm"]["future_llm_option"] == "keep"
        assert data["llm"]["model"] == "openai/gpt-4o-mini"

    def test_cleared_optional_value_is_removed_not_resurrected(self, tmp_path: Path) -> None:
        """Clearing base_url deletes the key instead of leaving the old value behind."""
        config = tmp_path / "config.yml"
        config.write_text('llm:\n  model: "openai/gpt-4o-mini"\n  api_key: "sk"\n  base_url: "http://old:11434"\n')
        settings = Settings(llm={"model": "openai/gpt-4o-mini", "api_key": "sk", "base_url": None})
        save_settings_to_yaml(settings, config)

        assert "base_url" not in yaml.safe_load(config.read_text())["llm"]

    def test_emptied_list_is_written_not_left_stale(self, tmp_path: Path) -> None:
        """Deleting every notification URL persists the empty list."""
        config = tmp_path / "config.yml"
        config.write_text('notifications:\n  urls:\n    - "ntfy://old"\n')
        settings = Settings(llm={"model": "openai/gpt-4o-mini", "api_key": "sk"})
        save_settings_to_yaml(settings, config)

        assert yaml.safe_load(config.read_text())["notifications"]["urls"] == []

    def test_non_mapping_file_is_replaced(self, tmp_path: Path) -> None:
        """A YAML file that is not a mapping degrades to a fresh document, no crash."""
        config = tmp_path / "config.yml"
        config.write_text("- just\n- a\n- list\n")
        settings = Settings(llm={"model": "openai/gpt-4o-mini", "api_key": "sk"})
        save_settings_to_yaml(settings, config)

        assert yaml.safe_load(config.read_text())["llm"]["model"] == "openai/gpt-4o-mini"


class TestAtomicPermissionSafeWrite:
    """AUG-198: settings writes are atomic and never widen file permissions."""

    def test_failed_write_leaves_the_previous_config_intact(self, tmp_path: Path) -> None:
        """An error mid-serialization must not truncate the last valid configuration."""
        from unittest.mock import patch

        config = tmp_path / "config.yml"
        original = 'llm:\n  model: "openai/gpt-4o-mini"\n  api_key: "sk-previous"\n'
        config.write_text(original)
        settings = Settings(llm={"model": "openai/new", "api_key": "sk-new"})

        with patch("app.config.yaml.dump", side_effect=OSError("disk full")), pytest.raises(OSError):
            save_settings_to_yaml(settings, config)

        assert config.read_text() == original

    def test_failed_write_leaves_no_temp_file_behind(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        config = tmp_path / "config.yml"
        config.write_text("llm:\n  model: x\n")
        settings = Settings(llm={"model": "openai/new", "api_key": "sk-new"})

        with patch("app.config.yaml.dump", side_effect=OSError("disk full")), pytest.raises(OSError):
            save_settings_to_yaml(settings, config)

        assert [p.name for p in tmp_path.iterdir()] == ["config.yml"]

    def test_new_config_is_owner_readable_only(self, tmp_path: Path) -> None:
        """A config file holding API keys is created 0600."""
        config = tmp_path / "config.yml"
        save_settings_to_yaml(Settings(llm={"model": "openai/m", "api_key": "sk"}), config)
        assert config.stat().st_mode & 0o777 == 0o600

    def test_existing_permissive_mode_is_tightened(self, tmp_path: Path) -> None:
        """A world-readable config (the 0644 copied example) stops being readable on save."""
        import os

        config = tmp_path / "config.yml"
        config.write_text("llm:\n  model: x\n")
        os.chmod(config, 0o644)
        save_settings_to_yaml(Settings(llm={"model": "openai/m", "api_key": "sk"}), config)
        assert config.stat().st_mode & 0o077 == 0

    def test_replacement_is_a_rename_not_a_truncate(self, tmp_path: Path) -> None:
        """The destination is replaced atomically, so no reader ever sees a partial file."""
        config = tmp_path / "config.yml"
        config.write_text("llm:\n  model: x\n")
        before = config.stat().st_ino
        save_settings_to_yaml(Settings(llm={"model": "openai/m", "api_key": "sk"}), config)
        assert config.stat().st_ino != before


class TestExaRequiresAKey:
    """AUG-099: enabled means usable — a keyless Exa source can never fetch."""

    def test_enabled_without_a_key_is_disabled(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="app.config"):
            settings = Settings(llm={"model": "openai/m", "api_key": "sk"}, exa={"enabled": True, "api_key": ""})
        assert settings.exa.enabled is False
        assert any("Exa is enabled but no API key" in r.message for r in caplog.records)

    def test_whitespace_key_does_not_count(self) -> None:
        settings = Settings(llm={"model": "openai/m", "api_key": "sk"}, exa={"enabled": True, "api_key": "   "})
        assert settings.exa.enabled is False

    def test_enabled_with_a_key_stays_enabled(self) -> None:
        settings = Settings(llm={"model": "openai/m", "api_key": "sk"}, exa={"enabled": True, "api_key": "exa-key"})
        assert settings.exa.enabled is True

    def test_keyless_yaml_config_still_loads(self, tmp_path: Path) -> None:
        """A hand-edited config is corrected, not rejected — startup must survive it."""
        config = tmp_path / "config.yml"
        config.write_text('llm:\n  model: "openai/m"\n  api_key: "sk"\nexa:\n  enabled: true\n')
        settings = load_settings(config_path=config)
        assert settings.exa.enabled is False

    def test_env_supplied_key_enables_it(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TOPIC_WATCH_EXA__API_KEY", "exa-env-key")
        config = tmp_path / "config.yml"
        config.write_text('llm:\n  model: "openai/m"\n  api_key: "sk"\nexa:\n  enabled: true\n')
        settings = load_settings(config_path=config)
        assert settings.exa.enabled is True


class TestKeylessProviders:
    """AUG-107: key requiredness is provider-aware."""

    def test_ollama_without_a_key_is_configured(self) -> None:
        settings = Settings(llm={"model": "ollama/llama3.3", "api_key": ""})
        assert settings.is_configured()

    def test_cloud_provider_without_a_key_is_not_configured(self) -> None:
        settings = Settings(llm={"model": "openai/gpt-4o-mini", "api_key": ""})
        assert not settings.is_configured()

    def test_provider_match_is_case_insensitive(self) -> None:
        from app.config import is_keyless_llm_provider

        assert is_keyless_llm_provider("Ollama/Llama3.3")
        assert not is_keyless_llm_provider("openai/gpt-4o-mini")
        assert not is_keyless_llm_provider("ollama")  # no provider prefix at all
