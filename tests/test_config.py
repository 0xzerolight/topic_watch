"""Tests for configuration loading and validation."""

import logging
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import Settings, load_settings


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
