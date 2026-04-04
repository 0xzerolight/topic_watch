"""Tests for configuration loading and validation."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import Settings, extract_provider, is_cloud_provider, load_settings


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

    def test_optional_base_url(self, tmp_path: Path) -> None:
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


class TestProviderDetection:
    """Tests for extract_provider and is_cloud_provider helpers."""

    def test_extract_provider_with_slash(self) -> None:
        assert extract_provider("openai/gpt-4") == "openai"

    def test_extract_provider_ollama(self) -> None:
        assert extract_provider("ollama/llama3") == "ollama"

    def test_extract_provider_no_slash(self) -> None:
        assert extract_provider("gpt-4") is None

    def test_extract_provider_empty(self) -> None:
        assert extract_provider("") is None

    def test_extract_provider_case_insensitive(self) -> None:
        assert extract_provider("Anthropic/claude-3") == "anthropic"

    def test_is_cloud_provider_openai(self) -> None:
        assert is_cloud_provider("openai/gpt-4o-mini") is True

    def test_is_cloud_provider_anthropic(self) -> None:
        assert is_cloud_provider("anthropic/claude-haiku-4-5") is True

    def test_is_cloud_provider_ollama(self) -> None:
        assert is_cloud_provider("ollama/llama3") is False

    def test_is_cloud_provider_unknown(self) -> None:
        assert is_cloud_provider("groq/mixtral") is False

    def test_is_cloud_provider_no_slash(self) -> None:
        assert is_cloud_provider("gpt-4") is False


class TestSaveSettingsBaseUrlGuard:
    """Test that save_settings_to_yaml strips base_url for cloud providers."""

    def test_cloud_provider_base_url_omitted(self, tmp_path: Path) -> None:
        """base_url is NOT written to YAML when model is a cloud provider."""
        import yaml

        from app.config import LLMSettings, save_settings_to_yaml

        settings = Settings(
            llm=LLMSettings(model="anthropic/claude-haiku-4-5", api_key="sk-test", base_url="http://localhost:11434"),
        )  # type: ignore[call-arg]
        config_file = tmp_path / "config.yml"
        save_settings_to_yaml(settings, config_file)

        data = yaml.safe_load(config_file.read_text())
        assert "base_url" not in data.get("llm", {})

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
