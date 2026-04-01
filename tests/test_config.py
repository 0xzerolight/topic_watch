"""Tests for configuration loading and validation."""

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
        assert settings.check_interval_hours == 6
        assert settings.max_articles_per_check == 10
        assert settings.knowledge_state_max_tokens == 2000
        assert len(settings.notifications.urls) == 1

    def test_load_minimal_config(self, minimal_config_yaml: Path) -> None:
        settings = load_settings(config_path=minimal_config_yaml)
        assert settings.llm.model == "openai/gpt-4o-mini"
        assert settings.check_interval_hours == 6
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
        config.write_text("check_interval_hours: 6\n")
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
        config.write_text('llm:\n  model: "openai/gpt-4o-mini"\n  api_key: "k"\ncheck_interval_hours: 0\n')
        with pytest.raises(ValidationError):
            load_settings(config_path=config)

    def test_invalid_check_interval_too_high(self, tmp_path: Path) -> None:
        config = tmp_path / "config.yml"
        config.write_text('llm:\n  model: "openai/gpt-4o-mini"\n  api_key: "k"\ncheck_interval_hours: 200\n')
        with pytest.raises(ValidationError):
            load_settings(config_path=config)

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
        monkeypatch.setenv("TOPIC_WATCH_CHECK_INTERVAL_HOURS", "12")
        settings = load_settings(config_path=sample_config_yaml)
        assert settings.check_interval_hours == 12
