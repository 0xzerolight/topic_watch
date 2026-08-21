"""Tests for the conftest ambient-environment scrub (AUG-039).

``Settings`` gives every ``TOPIC_WATCH_*`` env var precedence over YAML and
defaults, so a developer's configured shell must not leak into the suite.
``tests/conftest.py`` scrubs the namespace once at collection time before the
app is imported; these tests pin the scrub helper's own contract directly
(a subprocess-level end-to-end proof would only re-test Python's env dict).
"""

import pytest

from tests.conftest import _TEST_ENV_DEFAULTS, _scrub_ambient_env


def test_scrub_removes_every_ambient_topic_watch_key() -> None:
    fake_environ = {
        "TOPIC_WATCH_CHECK_INTERVAL": "10m",
        "TOPIC_WATCH_EXA__API_KEY": "a-real-developer-key",
        "PATH": "/usr/bin",
    }

    _scrub_ambient_env(fake_environ)

    assert "TOPIC_WATCH_CHECK_INTERVAL" not in fake_environ
    assert "TOPIC_WATCH_EXA__API_KEY" not in fake_environ
    assert fake_environ["PATH"] == "/usr/bin"  # unrelated vars are untouched


def test_scrub_overrides_rather_than_setdefaults(monkeypatch) -> None:
    """A pre-existing ambient value for a seeded key must not survive.

    ``os.environ.setdefault`` (the old behavior) would have kept a real
    developer key in place; the fix replaces it outright.
    """
    fake_environ = {"TOPIC_WATCH_LLM__API_KEY": "a-real-developer-key"}

    _scrub_ambient_env(fake_environ)

    assert fake_environ == _TEST_ENV_DEFAULTS


def test_settings_ignores_ambient_env_once_scrubbed(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: an ambient value for an unrelated setting never reaches a
    ``Settings()`` built after the scrub runs."""
    import os

    from app.config import Settings

    default_interval = Settings().check_interval

    monkeypatch.setenv("TOPIC_WATCH_CHECK_INTERVAL", "10m")
    assert Settings().check_interval == "10m"  # env does win before scrubbing

    _scrub_ambient_env(os.environ)

    assert Settings().check_interval == default_interval
