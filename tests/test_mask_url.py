"""Tests for the _mask_url Jinja2 filter."""

from app.web.routes import _mask_url


def test_ntfy_url_masked():
    assert _mask_url("ntfy://user:password@ntfy.example.com/topic") == "ntfy://****"


def test_discord_url_masked():
    assert _mask_url("discord://webhook_id/webhook_token") == "discord://****"


def test_slack_url_masked():
    assert _mask_url("slack://a/b/c") == "slack://****"


def test_mailto_url_masked():
    assert _mask_url("mailto://user:pass@gmail.com") == "mailto://****"


def test_empty_string_returns_masked():
    assert _mask_url("") == "****"


def test_no_scheme_returns_masked():
    assert _mask_url("no-scheme-here") == "****"


def test_invalid_url_returns_masked():
    # Pass something that won't have a scheme
    assert _mask_url("://broken") == "****"
