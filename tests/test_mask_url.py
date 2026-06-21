"""Tests for the _mask_url Jinja2 filter."""

from app.web.routers.templates import _mask_url


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


def test_hides_host_even_though_canonical_redact_would_show_it():
    # Fold-in: _mask_url builds on log_redaction.redact_url but stays stronger for
    # the UI by hiding the host too. The canonical helper would keep the host.
    from app.log_redaction import redact_url

    url = "ntfy://ntfy.example.com/sometopic"
    assert "ntfy.example.com" in redact_url(url)  # canonical keeps host
    assert _mask_url(url) == "ntfy://****"  # UI filter hides it
