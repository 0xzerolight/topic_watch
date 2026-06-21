"""Tests for the redact_url log-hygiene helper (OVH-038).

redact_url must keep webhook/notification URLs in logs informative (scheme +
host + a short path prefix) while never leaking embedded secrets: userinfo
(``user:token@``), query strings (``?token=...``), or the full path that often
*is* the secret for Slack/Discord webhooks.
"""

from app.log_redaction import redact_url


def test_strips_userinfo_credentials() -> None:
    # ntfy-style user:password@host — credentials must never survive.
    redacted = redact_url("https://user:s3cr3t@ntfy.example.com/topic")
    assert "s3cr3t" not in redacted
    assert "user" not in redacted
    assert redacted.startswith("https://ntfy.example.com")


def test_strips_query_string() -> None:
    redacted = redact_url("https://hooks.example.com/services/path?token=AABBCC")
    assert "AABBCC" not in redacted
    assert "?" not in redacted


def test_keeps_scheme_and_host() -> None:
    redacted = redact_url("https://hooks.slack.com/services/T000/B000/XXXX")
    assert redacted.startswith("https://hooks.slack.com")


def test_redacts_discord_webhook_token_in_path() -> None:
    # The long trailing token segment is the secret; it must be dropped.
    token = "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFG"
    redacted = redact_url(f"https://discord.com/api/webhooks/123456789/{token}")
    assert token not in redacted


def test_redacts_slack_webhook_token_in_path() -> None:
    secret = "T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX"
    redacted = redact_url(f"https://hooks.slack.com/services/{secret}")
    assert "XXXXXXXXXXXXXXXXXXXXXXXX" not in redacted


def test_short_path_prefix_is_kept_for_context() -> None:
    # A short, non-secret leading segment is useful context and may be shown.
    redacted = redact_url("https://discord.com/api/webhooks/123/secrettoken")
    assert redacted.startswith("https://discord.com")
    # Still drops the secret tail.
    assert "secrettoken" not in redacted


def test_handles_url_with_no_path() -> None:
    assert redact_url("https://example.com") == "https://example.com"


def test_handles_empty_string() -> None:
    # Must not raise; returns a safe placeholder.
    assert redact_url("") == "****"


def test_handles_no_scheme() -> None:
    redacted = redact_url("not-a-url-at-all")
    assert redacted == "****"


def test_handles_malformed_url_without_raising() -> None:
    # Must never raise from inside a log statement.
    redacted = redact_url("://broken")
    assert redacted == "****"


def test_drops_twelve_char_path_segment() -> None:
    # A 12-char path segment could be a short token; it must NOT be kept verbatim.
    token = "abcdefghijkl"  # exactly 12 chars
    assert len(token) == 12
    redacted = redact_url(f"https://ntfy.example.com/{token}")
    assert token not in redacted
    assert redacted.startswith("https://ntfy.example.com")


def test_ipv6_host_stays_bracketed() -> None:
    redacted = redact_url("https://[::1]/metadata")
    assert redacted.startswith("https://[::1]")


def test_strips_userinfo_with_port() -> None:
    redacted = redact_url("https://user:tok@host.example.com:8443/x")
    assert "tok" not in redacted
    assert "user" not in redacted
    assert redacted.startswith("https://host.example.com")


def test_full_url_never_appears_for_secret_bearing_webhook() -> None:
    full = "https://user:pw@hooks.slack.com/services/T1/B1/SECRETTOKEN12345?x=y"
    redacted = redact_url(full)
    assert redacted != full
    assert "SECRETTOKEN12345" not in redacted
    assert "pw" not in redacted
    assert "x=y" not in redacted
