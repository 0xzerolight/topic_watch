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


def test_pushover_app_token_not_kept_as_host() -> None:
    # pover://USERKEY@APPTOKEN — the app token parses as the URL's hostname,
    # which the old scheme-blind logic always kept (AUG-248).
    redacted = redact_url("pover://someuserkey1234@AppTokenSecretValue99")
    assert "apptokensecretvalue99" not in redacted.lower()
    assert redacted.startswith("pover://")


def test_ntfy_private_topic_not_kept_as_host() -> None:
    # ntfy://private-topic — the topic name IS the host; a private topic name
    # is itself the capability (anyone who knows it can publish/subscribe).
    redacted = redact_url("ntfy://my-private-topic-name")
    assert "my-private-topic-name" not in redacted
    assert redacted.startswith("ntfy://")


def test_non_http_scheme_redaction_is_stable_for_same_url() -> None:
    # Same target logged twice should fingerprint identically, so repeated
    # failures for one misconfigured notifier are recognizable as the same one.
    url = "tgram://123456789:AAFF-BOT-TOKEN/chat_id"
    assert redact_url(url) == redact_url(url)


def test_non_http_scheme_redaction_differs_for_different_urls() -> None:
    a = redact_url("tgram://123456789:AAFF-BOT-TOKEN-A/chat_id")
    b = redact_url("tgram://123456789:AAFF-BOT-TOKEN-B/chat_id")
    assert a != b


def test_http_scheme_still_shows_host_unchanged() -> None:
    # The scheme-aware branch must not regress ordinary HTTP webhook redaction.
    redacted = redact_url("https://hooks.slack.com/services/T000/B000/XXXX")
    assert redacted.startswith("https://hooks.slack.com")
