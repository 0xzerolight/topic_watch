"""Tests for SSRF protection in URL validation."""

import socket

from app.url_validation import (
    _resolved_ip_is_private,
    is_private_url,
    validate_feed_url,
    validate_feed_urls,
)


class TestIsPrivateUrl:
    """Unit tests for is_private_url() covering all bypass vectors."""

    # --- IPv4 private ranges (existing) ---

    def test_localhost(self) -> None:
        assert is_private_url("http://localhost/path") is True

    def test_localhost_with_port(self) -> None:
        assert is_private_url("http://localhost:8080/path") is True

    def test_loopback(self) -> None:
        assert is_private_url("http://127.0.0.1/path") is True

    def test_10_range(self) -> None:
        assert is_private_url("http://10.0.0.1/path") is True

    def test_172_range(self) -> None:
        assert is_private_url("http://172.16.0.1/path") is True

    def test_192_168_range(self) -> None:
        assert is_private_url("http://192.168.1.1/path") is True

    def test_link_local(self) -> None:
        assert is_private_url("http://169.254.1.1/path") is True

    def test_zero_address(self) -> None:
        assert is_private_url("http://0.0.0.0/path") is True

    # --- IPv6 private addresses (fixed in this patch) ---

    def test_ipv6_loopback(self) -> None:
        assert is_private_url("http://[::1]/path") is True

    def test_ipv6_loopback_with_port(self) -> None:
        assert is_private_url("http://[::1]:8080/path") is True

    def test_ipv6_ula(self) -> None:
        assert is_private_url("http://[fd00::1]/path") is True

    def test_ipv6_ula_full(self) -> None:
        assert is_private_url("http://[fdab:cdef:1234::1]/path") is True

    def test_ipv6_link_local(self) -> None:
        assert is_private_url("http://[fe80::1]/path") is True

    def test_ipv6_mapped_ipv4_loopback(self) -> None:
        assert is_private_url("http://[::ffff:127.0.0.1]/path") is True

    def test_ipv6_mapped_ipv4_private(self) -> None:
        assert is_private_url("http://[::ffff:10.0.0.1]/path") is True

    def test_ipv6_mapped_ipv4_192(self) -> None:
        assert is_private_url("http://[::ffff:192.168.1.1]/path") is True

    # --- Alternative IP encodings (caught by DNS resolution layer) ---

    def test_hex_ip_loopback(self, monkeypatch) -> None:
        """0x7f000001 = 127.0.0.1 in hex — caught by DNS resolution."""

        def _loopback(*_args, **_kwargs):
            return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 0))]

        monkeypatch.setattr(socket, "getaddrinfo", _loopback)
        assert is_private_url("http://0x7f000001/path") is True

    def test_decimal_ip_loopback(self, monkeypatch) -> None:
        """2130706433 = 127.0.0.1 in decimal — caught by DNS resolution."""

        def _loopback(*_args, **_kwargs):
            return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 0))]

        monkeypatch.setattr(socket, "getaddrinfo", _loopback)
        assert is_private_url("http://2130706433/path") is True

    # --- Public URLs should pass ---

    def test_public_url(self, monkeypatch) -> None:
        def _public(*_args, **_kwargs):
            return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))]

        monkeypatch.setattr(socket, "getaddrinfo", _public)
        assert is_private_url("https://example.com/feed.xml") is False

    def test_public_ip(self) -> None:
        assert is_private_url("http://8.8.8.8/feed.xml") is False

    def test_empty_url(self) -> None:
        assert is_private_url("") is False

    # --- DNS resolution failure = fail closed ---

    def test_unresolvable_host_blocked(self, monkeypatch) -> None:
        """Hosts that fail DNS resolution are blocked (fail-closed)."""

        def _raise(*_args, **_kwargs):
            raise socket.gaierror("name resolution failed")

        monkeypatch.setattr(socket, "getaddrinfo", _raise)
        assert is_private_url("http://this-definitely-does-not-resolve.invalid/feed") is True

    def test_resolved_ip_is_private_fails_closed_on_gaierror(self, monkeypatch) -> None:
        """_resolved_ip_is_private returns True when DNS resolution raises."""

        def _raise(*_args, **_kwargs):
            raise socket.gaierror("name resolution failed")

        monkeypatch.setattr(socket, "getaddrinfo", _raise)
        assert _resolved_ip_is_private("unresolvable.invalid") is True

    def test_resolved_public_ip_allowed(self, monkeypatch) -> None:
        """A host resolving to a public IP is still allowed (happy path)."""

        def _public(*_args, **_kwargs):
            return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))]

        monkeypatch.setattr(socket, "getaddrinfo", _public)
        assert _resolved_ip_is_private("example.com") is False
        assert is_private_url("https://example.com/feed.xml") is False


class TestValidateFeedUrl:
    """Tests for validate_feed_url()."""

    def test_rejects_non_http(self) -> None:
        error = validate_feed_url("ftp://example.com/feed.xml")
        assert error is not None
        assert "must be http or https" in error

    def test_rejects_file_scheme(self) -> None:
        error = validate_feed_url("file:///etc/passwd")
        assert error is not None

    def test_rejects_private(self) -> None:
        error = validate_feed_url("http://localhost/feed.xml")
        assert error is not None
        assert "private" in error.lower()

    def test_rejects_unresolvable_host(self, monkeypatch) -> None:
        def _raise(*_args, **_kwargs):
            raise socket.gaierror("name resolution failed")

        monkeypatch.setattr(socket, "getaddrinfo", _raise)
        error = validate_feed_url("https://nope.invalid/feed.xml")
        assert error is not None
        assert "could not be resolved" in error

    def test_accepts_valid_url(self, monkeypatch) -> None:
        def _public(*_args, **_kwargs):
            return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))]

        monkeypatch.setattr(socket, "getaddrinfo", _public)
        error = validate_feed_url("https://example.com/rss.xml")
        assert error is None


class TestValidateFeedUrls:
    """Tests for validate_feed_urls()."""

    def test_all_valid(self, monkeypatch) -> None:
        def _public(*_args, **_kwargs):
            return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))]

        monkeypatch.setattr(socket, "getaddrinfo", _public)
        errors = validate_feed_urls(["https://example.com/rss.xml", "https://other.com/feed"])
        assert errors == []

    def test_mixed_valid_invalid(self, monkeypatch) -> None:
        def _public(*_args, **_kwargs):
            return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))]

        monkeypatch.setattr(socket, "getaddrinfo", _public)
        errors = validate_feed_urls(["https://example.com/rss.xml", "http://localhost/feed.xml"])
        assert len(errors) == 1
        assert "private" in errors[0].lower()
