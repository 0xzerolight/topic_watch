"""Tests for SSRF protection in URL validation."""

import socket

import httpx
import pytest

from app.url_validation import (
    PrivateRedirectError,
    _resolved_ip_is_private,
    is_private_url,
    safe_get,
    safe_send,
    validate_feed_url,
    validate_feed_urls,
)


def _stub_resolves_to(monkeypatch, ip: str) -> None:
    """Make socket.getaddrinfo resolve any host to ``ip`` (mirrors real DNS).

    IPv6 literals resolve to themselves; this lets the IPv6-literal tests
    exercise the layer-2 ipaddress classification (which now does the work that
    the removed over-broad regexes used to do) rather than the autouse stub's
    default public IP.
    """
    family = socket.AF_INET6 if ":" in ip else socket.AF_INET
    sockaddr = (ip, 0, 0, 0) if family == socket.AF_INET6 else (ip, 0)

    def _resolve(*_args, **_kwargs):
        return [(family, socket.SOCK_STREAM, 0, "", sockaddr)]

    monkeypatch.setattr(socket, "getaddrinfo", _resolve)


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

    def test_ipv6_ula(self, monkeypatch) -> None:
        # ULA literals are classified by layer-2 (ipaddress), not the regex.
        _stub_resolves_to(monkeypatch, "fd00::1")
        assert is_private_url("http://[fd00::1]/path") is True

    def test_ipv6_ula_full(self, monkeypatch) -> None:
        _stub_resolves_to(monkeypatch, "fdab:cdef:1234::1")
        assert is_private_url("http://[fdab:cdef:1234::1]/path") is True

    def test_ipv6_ula_fc(self, monkeypatch) -> None:
        _stub_resolves_to(monkeypatch, "fc00::1")
        assert is_private_url("http://[fc00::1]/path") is True

    def test_ipv6_link_local(self) -> None:
        assert is_private_url("http://[fe80::1]/path") is True

    def test_ipv6_mapped_ipv4_loopback(self, monkeypatch) -> None:
        # Mapped literals are classified by layer-2 (ipaddress), not the regex.
        _stub_resolves_to(monkeypatch, "::ffff:127.0.0.1")
        assert is_private_url("http://[::ffff:127.0.0.1]/path") is True

    def test_ipv6_mapped_ipv4_private(self, monkeypatch) -> None:
        _stub_resolves_to(monkeypatch, "::ffff:10.0.0.1")
        assert is_private_url("http://[::ffff:10.0.0.1]/path") is True

    def test_ipv6_mapped_ipv4_192(self, monkeypatch) -> None:
        _stub_resolves_to(monkeypatch, "::ffff:192.168.1.1")
        assert is_private_url("http://[::ffff:192.168.1.1]/path") is True

    def test_ipv6_mapped_ipv4_public_allowed(self, monkeypatch) -> None:
        """A public IPv4-mapped IPv6 literal must NOT be blocked (OVH-169).

        The old blanket ``^::ffff:`` regex over-blocked every mapped address;
        layer-2 resolution distinguishes public from private.
        """
        _stub_resolves_to(monkeypatch, "::ffff:93.184.216.34")
        assert is_private_url("http://[::ffff:93.184.216.34]/path") is False

    def test_ipv6_mapped_ipv4_cgnat(self, monkeypatch) -> None:
        """A CGNAT IPv4-mapped IPv6 literal must be blocked (OVH-169 follow-up).

        ``::ffff:100.64.0.1`` keeps ``version == 6``, so the version==4 CGNAT
        gate was skipped and ipaddress flags no other predicate — the mapped
        address must be unwrapped to its embedded IPv4 and re-classified.
        """
        _stub_resolves_to(monkeypatch, "::ffff:100.64.0.1")
        assert is_private_url("http://[::ffff:100.64.0.1]/path") is True

    # --- fc-/fd- hostnames must not be mistaken for IPv6 ULA (OVH-142) ---

    def test_fc_hostname_allowed(self, monkeypatch) -> None:
        """A public hostname starting with 'fc' is a hostname, not an IPv6 ULA literal."""

        def _public(*_args, **_kwargs):
            return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))]

        monkeypatch.setattr(socket, "getaddrinfo", _public)
        assert is_private_url("https://fc-barcelona.example.com/feed.xml") is False

    def test_fd_hostname_allowed(self, monkeypatch) -> None:
        """A public hostname starting with 'fd' is a hostname, not an IPv6 ULA literal."""

        def _public(*_args, **_kwargs):
            return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))]

        monkeypatch.setattr(socket, "getaddrinfo", _public)
        assert is_private_url("https://fd-news.example.org/rss") is False

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

    def test_resolved_cgnat_blocked(self, monkeypatch) -> None:
        """A host resolving into the RFC 6598 CGNAT range (100.64.0.0/10) is blocked."""

        def _cgnat(*_args, **_kwargs):
            return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("100.64.0.1", 0))]

        monkeypatch.setattr(socket, "getaddrinfo", _cgnat)
        assert _resolved_ip_is_private("rebind.example.com") is True
        assert is_private_url("https://rebind.example.com/feed.xml") is True

    def test_resolver_timeout_fails_closed(self, monkeypatch) -> None:
        """A slow getaddrinfo is bounded by a resolver timeout and fails closed (OVH-148).

        A crafted host that never resolves in time must not occupy a worker for
        minutes: the bounded resolver gives up after the timeout and treats the
        host as unverifiable (blocked), rather than blocking indefinitely.
        """
        import time

        from app import url_validation

        monkeypatch.setattr(url_validation, "_RESOLVE_TIMEOUT", 0.1)

        def _slow(*_args, **_kwargs):
            time.sleep(5)  # would hang far beyond the timeout if not bounded
            return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))]

        monkeypatch.setattr(socket, "getaddrinfo", _slow)

        start = time.monotonic()
        result = _resolved_ip_is_private("slow.example.com")
        elapsed = time.monotonic() - start

        assert result is True  # fail closed
        assert elapsed < 2.0  # bounded — did not wait for the 5s sleep


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


class TestSafeSendInitialUrl:
    """safe_send validates its OWN initial request URL, not just redirects (OVH-140)."""

    async def test_rejects_private_initial_url_without_sending(self) -> None:
        """A private initial URL is blocked before any network send."""
        sends: list[str] = []

        def _handler(request: httpx.Request) -> httpx.Response:
            sends.append(str(request.url))
            return httpx.Response(200, text="should never reach here")

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
            with pytest.raises(PrivateRedirectError):
                await safe_get(client, "http://127.0.0.1:8080/internal")

        assert sends == []  # never sent

    async def test_rejects_non_http_initial_url_without_sending(self) -> None:
        """A non-http(s) initial scheme is blocked before any network send."""
        sends: list[str] = []

        def _handler(request: httpx.Request) -> httpx.Response:
            sends.append(str(request.url))
            return httpx.Response(200)

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
            request = client.build_request("GET", "file:///etc/passwd")
            with pytest.raises(PrivateRedirectError):
                await safe_send(client, request)

        assert sends == []

    async def test_allows_public_initial_url(self, monkeypatch) -> None:
        """A public initial URL still sends normally (no regression)."""

        def _public(*_args, **_kwargs):
            return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))]

        monkeypatch.setattr(socket, "getaddrinfo", _public)

        def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="ok")

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
            response = await safe_get(client, "https://example.com/feed.xml")

        assert response.status_code == 200
        assert response.text == "ok"

    async def test_safe_get_sends_custom_headers(self, monkeypatch) -> None:
        """Custom request headers (e.g. conditional-GET validators) reach the request."""

        def _public(*_args, **_kwargs):
            return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))]

        monkeypatch.setattr(socket, "getaddrinfo", _public)

        captured: dict[str, str] = {}

        def _handler(request: httpx.Request) -> httpx.Response:
            captured.update(request.headers)
            return httpx.Response(200, text="ok")

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
            await safe_get(client, "https://example.com/feed.xml", headers={"If-None-Match": 'W/"abc"'})

        assert captured.get("if-none-match") == 'W/"abc"'
