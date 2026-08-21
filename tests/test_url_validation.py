"""Tests for SSRF protection in URL validation."""

import gzip
import socket
import tracemalloc
import zlib

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


class TestNumericLabelHostnames:
    """A public hostname whose first label looks like a private IPv4 octet (AUG-189)."""

    @pytest.mark.parametrize(
        "url",
        [
            "http://127.example.com/feed.xml",
            "http://10.example.com/feed.xml",
            "http://192.168.example.com/feed.xml",
            "http://172.16.example.com/feed.xml",
            "http://169.254.example.com/feed.xml",
            "http://0.0.0.0.example.com/feed.xml",
        ],
    )
    def test_numeric_label_public_host_allowed(self, url: str, monkeypatch) -> None:
        _stub_resolves_to(monkeypatch, "93.184.216.34")
        assert is_private_url(url) is False

    def test_complete_ipv4_literal_still_blocked(self) -> None:
        assert is_private_url("http://127.0.0.1/x") is True
        assert is_private_url("http://192.168.1.1/x") is True


def _drain(uv) -> None:
    """Block until every slot of the process-wide resolver pool is free again."""
    for _ in range(uv._RESOLVER_POOL_SIZE):
        assert uv._resolver_slots.acquire(timeout=15)
    for _ in range(uv._RESOLVER_POOL_SIZE):
        uv._resolver_slots.release()


class TestResolverPoolBounded:
    """Timed-out resolver lookups must not spawn unbounded worker threads (AUG-013).

    Two independent mechanisms, pinned by one test each: a fixed process-wide pool
    caps how many resolver threads can exist, and an admission semaphore turns
    away a caller that cannot get a slot instead of letting it queue behind the
    lookups already stuck. Either one alone leaves the other's failure mode open.
    """

    @staticmethod
    def _run_against_a_full_pool(monkeypatch) -> tuple[list[str], set[int], int]:
        """Run ``pool size + 8`` callers against a resolver that never answers.

        Returns each caller's outcome, the distinct threads ``getaddrinfo`` ran
        on, and how many lookups reached it at all.
        """
        import threading
        from concurrent.futures import ThreadPoolExecutor

        from app import url_validation as uv

        release = threading.Event()
        worker_threads: set[int] = set()
        lock = threading.Lock()
        started = 0

        def _blocking(*_args, **_kwargs):
            nonlocal started
            with lock:
                started += 1
                worker_threads.add(threading.get_ident())
            release.wait(10)
            return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))]

        monkeypatch.setattr(socket, "getaddrinfo", _blocking)

        # The pool is process-wide and an earlier test abandons a slow lookup that
        # keeps its slot until the OS resolver returns. Start from a full pool so
        # the counts below are about this test's own callers.
        _drain(uv)

        callers = uv._RESOLVER_POOL_SIZE + 8

        def _lookup(index: int) -> str:
            try:
                uv._getaddrinfo_bounded(f"host{index}.example.com", 0.3)
            except uv.ResolverSaturatedError:
                return "saturated"
            except TimeoutError:
                return "timeout"
            return "ok"

        try:
            with ThreadPoolExecutor(max_workers=callers) as pool:
                outcomes = list(pool.map(_lookup, range(callers)))
        finally:
            release.set()
            # Hand every slot back before the next test asks for one.
            _drain(uv)

        return outcomes, worker_threads, started

    def test_overflow_callers_are_turned_away_not_queued(self, monkeypatch) -> None:
        from app import url_validation as uv

        outcomes, _threads, started = self._run_against_a_full_pool(monkeypatch)

        # One lookup per slot reached the resolver and nothing else did: without
        # admission control the extra callers queue behind the stuck lookups and
        # every one of them reports the ordinary lookup timeout instead. A
        # per-lookup executor makes this the caller count.
        assert started == uv._RESOLVER_POOL_SIZE
        assert outcomes.count("timeout") == uv._RESOLVER_POOL_SIZE
        assert outcomes.count("saturated") == len(outcomes) - uv._RESOLVER_POOL_SIZE
        assert outcomes.count("ok") == 0

    def test_resolver_threads_are_bounded(self, monkeypatch) -> None:
        from app import url_validation as uv

        _outcomes, worker_threads, _started = self._run_against_a_full_pool(monkeypatch)

        # A per-lookup executor gives every caller its own thread, so the count
        # tracks the callers rather than the pool.
        assert len(worker_threads) <= uv._RESOLVER_POOL_SIZE


class TestSafeSendByteCap:
    """safe_send streams and aborts an oversize body mid-transfer (AUG-006)."""

    async def test_oversize_body_aborted_mid_stream(self, monkeypatch) -> None:
        from app.url_validation import MAX_RESPONSE_BYTES, ResponseTooLargeError

        assert MAX_RESPONSE_BYTES > 0
        _stub_resolves_to(monkeypatch, "93.184.216.34")

        chunks_produced = 0

        async def _body():
            nonlocal chunks_produced
            for _ in range(500):
                chunks_produced += 1
                yield b"x" * 1024

        def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=_body())

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
            with pytest.raises(ResponseTooLargeError):
                await safe_get(client, "https://example.com/huge.xml", max_bytes=4096)

        # Aborted mid-stream: only the chunks needed to cross the cap were pulled,
        # not the whole 500 KiB body.
        assert chunks_produced <= 8

    async def test_body_under_cap_returned_intact(self, monkeypatch) -> None:
        _stub_resolves_to(monkeypatch, "93.184.216.34")

        async def _body():
            yield b"hello "
            yield b"world"

        def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=_body(), headers={"etag": 'W/"v1"'})

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
            response = await safe_get(client, "https://example.com/small.xml", max_bytes=4096)

        assert response.status_code == 200
        assert response.content == b"hello world"
        assert response.text == "hello world"
        assert response.headers.get("etag") == 'W/"v1"'

    async def test_redirect_body_is_not_downloaded(self, monkeypatch) -> None:
        _stub_resolves_to(monkeypatch, "93.184.216.34")

        redirect_chunks = 0

        async def _redirect_body():
            nonlocal redirect_chunks
            for _ in range(200):
                redirect_chunks += 1
                yield b"y" * 1024

        def _handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/start":
                return httpx.Response(
                    302,
                    headers={"location": "https://example.com/final"},
                    content=_redirect_body(),
                )
            return httpx.Response(200, text="final")

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
            response = await safe_get(client, "https://example.com/start")

        assert response.text == "final"
        assert redirect_chunks == 0


def _gzip_expanding_to(total_bytes: int) -> bytes:
    """A gzip stream that decodes to ``total_bytes`` of zeros, built incrementally.

    Built a megabyte at a time so the fixture itself never holds the expansion it
    describes.
    """
    compressor = zlib.compressobj(9, zlib.DEFLATED, zlib.MAX_WBITS | 16)
    block = b"\0" * (1024 * 1024)
    parts = []
    remaining = total_bytes
    while remaining > 0:
        take = min(len(block), remaining)
        remaining -= take
        parts.append(compressor.compress(block[:take]))
    parts.append(compressor.flush())
    return b"".join(parts)


class TestSafeSendCompressedBodyCap:
    """The byte budget bounds the DECODER, not just what it hands back (AUG-006).

    Counting decoded bytes *between* chunks cannot bound a decoder that is itself
    unbounded: httpx decodes a whole raw chunk in one unbounded ``decompress()``
    call, and a stacked ``Content-Encoding`` multiplies that per layer.
    """

    async def test_stacked_content_encoding_refused(self, monkeypatch) -> None:
        """A tiny body naming three gzip layers must never reach a decoder."""
        from app.url_validation import ResponseTooLargeError

        _stub_resolves_to(monkeypatch, "93.184.216.34")
        nested = gzip.compress(gzip.compress(_gzip_expanding_to(64 * 1024 * 1024), 9), 9)
        assert len(nested) < 64 * 1024
        pulled = 0

        async def _body():
            nonlocal pulled
            pulled += 1
            yield nested

        def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, headers={"content-encoding": "gzip, gzip, gzip"}, content=_body())

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
            with pytest.raises(ResponseTooLargeError, match="[Ss]tacked"):
                await safe_get(client, "https://example.com/feed.xml", max_bytes=65536)

        # Refused on the headers alone: not one byte was pulled or decoded.
        assert pulled == 0

    async def test_single_layer_bomb_peaks_near_the_budget(self, monkeypatch) -> None:
        """One raw chunk expanding far past the budget must not be materialised."""
        from app.url_validation import ResponseTooLargeError

        _stub_resolves_to(monkeypatch, "93.184.216.34")
        bomb = _gzip_expanding_to(64 * 1024 * 1024)

        async def _body():
            yield bomb

        def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, headers={"content-encoding": "gzip"}, content=_body())

        transport = httpx.MockTransport(_handler)
        tracemalloc.start()
        try:
            async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
                with pytest.raises(ResponseTooLargeError):
                    await safe_get(client, "https://example.com/feed.xml", max_bytes=65536)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        # An unbounded decompress() puts the whole 64 MiB expansion in one bytes
        # object before any counter runs. A bounded one never allocates past the
        # budget, so the peak stays within a small multiple of it.
        assert peak < 4 * 1024 * 1024, f"peak {peak} bytes for a 65536-byte budget"

    async def test_wire_bytes_are_bounded_even_when_nothing_decodes(self, monkeypatch) -> None:
        """A stream of empty deflate blocks decodes to ~0, so only a wire bound stops it."""
        from app.url_validation import ResponseTooLargeError

        _stub_resolves_to(monkeypatch, "93.184.216.34")
        compressor = zlib.compressobj(9, zlib.DEFLATED, zlib.MAX_WBITS | 16)
        head = compressor.compress(b"hello") + compressor.flush(zlib.Z_SYNC_FLUSH)
        # A byte-aligned stored block with LEN=0: valid deflate, zero output.
        empty_block = b"\x00\x00\x00\xff\xff"
        chunks_produced = 0

        async def _body():
            nonlocal chunks_produced
            yield head
            for _ in range(500):
                chunks_produced += 1
                yield empty_block * 1024

        def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, headers={"content-encoding": "gzip"}, content=_body())

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
            with pytest.raises(ResponseTooLargeError):
                await safe_get(client, "https://example.com/feed.xml", max_bytes=65536)

        assert chunks_produced <= 16

    async def test_gzip_body_under_cap_decoded_intact(self, monkeypatch) -> None:
        _stub_resolves_to(monkeypatch, "93.184.216.34")
        payload = b"<rss><channel><title>ok</title></channel></rss>" * 100
        body = gzip.compress(payload)

        async def _stream():
            # Split so the decoder is driven across chunk boundaries, as a real
            # transport does at 64 KiB.
            yield body[: len(body) // 2]
            yield body[len(body) // 2 :]

        def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, headers={"content-encoding": "gzip"}, content=_stream())

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
            response = await safe_get(client, "https://example.com/feed.xml", max_bytes=65536)

        assert response.content == payload
        assert "content-encoding" not in response.headers

    @pytest.mark.parametrize("wbits", [zlib.MAX_WBITS, -zlib.MAX_WBITS])
    async def test_deflate_body_decoded_in_both_framings(self, monkeypatch, wbits: int) -> None:
        """``deflate`` is zlib-wrapped or raw in the wild; both must still decode."""
        _stub_resolves_to(monkeypatch, "93.184.216.34")
        payload = b"<rss>deflate</rss>" * 50
        compressor = zlib.compressobj(9, zlib.DEFLATED, wbits)
        body = compressor.compress(payload) + compressor.flush()

        async def _stream():
            yield body

        def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, headers={"content-encoding": "deflate"}, content=_stream())

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
            response = await safe_get(client, "https://example.com/feed.xml", max_bytes=65536)

        assert response.content == payload

    async def test_identity_alongside_one_codec_is_not_stacking(self, monkeypatch) -> None:
        _stub_resolves_to(monkeypatch, "93.184.216.34")
        payload = b"<rss>ok</rss>"

        async def _stream():
            yield gzip.compress(payload)

        def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, headers={"content-encoding": "identity, gzip"}, content=_stream())

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
            response = await safe_get(client, "https://example.com/feed.xml", max_bytes=65536)

        assert response.content == payload


class TestSafeSendOwnsRedirectFollowing:
    """Per-hop SSRF checks must not depend on how the caller built its client.

    ``client.send()`` inherits ``client.follow_redirects``; when that is True httpx
    follows the chain internally, ``next_request`` is never populated, and the loop
    below hands back the final body as an ordinary result. The invariant used to be
    a docstring line only.
    """

    async def test_redirect_to_private_blocked_on_a_following_client(self, monkeypatch) -> None:
        _stub_resolves_to(monkeypatch, "93.184.216.34")
        private_hits = 0

        def _handler(request: httpx.Request) -> httpx.Response:
            nonlocal private_hits
            if request.url.host == "127.0.0.1":
                private_hits += 1
                return httpx.Response(200, text="INTERNAL-DATA")
            return httpx.Response(302, headers={"location": "http://127.0.0.1/internal"})

        transport = httpx.MockTransport(_handler)
        # Deliberately NOT follow_redirects=False: the helper must hold the line
        # on its own.
        async with httpx.AsyncClient(transport=transport, follow_redirects=True) as client:
            with pytest.raises(PrivateRedirectError):
                await safe_get(client, "https://example.com/start")

        assert private_hits == 0

    async def test_redirect_limit_holds_on_a_following_client(self, monkeypatch) -> None:
        _stub_resolves_to(monkeypatch, "93.184.216.34")

        def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(302, headers={"location": "https://example.com/next"})

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport, follow_redirects=True) as client:
            with pytest.raises(PrivateRedirectError, match="maximum of 3 redirects"):
                await safe_get(client, "https://example.com/start", max_redirects=3)


class TestSafeSendRedirectCredentials:
    """Cross-origin redirects must not carry origin-bound credentials (AUG-005)."""

    async def test_cross_origin_redirect_drops_credentials(self, monkeypatch) -> None:
        _stub_resolves_to(monkeypatch, "93.184.216.34")
        seen: list[httpx.Request] = []

        def _handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            if request.url.host == "feeds.example.com":
                return httpx.Response(302, headers={"location": "https://attacker.example.net/steal"})
            return httpx.Response(200, text="ok")

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
            request = client.build_request(
                "GET",
                "https://feeds.example.com/private.xml",
                headers={"Authorization": "Basic dXNlcjpwYXNz", "Cookie": "session=abc"},
            )
            response = await safe_send(client, request)

        assert response.status_code == 200
        assert len(seen) == 2
        assert "authorization" not in seen[1].headers
        assert "cookie" not in seen[1].headers
        assert seen[1].headers["host"] == "attacker.example.net"

    async def test_userinfo_credentials_not_forwarded_cross_origin(self, monkeypatch) -> None:
        _stub_resolves_to(monkeypatch, "93.184.216.34")
        seen: list[httpx.Request] = []

        def _handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            if request.url.host == "feeds.example.com":
                return httpx.Response(301, headers={"location": "https://attacker.example.net/steal"})
            return httpx.Response(200, text="ok")

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
            await safe_get(client, "https://user:pass@feeds.example.com/private.xml")

        assert "authorization" in seen[0].headers
        assert "authorization" not in seen[1].headers

    async def test_same_origin_redirect_keeps_credentials(self, monkeypatch) -> None:
        _stub_resolves_to(monkeypatch, "93.184.216.34")
        seen: list[httpx.Request] = []

        def _handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            if request.url.path == "/old":
                return httpx.Response(302, headers={"location": "https://feeds.example.com/new"})
            return httpx.Response(200, text="ok")

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
            request = client.build_request(
                "GET",
                "https://feeds.example.com/old",
                headers={"Authorization": "Basic dXNlcjpwYXNz"},
            )
            await safe_send(client, request)

        assert seen[1].headers.get("authorization") == "Basic dXNlcjpwYXNz"


class TestPrivateRedirectErrorRedaction:
    """AUG-312: PrivateRedirectError's own message must not carry the raw
    blocked URL — a caller logging it with exc_info=True (a traceback renders
    the exception's str()) would otherwise reproduce userinfo/query secrets
    that the warning logged alongside the raise already redacts."""

    async def test_blocked_initial_private_url_message_is_redacted(self) -> None:
        secret_url = "http://user:s3cr3tpass@127.0.0.1:8080/internal?token=QUERYSECRET"
        transport = httpx.MockTransport(lambda request: httpx.Response(200))
        async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
            with pytest.raises(PrivateRedirectError) as exc_info:
                await safe_get(client, secret_url)

        message = str(exc_info.value)
        assert "s3cr3tpass" not in message
        assert "QUERYSECRET" not in message

    async def test_blocked_initial_non_http_scheme_message_is_redacted(self) -> None:
        transport = httpx.MockTransport(lambda request: httpx.Response(200))
        async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
            request = client.build_request("GET", "file:///etc/passwd?token=QUERYSECRET")
            with pytest.raises(PrivateRedirectError) as exc_info:
                await safe_send(client, request)

        assert "QUERYSECRET" not in str(exc_info.value)

    async def test_blocked_redirect_target_message_is_redacted(self, monkeypatch) -> None:
        _stub_resolves_to(monkeypatch, "93.184.216.34")

        def _handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "feeds.example.com":
                return httpx.Response(
                    302,
                    headers={"location": "http://user:s3cr3tpass@127.0.0.1/internal?token=QUERYSECRET"},
                )
            return httpx.Response(200, text="ok")

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
            with pytest.raises(PrivateRedirectError) as exc_info:
                await safe_get(client, "https://feeds.example.com/feed.xml")

        message = str(exc_info.value)
        assert "s3cr3tpass" not in message
        assert "QUERYSECRET" not in message


class TestValidateFeedUrlIsTotal:
    """A malformed URL yields an error string, never an exception (AUG-205)."""

    @pytest.mark.parametrize("url", ["http://[::1", "http://[fe80::1%25]:notaport/", "http://[:::]/x"])
    def test_malformed_url_returns_error(self, url: str) -> None:
        error = validate_feed_url(url)
        assert error is not None
        assert url in error


class TestValidateFeedUrlsBounded:
    """Manual feed lists are deduped and capped before any DNS work (AUG-193)."""

    def test_duplicate_urls_resolved_once(self, monkeypatch) -> None:
        from app import url_validation as uv

        calls: list[str] = []

        def _fake(url: str) -> str | None:
            calls.append(url)
            return None

        monkeypatch.setattr(uv, "validate_feed_url", _fake)
        errors = uv.validate_feed_urls(["https://a.example.com/f"] * 5)

        assert errors == []
        assert calls == ["https://a.example.com/f"]

    def test_list_longer_than_cap_is_rejected(self, monkeypatch) -> None:
        from app import url_validation as uv

        urls = [f"https://a{i}.example.com/f" for i in range(uv.MAX_FEED_URLS_PER_TOPIC + 3)]
        errors = uv.validate_feed_urls(urls)

        assert any(str(uv.MAX_FEED_URLS_PER_TOPIC) in e for e in errors)


class TestValidateOutboundUrl:
    """Shared credential-bearing endpoint gate."""

    def test_rejects_non_http_scheme(self) -> None:
        from app.url_validation import validate_outbound_url

        with pytest.raises(ValueError, match="http"):
            validate_outbound_url("ftp://example.com/x", purpose="the LLM endpoint")

    def test_rejects_public_cleartext_http(self, monkeypatch) -> None:
        from app.url_validation import validate_outbound_url

        _stub_resolves_to(monkeypatch, "93.184.216.34")
        with pytest.raises(ValueError, match="cleartext"):
            validate_outbound_url("http://gateway.example.com/v1", purpose="the LLM endpoint", require_https=True)

    def test_allows_public_https(self, monkeypatch) -> None:
        from app.url_validation import validate_outbound_url

        _stub_resolves_to(monkeypatch, "93.184.216.34")
        validate_outbound_url("https://gateway.example.com/v1", purpose="the LLM endpoint", require_https=True)

    def test_allows_public_cleartext_without_require_https(self, monkeypatch) -> None:
        """Notification targets keep the plain SSRF gate, not a transport policy."""
        from app.url_validation import validate_outbound_url

        _stub_resolves_to(monkeypatch, "93.184.216.34")
        validate_outbound_url("http://hooks.example.com/x", purpose="Notification target")

    def test_allows_local_cleartext_when_private_permitted(self) -> None:
        from app.url_validation import validate_outbound_url

        validate_outbound_url(
            "http://localhost:11434", purpose="the LLM endpoint", allow_private=True, require_https=True
        )

    def test_rejects_private_when_not_permitted(self) -> None:
        from app.url_validation import validate_outbound_url

        with pytest.raises(ValueError, match="private"):
            validate_outbound_url("http://127.0.0.1:8080/v1", purpose="the Exa API")

    def test_allows_resolved_private_cleartext(self, monkeypatch) -> None:
        """A LAN gateway that really resolves private keeps the documented http path."""
        from app.url_validation import validate_outbound_url

        _stub_resolves_to(monkeypatch, "192.168.1.50")
        validate_outbound_url(
            "http://ollama.lan:11434", purpose="the LLM endpoint", allow_private=True, require_https=True
        )

    def test_unresolvable_host_does_not_waive_require_https(self, monkeypatch) -> None:
        """An absent DNS answer is not a private-address verdict.

        ``allow_private`` lets an unresolvable host through the SSRF rule, and
        folding "private" and "unresolvable" into one bool then let it skip the
        cleartext rule as well: one DNS blip sent the provider key and the whole
        prompt to a public endpoint over plain http.
        """
        from app.url_validation import validate_outbound_url

        def _fails(*_args, **_kwargs):
            raise socket.gaierror("name resolution failed")

        monkeypatch.setattr(socket, "getaddrinfo", _fails)
        with pytest.raises(ValueError, match="cleartext"):
            validate_outbound_url(
                "http://llm.corp.example:8000/v1",
                purpose="The LLM base URL",
                allow_private=True,
                require_https=True,
            )

    def test_unresolvable_host_still_allowed_over_https(self, monkeypatch) -> None:
        """The rule is about the transport, so https is unaffected by a DNS blip."""
        from app.url_validation import validate_outbound_url

        def _fails(*_args, **_kwargs):
            raise socket.gaierror("name resolution failed")

        monkeypatch.setattr(socket, "getaddrinfo", _fails)
        validate_outbound_url(
            "https://llm.corp.example:8000/v1",
            purpose="The LLM base URL",
            allow_private=True,
            require_https=True,
        )

    def test_saturated_resolver_does_not_waive_require_https(self, monkeypatch) -> None:
        """A busy pool is even less of a private-address verdict than a NXDOMAIN."""
        from app import url_validation as uv

        def _saturated(*_args, **_kwargs):
            raise uv.ResolverSaturatedError("saturated")

        monkeypatch.setattr(uv, "_getaddrinfo_bounded", _saturated)
        with pytest.raises(ValueError, match="cleartext"):
            uv.validate_outbound_url(
                "http://llm.corp.example:8000/v1",
                purpose="The LLM base URL",
                allow_private=True,
                require_https=True,
            )


class TestIsAbsoluteHttpUrl:
    """Discovered URLs need a real host, not just an http(s) scheme (AUG-182)."""

    @pytest.mark.parametrize(
        "url",
        [
            "https:///path",
            "http:foo",
            "http://",
            "https://",
            "javascript:alert(1)",
            "",
            "//example.com/x",
            "http://example.com:notaport/x",
        ],
    )
    def test_rejects_non_absolute(self, url: str) -> None:
        from app.url_validation import is_absolute_http_url

        assert is_absolute_http_url(url) is False

    @pytest.mark.parametrize(
        "url",
        ["https://example.com", "http://example.com/a?b=c", "https://example.com:8443/x", "http://[::1]:80/x"],
    )
    def test_accepts_absolute(self, url: str) -> None:
        from app.url_validation import is_absolute_http_url

        assert is_absolute_http_url(url) is True
