"""URL validation utilities shared across the application.

SSRF defense and its residual risk: :func:`is_private_url` resolves DNS at
*check time* and classifies the resulting IPs, while httpx re-resolves the
hostname at *connect time*. A TOCTOU / DNS-rebinding window therefore remains
between validation and the actual fetch -- an attacker controlling DNS could
return a public IP during validation and a private one at connect. Per-hop
redirect re-validation (:func:`safe_send`) and fail-closed resolution
(:func:`_resolved_ip_is_private`) reduce but do not eliminate this window.
Closing it fully would require a pinned-IP / custom-resolver transport, which
risks breaking HTTPS feed fetching (SNI / cert verification); this is an
accepted limitation for a single-user self-hosted tool.
"""

import asyncio
import enum
import ipaddress
import logging
import re
import socket
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from urllib.parse import urlparse

import httpx

from app.log_redaction import redact_url

logger = logging.getLogger(__name__)

# Bound on redirect hops we will follow while re-validating each target.
# Mirrors httpx's own default to keep behaviour familiar.
_MAX_REDIRECTS = 20

# Hard ceiling on the decoded bytes a single response body may contribute
# (AUG-006). Untrusted publishers control both the length and the compression
# ratio of what they return, and a normal httpx send buffers the whole decoded
# body before the caller sees a status line -- one hostile feed could exhaust the
# documented 512 MiB container. Enforced while streaming, so an oversize body is
# abandoned mid-transfer rather than downloaded and then discarded. A module
# constant, not a setting: it is a liveness bound on a shared runtime, and 10 MiB
# is roughly two orders of magnitude above the largest real RSS/article payload.
MAX_RESPONSE_BYTES = 10 * 1024 * 1024

# Feed URLs one topic may carry. The manual-feed textarea and OPML import both
# enforce it, because every stored URL becomes one DNS resolution and one HTTP
# fetch on every single check (AUG-193 / AUG-012).
MAX_FEED_URLS_PER_TOPIC = 20

# Bound on concurrent feed-URL validations (each does a blocking getaddrinfo).
# Caps wall-clock time and resolver fan-out for a bulk validation so a handful of
# slow/unresolvable hosts no longer serialize (OVH-053).
VALIDATION_CONCURRENCY = 16

# Wall-clock bound (seconds) on a single blocking getaddrinfo. socket.getaddrinfo
# ignores socket.setdefaulttimeout, so we run it on a dedicated pool and abandon
# the lookup past this deadline. Caps how long one crafted slow/non-resolving host
# can occupy a worker (OVH-148); on timeout we fail closed.
_RESOLVE_TIMEOUT = 5.0

# Process-wide resolver pool (AUG-013). A per-lookup executor bounded only the
# *caller's* wait: ``shutdown(wait=False)`` cannot cancel a running getaddrinfo,
# so concurrent or bulk work (an OPML import, a pasted feed list) piled up waves
# of abandoned worker threads with no process-wide ceiling. One fixed pool plus
# admission control puts a real bound on it: a lookup runs only once it holds a
# slot, an abandoned lookup keeps its slot until the OS resolver returns, and a
# caller that cannot get a slot within its own deadline fails closed instead of
# spawning another thread.
_RESOLVER_POOL_SIZE = 16
_resolver_pool = ThreadPoolExecutor(max_workers=_RESOLVER_POOL_SIZE, thread_name_prefix="dns-resolve")
_resolver_slots = threading.BoundedSemaphore(_RESOLVER_POOL_SIZE)

# ``localhost`` is a name, not an address, so no IP parse can catch it. Every
# other private form is a real IP literal and is classified by ``ipaddress``
# below: prefix regexes such as ``^127\\.`` also matched public hostnames like
# ``127.example.com`` and blocked them before DNS ever ran (AUG-189).
_LOCALHOST_RE = re.compile(r"^localhost(:\d+)?$", re.IGNORECASE)

# RFC 6598 carrier-grade NAT range. Not flagged by ipaddress.is_private/.is_reserved,
# so it is checked explicitly — on CGNAT hosts it can reach carrier infrastructure.
_CGNAT_NETWORK = ipaddress.ip_network("100.64.0.0/10")


class _Destination(enum.Enum):
    """What resolution actually said about a host.

    ``PRIVATE`` is a verdict about the destination; ``UNRESOLVABLE`` is the
    absence of one. Both are blocked, so most callers only want the bool that
    :func:`is_private_url` gives them — but a rule that exists to protect a
    PUBLIC destination must not be waived by the absence of an answer.
    """

    PUBLIC = "public"
    PRIVATE = "private"
    UNRESOLVABLE = "unresolvable"


class ResolverSaturatedError(httpx.HTTPError):
    """Raised when no resolver slot came free within the caller's deadline.

    Every other blocked outcome is something we learned about the destination;
    this one is the shared pool being busy with unrelated work — a bulk OPML
    import holds all sixteen slots for as long as it runs — and says nothing
    about the host at all. Blocking is still right, but a caller that keeps a
    per-feed failure streak has to charge this to the run instead, or an
    untouched healthy feed walks into exponential backoff.

    An ``httpx.HTTPError`` for the same reason as :class:`PrivateRedirectError`:
    to a call site with no special handling it is an ordinary fetch failure.
    """


def _getaddrinfo_bounded(hostname: str, timeout: float) -> list:
    """Run ``socket.getaddrinfo`` with a wall-clock timeout on the shared pool.

    ``getaddrinfo`` is blocking and ignores ``socket.setdefaulttimeout``, so a
    slow/non-resolving host could otherwise pin a worker for the OS resolver's
    full default timeout. The whole call — waiting for a resolver slot plus the
    lookup itself — is bounded by ``timeout`` (OVH-148), and the pool bounds how
    many resolver threads can exist at once (AUG-013).

    Raises ``TimeoutError`` when the lookup itself ran out of time and
    :class:`ResolverSaturatedError` when no slot came free — two different
    events, because only the first says anything about ``hostname``.
    """
    deadline = time.monotonic() + timeout
    if not _resolver_slots.acquire(timeout=timeout):
        # Every resolver thread is occupied, most likely by earlier abandoned
        # lookups. Spawning another thread is exactly what the bound exists to
        # prevent, so this is unverifiable -> blocked, like any other timeout.
        raise ResolverSaturatedError(f"DNS resolver saturated; {hostname!r} not resolved within {timeout}s")
    try:
        future = _resolver_pool.submit(socket.getaddrinfo, hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except BaseException:
        _resolver_slots.release()
        raise
    # The slot is returned when the lookup actually finishes, NOT when we stop
    # waiting for it: an abandoned getaddrinfo still owns its thread, and the
    # bound is only real if it keeps owning its slot too.
    future.add_done_callback(lambda _future: _resolver_slots.release())
    try:
        return future.result(timeout=max(deadline - time.monotonic(), 0.0))
    except FuturesTimeoutError:
        raise TimeoutError(f"DNS resolution for {hostname!r} exceeded {timeout}s") from None


def _addr_is_private(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Classify a single resolved IP as private/reserved/CGNAT.

    Applies the standard ipaddress predicates plus the explicit RFC 6598 CGNAT
    range (not covered by is_private/.is_reserved). Used for both the resolved
    address and any IPv4 unwrapped from an IPv4-mapped IPv6 address.
    """
    return bool(
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_unspecified
        or (addr.version == 4 and addr in _CGNAT_NETWORK)
    )


def _addr_or_mapped_is_private(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """``_addr_is_private`` plus the IPv4 embedded in an IPv4-mapped IPv6 address.

    An IPv4-mapped IPv6 address (e.g. ``::ffff:100.64.0.1``) keeps ``version == 6``,
    so the IPv4-only CGNAT gate and most predicates never fire on the wrapper.
    Unwrapping re-classifies mapped CGNAT/private/etc. as private while a mapped
    PUBLIC address (``::ffff:93.184.216.34``) stays allowed (OVH-169 follow-up —
    do NOT reintroduce a blanket ``::ffff:`` block).
    """
    if _addr_is_private(addr):
        return True
    mapped = getattr(addr, "ipv4_mapped", None)
    return mapped is not None and _addr_is_private(mapped)


def _ip_literal(hostname: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """Parse ``hostname`` as a complete IP literal, or ``None`` if it is a name."""
    try:
        return ipaddress.ip_address(hostname.strip("[]"))
    except ValueError:
        return None


def _classify_hostname(hostname: str) -> _Destination:
    """Resolve a hostname and say what it points at.

    ``UNRESOLVABLE`` on DNS resolution failure (fail-closed): a host we cannot
    resolve cannot be verified as public, so we treat it as blocked rather
    than silently allowing it. This also closes one DNS-rebinding variant
    where resolution fails at check time but later succeeds (to a private
    address) at connect time. This layer also covers encoding bypasses (hex IP,
    decimal IP, DNS rebinding) that resolve to private addresses.

    DNS resolution is bounded by ``_RESOLVE_TIMEOUT`` (OVH-148): a slow lookup
    times out and is treated as unverifiable (blocked) rather than hanging.

    Raises :class:`ResolverSaturatedError` when the lookup never ran because no
    resolver slot came free — the one outcome that is not about this host.
    """
    try:
        infos = _getaddrinfo_bounded(hostname, _RESOLVE_TIMEOUT)
        private = any(_addr_or_mapped_is_private(ipaddress.ip_address(sockaddr[0])) for *_head, sockaddr in infos)
    except (socket.gaierror, ValueError, OSError):
        # Fail closed: an unresolvable host cannot be verified as public.
        # TimeoutError (raised by _getaddrinfo_bounded) is an OSError subclass,
        # so a bounded-out slow resolver also lands here and is treated as blocked.
        return _Destination.UNRESOLVABLE
    return _Destination.PRIVATE if private else _Destination.PUBLIC


def _resolved_ip_is_private(hostname: str) -> bool:
    """``_classify_hostname`` as the bool that :func:`is_private_url` needs."""
    try:
        return _classify_hostname(hostname) is not _Destination.PUBLIC
    except ResolverSaturatedError:
        return True


def _classify_url(url: str) -> _Destination:
    """Classify a URL's destination. Raises :class:`ResolverSaturatedError`.

    Three layers, cheapest first: the ``localhost`` name, a complete IP literal
    classified by ``ipaddress``, and otherwise DNS resolution of the name. A URL
    with no host at all resolves to nothing and reaches nothing, so it is not
    treated as private; the shape gate for that is :func:`is_absolute_http_url`.
    """
    parsed = urlparse(url)
    netloc = parsed.hostname or parsed.netloc
    if not netloc:
        return _Destination.PUBLIC
    if _LOCALHOST_RE.match(netloc):
        return _Destination.PRIVATE
    literal = _ip_literal(netloc)
    if literal is not None:
        return _Destination.PRIVATE if _addr_or_mapped_is_private(literal) else _Destination.PUBLIC
    return _classify_hostname(netloc)


def is_private_url(url: str) -> bool:
    """Check if a URL points to a private/reserved network address.

    Total: a saturated resolver is unverifiable and therefore blocked, same as an
    unresolvable host. Call sites that must tell "we could not check" apart from
    "we checked" use :func:`_classify_url` and handle
    :class:`ResolverSaturatedError` themselves.
    """
    try:
        return _classify_url(url) is not _Destination.PUBLIC
    except ResolverSaturatedError:
        return True


def is_absolute_http_url(url: str) -> bool:
    """True if ``url`` is an absolute http(s) URL with a real host.

    A scheme check alone accepts ``https:///path`` and ``http:foo`` — values with
    no fetchable authority that were stored as article links, failed extraction
    on every check, and shipped in notifications as an unusable destination
    (AUG-182). Purely structural and DNS-free: this is the shape gate that runs
    before the SSRF gate, not a replacement for it.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if parsed.scheme.lower() not in ("http", "https"):
        return False
    try:
        _ = parsed.port  # a non-numeric or out-of-range port makes this raise
    except ValueError:
        return False
    return bool(parsed.hostname)


def validate_feed_url(url: str) -> str | None:
    """Validate a single feed URL.

    Returns an error message string if invalid, or None if valid. Total by
    contract (AUG-205): a malformed URL that makes ``urlparse``/``is_private_url``
    raise is reported as an invalid candidate, so one bad outline in an OPML file
    or one bad line in a pasted list cannot abort the whole import.
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return f"Invalid feed URL (must be http or https): {url}"
        if is_private_url(url):
            return f"Feed URL points to a private/reserved address or could not be resolved: {url}"
    except Exception:
        logger.warning("Rejecting unparseable feed URL: %s", redact_url(url), exc_info=True)
        return f"Invalid feed URL (could not be parsed): {url}"
    return None


def validate_urls_concurrently(urls: list[str], validator, workers: int = VALIDATION_CONCURRENCY) -> dict:
    """Validate a deduped URL list concurrently, returning ``{url: error|None}``.

    ``validator`` is the per-URL check (``validate_feed_url`` and its test
    doubles). Bounded concurrency caps wall-clock time and resolver fan-out for a
    bulk validation; the SSRF invariant is preserved because every URL still
    flows through ``validator``. A tiny/empty list skips the pool entirely.
    """
    if not urls:
        return {}
    if len(urls) == 1:
        return {urls[0]: validator(urls[0])}

    with ThreadPoolExecutor(max_workers=min(workers, len(urls))) as pool:
        results = list(pool.map(validator, urls))
    return dict(zip(urls, results, strict=True))


def validate_feed_urls(urls: list[str]) -> list[str]:
    """Validate a list of feed URLs.

    Returns a list of error messages (empty if all valid). Duplicates are
    resolved once and the list is capped at ``MAX_FEED_URLS_PER_TOPIC`` before
    any DNS work, so a pasted list cannot turn one form submission into an
    unbounded run of blocking lookups — or into that many fetches on every later
    check (AUG-193).
    """
    deduped = list(dict.fromkeys(urls))
    if len(deduped) > MAX_FEED_URLS_PER_TOPIC:
        return [f"Too many feed URLs: {len(deduped)} (maximum {MAX_FEED_URLS_PER_TOPIC} per topic)"]
    errors_by_url = validate_urls_concurrently(deduped, validate_feed_url)
    return [errors_by_url[url] for url in deduped if errors_by_url[url]]


def validate_outbound_url(
    url: str,
    *,
    purpose: str,
    allow_private: bool = False,
    require_https: bool = False,
) -> None:
    """Gate one operator-configured outbound endpoint. Raises ``ValueError``.

    The single policy behind the Apprise, LLM and Exa call sites, which each had
    their own (or no) check before (AUG-004 / AUG-333 / AUG-304). Rules:

    1. the scheme must be http(s);
    2. a private/reserved/unresolvable destination is refused unless
       ``allow_private``;
    3. with ``require_https``, https is required unless the destination actually
       RESOLVED to a private address — an endpoint that carries our own API key
       must never be reached in cleartext across the internet.

    Rule 3 turns on the resolution verdict, not on "rule 2 would have blocked
    it". An unresolvable host is blocked by rule 2 for want of proof, and with
    ``allow_private`` that block is waived; letting the same missing proof waive
    rule 3 as well meant one DNS blip put a public endpoint on plain http, and
    the caller's cache made it permanent.

    The two flags are what the three callers actually differ on, and both cases
    are real: ``allow_private=True`` keeps the documented local LLM path working
    (``http://localhost:11434``, ``http://host.docker.internal:11434``), where
    cleartext on your own machine is the intended setup, while ``require_https``
    is off for notification targets because the webhook sender already accepts
    plain-http public endpoints and this is the SSRF gate, not a transport policy.

    Performs blocking DNS via :func:`_classify_url` — call it through
    ``asyncio.to_thread`` from async code.
    """
    scheme = urlparse(url).scheme
    if scheme not in ("http", "https"):
        raise ValueError(f"{purpose} must be an http(s) URL, got scheme {scheme!r}")
    try:
        destination = _classify_url(url)
    except ResolverSaturatedError:
        # Nothing was asked of DNS, so nothing is known: the same standing as a
        # host that would not resolve, and refused by the same rules.
        destination = _Destination.UNRESOLVABLE
    if destination is not _Destination.PUBLIC and not allow_private:
        raise ValueError(f"{purpose} points to a private/reserved address or could not be resolved")
    if require_https and scheme != "https" and destination is not _Destination.PRIVATE:
        raise ValueError(f"{purpose} would be sent in cleartext to a host that is not private; use https")


class PrivateRedirectError(httpx.HTTPError):
    """Raised when a redirect target points to a private/reserved address.

    Subclasses ``httpx.HTTPError`` so existing call sites that catch
    ``httpx.HTTPError`` (e.g. google_news) treat a blocked redirect as a
    fetch failure rather than crashing.
    """


class ResponseTooLargeError(httpx.HTTPError):
    """Raised when a response body exceeds the caller's byte budget (AUG-006).

    An ``httpx.HTTPError`` for the same reason as ``PrivateRedirectError``: to
    every fetch call site an oversize body is a failed fetch, not a crash.
    """


# Headers that describe the framing of the body we just decoded and re-buffered.
# Carrying ``content-encoding`` onto the rebuilt response would make httpx try to
# decompress already-decompressed bytes; the lengths are recomputed from the body.
_BODY_FRAMING_HEADERS = frozenset({"content-encoding", "content-length", "transfer-encoding"})

# Content codecs we can decode under a hard output bound. ``zlib`` is the only
# incremental decompressor whose API takes a ``max_length``; anything else httpx
# would hand a whole raw chunk to an unbounded ``decompress()``. ``deflate`` is
# zlib-wrapped or raw in the wild, so both window sizes are tried in httpx's own
# order. Any other codec is passed through undecoded -- which is what httpx itself
# does with an encoding it has no decoder for, and what our own Accept-Encoding
# means we should never be sent.
_BOUNDED_CODECS: dict[str, tuple[int, ...]] = {
    "gzip": (zlib.MAX_WBITS | 16,),
    "deflate": (zlib.MAX_WBITS, -zlib.MAX_WBITS),
}


class _BoundedZlibDecoder:
    """Incremental gzip/deflate decoder that cannot allocate past a budget.

    httpx decodes with ``zlib.decompressobj().decompress(data)`` -- no
    ``max_length`` -- so the entire expansion of one 64 KiB raw chunk is
    materialised inside a single iteration step, before any caller-side counter
    can run. Passing ``max_length`` makes the caller's remaining budget the real
    ceiling on every allocation. Output past the budget is simply not produced;
    the caller raises rather than resuming, so the dropped ``unconsumed_tail`` is
    never wanted.
    """

    def __init__(self, wbits: tuple[int, ...]) -> None:
        self._decompressor = zlib.decompressobj(wbits[0])
        self._fallback = wbits[1:]

    def decompress(self, data: bytes, max_length: int) -> bytes:
        # The alternate window size is only ever tried on the first chunk, as in
        # httpx: retrying mid-stream would decode from a reset dictionary.
        fallback, self._fallback = self._fallback, ()
        try:
            return self._decompressor.decompress(data, max_length)
        except zlib.error as exc:
            if not fallback:
                raise httpx.DecodingError(str(exc)) from exc
            try:
                self._decompressor = zlib.decompressobj(fallback[0])
                return self._decompressor.decompress(data, max_length)
            except zlib.error as retry_exc:
                raise httpx.DecodingError(str(retry_exc)) from retry_exc


async def _read_capped(response: httpx.Response, request: httpx.Request, max_bytes: int) -> httpx.Response:
    """Buffer a streamed response body, aborting once it exceeds ``max_bytes``.

    Bounds the bytes on the wire AND the DECODED bytes at ``max_bytes``, so a
    small compressed body that would expand to gigabytes is stopped at the same
    budget as a plainly large one -- and stopped INSIDE the decoder rather than
    after it, because counting between chunks cannot bound a decompressor that is
    itself unbounded. The wire bound also stops a stream that decodes to nothing
    (empty deflate blocks) from being read forever. The connection is closed as
    soon as either budget is crossed, so the rest of a hostile body is never
    pulled.

    Stacked ``Content-Encoding`` (``gzip, gzip, gzip``) is refused outright: httpx
    answers it with a MultiDecoder whose layers multiply -- 298 bytes on the wire
    reached gigabytes -- and no legitimate publisher nests compression.

    Returns an equivalent non-streaming response, because every caller reads
    ``.text`` / ``.content`` after the fact.
    """
    codecs = [value.strip().lower() for value in response.headers.get_list("content-encoding", split_commas=True)]
    codecs = [value for value in codecs if value and value != "identity"]
    if len(codecs) > 1:
        await response.aclose()
        raise ResponseTooLargeError(f"Stacked content-encoding refused: {redact_url(str(request.url))}")
    wbits = _BOUNDED_CODECS.get(codecs[0]) if codecs else None
    decoder = _BoundedZlibDecoder(wbits) if wbits is not None else None

    oversize = f"Response body exceeded the {max_bytes}-byte limit: {redact_url(str(request.url))}"
    chunks: list[bytes] = []
    total = 0
    wire_total = 0
    try:
        if response.is_stream_consumed:
            # The transport handed back a body httpx had already buffered and
            # decoded -- it does that for any response built from in-memory
            # content. There is no stream left to bound, so the size check is all
            # that remains. A live transport always streams here.
            chunks.append(response.content)
            total = len(response.content)
            if total > max_bytes:
                raise ResponseTooLargeError(oversize)
        else:
            async for raw in response.aiter_raw():
                wire_total += len(raw)
                if wire_total > max_bytes:
                    raise ResponseTooLargeError(oversize)
                # ``total <= max_bytes`` here, so max_length is >= 1 -- a zlib
                # max_length of 0 would mean "unbounded". The extra byte is what
                # makes crossing the budget observable.
                decoded = decoder.decompress(raw, max_bytes - total + 1) if decoder is not None else raw
                total += len(decoded)
                if total > max_bytes:
                    raise ResponseTooLargeError(oversize)
                if decoded:
                    chunks.append(decoded)
    finally:
        await response.aclose()

    headers = [
        (key, value) for key, value in response.headers.multi_items() if key.lower() not in _BODY_FRAMING_HEADERS
    ]
    return httpx.Response(
        status_code=response.status_code,
        headers=headers,
        content=b"".join(chunks),
        request=request,
        extensions=response.extensions,
    )


async def safe_send(
    client: httpx.AsyncClient,
    request: httpx.Request,
    *,
    max_redirects: int = _MAX_REDIRECTS,
    max_bytes: int = MAX_RESPONSE_BYTES,
) -> httpx.Response:
    """Send a request, manually following redirects with per-hop SSRF checks.

    Every hop is sent with ``follow_redirects=False`` regardless of how the client
    was configured, so the per-hop checks cannot be lost to a caller's client
    setting. Each redirect target is validated with :func:`is_private_url` BEFORE
    the next hop is sent, so an attacker-controlled public host cannot 3xx-redirect
    into loopback/RFC-1918.

    The initial request URL is validated with the SAME scheme + private-host
    checks as every redirect hop BEFORE the first send (OVH-140), so a caller
    that forgets the separate :func:`is_private_url` guard can never have this
    helper fetch a private/loopback or non-http(s) initial target.

    Each hop is sent with ``stream=True`` and only the FINAL body is buffered,
    under a hard ``max_bytes`` budget (AUG-006): redirect bodies are closed
    unread, and an oversize final body is abandoned mid-transfer.

    The next hop is httpx's own prepared ``next_request`` rather than a copy of
    the previous request's headers (AUG-005). httpx applies the method change and
    strips origin-bound credentials — the Authorization header it derives from
    URL userinfo, the Cookie header, and the stale Host — when the target is a
    different origin; rebuilding from ``request.headers`` forwarded all three.

    Raises :class:`PrivateRedirectError` if the initial URL or any redirect
    target is private/non-http(s), or if the redirect limit is exceeded,
    :class:`ResponseTooLargeError` if the body exceeds ``max_bytes``, and
    :class:`ResolverSaturatedError` if no resolver slot came free — kept separate
    so a caller that keeps a failure streak does not charge it to the host.
    """
    initial_url = str(request.url)
    if urlparse(initial_url).scheme not in ("http", "https"):
        logger.warning("Blocked request to non-http(s) URL: %s", redact_url(initial_url))
        raise PrivateRedirectError(f"Non-http(s) scheme blocked: {initial_url}")
    # Classification does blocking DNS; offload so the event loop is not stalled.
    # _classify_url rather than is_private_url so a saturated resolver -- which is
    # about us, not about this host -- leaves as its own error instead of being
    # flattened into "this URL is private".
    if await asyncio.to_thread(_classify_url, initial_url) is not _Destination.PUBLIC:
        logger.warning("Blocked request to private/reserved URL: %s", redact_url(initial_url))
        raise PrivateRedirectError(f"Request to private/reserved address blocked: {initial_url}")

    sent = request
    # follow_redirects is pinned per send, not left to the client: client.send()
    # inherits client.follow_redirects, and a caller that built its client without
    # the flag had httpx follow the chain internally -- next_request stays None and
    # the loop below returns the final (possibly private) body as a normal result.
    response = await client.send(sent, stream=True, follow_redirects=False)
    redirects = 0
    while True:
        next_request = response.next_request
        if next_request is None:
            return await _read_capped(response, sent, max_bytes)
        next_url = str(next_request.url)
        # Re-validate the scheme: is_private_url() returns False for URLs with no
        # netloc (e.g. file:///etc/passwd, gopher://...), so a redirect to a
        # non-http(s) scheme would otherwise slip past the private-host check.
        if urlparse(next_url).scheme not in ("http", "https"):
            await response.aclose()
            logger.warning("Blocked redirect to non-http(s) URL: %s", redact_url(next_url))
            raise PrivateRedirectError(f"Redirect to non-http(s) scheme blocked: {next_url}")
        try:
            hop = await asyncio.to_thread(_classify_url, next_url)
        except BaseException:
            await response.aclose()
            raise
        if hop is not _Destination.PUBLIC:
            await response.aclose()
            logger.warning("Blocked redirect to private/reserved URL: %s", redact_url(next_url))
            raise PrivateRedirectError(f"Redirect to private/reserved address blocked: {next_url}")
        redirects += 1
        if redirects > max_redirects:
            await response.aclose()
            raise PrivateRedirectError(f"Exceeded maximum of {max_redirects} redirects")
        # Nothing of the redirect body is read; closing it here discards it.
        await response.aclose()
        sent = next_request
        response = await client.send(sent, stream=True, follow_redirects=False)


async def safe_get(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    max_redirects: int = _MAX_REDIRECTS,
    max_bytes: int = MAX_RESPONSE_BYTES,
) -> httpx.Response:
    """GET ``url`` with redirect-target SSRF validation on every hop.

    ``headers`` are merged onto the request (e.g. conditional-GET validators);
    they do not affect host validation, so no SSRF surface is added.
    """
    request = client.build_request("GET", url, headers=headers)
    return await safe_send(client, request, max_redirects=max_redirects, max_bytes=max_bytes)
