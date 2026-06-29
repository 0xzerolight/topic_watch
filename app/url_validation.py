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
import ipaddress
import logging
import re
import socket
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from urllib.parse import urljoin, urlparse

import httpx

from app.log_redaction import redact_url

logger = logging.getLogger(__name__)

# Bound on redirect hops we will follow while re-validating each target.
# Mirrors httpx's own default to keep behaviour familiar.
_MAX_REDIRECTS = 20

# Wall-clock bound (seconds) on a single blocking getaddrinfo. socket.getaddrinfo
# ignores socket.setdefaulttimeout, so we run it under a dedicated executor and
# abandon the lookup past this deadline. Caps how long one crafted slow/non-
# resolving host can occupy a worker (OVH-148); on timeout we fail closed.
_RESOLVE_TIMEOUT = 5.0

# Patterns that indicate a private/reserved network address.
#
# Layer-1 string match is a fast pre-filter for the canonical IPv4 private/
# reserved forms and the two unambiguous IPv6 *literal* forms (loopback ``::1``
# and link-local ``fe80:`` — both contain a colon and so can never match a bare
# hostname). IPv6 ULA (fc00::/7) and IPv4-mapped (``::ffff:``) literals are
# intentionally NOT matched here: a bare ``^f[cd]`` over-blocked legitimate
# fc-/fd- hostnames (OVH-142) and a blanket ``^::ffff:`` over-blocked public
# mapped addresses (OVH-169). Those literals arrive bracketed, so urlparse hands
# us the bare IP and layer-2 (getaddrinfo + ipaddress) classifies them correctly
# — private mapped/ULA still blocked, public mapped allowed.
_PRIVATE_NETLOC_PATTERNS = [
    re.compile(r"^localhost(:\d+)?$", re.IGNORECASE),
    re.compile(r"^127\."),
    re.compile(r"^10\."),
    re.compile(r"^172\.(1[6-9]|2\d|3[01])\."),
    re.compile(r"^192\.168\."),
    re.compile(r"^169\.254\."),
    re.compile(r"^0\.0\.0\.0"),
    re.compile(r"^::1$"),  # IPv6 loopback
    re.compile(r"^fe80:", re.IGNORECASE),  # IPv6 link-local
]

# RFC 6598 carrier-grade NAT range. Not flagged by ipaddress.is_private/.is_reserved,
# so it is checked explicitly — on CGNAT hosts it can reach carrier infrastructure.
_CGNAT_NETWORK = ipaddress.ip_network("100.64.0.0/10")


def _getaddrinfo_bounded(hostname: str, timeout: float) -> list:
    """Run ``socket.getaddrinfo`` with a wall-clock timeout.

    ``getaddrinfo`` is blocking and ignores ``socket.setdefaulttimeout``, so a
    slow/non-resolving host could otherwise pin a worker for the OS resolver's
    full default timeout. Running it in a single-shot executor and waiting only
    ``timeout`` seconds bounds the *caller's* wait (OVH-148). On timeout we raise
    ``TimeoutError`` (handled by the fail-closed caller) WITHOUT joining the
    executor — ``shutdown(wait=False)`` returns immediately so the caller is not
    stuck behind the lookup; the abandoned worker exits on its own once the OS
    resolver itself times out (bounded, not unbounded).
    """
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dns-resolve")
    future = pool.submit(socket.getaddrinfo, hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    try:
        result = future.result(timeout=timeout)
    except FuturesTimeoutError:
        # Abandon without joining: a blocked getaddrinfo would make a waiting
        # shutdown hang for the full OS resolver timeout, defeating the bound.
        pool.shutdown(wait=False)
        raise TimeoutError(f"DNS resolution for {hostname!r} exceeded {timeout}s") from None
    pool.shutdown(wait=False)
    return result


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


def _resolved_ip_is_private(hostname: str) -> bool:
    """Resolve a hostname and check if any resulting IP is private/reserved.

    Returns True on DNS resolution failure (fail-closed): a host we cannot
    resolve cannot be verified as public, so we treat it as blocked rather
    than silently allowing it. This also closes one DNS-rebinding variant
    where resolution fails at check time but later succeeds (to a private
    address) at connect time. The pattern-based check already covers known
    private hostname formats; this layer adds protection against encoding
    bypasses (hex IP, decimal IP, DNS rebinding) that resolve to private
    addresses.

    DNS resolution is bounded by ``_RESOLVE_TIMEOUT`` (OVH-148): a slow lookup
    times out and is treated as unverifiable (blocked) rather than hanging.
    """
    try:
        infos = _getaddrinfo_bounded(hostname, _RESOLVE_TIMEOUT)
        for _family, _type, _proto, _canonname, sockaddr in infos:
            addr = ipaddress.ip_address(sockaddr[0])
            if _addr_is_private(addr):
                return True
            # An IPv4-mapped IPv6 address (e.g. ::ffff:100.64.0.1) keeps
            # version == 6, so the IPv4-only CGNAT gate and most predicates
            # never fire on the wrapper. Unwrap to the embedded IPv4 and
            # re-classify so mapped CGNAT/private/etc. is blocked while a
            # mapped PUBLIC address (::ffff:93.184.216.34) stays allowed
            # (OVH-169 follow-up — do NOT reintroduce a blanket ::ffff: block).
            mapped = getattr(addr, "ipv4_mapped", None)
            if mapped is not None and _addr_is_private(mapped):
                return True
        return False
    except (socket.gaierror, ValueError, OSError):
        # Fail closed: an unresolvable host cannot be verified as public.
        # TimeoutError (raised by _getaddrinfo_bounded) is an OSError subclass,
        # so a bounded-out slow resolver also lands here and is treated as blocked.
        return True


def is_private_url(url: str) -> bool:
    """Check if a URL points to a private/reserved network address.

    Uses a two-layer check: fast regex patterns on the hostname string,
    then DNS resolution to catch alternative IP encodings and rebinding.
    """
    parsed = urlparse(url)
    netloc = parsed.hostname or parsed.netloc
    if not netloc:
        return False
    # Layer 1: fast pattern match
    if any(pattern.search(netloc) for pattern in _PRIVATE_NETLOC_PATTERNS):
        return True
    # Layer 2: resolve and check the actual IP address
    return _resolved_ip_is_private(netloc)


def validate_feed_url(url: str) -> str | None:
    """Validate a single feed URL.

    Returns an error message string if invalid, or None if valid.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return f"Invalid feed URL (must be http or https): {url}"
    if is_private_url(url):
        return f"Feed URL points to a private/reserved address or could not be resolved: {url}"
    return None


def validate_feed_urls(urls: list[str]) -> list[str]:
    """Validate a list of feed URLs.

    Returns a list of error messages (empty if all valid).
    """
    errors = []
    for url in urls:
        error = validate_feed_url(url)
        if error:
            errors.append(error)
    return errors


class PrivateRedirectError(httpx.HTTPError):
    """Raised when a redirect target points to a private/reserved address.

    Subclasses ``httpx.HTTPError`` so existing call sites that catch
    ``httpx.HTTPError`` (e.g. google_news) treat a blocked redirect as a
    fetch failure rather than crashing.
    """


def _is_redirect_status(status_code: int) -> bool:
    return status_code in (301, 302, 303, 307, 308)


async def safe_send(
    client: httpx.AsyncClient,
    request: httpx.Request,
    *,
    max_redirects: int = _MAX_REDIRECTS,
) -> httpx.Response:
    """Send a request, manually following redirects with per-hop SSRF checks.

    The client MUST be configured with ``follow_redirects=False`` (the default
    of this helper assumes httpx will not auto-follow). Each ``Location`` target
    is validated with :func:`is_private_url` BEFORE the next hop is sent, so an
    attacker-controlled public host cannot 3xx-redirect into loopback/RFC-1918.

    The initial request URL is validated with the SAME scheme + private-host
    checks as every redirect hop BEFORE the first send (OVH-140), so a caller
    that forgets the separate :func:`is_private_url` guard can never have this
    helper fetch a private/loopback or non-http(s) initial target.

    Raises :class:`PrivateRedirectError` if the initial URL or any redirect
    target is private/non-http(s), or if the redirect limit is exceeded.
    """
    initial_url = str(request.url)
    if urlparse(initial_url).scheme not in ("http", "https"):
        logger.warning("Blocked request to non-http(s) URL: %s", redact_url(initial_url))
        raise PrivateRedirectError(f"Non-http(s) scheme blocked: {initial_url}")
    # is_private_url does blocking DNS; offload so the event loop is not stalled.
    if await asyncio.to_thread(is_private_url, initial_url):
        logger.warning("Blocked request to private/reserved URL: %s", redact_url(initial_url))
        raise PrivateRedirectError(f"Request to private/reserved address blocked: {initial_url}")

    response = await client.send(request)
    redirects = 0
    while _is_redirect_status(response.status_code):
        location = response.headers.get("location")
        if not location:
            return response
        next_url = urljoin(str(request.url), location)
        # Re-validate the scheme: is_private_url() returns False for URLs with no
        # netloc (e.g. file:///etc/passwd, gopher://...), so a redirect to a
        # non-http(s) scheme would otherwise slip past the private-host check.
        if urlparse(next_url).scheme not in ("http", "https"):
            await response.aclose()
            logger.warning("Blocked redirect to non-http(s) URL: %s", redact_url(next_url))
            raise PrivateRedirectError(f"Redirect to non-http(s) scheme blocked: {next_url}")
        if await asyncio.to_thread(is_private_url, next_url):
            await response.aclose()
            logger.warning("Blocked redirect to private/reserved URL: %s", redact_url(next_url))
            raise PrivateRedirectError(f"Redirect to private/reserved address blocked: {next_url}")
        redirects += 1
        if redirects > max_redirects:
            await response.aclose()
            raise PrivateRedirectError(f"Exceeded maximum of {max_redirects} redirects")
        # 303, and 301/302 in practice, downgrade to GET with no body.
        method = "GET" if response.status_code == 303 else request.method
        new_headers = dict(request.headers)
        new_content = None if method == "GET" else request.content
        if method == "GET":
            new_headers.pop("content-length", None)
            new_headers.pop("content-type", None)
        await response.aclose()
        request = client.build_request(method, next_url, headers=new_headers, content=new_content)
        response = await client.send(request)
    return response


async def safe_get(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    max_redirects: int = _MAX_REDIRECTS,
) -> httpx.Response:
    """GET ``url`` with redirect-target SSRF validation on every hop.

    ``headers`` are merged onto the request (e.g. conditional-GET validators);
    they do not affect host validation, so no SSRF surface is added.
    """
    request = client.build_request("GET", url, headers=headers)
    return await safe_send(client, request, max_redirects=max_redirects)
