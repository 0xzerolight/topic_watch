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

import ipaddress
import logging
import re
import socket
from urllib.parse import urljoin, urlparse

import httpx

logger = logging.getLogger(__name__)

# Bound on redirect hops we will follow while re-validating each target.
# Mirrors httpx's own default to keep behaviour familiar.
_MAX_REDIRECTS = 20

# Patterns that indicate a private/reserved network address
_PRIVATE_NETLOC_PATTERNS = [
    re.compile(r"^localhost(:\d+)?$", re.IGNORECASE),
    re.compile(r"^127\."),
    re.compile(r"^10\."),
    re.compile(r"^172\.(1[6-9]|2\d|3[01])\."),
    re.compile(r"^192\.168\."),
    re.compile(r"^169\.254\."),
    re.compile(r"^0\.0\.0\.0"),
    re.compile(r"^::1$"),  # IPv6 loopback
    re.compile(r"^::ffff:", re.IGNORECASE),  # IPv6-mapped IPv4
    re.compile(r"^f[cd]", re.IGNORECASE),  # IPv6 ULA (fc00::/7)
    re.compile(r"^fe80:", re.IGNORECASE),  # IPv6 link-local
]

# RFC 6598 carrier-grade NAT range. Not flagged by ipaddress.is_private/.is_reserved,
# so it is checked explicitly — on CGNAT hosts it can reach carrier infrastructure.
_CGNAT_NETWORK = ipaddress.ip_network("100.64.0.0/10")


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
    """
    try:
        infos = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        for _family, _type, _proto, _canonname, sockaddr in infos:
            addr = ipaddress.ip_address(sockaddr[0])
            if (
                addr.is_private
                or addr.is_loopback
                or addr.is_link_local
                or addr.is_reserved
                or (addr.version == 4 and addr in _CGNAT_NETWORK)
            ):
                return True
        return False
    except (socket.gaierror, ValueError, OSError):
        return True  # fail closed: an unresolvable host cannot be verified as public


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

    Raises :class:`PrivateRedirectError` if any redirect target is private or if
    the redirect limit is exceeded.
    """
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
            logger.warning("Blocked redirect to non-http(s) URL: %s", next_url)
            raise PrivateRedirectError(f"Redirect to non-http(s) scheme blocked: {next_url}")
        if is_private_url(next_url):
            await response.aclose()
            logger.warning("Blocked redirect to private/reserved URL: %s", next_url)
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
    max_redirects: int = _MAX_REDIRECTS,
) -> httpx.Response:
    """GET ``url`` with redirect-target SSRF validation on every hop."""
    request = client.build_request("GET", url)
    return await safe_send(client, request, max_redirects=max_redirects)
