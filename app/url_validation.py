"""URL validation utilities shared across the application."""

import ipaddress
import re
import socket
from urllib.parse import urlparse

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
    re.compile(r"^fd", re.IGNORECASE),  # IPv6 ULA
    re.compile(r"^fe80:", re.IGNORECASE),  # IPv6 link-local
]


def _resolved_ip_is_private(hostname: str) -> bool:
    """Resolve a hostname and check if any resulting IP is private/reserved.

    Returns False on DNS resolution failure (fail-open). The pattern-based
    check already covers known private hostname formats. This layer adds
    protection against encoding bypasses (hex IP, decimal IP, DNS rebinding)
    that happen to resolve to private addresses.
    """
    try:
        infos = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        for _family, _type, _proto, _canonname, sockaddr in infos:
            addr = ipaddress.ip_address(sockaddr[0])
            if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
                return True
        return False
    except (socket.gaierror, ValueError, OSError):
        return False  # fail open: DNS failure is handled by the HTTP client


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
        return f"Feed URL points to a private/reserved address: {url}"
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
