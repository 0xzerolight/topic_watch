"""URL validation utilities shared across the application."""

import re
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
    re.compile(r"^\[::1\]"),
    re.compile(r"^\[fd"),  # IPv6 ULA
    re.compile(r"^\[fe80:", re.IGNORECASE),  # IPv6 link-local
]


def is_private_url(url: str) -> bool:
    """Check if a URL points to a private/reserved network address."""
    parsed = urlparse(url)
    netloc = parsed.hostname or parsed.netloc
    if not netloc:
        return False
    return any(pattern.search(netloc) for pattern in _PRIVATE_NETLOC_PATTERNS)


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
