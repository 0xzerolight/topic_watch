"""Log-hygiene helpers for redacting secret-bearing URLs (OVH-038).

Webhook and notification URLs frequently embed credentials: ``user:token@host``
userinfo, ``?token=...`` query strings, and — for Slack/Discord webhooks — a
long opaque token *as the trailing path segment*. Logging such a URL in full
leaks the secret into log files/aggregators.

``redact_url`` keeps just enough to identify the destination (scheme + host,
plus a short leading path prefix for context) while dropping userinfo, the query
string, and any long path segments that are likely the secret.
"""

from urllib.parse import urlparse

# Path segments shorter than this are treated as routing context (e.g.
# ``services``, ``api``, ``webhooks``) and kept; segments at or above it are
# assumed to be opaque tokens and dropped. Chosen to clear common routing words
# (≤8 chars) while staying below realistic webhook/ntfy tokens.
_MAX_SAFE_SEGMENT = 12

# How many leading "safe" path segments to keep for context before truncating.
_MAX_PREFIX_SEGMENTS = 2


def redact_url(url: str) -> str:
    """Return a log-safe form of ``url``.

    Shows ``scheme://host`` plus a short, non-secret leading path prefix.
    Strips userinfo, query string, fragment, and any long (likely-secret) path
    segments. Never raises — returns ``"****"`` for unparseable input so it is
    safe to call directly inside a log statement.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return "****"

    scheme = parsed.scheme
    host = parsed.hostname  # hostname drops userinfo and port
    if not scheme or not host:
        return "****"

    # hostname strips the RFC-required brackets from IPv6 literals; re-add them
    # so the logged URL stays well-formed.
    if ":" in host:
        host = f"[{host}]"

    base = f"{scheme}://{host}"

    # Walk leading path segments; keep short routing-context ones, stop at the
    # first long (likely-secret) segment.
    segments = [seg for seg in parsed.path.split("/") if seg]
    kept: list[str] = []
    truncated = False
    for seg in segments:
        if len(kept) >= _MAX_PREFIX_SEGMENTS or len(seg) >= _MAX_SAFE_SEGMENT:
            truncated = True
            break
        kept.append(seg)
    if len(kept) < len(segments):
        truncated = True

    if kept:
        base = base + "/" + "/".join(kept)
    if truncated:
        base = base + "/…"
    return base
