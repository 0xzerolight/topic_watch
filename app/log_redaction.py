"""Log-hygiene helpers for redacting secret-bearing URLs (OVH-038).

Webhook and notification URLs frequently embed credentials: ``user:token@host``
userinfo, ``?token=...`` query strings, and — for Slack/Discord webhooks — a
long opaque token *as the trailing path segment*. Logging such a URL in full
leaks the secret into log files/aggregators.

``redact_url`` keeps just enough to identify the destination (scheme + host,
plus a short leading path prefix for context) while dropping userinfo, the query
string, and any long path segments that are likely the secret.
"""

import hashlib
from urllib.parse import urlparse

# Path segments shorter than this are treated as routing context (e.g.
# ``services``, ``api``, ``webhooks``) and kept; segments at or above it are
# assumed to be opaque tokens and dropped. Chosen to clear common routing words
# (≤8 chars) while staying below realistic webhook/ntfy tokens.
_MAX_SAFE_SEGMENT = 12

# How many leading "safe" path segments to keep for context before truncating.
_MAX_PREFIX_SEGMENTS = 2

# Schemes whose authority (host) component is a normal server name — safe to
# show, because the *path* carries whatever credential the service uses.
# Every other scheme is an Apprise notifier where the authority itself is
# frequently the secret: ``pover://USERKEY@APPTOKEN`` puts the app token in
# the host, ``ntfy://private-topic`` puts the whole private topic name there.
# ``urlparse().hostname`` cannot tell "host" from "capability token" apart, so
# non-HTTP schemes get a fingerprint instead of their authority (AUG-248).
#
# The generic-HTTP aliases (``app.notifications._GENERIC_HTTP_SCHEMES``) are
# the one exception: Topic Watch resolves them to a real ``http(s)`` request
# before ever sending one (SSRF-gated in ``_generic_http_target``), so their
# host is a genuine server address, not a token.
_HTTP_SCHEMES = frozenset({"http", "https", "json", "jsons", "form", "forms", "xml", "xmls"})


def _scheme_fingerprint(scheme: str, url: str) -> str:
    """Identify a non-HTTP target by provider (scheme) plus a stable, non-reversible tag.

    Lets repeated log lines for the same misconfigured target be recognized as
    the same target — useful for a retry/backoff story — without exposing the
    capability-bearing host or path that names it.
    """
    digest = hashlib.sha256(url.encode("utf-8", errors="surrogateescape")).hexdigest()[:8]
    return f"{scheme}://***{digest}"


def redact_url(url: str) -> str:
    """Return a log-safe form of ``url``.

    For ``http(s)`` targets, shows ``scheme://host`` plus a short, non-secret
    leading path prefix, stripping userinfo, query string, fragment, and any
    long (likely-secret) path segments. Non-HTTP schemes (most Apprise
    notifiers) instead get a provider tag plus a non-reversible fingerprint,
    because their authority itself commonly carries the secret (AUG-248).
    Never raises — returns ``"****"`` for unparseable input so it is safe to
    call directly inside a log statement.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return "****"

    scheme = parsed.scheme
    if not scheme:
        return "****"
    if scheme.lower() not in _HTTP_SCHEMES:
        return _scheme_fingerprint(scheme, url)

    host = parsed.hostname  # hostname drops userinfo and port
    if not host:
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
