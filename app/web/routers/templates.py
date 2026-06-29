"""Shared Jinja2 template environment and template filters for web routers.

Centralizes the single ``Jinja2Templates`` instance plus the custom filters
so every router renders against the same environment. The filter helper
functions are module-level (and importable) because they are unit-tested
directly.
"""

import json as json_mod
import re
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from fastapi.templating import Jinja2Templates
from markdown_it import MarkdownIt
from markupsafe import Markup, escape

from app import __version__
from app.log_redaction import redact_url

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent.parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))

templates.env.globals["version"] = __version__

# Markdown renderer for LLM-generated knowledge summaries. ``html=False`` escapes
# any raw HTML in the (article-derived) source and rejects unsafe link schemes
# (javascript:/data:), so no separate HTML sanitizer is needed. Images are
# disabled and hard breaks are off — list structure is restored by
# ``_normalize_markdown`` instead. Built once and shared; ``render()`` is
# stateless per call, so it is safe across concurrent requests.
_MD = MarkdownIt("commonmark", {"html": False, "linkify": False, "breaks": False})
_MD.disable("image")
_MD_LABEL_RE = re.compile(r"^\s*\*\*[^*]+\*\*")
_MD_LIST_RE = re.compile(r"^\s*([-*+]|\d+\.)\s+")


def _timeago(dt: datetime) -> str:
    """Format a datetime as a human-readable relative time."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    now = datetime.now(UTC)
    diff = now - dt
    seconds = int(diff.total_seconds())
    if seconds < 0:
        return "just now"
    if seconds < 60:
        return "just now"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    if days < 30:
        return f"{days}d ago"
    return dt.strftime("%Y-%m-%d")


def _sanitize_error(error_message: str | None) -> Markup:
    """Format error messages for display, collapsing long tracebacks."""
    if not error_message:
        return Markup("<p>An unknown error occurred.</p>")

    if len(error_message) < 200:
        return Markup(f"<p>{escape(error_message)}</p>")

    # Extract last non-empty line as the summary (usually the actual error)
    lines = error_message.strip().splitlines()
    summary = ""
    for line in reversed(lines):
        stripped = line.strip()
        if stripped:
            summary = stripped
            break
    if not summary:
        summary = error_message[:100] + "..."

    escaped_summary = escape(summary)
    escaped_full = escape(error_message)

    return Markup(
        f"<p>{escaped_summary}</p>"
        f"<details><summary><small>Show full error</small></summary>"
        f"<pre><code>{escaped_full}</code></pre></details>"
    )


def _normalize_markdown(text: str) -> str:
    """Insert blank lines so the LLM's label+bullet markdown parses correctly.

    The knowledge prompt emits ``**Label:**`` headers immediately followed by
    ``-`` bullets with no blank line between them. Raw CommonMark then merges
    adjacent labels into one paragraph and swallows the next label into the
    preceding list item. Inserting a blank line before each label line and
    before each list run (but never between consecutive bullets) restores one
    paragraph per category and a real ``<ul>``. Idempotent on already-spaced
    input.
    """
    out: list[str] = []
    prev = ""
    for line in text.splitlines():
        is_label = bool(_MD_LABEL_RE.match(line))
        is_item = bool(_MD_LIST_RE.match(line))
        prev_item = bool(_MD_LIST_RE.match(prev))
        if prev.strip() and (is_label or (is_item and not prev_item)):
            out.append("")
        out.append(line)
        prev = line
    return "\n".join(out)


def _markdown(text: str | None) -> Markup:
    """Render an LLM-generated markdown summary to sanitized HTML.

    ``_MD`` is configured ``html=False`` with images disabled, so raw HTML is
    escaped and unsafe link schemes are rejected at render time — the result is
    safe to mark as ``Markup`` without a separate sanitizer. ``None``/empty
    input yields an empty fragment.
    """
    if not text:
        return Markup("")
    return Markup(_MD.render(_normalize_markdown(text)))


def _mask_url(url: str) -> str:
    """Mask a notification URL for the UI, showing only the scheme.

    Built on the single canonical ``app.log_redaction.redact_url`` (fold-in): that
    helper already strips userinfo/query/secret path segments and never raises.
    For the UI this filter is deliberately *stronger* — it also hides the host —
    so it collapses everything after the scheme to ``****``. ``redact_url``
    returns ``"****"`` (no ``://``) for schemeless/garbage input, which maps to the
    same masked placeholder here.
    """
    redacted = redact_url(url)
    scheme, sep, _rest = redacted.partition("://")
    if sep and scheme:
        return f"{scheme}://****"
    return "****"


def _safe_href(url: str | None) -> str:
    """Return ``url`` only if its scheme is http(s), else ``"#"``.

    Jinja autoescape neutralizes quotes/angle brackets but NOT a ``javascript:``
    or ``data:text/html`` scheme inside an href, so an attacker-controlled feed
    link could otherwise plant a clickable script in the app origin. Allowlist
    the scheme before render, mirroring url_validation.validate_feed_url.
    """
    if not url:
        return "#"
    try:
        scheme = urlparse(url.strip()).scheme.lower()
    except Exception:
        return "#"
    return url if scheme in ("http", "https") else "#"


def _confidence_value(confidence: float | int | None) -> str:
    """Render a confidence scalar (already extracted) as a colored badge.

    Used on the dashboard, where the confidence is read via SQL ``json_extract``
    so the full ``llm_response`` blob is never shipped/parsed per topic
    (OVH-052). ``None`` (no check / missing confidence) renders as ``-``.
    """
    if confidence is None:
        return "-"
    try:
        score = float(confidence)
    except (ValueError, TypeError):
        return "-"

    if score >= 0.8:
        bg, color = "#2ecc40", "#fff"
    elif score >= 0.5:
        bg, color = "#ffdc00", "#111"
    else:
        bg, color = "#ff4136", "#fff"

    score_text = f"{score:.2f}"
    return Markup(  # type: ignore[no-any-return]
        f'<span style="background:{bg};color:{color};padding:0.15em 0.5em;'
        f'border-radius:0.25em;font-size:0.85em;font-weight:600;" '
        f'title="Confidence: {score_text}">{score_text}</span>'
    )


def _confidence_badge(llm_response: str | None) -> str:
    """Render a confidence badge from a full ``llm_response`` JSON blob.

    Used on paths that already hold the blob (e.g. the per-check history table).
    The dashboard listing uses :func:`_confidence_value` on a pre-extracted
    scalar instead so it never ships the blob (OVH-052).
    """
    if not llm_response:
        return "-"

    try:
        data = json_mod.loads(llm_response)
        confidence = data.get("confidence")
    except json_mod.JSONDecodeError:
        return "-"

    return _confidence_value(confidence)


def _feed_source_name(feed_url: str) -> str:
    """Convert a feed URL to a human-readable source name."""
    try:
        host = urlparse(feed_url).hostname or feed_url
    except Exception:
        return feed_url
    host = host.lower()
    if "google.com" in host:
        return "Google News"
    if "bing.com" in host:
        return "Bing News"
    # Strip common prefixes for other feeds
    for prefix in ("www.", "news.", "feeds.", "rss.", "feed."):
        if host.startswith(prefix):
            host = host[len(prefix) :]
    return host


templates.env.filters["timeago"] = _timeago
templates.env.filters["sanitize_error"] = _sanitize_error
templates.env.filters["markdown"] = _markdown
templates.env.filters["mask_url"] = _mask_url
templates.env.filters["safe_href"] = _safe_href
templates.env.filters["confidence_badge"] = _confidence_badge
templates.env.filters["confidence_value"] = _confidence_value
templates.env.filters["feed_source_name"] = _feed_source_name
