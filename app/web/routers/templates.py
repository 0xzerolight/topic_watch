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
from typing import Any
from urllib.parse import urlparse

from fastapi.templating import Jinja2Templates
from markdown_it import MarkdownIt
from markupsafe import Markup, escape

from app import __version__
from app.log_redaction import redact_url
from app.scraping.google_news import GOOGLE_NEWS_HOST
from app.scraping.rss import BING_HOST
from app.scraping.source import host_matches, url_hostname

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent.parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))

templates.env.globals["version"] = __version__

# Markdown renderer for LLM-generated knowledge summaries. ``html=False`` escapes
# any raw HTML in the (article-derived) source and rejects unsafe link schemes
# (javascript:/data:), so no separate HTML sanitizer is needed. Images are
# disabled and hard breaks are off — list structure is restored by
# ``_normalize_markdown`` instead. Anchors are not produced at all: see the two
# render rules below. Built once and shared; ``render()`` is stateless per call
# (``env`` is a fresh dict per render), so it is safe across concurrent requests.
_MD = MarkdownIt("commonmark", {"html": False, "linkify": False, "breaks": False})
_MD.disable("image")


def _render_inert_link_open(self: object, tokens: list[Any], idx: int, options: object, env: dict[str, Any]) -> str:
    """Open a link as nothing at all, remembering where it pointed.

    The knowledge summary is written by the model from article text an attacker
    controls, so a link inside it is a link a feed can choose. Rendered as an
    anchor it becomes a clickable destination on a page the user trusts — the
    phishing half of AUG-016. Anchors are therefore not produced at all.
    """
    token = tokens[idx]
    # An autolink already IS its URL on screen, so repeating it in parentheses
    # would only render the same string twice.
    href = "" if token.markup == "autolink" else (token.attrGet("href") or "")
    env.setdefault("_inert_link_hrefs", []).append(href)
    return ""


def _render_inert_link_close(self: object, tokens: list[Any], idx: int, options: object, env: dict[str, Any]) -> str:
    """Close it by showing the destination as inert text.

    Dropping the URL entirely would leave friendly link text with its target
    hidden, which is the shape the attack wants. Showing it beside the text is
    what lets a reader see that "your account portal" points somewhere else.
    """
    hrefs: list[str] = env.get("_inert_link_hrefs") or []
    href = hrefs.pop() if hrefs else ""
    return f" ({escape(href)})" if href else ""


_MD.add_render_rule("link_open", _render_inert_link_open)
_MD.add_render_rule("link_close", _render_inert_link_close)

_MD_LABEL_RE = re.compile(r"^\s*\*\*[^*]+\*\*")
_MD_LIST_RE = re.compile(r"^\s*([-*+]|\d+\.)\s+")


# How far ahead of now a timestamp may sit and still read as "just now". Covers
# the ordinary case — a row written moments ago, a container clock a few seconds
# off — and nothing else.
_FUTURE_TOLERANCE_SECONDS = 60


def _coarse_span(seconds: int) -> str | None:
    """``5m`` / ``3h`` / ``5d``, or None when a date reads better than a span."""
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h"
    days = hours // 24
    if days < 30:
        return f"{days}d"
    return None


def _timeago(dt: datetime) -> str:
    """Format a datetime as a human-readable relative time.

    A timestamp ahead of now is rendered as a future one ("in 3h", "on
    2027-01-01") rather than collapsed into "just now". Every negative age used
    to read as current, so a clock rollback or a bad stored value made the
    dashboard, check history, feed health and status pages all claim freshness
    they did not have — for as long as it took the wall clock to catch up, which
    is precisely when the operator needs to see it (AUG-284).
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    now = datetime.now(UTC)
    seconds = int((now - dt).total_seconds())
    if seconds < 0:
        ahead = -seconds
        if ahead <= _FUTURE_TOLERANCE_SECONDS:
            return "just now"
        span = _coarse_span(ahead)
        return f"in {span}" if span else f"on {dt.strftime('%Y-%m-%d')}"
    if seconds < 60:
        return "just now"
    span = _coarse_span(seconds)
    return f"{span} ago" if span else dt.strftime("%Y-%m-%d")


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
    safe to mark as ``Markup`` without a separate sanitizer. Links render as
    inert text rather than anchors (AUG-016), because the summary is model-derived
    from attacker-controllable article text. ``None``/empty input yields an empty
    fragment.
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
    """Render a confidence scalar (already extracted) as a status badge.

    Used on the dashboard, where the confidence is read via SQL ``json_extract``
    so the full ``llm_response`` blob is never shipped/parsed per topic
    (OVH-052). ``None`` (no check / missing confidence) renders as ``-``.

    Emits classes only — never inline hex. Colors live in ``components.css``
    (``.badge--conf-{high,mid,low}``) so every theme inherits them.
    """
    if confidence is None:
        return "-"
    try:
        score = float(confidence)
    except (ValueError, TypeError):
        return "-"

    if score >= 0.8:
        level = "high"
    elif score >= 0.5:
        level = "mid"
    else:
        level = "low"

    score_text = f"{score:.2f}"
    return Markup(  # type: ignore[no-any-return]
        f'<span class="badge badge--conf-{level}" title="Confidence: {score_text}">{score_text}</span>'
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
    except json_mod.JSONDecodeError:
        return "-"
    # json.loads() also accepts arrays, scalars, booleans, and null; only a dict
    # has a ``confidence`` key to read (AUG-215, mirrors _importance_score()).
    if not isinstance(data, dict):
        return "-"

    return _confidence_value(data.get("confidence"))


def _importance_score(llm_response: str | None) -> int | None:
    """Pull the 1-5 importance score out of an ``llm_response`` JSON blob.

    ``None`` when the blob is missing, unparseable, or predates m023 (checks
    recorded before importance scoring existed). The check-history table renders
    it directly and uses it to tell an importance-suppressed check apart from a
    failed delivery.
    """
    if not llm_response:
        return None

    try:
        data = json_mod.loads(llm_response)
    except json_mod.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    try:
        return int(data["importance"])
    except (KeyError, TypeError, ValueError):
        return None


def _feed_source_name(feed_url: str) -> str:
    """Convert a feed URL to a human-readable source name.

    The brand labels are claimed by hostname, not by a substring of the URL
    (TW-AUD-031): ``fake-google.com`` and ``news.google.com.example.net`` used to
    display as Google News, as did any feed with ``bing.com`` anywhere in its
    path or query. Everything else renders by canonical host.
    """
    host = url_hostname(feed_url)
    if not host:
        return feed_url
    if host_matches(host, GOOGLE_NEWS_HOST):
        return "Google News"
    if host_matches(host, BING_HOST):
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
templates.env.filters["importance_score"] = _importance_score
templates.env.filters["feed_source_name"] = _feed_source_name
