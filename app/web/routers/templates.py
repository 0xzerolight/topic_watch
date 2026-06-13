"""Shared Jinja2 template environment and template filters for web routers.

Centralizes the single ``Jinja2Templates`` instance plus the custom filters
so every router renders against the same environment. The filter helper
functions are module-level (and importable) because they are unit-tested
directly.
"""

import json as json_mod
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape

from app import __version__

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent.parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))

templates.env.globals["version"] = __version__


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


def _mask_url(url: str) -> str:
    """Mask a notification URL, showing only the scheme."""
    try:
        parsed = urlparse(url)
        scheme = parsed.scheme
        if scheme:
            return f"{scheme}://****"
        return "****"
    except Exception:
        return "****"


def _confidence_badge(llm_response: str | None) -> str:
    """Render a confidence score as a colored badge from llm_response JSON."""
    if not llm_response:
        return "-"

    try:
        data = json_mod.loads(llm_response)
        confidence = data.get("confidence")
        if confidence is None:
            return "-"
        confidence = float(confidence)
    except (json_mod.JSONDecodeError, ValueError, TypeError):
        return "-"

    if confidence >= 0.8:
        bg, color = "#2ecc40", "#fff"
    elif confidence >= 0.5:
        bg, color = "#ffdc00", "#111"
    else:
        bg, color = "#ff4136", "#fff"

    score_text = f"{confidence:.2f}"
    return Markup(  # type: ignore[no-any-return]
        f'<span style="background:{bg};color:{color};padding:0.15em 0.5em;'
        f'border-radius:0.25em;font-size:0.85em;font-weight:600;" '
        f'title="Confidence: {score_text}">{score_text}</span>'
    )


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
templates.env.filters["mask_url"] = _mask_url
templates.env.filters["confidence_badge"] = _confidence_badge
templates.env.filters["feed_source_name"] = _feed_source_name
