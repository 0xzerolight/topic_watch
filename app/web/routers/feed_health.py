"""Feed health dashboard and feed-URL validation routes."""

import asyncio
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse

from app.config import Settings
from app.crud import list_all_feed_health, list_topics
from app.feed_backoff import feed_backoff_until
from app.models import FeedHealth, FeedMode, Topic
from app.scraping.exa import exa_search_endpoint
from app.scraping.routing import topic_owned_feed_urls
from app.web.csrf import verify_csrf
from app.web.dependencies import get_db_conn, get_settings
from app.web.routers.templates import templates
from app.web.state import _check_rate_limit

router = APIRouter()


@dataclass
class FeedHealthRow:
    """One Feed Health table row: a feed_health record plus its topic ownership.

    feed_health has no topic_id column — a feed URL can be shared by topics, or
    left behind by a topic edit/delete — so ownership is derived at render time
    from current topic state instead of stored (AUG-105).
    """

    feed: FeedHealth
    is_exa: bool
    owners: list[Topic]

    @property
    def orphaned(self) -> bool:
        return not self.owners


def _build_feed_health_rows(feeds: list[FeedHealth], topics: list[Topic], exa_endpoint: str) -> list[FeedHealthRow]:
    owners_by_url: dict[str, list[Topic]] = {}
    for topic in topics:
        for url in topic_owned_feed_urls(topic, exa_endpoint):
            owners_by_url.setdefault(url, []).append(topic)

    rows = []
    for feed in feeds:
        owners = owners_by_url.get(feed.feed_url, [])
        is_exa = feed.feed_url == exa_endpoint or any(t.feed_mode == FeedMode.EXA for t in owners)
        rows.append(FeedHealthRow(feed=feed, is_exa=is_exa, owners=owners))
    return rows


def _format_backoff_label(until: datetime, now: datetime) -> str:
    """Format a retry ETA, minutes below two hours and hours thereafter (AUG-123).

    The default first backoff is 15 minutes; rounding straight to hours and
    clamping to a minimum of one made every sub-hour retry read as "~1h".
    """
    minutes = max(1, round((until - now).total_seconds() / 60))
    if minutes < 120:
        return f"next retry ~{minutes}m"
    return f"next retry ~{round(minutes / 60)}h"


@router.get("/feeds", response_class=HTMLResponse)
async def feed_health_page(
    request: Request,
    conn: sqlite3.Connection = Depends(get_db_conn, scope="function"),
    settings: Settings = Depends(get_settings),
):
    """Global feed health dashboard: per-source health with owning-topic links."""
    feeds = list_all_feed_health(conn)
    topics = list_topics(conn)
    rows = _build_feed_health_rows(feeds, topics, exa_search_endpoint(settings.exa))

    # AUG-213: feed_backoff_until() is a MANUAL-feed-only formula (its own module
    # docstring says so) — AUTO uses separate provider cooldown state and EXA is
    # attempted without this backoff, so applying it to every row could label an
    # AUTO/EXA source as "backing off" until a time the runtime never honors.
    manual_urls = {url for topic in topics if topic.feed_mode == FeedMode.MANUAL for url in topic.feed_urls}

    now = datetime.now(UTC)
    backoff_map: dict[str, str] = {}
    for feed in feeds:
        if feed.feed_url not in manual_urls:
            continue
        until = feed_backoff_until(
            feed,
            base_minutes=settings.feed_backoff_base_minutes,
            cap_hours=settings.feed_backoff_cap_hours,
        )
        if until is not None and until > now:
            backoff_map[feed.feed_url] = _format_backoff_label(until, now)
    return templates.TemplateResponse(
        request,
        "feed_health.html",
        {"rows": rows, "backoff_map": backoff_map},
    )


_MAX_VALIDATION_ERROR_CHARS = 150


async def _validate_one(url: str) -> dict:
    """Fetch one URL and report whether it is a feed worth saving.

    Reads the fetch's own outcome rather than the length of the list it returned.
    The list is empty for a blocked URL, a timeout, a 404 and an unparseable body
    alike, so treating "no exception" as valid told users an unreachable or
    malformed URL was a 'Valid RSS feed with 0 entries' and let them save a source
    that will never deliver news (AUG-175). Only a fetch the source actually
    answered counts as valid.
    """
    from app.scraping.rss import fetch_feed_outcome
    from app.scraping.source import FeedHealthOutcome

    reports: list[FeedHealthOutcome] = []
    try:
        result = await fetch_feed_outcome(url, timeout=10.0, health_callback=reports.append)
    except Exception as exc:  # the fetch layer is fail-safe, but never trust that here
        message = str(exc)[:_MAX_VALIDATION_ERROR_CHARS]
        return {"url": url, "valid": False, "message": message or type(exc).__name__}

    if result.status.succeeded:
        return {"url": url, "valid": True, "message": f"Valid RSS feed with {len(result.entries)} entries"}

    # The reason lives on the health report the fetch just filed for this URL.
    reason = next((r.error_msg for r in reports if r.error_msg), None) or "Feed could not be fetched"
    return {"url": url, "valid": False, "message": reason[:_MAX_VALIDATION_ERROR_CHARS]}


@router.post("/feeds/validate", response_class=HTMLResponse, dependencies=[Depends(verify_csrf)])
async def validate_feed_url(
    request: Request,
    feed_urls: str = Form(""),
):
    """Validate feed URLs by attempting to fetch them. Returns HTMX partial."""
    client_ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(client_ip):
        return HTMLResponse(
            '<div style="color: var(--pico-del-color, red);"><small>Rate limit exceeded. Please wait before validating again.</small></div>',
            status_code=429,
        )

    urls = [u.strip() for u in feed_urls.strip().splitlines() if u.strip()]
    if not urls:
        return templates.TemplateResponse(
            request,
            "_feed_validation.html",
            {"results": [{"url": "", "valid": False, "message": "No URLs provided"}]},
        )

    from app.url_validation import is_private_url

    results = []
    for url in urls:
        if await asyncio.to_thread(is_private_url, url):
            results.append({"url": url, "valid": False, "message": "Private/local URLs are not allowed"})
            continue
        results.append(await _validate_one(url))

    return templates.TemplateResponse(
        request,
        "_feed_validation.html",
        {"results": results},
    )
