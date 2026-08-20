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
from app.scraping.routing import router as provider_router
from app.web.csrf import verify_csrf
from app.web.dependencies import get_db_conn, get_settings
from app.web.routers.templates import templates
from app.web.state import _check_rate_limit

router = APIRouter()

# Mirrors app.scraping.exa's own fallback. Kept as a local literal rather than
# importing that module's private constant — Exa's shared search endpoint is
# what every EXA-mode topic's feed_health row is keyed on.
_EXA_DEFAULT_BASE_URL = "https://api.exa.ai"


def _exa_endpoint(settings: Settings) -> str:
    """The endpoint every EXA-mode topic's health is recorded under."""
    base = (settings.exa.base_url or _EXA_DEFAULT_BASE_URL).rstrip("/")
    return f"{base}/search"


def _topic_owned_urls(topic: Topic, exa_endpoint: str) -> list[str]:
    """URLs a topic currently resolves to (mirrors topics._feed_source_context).

    AUTO topics own every provider URL, not just the currently-active one, so a
    standby provider's health still traces back to its topic. EXA topics all
    share one endpoint. MANUAL topics own their configured list.
    """
    if topic.feed_mode == FeedMode.AUTO:
        return [p.build_feed_url(topic) for p in provider_router.providers]
    if topic.feed_mode == FeedMode.EXA:
        return [exa_endpoint]
    return list(topic.feed_urls)


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
        for url in _topic_owned_urls(topic, exa_endpoint):
            owners_by_url.setdefault(url, []).append(topic)

    rows = []
    for feed in feeds:
        owners = owners_by_url.get(feed.feed_url, [])
        is_exa = feed.feed_url == exa_endpoint or any(t.feed_mode == FeedMode.EXA for t in owners)
        rows.append(FeedHealthRow(feed=feed, is_exa=is_exa, owners=owners))
    return rows


@router.get("/feeds", response_class=HTMLResponse)
async def feed_health_page(
    request: Request,
    conn: sqlite3.Connection = Depends(get_db_conn),
    settings: Settings = Depends(get_settings),
):
    """Global feed health dashboard: per-source health with owning-topic links."""
    feeds = list_all_feed_health(conn)
    topics = list_topics(conn)
    rows = _build_feed_health_rows(feeds, topics, _exa_endpoint(settings))

    now = datetime.now(UTC)
    backoff_map: dict[str, str] = {}
    for feed in feeds:
        until = feed_backoff_until(
            feed,
            base_minutes=settings.feed_backoff_base_minutes,
            cap_hours=settings.feed_backoff_cap_hours,
        )
        if until is not None and until > now:
            hours = max(1, round((until - now).total_seconds() / 3600))
            backoff_map[feed.feed_url] = f"next retry ~{hours}h"
    return templates.TemplateResponse(
        request,
        "feed_health.html",
        {"rows": rows, "backoff_map": backoff_map},
    )


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

    from app.scraping.rss import fetch_feed
    from app.url_validation import is_private_url

    results = []
    for url in urls:
        if await asyncio.to_thread(is_private_url, url):
            results.append({"url": url, "valid": False, "message": "Private/local URLs are not allowed"})
            continue
        try:
            entries = await fetch_feed(url, timeout=10.0)
            results.append(
                {
                    "url": url,
                    "valid": True,
                    "message": f"Valid RSS feed with {len(entries)} entries",
                }
            )
        except Exception as exc:
            error_msg = str(exc)
            if len(error_msg) > 150:
                error_msg = error_msg[:150] + "..."
            results.append({"url": url, "valid": False, "message": error_msg})

    return templates.TemplateResponse(
        request,
        "_feed_validation.html",
        {"results": results},
    )
