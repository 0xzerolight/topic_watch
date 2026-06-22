"""Feed health dashboard and feed-URL validation routes."""

import asyncio
import sqlite3

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse

from app.crud import list_all_feed_health
from app.web.csrf import verify_csrf
from app.web.dependencies import get_db_conn
from app.web.routers.templates import templates
from app.web.state import _check_rate_limit

router = APIRouter()


@router.get("/feeds", response_class=HTMLResponse)
async def feed_health_page(
    request: Request,
    conn: sqlite3.Connection = Depends(get_db_conn),
):
    """Global feed health dashboard."""
    feeds = list_all_feed_health(conn)
    return templates.TemplateResponse(
        request,
        "feed_health.html",
        {"feeds": feeds},
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
