"""Dashboard, health check, and topic search routes."""

import sqlite3

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse

from app.crud import (
    get_dashboard_data,
    get_dashboard_stats,
    search_dashboard_data,
)
from app.web.dependencies import get_db_conn
from app.web.routers.templates import templates

router = APIRouter()


@router.get("/health")
async def health_check(conn: sqlite3.Connection = Depends(get_db_conn)):
    """Health check endpoint for load balancers and container orchestrators."""
    topic_count = conn.execute("SELECT COUNT(*) FROM topics").fetchone()[0]
    return {"status": "ok", "topics": topic_count}


@router.get("/", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    conn: sqlite3.Connection = Depends(get_db_conn),
    tag: str | None = None,
    error: str | None = None,
):
    """Dashboard showing all topics with status and last check time."""
    topic_data = get_dashboard_data(conn)

    # Collect all unique tags across all topics for the filter bar
    all_tags: list[str] = []
    seen: set[str] = set()
    for item in topic_data:
        for t in item["topic"].tags:
            if t not in seen:
                all_tags.append(t)
                seen.add(t)
    all_tags.sort()

    # Filter topic_data by tag if requested
    if tag:
        topic_data = [item for item in topic_data if tag in item["topic"].tags]

    status_counts = {"ready": 0, "researching": 0, "error": 0, "new": 0}
    for item in topic_data:
        status_counts[item["topic"].status.value] += 1

    # Dashboard stats: a handful of COUNT/MAX subqueries over topics + check_results,
    # run only on this page load. Query directly so the numbers are always fresh
    # (a TTL cache here lagged every stat card for up to 60s after any mutation).
    stats = get_dashboard_stats(conn)

    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "topic_data": topic_data,
            "status_counts": status_counts,
            "all_tags": all_tags,
            "active_tag": tag,
            "stats": stats,
            "error": error,
        },
    )


@router.get("/topics/search", response_class=HTMLResponse)
async def search_topics(
    request: Request,
    conn: sqlite3.Connection = Depends(get_db_conn),
    q: str = "",
    status: str = "all",
):
    """HTMX partial: filtered topic list for search/filter."""
    topic_data = search_dashboard_data(
        conn,
        query=q if q.strip() else None,
        status=status if status != "all" else None,
    )
    return templates.TemplateResponse(
        request,
        "_topic_list.html",
        {"topic_data": topic_data},
    )
