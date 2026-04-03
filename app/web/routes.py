"""Web route handlers for Topic Watch dashboard."""

import asyncio
import csv
import io
import json
import logging
import re
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape

from app import __version__
from app.analysis.llm import NoveltyResult
from app.checker import check_all_topics, check_topic
from app.config import (
    CLOUD_PROVIDERS,
    DEFAULT_CONFIG_PATH,
    LOCAL_PROVIDER_DEFAULTS,
    Settings,
    is_cloud_provider,
    load_settings,
    save_settings_to_yaml,
)
from app.crud import (
    count_articles_for_topic,
    count_check_results,
    create_topic,
    delete_topic,
    get_check_result,
    get_dashboard_data,
    get_feed_health,
    get_knowledge_state,
    get_topic,
    list_all_feed_health,
    list_articles_for_topic,
    list_check_results,
    list_topics,
    search_dashboard_data,
    update_topic,
)
from app.models import FeedMode, Topic, TopicStatus
from app.notifications import format_notification, send_notification
from app.scraping.rss import build_google_news_url
from app.url_validation import validate_feed_urls
from app.web.csrf import verify_csrf
from app.web.dependencies import get_db_conn, get_settings

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
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


templates.env.filters["timeago"] = _timeago


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


templates.env.filters["sanitize_error"] = _sanitize_error


def _mask_url(url: str) -> str:
    """Mask a notification URL, showing only the scheme."""
    try:
        from urllib.parse import urlparse

        parsed = urlparse(url)
        scheme = parsed.scheme
        if scheme:
            return f"{scheme}://****"
        return "****"
    except Exception:
        return "****"


templates.env.filters["mask_url"] = _mask_url


def _confidence_badge(llm_response: str | None) -> str:
    """Render a confidence score as a colored badge from llm_response JSON."""
    import json as json_mod

    from markupsafe import Markup

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
    return str(
        Markup(
            f'<span style="background:{bg};color:{color};padding:0.15em 0.5em;'
            f'border-radius:0.25em;font-size:0.85em;font-weight:600;" '
            f'title="Confidence: {score_text}">{score_text}</span>'
        )
    )


templates.env.filters["confidence_badge"] = _confidence_badge

router = APIRouter()

# Simple in-memory rate limiter for feed validation
_rate_limit_store: dict[str, list[float]] = {}
_RATE_LIMIT_MAX = 10
_RATE_LIMIT_WINDOW = 60  # seconds


def _check_rate_limit(ip: str) -> bool:
    """Check if IP is within rate limit. Returns True if allowed."""
    now = time.time()
    timestamps = _rate_limit_store.get(ip, [])
    active = [t for t in timestamps if now - t < _RATE_LIMIT_WINDOW]
    if len(active) >= _RATE_LIMIT_MAX:
        _rate_limit_store[ip] = active
        return False
    active.append(now)
    _rate_limit_store[ip] = active
    # Evict stale IPs to prevent unbounded memory growth
    if len(_rate_limit_store) > 10000:
        stale = [k for k, v in _rate_limit_store.items() if not v or now - v[-1] >= _RATE_LIMIT_WINDOW]
        for k in stale:
            del _rate_limit_store[k]
    return True


class CheckingState:
    """Async-safe state tracker for in-progress topic checks."""

    def __init__(self) -> None:
        self._topics: set[int] = set()
        self._start_times: dict[int, float] = {}
        self._checking_all: bool = False
        self._lock = asyncio.Lock()

    async def start_check(self, topic_id: int) -> bool:
        """Mark topic as being checked. Returns False if already checking."""
        async with self._lock:
            if topic_id in self._topics:
                return False
            self._topics.add(topic_id)
            self._start_times[topic_id] = time.monotonic()
            return True

    async def finish_check(self, topic_id: int) -> None:
        """Mark topic check as finished."""
        async with self._lock:
            self._topics.discard(topic_id)
            self._start_times.pop(topic_id, None)

    async def is_checking(self, topic_id: int) -> bool:
        """Return True if topic is currently being checked."""
        async with self._lock:
            return topic_id in self._topics

    async def start_check_all(self) -> bool:
        """Mark check-all as running. Returns False if already running."""
        async with self._lock:
            if self._checking_all:
                return False
            self._checking_all = True
            return True

    async def finish_check_all(self) -> None:
        """Mark check-all as finished."""
        async with self._lock:
            self._checking_all = False

    async def is_checking_all(self) -> bool:
        """Return True if a check-all is currently running."""
        async with self._lock:
            return self._checking_all

    async def clear_stale(self, timeout_seconds: float) -> list[int]:
        """Remove topic entries older than timeout_seconds. Returns cleared IDs."""
        now = time.monotonic()
        async with self._lock:
            stale = [tid for tid, start in self._start_times.items() if now - start > timeout_seconds]
            for tid in stale:
                self._topics.discard(tid)
                self._start_times.pop(tid, None)
        return stale


_checking_state = CheckingState()

_INIT_TIMEOUT_SECONDS = 600  # 10 minutes
_CHECK_ALL_TIMEOUT_SECONDS = 1800  # 30 minutes


# --- Health ---


@router.get("/health")
async def health_check(conn: sqlite3.Connection = Depends(get_db_conn)):
    """Health check endpoint for load balancers and container orchestrators."""
    topic_count = conn.execute("SELECT COUNT(*) FROM topics").fetchone()[0]
    return {"status": "ok", "topics": topic_count}


# --- Feed Health page ---


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


# --- Feed validation ---


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
        if is_private_url(url):
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


# --- Pages ---


@router.get("/", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    conn: sqlite3.Connection = Depends(get_db_conn),
    tag: str | None = None,
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

    status_counts = {"ready": 0, "researching": 0, "error": 0}
    for item in topic_data:
        status_counts[item["topic"].status.value] += 1

    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "topic_data": topic_data,
            "status_counts": status_counts,
            "all_tags": all_tags,
            "active_tag": tag,
        },
    )


@router.get("/topics/new", response_class=HTMLResponse)
async def topic_add_form(request: Request):
    """Render the add topic form."""
    return templates.TemplateResponse(request, "topic_add.html", {})


@router.post("/topics", dependencies=[Depends(verify_csrf)])
async def create_topic_handler(
    request: Request,
    background_tasks: BackgroundTasks,
    conn: sqlite3.Connection = Depends(get_db_conn),
    settings: Settings = Depends(get_settings),
    name: str = Form(...),
    description: str = Form(...),
    feed_urls: str = Form(""),
    feed_mode: str = Form("auto"),
    check_interval_minutes: str = Form(""),
    tags: str = Form(""),
):
    """Create a new topic and kick off initial research in the background."""
    mode = FeedMode.AUTO if feed_mode == "auto" else FeedMode.MANUAL

    urls: list[str] = []
    errors: list[str] = []
    if mode == FeedMode.MANUAL:
        urls = [u.strip() for u in feed_urls.strip().splitlines() if u.strip()]
        errors = validate_feed_urls(urls)

    parsed_interval: int | None = None
    if check_interval_minutes.strip():
        try:
            parsed_interval = int(check_interval_minutes)
            if parsed_interval < 10 or parsed_interval > 10080:
                errors.append("Check interval must be between 10 and 10080 minutes.")
                parsed_interval = None
        except ValueError:
            errors.append("Check interval must be a whole number.")

    tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    if errors:
        return templates.TemplateResponse(
            request,
            "topic_add.html",
            {
                "errors": errors,
                "name": name,
                "description": description,
                "feed_urls": feed_urls,
                "feed_mode": feed_mode,
                "check_interval_minutes": check_interval_minutes,
                "tags": tags,
            },
            status_code=422,
        )

    topic = Topic(
        name=name,
        description=description,
        feed_urls=urls,
        feed_mode=mode,
        status=TopicStatus.RESEARCHING,
        status_changed_at=datetime.now(UTC),
        check_interval_minutes=parsed_interval,
        tags=tag_list,
    )
    created = create_topic(conn, topic)
    conn.commit()

    assert created.id is not None
    db_path = getattr(request.app.state, "db_path", None)
    background_tasks.add_task(_run_init, created.id, settings, db_path)

    return RedirectResponse(url=f"/topics/{created.id}", status_code=303)


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


@router.get("/topics/{topic_id}", response_class=HTMLResponse)
async def topic_detail(
    request: Request,
    topic_id: int,
    conn: sqlite3.Connection = Depends(get_db_conn),
    settings: Settings = Depends(get_settings),
    page: int = 1,
):
    """Topic detail page: knowledge state, check history, actions."""
    topic = get_topic(conn, topic_id)
    if topic is None:
        raise HTTPException(status_code=404, detail="Topic not found")

    auto_feed_url = None
    if topic.feed_mode == FeedMode.AUTO:
        auto_feed_url = build_google_news_url(topic.name)

    per_page = settings.web_page_size
    offset = (max(1, page) - 1) * per_page

    knowledge = get_knowledge_state(conn, topic_id)
    checks = list_check_results(conn, topic_id, limit=per_page, offset=offset)
    total_checks = count_check_results(conn, topic_id)
    articles = list_articles_for_topic(conn, topic_id, limit=per_page)
    article_count = count_articles_for_topic(conn, topic_id)
    total_pages = max(1, (total_checks + per_page - 1) // per_page)

    feed_health_map = {}
    if topic.feed_mode == FeedMode.AUTO and auto_feed_url:
        health = get_feed_health(conn, auto_feed_url)
        if health:
            feed_health_map[auto_feed_url] = health
    else:
        for url in topic.feed_urls:
            health = get_feed_health(conn, url)
            if health:
                feed_health_map[url] = health

    return templates.TemplateResponse(
        request,
        "topic_detail.html",
        {
            "topic": topic,
            "knowledge": knowledge,
            "checks": checks,
            "articles": articles,
            "article_count": article_count,
            "page": page,
            "total_pages": total_pages,
            "auto_feed_url": auto_feed_url,
            "default_interval": settings.check_interval_hours,
            "knowledge_state_max_tokens": settings.knowledge_state_max_tokens,
            "feed_health_map": feed_health_map,
        },
    )


# --- HTMX partials ---


@router.get("/topics/{topic_id}/status", response_class=HTMLResponse)
async def topic_status(
    request: Request,
    topic_id: int,
    conn: sqlite3.Connection = Depends(get_db_conn),
    settings: Settings = Depends(get_settings),
):
    """HTMX partial: knowledge state fragment for polling during research."""
    topic = get_topic(conn, topic_id)
    if topic is None:
        raise HTTPException(status_code=404, detail="Topic not found")

    knowledge = get_knowledge_state(conn, topic_id)

    return templates.TemplateResponse(
        request,
        "topic_status.html",
        {
            "topic": topic,
            "knowledge": knowledge,
            "knowledge_state_max_tokens": settings.knowledge_state_max_tokens,
        },
    )


@router.post("/topics/{topic_id}/check", response_class=HTMLResponse, dependencies=[Depends(verify_csrf)])
async def check_topic_handler(
    request: Request,
    topic_id: int,
    conn: sqlite3.Connection = Depends(get_db_conn),
    settings: Settings = Depends(get_settings),
):
    """Manual check trigger. Returns HTMX partial for the topic's table row."""
    await _checking_state.clear_stale(600)

    topic = get_topic(conn, topic_id)
    if topic is None:
        raise HTTPException(status_code=404, detail="Topic not found")

    if not await _checking_state.start_check(topic_id):
        # Already checking — return current state without re-checking
        checks = list_check_results(conn, topic_id, limit=1)
        last_check = checks[0] if checks else None
        article_count = len(list_articles_for_topic(conn, topic_id))
        return templates.TemplateResponse(
            request,
            "_topic_row.html",
            {
                "topic": topic,
                "last_check": last_check,
                "article_count": article_count,
            },
        )

    try:
        await check_topic(topic, conn, settings)
    finally:
        await _checking_state.finish_check(topic_id)

    # Re-fetch topic (status may have changed) and latest check
    topic = get_topic(conn, topic_id)
    checks = list_check_results(conn, topic_id, limit=1)
    last_check = checks[0] if checks else None
    article_count = len(list_articles_for_topic(conn, topic_id))

    return templates.TemplateResponse(
        request,
        "_topic_row.html",
        {
            "topic": topic,
            "last_check": last_check,
            "article_count": article_count,
        },
    )


# --- Actions ---


@router.post("/topics/{topic_id}/toggle-active", dependencies=[Depends(verify_csrf)])
async def toggle_active(
    request: Request,
    topic_id: int,
    conn: sqlite3.Connection = Depends(get_db_conn),
):
    """Toggle a topic's is_active flag."""
    topic = get_topic(conn, topic_id)
    if topic is None:
        raise HTTPException(status_code=404, detail="Topic not found")

    topic.is_active = not topic.is_active
    update_topic(conn, topic)
    conn.commit()

    # HTMX request from dashboard — return updated row partial
    if request.headers.get("HX-Request"):
        checks = list_check_results(conn, topic_id, limit=1)
        last_check = checks[0] if checks else None
        article_count = count_articles_for_topic(conn, topic_id)
        return templates.TemplateResponse(
            request,
            "_topic_row.html",
            {
                "topic": topic,
                "last_check": last_check,
                "article_count": article_count,
            },
        )

    return RedirectResponse(url=f"/topics/{topic_id}", status_code=303)


@router.post("/topics/{topic_id}/init", dependencies=[Depends(verify_csrf)])
async def reinit_topic(
    request: Request,
    topic_id: int,
    background_tasks: BackgroundTasks,
    conn: sqlite3.Connection = Depends(get_db_conn),
    settings: Settings = Depends(get_settings),
):
    """Re-trigger initial research for error recovery."""
    topic = get_topic(conn, topic_id)
    if topic is None:
        raise HTTPException(status_code=404, detail="Topic not found")

    topic.status = TopicStatus.RESEARCHING
    topic.status_changed_at = datetime.now(UTC)
    topic.error_message = None
    update_topic(conn, topic)
    conn.commit()

    assert topic.id is not None
    db_path = getattr(request.app.state, "db_path", None)
    background_tasks.add_task(_run_init, topic.id, settings, db_path)

    return RedirectResponse(url=f"/topics/{topic_id}", status_code=303)


@router.post("/topics/{topic_id}/delete", dependencies=[Depends(verify_csrf)])
async def delete_topic_handler(
    topic_id: int,
    conn: sqlite3.Connection = Depends(get_db_conn),
):
    """Delete a topic and redirect to dashboard."""
    delete_topic(conn, topic_id)
    conn.commit()
    return RedirectResponse(url="/", status_code=303)


@router.get("/topics/{topic_id}/edit", response_class=HTMLResponse)
async def topic_edit_form(
    request: Request,
    topic_id: int,
    conn: sqlite3.Connection = Depends(get_db_conn),
    settings: Settings = Depends(get_settings),
):
    """Render the edit topic form."""
    topic = get_topic(conn, topic_id)
    if topic is None:
        raise HTTPException(status_code=404, detail="Topic not found")
    return templates.TemplateResponse(
        request,
        "topic_edit.html",
        {
            "topic": topic,
            "default_interval": settings.check_interval_hours,
            "tags_string": ", ".join(topic.tags),
        },
    )


@router.post("/topics/{topic_id}/edit", dependencies=[Depends(verify_csrf)])
async def edit_topic_handler(
    request: Request,
    topic_id: int,
    conn: sqlite3.Connection = Depends(get_db_conn),
    settings: Settings = Depends(get_settings),
    name: str = Form(...),
    description: str = Form(...),
    feed_urls: str = Form(""),
    feed_mode: str = Form("auto"),
    check_interval_minutes: str = Form(""),
    tags: str = Form(""),
):
    """Update an existing topic's name, description, feed URLs, and feed mode."""
    topic = get_topic(conn, topic_id)
    if topic is None:
        raise HTTPException(status_code=404, detail="Topic not found")

    mode = FeedMode.AUTO if feed_mode == "auto" else FeedMode.MANUAL

    urls: list[str] = []
    errors: list[str] = []
    if mode == FeedMode.MANUAL:
        urls = [u.strip() for u in feed_urls.strip().splitlines() if u.strip()]
        errors = validate_feed_urls(urls)

    parsed_interval: int | None = None
    if check_interval_minutes.strip():
        try:
            parsed_interval = int(check_interval_minutes)
            if parsed_interval < 10 or parsed_interval > 10080:
                errors.append("Check interval must be between 10 and 10080 minutes.")
                parsed_interval = None
        except ValueError:
            errors.append("Check interval must be a whole number.")

    tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    if errors:
        return templates.TemplateResponse(
            request,
            "topic_edit.html",
            {
                "topic": topic,
                "errors": errors,
                "name": name,
                "description": description,
                "feed_urls": feed_urls,
                "feed_mode": feed_mode,
                "check_interval_minutes": check_interval_minutes,
                "tags": tags,
                "default_interval": settings.check_interval_hours,
            },
            status_code=422,
        )

    topic.name = name
    topic.description = description
    topic.feed_urls = urls
    topic.feed_mode = mode
    topic.check_interval_minutes = parsed_interval
    topic.tags = tag_list
    update_topic(conn, topic)
    conn.commit()

    return RedirectResponse(url=f"/topics/{topic_id}", status_code=303)


@router.post("/topics/bulk-delete", dependencies=[Depends(verify_csrf)])
async def bulk_delete_handler(
    request: Request,
    conn: sqlite3.Connection = Depends(get_db_conn),
):
    """Delete multiple topics at once."""
    form = await request.form()
    topic_ids = form.getlist("topic_ids")
    for tid in topic_ids:
        try:
            delete_topic(conn, int(str(tid)))
        except (ValueError, Exception) as exc:
            logger.warning("Failed to delete topic %s: %s", tid, exc)
    conn.commit()
    return RedirectResponse(url="/", status_code=303)


@router.post("/topics/bulk-check", dependencies=[Depends(verify_csrf)])
async def bulk_check_handler(
    request: Request,
    background_tasks: BackgroundTasks,
    conn: sqlite3.Connection = Depends(get_db_conn),
    settings: Settings = Depends(get_settings),
):
    """Trigger checks for multiple topics."""
    form = await request.form()
    topic_ids = form.getlist("topic_ids")
    db_path = getattr(request.app.state, "db_path", None)
    for tid in topic_ids:
        try:
            topic = get_topic(conn, int(str(tid)))
            if topic and topic.id is not None and topic.status == TopicStatus.READY:
                background_tasks.add_task(_run_single_check, topic.id, settings, db_path)
        except (ValueError, Exception) as exc:
            logger.warning("Failed to queue check for topic %s: %s", tid, exc)
    return RedirectResponse(url="/", status_code=303)


@router.post("/check-all", dependencies=[Depends(verify_csrf)])
async def check_all_handler(
    request: Request,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings),
):
    """Trigger a check of all ready topics in the background."""
    if await _checking_state.start_check_all():
        db_path = getattr(request.app.state, "db_path", None)
        background_tasks.add_task(_run_check_all, settings, db_path)
    return RedirectResponse(url="/", status_code=303)


@router.get("/export/topics/json")
async def export_all_topics_json(
    conn: sqlite3.Connection = Depends(get_db_conn),
):
    """Export all topics as JSON."""
    topics = list_topics(conn)

    data = {
        "topics": [t.model_dump(mode="json") for t in topics],
        "exported_at": datetime.now(UTC).isoformat(),
    }

    content = json.dumps(data, indent=2, default=str)

    return StreamingResponse(
        iter([content]),
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="topics_export.json"'},
    )


@router.get("/topics/{topic_id}/export/json")
async def export_topic_json(
    topic_id: int,
    conn: sqlite3.Connection = Depends(get_db_conn),
):
    """Export a topic with articles and check results as JSON."""
    topic = get_topic(conn, topic_id)
    if topic is None:
        raise HTTPException(status_code=404, detail="Topic not found")

    articles = list_articles_for_topic(conn, topic_id)
    checks = list_check_results(conn, topic_id, limit=10000, offset=0)
    knowledge = get_knowledge_state(conn, topic_id)

    data = {
        "topic": topic.model_dump(mode="json"),
        "knowledge_state": knowledge.model_dump(mode="json") if knowledge else None,
        "articles": [a.model_dump(mode="json") for a in articles],
        "check_results": [c.model_dump(mode="json") for c in checks],
    }

    content = json.dumps(data, indent=2, default=str)
    safe_name = re.sub(r"[^a-z0-9_-]", "", topic.name.replace(" ", "_").lower())
    filename = f"topic_{topic_id}_{safe_name}.json"

    return StreamingResponse(
        iter([content]),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/topics/{topic_id}/export/csv")
async def export_topic_csv(
    topic_id: int,
    conn: sqlite3.Connection = Depends(get_db_conn),
):
    """Export check results for a topic as CSV."""
    topic = get_topic(conn, topic_id)
    if topic is None:
        raise HTTPException(status_code=404, detail="Topic not found")

    checks = list_check_results(conn, topic_id, limit=10000, offset=0)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "id",
            "topic_id",
            "checked_at",
            "articles_found",
            "articles_new",
            "has_new_info",
            "notification_sent",
            "notification_error",
        ]
    )
    for check in checks:
        writer.writerow(
            [
                check.id,
                check.topic_id,
                check.checked_at.isoformat(),
                check.articles_found,
                check.articles_new,
                check.has_new_info,
                check.notification_sent,
                check.notification_error or "",
            ]
        )

    content = output.getvalue()
    safe_name = re.sub(r"[^a-z0-9_-]", "", topic.name.replace(" ", "_").lower())
    filename = f"checks_{topic_id}_{safe_name}.csv"

    return StreamingResponse(
        iter([content]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/setup", response_class=HTMLResponse)
async def setup_view(request: Request):
    """Display the first-run setup wizard, or redirect to dashboard if already configured."""
    if not getattr(request.app.state, "setup_required", False):
        return RedirectResponse(url="/", status_code=303)
    _provider_ctx = {"cloud_providers": sorted(CLOUD_PROVIDERS), "local_provider_defaults": LOCAL_PROVIDER_DEFAULTS}
    return templates.TemplateResponse(
        request,
        "setup.html",
        {"setup_mode": True, **_provider_ctx},
    )


@router.post("/setup", dependencies=[Depends(verify_csrf)])
async def complete_setup(
    request: Request,
    llm_model: str = Form(...),
    llm_api_key: str = Form(...),
    llm_base_url: str = Form(""),
):
    """Process setup form and start the application."""
    from pydantic import ValidationError

    from app.config import LLMSettings, NotificationSettings
    from app.scheduler import start_scheduler

    # Strip base_url for cloud providers (e.g. stale Ollama URL when switching to Anthropic)
    effective_base_url = llm_base_url.strip() or None
    if effective_base_url and is_cloud_provider(llm_model):
        effective_base_url = None

    form_values = {
        "llm_model": llm_model,
        "llm_api_key": llm_api_key,
        "llm_base_url": llm_base_url,
    }
    _provider_ctx = {"cloud_providers": sorted(CLOUD_PROVIDERS), "local_provider_defaults": LOCAL_PROVIDER_DEFAULTS}
    try:
        new_settings = Settings(  # type: ignore[call-arg]
            llm=LLMSettings(
                model=llm_model,
                api_key=llm_api_key,
                base_url=effective_base_url,
            ),
            notifications=NotificationSettings(),
        )
        save_settings_to_yaml(new_settings)
        request.app.state.settings = new_settings
        request.app.state.setup_required = False
        start_scheduler(new_settings, db_path=request.app.state.db_path)
    except ValidationError as exc:
        errors = [f"{' → '.join(str(loc) for loc in e['loc'])}: {e['msg']}" for e in exc.errors()]
        return templates.TemplateResponse(
            request,
            "setup.html",
            {"setup_mode": True, "errors": errors, "form": form_values, **_provider_ctx},
            status_code=422,
        )
    except Exception as exc:
        logger.exception("Setup failed: %s", exc)
        return templates.TemplateResponse(
            request,
            "setup.html",
            {"setup_mode": True, "errors": [f"Setup failed: {exc}"], "form": form_values, **_provider_ctx},
            status_code=422,
        )

    logger.info("Setup completed — application is now configured")
    return RedirectResponse(url="/", status_code=303)


@router.get("/settings", response_class=HTMLResponse)
async def settings_view(request: Request):
    """Display of current configuration as an editable form."""
    settings = load_settings()
    return templates.TemplateResponse(
        request,
        "settings.html",
        {
            "settings": settings,
            "config_path": str(DEFAULT_CONFIG_PATH),
            "cloud_providers": sorted(CLOUD_PROVIDERS),
            "local_provider_defaults": LOCAL_PROVIDER_DEFAULTS,
        },
    )


@router.post("/settings", dependencies=[Depends(verify_csrf)])
async def update_settings(
    request: Request,
    llm_model: str = Form(...),
    llm_api_key: str = Form(""),
    llm_base_url: str = Form(""),
    notification_urls: str = Form(""),
    webhook_urls: str = Form(""),
    check_interval_hours: int = Form(...),
    max_articles_per_check: int = Form(...),
    knowledge_state_max_tokens: int = Form(2000),
    article_retention_days: int = Form(90),
    feed_fetch_timeout: float = Form(15.0),
    article_fetch_timeout: float = Form(20.0),
    llm_analysis_timeout: int = Form(60),
    llm_knowledge_timeout: int = Form(120),
    web_page_size: int = Form(20),
):
    """Save updated settings to config file and reload into app state."""
    from pydantic import ValidationError

    from app.config import LLMSettings, NotificationSettings

    parsed_notification_urls = [u.strip() for u in notification_urls.splitlines() if u.strip()]
    parsed_webhook_urls = [u.strip() for u in webhook_urls.splitlines() if u.strip()]

    # If API key field is empty, retain existing key
    effective_api_key = llm_api_key.strip() or request.app.state.settings.llm.api_key

    # Strip base_url for cloud providers (e.g. stale Ollama URL when switching to Anthropic)
    effective_base_url = llm_base_url.strip() or None
    if effective_base_url and is_cloud_provider(llm_model):
        effective_base_url = None

    # Build a new Settings object to validate via Pydantic, then save
    form_values = {
        "llm_model": llm_model,
        "llm_api_key": llm_api_key,
        "llm_base_url": llm_base_url,
        "notification_urls": notification_urls,
        "webhook_urls": webhook_urls,
        "check_interval_hours": check_interval_hours,
        "max_articles_per_check": max_articles_per_check,
        "knowledge_state_max_tokens": knowledge_state_max_tokens,
        "article_retention_days": article_retention_days,
        "feed_fetch_timeout": feed_fetch_timeout,
        "article_fetch_timeout": article_fetch_timeout,
        "llm_analysis_timeout": llm_analysis_timeout,
        "llm_knowledge_timeout": llm_knowledge_timeout,
        "web_page_size": web_page_size,
    }
    try:
        new_settings = Settings(  # type: ignore[call-arg]
            llm=LLMSettings(
                model=llm_model,
                api_key=effective_api_key,
                base_url=effective_base_url,
            ),
            notifications=NotificationSettings(
                urls=parsed_notification_urls,
                webhook_urls=parsed_webhook_urls,
            ),
            check_interval_hours=check_interval_hours,
            max_articles_per_check=max_articles_per_check,
            knowledge_state_max_tokens=knowledge_state_max_tokens,
            article_retention_days=article_retention_days,
            feed_fetch_timeout=feed_fetch_timeout,
            article_fetch_timeout=article_fetch_timeout,
            llm_analysis_timeout=llm_analysis_timeout,
            llm_knowledge_timeout=llm_knowledge_timeout,
            web_page_size=web_page_size,
            # Preserve values not in form from current app settings
            db_path=request.app.state.settings.db_path,
            feed_max_retries=request.app.state.settings.feed_max_retries,
            content_fetch_concurrency=request.app.state.settings.content_fetch_concurrency,
            scheduler_misfire_grace_time=request.app.state.settings.scheduler_misfire_grace_time,
            scheduler_jitter_seconds=request.app.state.settings.scheduler_jitter_seconds,
            llm_max_retries=request.app.state.settings.llm_max_retries,
        )
        save_settings_to_yaml(new_settings)
        request.app.state.settings = new_settings
    except ValidationError as exc:
        errors = [f"{' → '.join(str(loc) for loc in e['loc'])}: {e['msg']}" for e in exc.errors()]
        return templates.TemplateResponse(
            request,
            "settings.html",
            {
                "settings": request.app.state.settings,
                "config_path": str(DEFAULT_CONFIG_PATH),
                "errors": errors,
                "form": form_values,
                "cloud_providers": sorted(CLOUD_PROVIDERS),
                "local_provider_defaults": LOCAL_PROVIDER_DEFAULTS,
            },
            status_code=422,
        )
    except Exception as exc:
        logger.exception("Failed to save settings: %s", exc)
        return templates.TemplateResponse(
            request,
            "settings.html",
            {
                "settings": request.app.state.settings,
                "config_path": str(DEFAULT_CONFIG_PATH),
                "errors": [f"Failed to save settings: {exc}"],
                "form": form_values,
                "cloud_providers": sorted(CLOUD_PROVIDERS),
                "local_provider_defaults": LOCAL_PROVIDER_DEFAULTS,
            },
            status_code=422,
        )

    return RedirectResponse(url="/settings?saved=1", status_code=303)


@router.post("/notifications/test", dependencies=[Depends(verify_csrf)])
async def test_notification(
    request: Request,
    settings: Settings = Depends(get_settings),
):
    """Send a test notification to verify notification configuration."""
    if not settings.notifications.urls:
        return HTMLResponse(
            '<article style="border-left: 4px solid var(--pico-color-orange-500, #f57c00); padding: 1rem;">'
            "<strong>No notification URLs configured.</strong>"
            "<p>To receive notifications, add one or more Apprise notification URLs to your config file "
            "(<code>data/config.yml</code>) under <code>notifications.urls</code>.</p>"
            "<p><small>Supported services include: Ntfy, Discord, Telegram, Slack, Email, Pushover, Gotify, "
            "and <a href='https://github.com/caronc/apprise/wiki#notification-services' target='_blank'>"
            "90+ more via Apprise</a>.</small></p>"
            "<p><small>Example: <code>ntfy://your-topic-name</code></small></p>"
            "</article>",
            status_code=200,
        )

    try:
        success = await send_notification(
            "Topic Watch Test",
            "This is a test notification from Topic Watch. If you received this, notifications are working correctly.",
            settings,
        )
        if success:
            return HTMLResponse(
                '<article style="border-left: 4px solid var(--pico-ins-color, #2e7d32); padding: 1rem;">'
                "<strong>&#10003; Notification sent successfully!</strong>"
                "<p><small>Check your notification service to confirm delivery.</small></p>"
                "</article>",
                status_code=200,
            )
        else:
            return HTMLResponse(
                '<article style="border-left: 4px solid var(--pico-color-orange-500, #f57c00); padding: 1rem;">'
                "<strong>Notification delivery failed.</strong>"
                "<p><small>The notification service rejected the message. Check that your notification URLs "
                "are correct and the service is reachable.</small></p>"
                "</article>",
                status_code=200,
            )
    except Exception:
        return HTMLResponse(
            '<article style="border-left: 4px solid var(--pico-del-color, #c62828); padding: 1rem;">'
            "<strong>Notification error.</strong>"
            "<p><small>An unexpected error occurred. Check the server logs for details.</small></p>"
            "</article>",
            status_code=200,
        )


@router.post("/topics/{topic_id}/checks/{check_id}/notify", dependencies=[Depends(verify_csrf)])
async def force_notify(
    request: Request,
    topic_id: int,
    check_id: int,
    conn: sqlite3.Connection = Depends(get_db_conn),
    settings: Settings = Depends(get_settings),
):
    """Re-send notification for a specific check result."""
    topic = get_topic(conn, topic_id)
    if topic is None:
        raise HTTPException(status_code=404, detail="Topic not found")

    check_result = get_check_result(conn, check_id)
    if check_result is None or check_result.topic_id != topic_id:
        raise HTTPException(status_code=404, detail="Check result not found")

    if not check_result.has_new_info or not check_result.llm_response:
        return HTMLResponse(
            '<span style="color: var(--pico-del-color, red);">No new info to notify about</span>',
            status_code=400,
        )

    try:
        novelty = NoveltyResult.model_validate_json(check_result.llm_response)
        title, body = format_notification(topic.name, novelty)
        sent = await send_notification(title, body, settings)

        if sent:
            return HTMLResponse('<span style="color: var(--pico-ins-color, green);">Sent!</span>')
        else:
            return HTMLResponse('<span style="color: var(--pico-del-color, red);">Delivery failed</span>')
    except Exception as exc:
        logger.warning("Force notify failed for check %d", check_id, exc_info=True)
        from markupsafe import escape

        return HTMLResponse(f'<span style="color: var(--pico-del-color, red);">Error: {escape(str(exc))}</span>')


# --- Background tasks ---


async def _run_init(topic_id: int, settings: Settings, db_path: Path | None = None) -> None:
    """Background task: fetch articles and build initial knowledge state.

    Creates its own database connection since the request connection
    is closed by the time this runs.
    """
    from app.analysis.knowledge import initialize_knowledge
    from app.crud import mark_articles_processed
    from app.database import get_db
    from app.scraping import fetch_new_articles_for_topic

    if not await _checking_state.start_check(topic_id):
        logger.info("Init background task: topic %d already being initialized, skipping", topic_id)
        return

    try:
        with get_db(db_path) as conn:
            topic = get_topic(conn, topic_id)
            if topic is None:
                logger.error("Init background task: topic %d not found", topic_id)
                return

            async def _do_init_work(topic, conn):
                articles = await fetch_new_articles_for_topic(
                    topic,
                    conn,
                    max_articles=settings.max_articles_per_check,
                    feed_fetch_timeout=settings.feed_fetch_timeout,
                    article_fetch_timeout=settings.article_fetch_timeout,
                    feed_max_retries=settings.feed_max_retries,
                    concurrency=settings.content_fetch_concurrency,
                )

                if not articles:
                    topic.status = TopicStatus.ERROR
                    topic.error_message = "No articles found during initialization"
                    update_topic(conn, topic)
                    return

                await initialize_knowledge(topic, articles, conn, settings)

                article_ids = [a.id for a in articles if a.id is not None]
                if article_ids:
                    mark_articles_processed(conn, article_ids)

                topic.status = TopicStatus.READY
                topic.error_message = None
                update_topic(conn, topic)

            try:
                await asyncio.wait_for(_do_init_work(topic, conn), timeout=_INIT_TIMEOUT_SECONDS)
            except TimeoutError:
                logger.error(
                    "Init timed out for topic '%s' after %d seconds",
                    topic.name,
                    _INIT_TIMEOUT_SECONDS,
                )
                topic.status = TopicStatus.ERROR
                topic.error_message = "Research timed out. Click Retry."
                update_topic(conn, topic)
            except Exception as exc:
                logger.error("Init failed for topic '%s'", topic.name, exc_info=True)
                topic.status = TopicStatus.ERROR
                topic.error_message = str(exc)
                update_topic(conn, topic)
    finally:
        await _checking_state.finish_check(topic_id)


async def _run_single_check(topic_id: int, settings: Settings, db_path: Path | None = None) -> None:
    """Background task: check a single topic by ID."""
    from app.database import get_db

    try:
        with get_db(db_path) as conn:
            topic = get_topic(conn, topic_id)
            if topic:
                await check_topic(topic, conn, settings)
    except Exception:
        logger.error("Background check failed for topic %d", topic_id, exc_info=True)


async def _run_check_all(settings: Settings, db_path: Path | None = None) -> None:
    """Background task: check all topics for new information."""
    from app.database import get_db

    try:
        with get_db(db_path) as conn:
            try:
                await asyncio.wait_for(check_all_topics(conn, settings), timeout=_CHECK_ALL_TIMEOUT_SECONDS)
            except TimeoutError:
                logger.error("Check all timed out after %d seconds", _CHECK_ALL_TIMEOUT_SECONDS)
    except Exception:
        logger.error("Check all background task failed", exc_info=True)
    finally:
        await _checking_state.finish_check_all()
