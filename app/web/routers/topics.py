"""Topic CRUD, detail, articles, check/init triggers, and per-topic exports."""

import csv
import io
import json
import logging
import re
import sqlite3
from datetime import UTC, datetime

from fastapi import APIRouter, BackgroundTasks, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response, StreamingResponse

from app.analysis.llm import NoveltyResult
from app.config import Settings
from app.crud import (
    count_articles_for_topic,
    count_check_results,
    create_topic,
    delete_topic,
    get_check_result,
    get_feed_health,
    get_knowledge_state,
    get_topic,
    get_topic_by_name,
    list_articles_for_topic,
    list_check_results,
    sum_check_tokens,
    update_topic,
)
from app.models import FeedMode, Topic, TopicStatus
from app.notifications import format_notification, send_notification
from app.scraping.routing import router as provider_router
from app.web.csrf import verify_csrf
from app.web.dependencies import get_db_conn, get_settings
from app.web.routers import background
from app.web.routers._validation import parse_threshold, validate_topic_form
from app.web.routers.templates import templates
from app.web.state import _checking_state

logger = logging.getLogger(__name__)

router = APIRouter()

# Upper bound on rows pulled into memory for a single-topic export, so a large
# article/check history can't materialise an unbounded result set (OVH-051).
# Comfortably above any single-user volume; index-backed (m014) so the LIMIT is
# index-ordered, not a full sort.
_EXPORT_ROW_CAP = 10000


@router.get("/topics/new", response_class=HTMLResponse)
async def topic_add_form(request: Request, settings: Settings = Depends(get_settings)):
    """Render the add topic form."""
    return templates.TemplateResponse(
        request,
        "topic_add.html",
        {
            "global_confidence_threshold": settings.min_confidence_threshold,
            "global_relevance_threshold": settings.min_relevance_threshold,
        },
    )


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
    check_interval: str = Form(""),
    tags: str = Form(""),
    confidence_threshold: str = Form(""),
    relevance_threshold: str = Form(""),
):
    """Create a new topic and kick off initial research in the background."""
    mode, urls, parsed_interval, errors = validate_topic_form(feed_mode, feed_urls, check_interval)
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    conf_threshold = parse_threshold(confidence_threshold, "Confidence threshold", errors)
    rel_threshold = parse_threshold(relevance_threshold, "Relevance threshold", errors)

    def _render_errors() -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "topic_add.html",
            {
                "errors": errors,
                "name": name,
                "description": description,
                "feed_urls": feed_urls,
                "feed_mode": feed_mode,
                "check_interval": check_interval,
                "tags": tags,
                "confidence_threshold": confidence_threshold,
                "relevance_threshold": relevance_threshold,
                "global_confidence_threshold": settings.min_confidence_threshold,
                "global_relevance_threshold": settings.min_relevance_threshold,
            },
            status_code=422,
        )

    if errors:
        return _render_errors()

    if get_topic_by_name(conn, name) is not None:
        errors.append("A topic with that name already exists")
        return _render_errors()

    topic = Topic(
        name=name,
        description=description,
        feed_urls=urls,
        feed_mode=mode,
        status=TopicStatus.RESEARCHING,
        status_changed_at=datetime.now(UTC),
        check_interval_minutes=parsed_interval,
        tags=tag_list,
        confidence_threshold=conf_threshold,
        relevance_threshold=rel_threshold,
    )
    try:
        created = create_topic(conn, topic)
        conn.commit()
    except sqlite3.IntegrityError:
        # Defense-in-depth against a name race between the pre-check and INSERT.
        conn.rollback()
        errors.append("A topic with that name already exists")
        return _render_errors()

    assert created.id is not None
    db_path = getattr(request.app.state, "db_path", None)
    background_tasks.add_task(background._run_init, created.id, settings, db_path)

    return RedirectResponse(url=f"/topics/{created.id}", status_code=303)


@router.get("/topics/{topic_id}", response_class=HTMLResponse)
async def topic_detail(
    request: Request,
    topic_id: int,
    conn: sqlite3.Connection = Depends(get_db_conn),
    settings: Settings = Depends(get_settings),
    page: int = 1,
):
    """Topic detail page: knowledge state, check history, actions."""
    from app.interval import format_interval

    topic = get_topic(conn, topic_id)
    if topic is None:
        raise HTTPException(status_code=404, detail="Topic not found")

    auto_feed_url = None
    auto_feed_urls: list[str] = []
    if topic.feed_mode == FeedMode.AUTO:
        auto_feed_url = provider_router.get_provider().build_feed_url(topic)
        auto_feed_urls = [p.build_feed_url(topic) for p in provider_router.providers]

    per_page = settings.web_page_size
    offset = (max(1, page) - 1) * per_page

    knowledge = get_knowledge_state(conn, topic_id)
    checks = list_check_results(conn, topic_id, limit=per_page, offset=offset)
    total_checks = count_check_results(conn, topic_id)
    total_prompt_tokens, total_completion_tokens = sum_check_tokens(conn, topic_id)
    articles = list_articles_for_topic(conn, topic_id, limit=per_page)
    article_count = count_articles_for_topic(conn, topic_id)
    total_pages = max(1, (total_checks + per_page - 1) // per_page)

    feed_health_map = {}
    if topic.feed_mode == FeedMode.AUTO:
        # Show health for all provider URLs, not just the active one
        for provider in provider_router.providers:
            url = provider.build_feed_url(topic)
            health = get_feed_health(conn, url)
            if health:
                feed_health_map[url] = health
    else:
        for url in topic.feed_urls:
            health = get_feed_health(conn, url)
            if health:
                feed_health_map[url] = health

    formatted = format_interval(topic.check_interval_minutes) if topic.check_interval_minutes else ""
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
            "auto_feed_urls": auto_feed_urls,
            "formatted_interval": formatted,
            "default_interval": settings.check_interval,
            "knowledge_state_max_tokens": settings.knowledge_state_max_tokens,
            "feed_health_map": feed_health_map,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
        },
    )


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
        # Topic deleted mid-research: return a 200 terminal fragment (no polling
        # trigger) so the every-3s HTMX poll swaps it in and stops (OVH-048).
        return templates.TemplateResponse(
            request,
            "topic_status.html",
            {
                "topic": None,
                "knowledge": None,
                "knowledge_state_max_tokens": settings.knowledge_state_max_tokens,
            },
        )

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


def _topic_row_response(request: Request, conn: sqlite3.Connection, topic: Topic, topic_id: int) -> Response:
    """Render the topic-row partial for HTMX, or redirect to the detail page for a full navigation."""
    if not request.headers.get("HX-Request"):
        return RedirectResponse(url=f"/topics/{topic_id}", status_code=303)

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


@router.post("/topics/{topic_id}/check", response_class=HTMLResponse, dependencies=[Depends(verify_csrf)])
async def check_topic_handler(
    request: Request,
    topic_id: int,
    background_tasks: BackgroundTasks,
    conn: sqlite3.Connection = Depends(get_db_conn),
    settings: Settings = Depends(get_settings),
):
    """Manual check trigger.

    Enqueues the fetch+LLM pipeline as a background task (it opens its own
    connection) and returns immediately, so the request connection is never
    held across the long awaits. HTMX polling (``topic_status``) surfaces the
    result. HTMX requests get the topic-row partial; plain-form submissions
    redirect to the topic detail page.
    """
    await _checking_state.clear_stale(600)

    topic = get_topic(conn, topic_id)
    if topic is None:
        raise HTTPException(status_code=404, detail="Topic not found")

    if not await _checking_state.start_check(topic_id):
        # Already checking — return current state without re-checking.
        return _topic_row_response(request, conn, topic, topic_id)

    # Defer the pipeline to a background task with its own connection; the
    # task releases the per-topic guard when it completes.
    db_path = getattr(request.app.state, "db_path", None)
    background_tasks.add_task(background._run_single_check, topic_id, settings, db_path)

    return _topic_row_response(request, conn, topic, topic_id)


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
    # OVH-098: explicit Retry restores the full thin-data retry budget, so a topic
    # that previously bumped init_attempts before erroring does not start the retry
    # with a reduced budget.
    topic.init_attempts = 0
    update_topic(conn, topic)
    conn.commit()

    assert topic.id is not None
    db_path = getattr(request.app.state, "db_path", None)
    background_tasks.add_task(background._run_init, topic.id, settings, db_path)

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
    from app.interval import format_interval

    topic = get_topic(conn, topic_id)
    if topic is None:
        raise HTTPException(status_code=404, detail="Topic not found")
    formatted = format_interval(topic.check_interval_minutes) if topic.check_interval_minutes else ""
    return templates.TemplateResponse(
        request,
        "topic_edit.html",
        {
            "topic": topic,
            "formatted_interval": formatted,
            "default_interval": settings.check_interval,
            "tags_string": ", ".join(topic.tags),
            "global_confidence_threshold": settings.min_confidence_threshold,
            "global_relevance_threshold": settings.min_relevance_threshold,
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
    check_interval: str = Form(""),
    tags: str = Form(""),
    confidence_threshold: str = Form(""),
    relevance_threshold: str = Form(""),
):
    """Update an existing topic's name, description, feed URLs, and feed mode."""
    topic = get_topic(conn, topic_id)
    if topic is None:
        raise HTTPException(status_code=404, detail="Topic not found")

    mode, urls, parsed_interval, errors = validate_topic_form(feed_mode, feed_urls, check_interval)
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    conf_threshold = parse_threshold(confidence_threshold, "Confidence threshold", errors)
    rel_threshold = parse_threshold(relevance_threshold, "Relevance threshold", errors)

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
                "check_interval": check_interval,
                "tags": tags,
                "confidence_threshold": confidence_threshold,
                "relevance_threshold": relevance_threshold,
                "default_interval": settings.check_interval,
                "global_confidence_threshold": settings.min_confidence_threshold,
                "global_relevance_threshold": settings.min_relevance_threshold,
            },
            status_code=422,
        )

    topic.name = name
    topic.description = description
    topic.feed_urls = urls
    topic.feed_mode = mode
    topic.check_interval_minutes = parsed_interval
    topic.tags = tag_list
    topic.confidence_threshold = conf_threshold
    topic.relevance_threshold = rel_threshold
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
        except Exception as exc:
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
                background_tasks.add_task(background._run_single_check, topic.id, settings, db_path)
        except Exception as exc:
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
        background_tasks.add_task(background._run_check_all, settings, db_path)
    return RedirectResponse(url="/", status_code=303)


@router.get("/topics/{topic_id}/export/json")
async def export_topic_json(
    topic_id: int,
    conn: sqlite3.Connection = Depends(get_db_conn),
):
    """Export a topic with articles and check results as JSON."""
    topic = get_topic(conn, topic_id)
    if topic is None:
        raise HTTPException(status_code=404, detail="Topic not found")

    articles = list_articles_for_topic(conn, topic_id, limit=_EXPORT_ROW_CAP, offset=0)
    checks = list_check_results(conn, topic_id, limit=_EXPORT_ROW_CAP, offset=0)
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

    checks = list_check_results(conn, topic_id, limit=_EXPORT_ROW_CAP, offset=0)

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
            "stage_error",
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
                check.stage_error or "",
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
