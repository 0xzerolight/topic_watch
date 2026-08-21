"""Topic CRUD, detail, articles, check/init triggers, and per-topic exports."""

import asyncio
import logging
import sqlite3
from datetime import UTC, datetime

from fastapi import APIRouter, BackgroundTasks, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from app.analysis.knowledge_diff import diff_segments
from app.analysis.llm import NoveltyResult
from app.checker import build_notification_intents, deliver_notification_intents
from app.config import Settings
from app.crud import (
    claim_topic_for_init,
    count_articles_for_topic,
    count_check_results,
    create_notification_intents,
    create_topic,
    create_webhook_intents,
    delete_topic,
    get_check_result,
    get_feed_health,
    get_knowledge_revision,
    get_knowledge_state,
    get_previous_knowledge_revision,
    get_topic,
    get_topic_by_name,
    list_article_headers_for_topic,
    list_check_results,
    list_knowledge_revision_headers,
    mark_check_seen,
    sum_check_tokens,
    update_topic,
    update_topic_config,
    update_topic_init_status,
)
from app.database import short_conn
from app.models import (
    NOVELTY_INSTRUCTION_MAX_CHARS,
    FeedMode,
    KnowledgeRevisionSource,
    Topic,
    TopicStatus,
    normalize_tags,
)
from app.notifications import format_notification
from app.scraping.routing import router as provider_router
from app.web.csrf import verify_csrf
from app.web.dependencies import get_db_conn, get_settings
from app.web.routers import background
from app.web.routers._validation import (
    parse_importance,
    parse_novelty_instruction,
    parse_threshold,
    parse_topic_name,
    validate_topic_form,
)
from app.web.routers.templates import templates
from app.web.state import _checking_state
from app.webhooks import build_webhook_intents, deliver_webhook_intents

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/topics/new", response_class=HTMLResponse)
async def topic_add_form(request: Request, settings: Settings = Depends(get_settings)):
    """Render the add topic form."""
    return templates.TemplateResponse(
        request,
        "topic_add.html",
        {
            "global_confidence_threshold": settings.min_confidence_threshold,
            "global_relevance_threshold": settings.min_relevance_threshold,
            "novelty_instruction_max": NOVELTY_INSTRUCTION_MAX_CHARS,
            "exa_enabled": settings.exa.enabled,
        },
    )


@router.post("/topics", dependencies=[Depends(verify_csrf)])
async def create_topic_handler(
    request: Request,
    background_tasks: BackgroundTasks,
    conn: sqlite3.Connection = Depends(get_db_conn, scope="function"),
    settings: Settings = Depends(get_settings),
    name: str = Form(...),
    description: str = Form(...),
    feed_urls: str = Form(""),
    feed_mode: str = Form("auto"),
    check_interval: str = Form(""),
    tags: str = Form(""),
    confidence_threshold: str = Form(""),
    relevance_threshold: str = Form(""),
    novelty_instruction: str = Form(""),
    importance_threshold: str = Form(""),
):
    """Create a new topic and kick off initial research in the background."""
    from app.interval import format_interval

    mode, urls, parsed_interval, errors = await validate_topic_form(feed_mode, feed_urls, check_interval)
    name = parse_topic_name(name, errors)
    tag_list = normalize_tags(tags.splitlines())
    conf_threshold = parse_threshold(confidence_threshold, "Confidence threshold", errors)
    rel_threshold = parse_threshold(relevance_threshold, "Relevance threshold", errors)
    instruction = parse_novelty_instruction(novelty_instruction, errors)
    imp_threshold = parse_importance(importance_threshold, errors)

    # Guard: a brand-new EXA topic while Exa is disabled would instantly ERROR (no source).
    if mode == FeedMode.EXA and not settings.exa.enabled:
        errors.append("Exa search is not enabled. Configure an Exa API key in Settings first.")

    def _render_errors() -> HTMLResponse:
        # Reuse the already-parsed interval (no re-parse) for the schedule preview.
        formatted = format_interval(parsed_interval) if parsed_interval else ""
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
                "interval_preview": formatted,
                "tags": tags,
                "confidence_threshold": confidence_threshold,
                "relevance_threshold": relevance_threshold,
                "novelty_instruction": novelty_instruction,
                "importance_threshold": importance_threshold,
                "global_confidence_threshold": settings.min_confidence_threshold,
                "global_relevance_threshold": settings.min_relevance_threshold,
                "novelty_instruction_max": NOVELTY_INSTRUCTION_MAX_CHARS,
                "exa_enabled": settings.exa.enabled,
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
        novelty_instruction=instruction,
        importance_threshold=imp_threshold,
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
    # The INSERT above created the row already in RESEARCHING, so this request owns
    # the claim outright — nobody else can have seen the topic yet.
    background_tasks.add_task(background._run_init, created.id, settings, db_path, None, claimed=True)

    return RedirectResponse(url=f"/topics/{created.id}", status_code=303)


def _feed_source_context(conn: sqlite3.Connection, topic: Topic, topic_id: int) -> dict:
    """Feed Source section context, shared by the detail page and its poll endpoint.

    Single source for auto_feed_url / auto_feed_urls / feed_health_map so the full-page
    render (topic_detail) and the /feed-source HTMX fragment can never drift. Mirrors the
    _topic_row_context anti-drift helper (OVH-154). Takes ``topic_id`` explicitly
    (rather than ``topic.id``, which is Optional) like that helper does.

    Also carries ``latest_check`` (AUG-224): the "sources failing" callout used to
    read ``checks[0]`` from the paginated, load-time history and only render on
    page 1, so it vanished on older pages and never updated after a later check
    failed or recovered while the tab stayed open. Querying it here instead means
    it renders on every page and refreshes on this fragment's own 30s poll cadence.
    """
    auto_feed_url = None
    auto_feed_urls: list[str] = []
    feed_health_map = {}
    if topic.feed_mode == FeedMode.AUTO:
        auto_feed_url = provider_router.get_provider().build_feed_url(topic)
        auto_feed_urls = [p.build_feed_url(topic) for p in provider_router.providers]
        # Show health for all provider URLs, not just the active one.
        for url in auto_feed_urls:
            health = get_feed_health(conn, url)
            if health:
                feed_health_map[url] = health
    else:
        for url in topic.feed_urls:
            health = get_feed_health(conn, url)
            if health:
                feed_health_map[url] = health

    latest_checks = list_check_results(conn, topic_id, limit=1)

    return {
        "auto_feed_url": auto_feed_url,
        "auto_feed_urls": auto_feed_urls,
        "feed_health_map": feed_health_map,
        "latest_check": latest_checks[0] if latest_checks else None,
    }


@router.get("/topics/{topic_id}", response_class=HTMLResponse)
async def topic_detail(
    request: Request,
    topic_id: int,
    conn: sqlite3.Connection = Depends(get_db_conn, scope="function"),
    settings: Settings = Depends(get_settings),
    page: int = 1,
):
    """Topic detail page: knowledge state, check history, actions."""
    from app.interval import format_interval

    topic = get_topic(conn, topic_id)
    if topic is None:
        raise HTTPException(status_code=404, detail="Topic not found")

    # TW-AUD-024: no mutation on GET. This used to stamp seen_at (clearing the
    # dashboard "new info" badge) here, before the reads/render below that can
    # still fail — a prefetch, retry, or render failure could clear the badge
    # without the user ever seeing the detail. The page instead fires an
    # idempotent POST to /checks/{check_id}/seen once it has actually rendered
    # (see the hidden trigger near Check History), keyed to the displayed check.

    per_page = settings.web_page_size
    total_checks = count_check_results(conn, topic_id)
    total_pages = max(1, (total_checks + per_page - 1) // per_page)
    # Count first, then clamp into range. An unbounded page rendered an empty list
    # as the "No checks performed yet" empty state even when history existed, and a
    # very large one produced an OFFSET too big for SQLite to bind (TW-AUD-023).
    page = min(max(1, page), total_pages)
    offset = (page - 1) * per_page

    knowledge = get_knowledge_state(conn, topic_id)
    # Fetch the full retained set, NOT web_page_size (default 20 vs a cap of up
    # to 200): the page's single ``page`` param already drives check history, so
    # a second paginated list would strand later revisions. This query omits
    # summary_text and is covered by idx_knowledge_revisions_topic.
    revisions = list_knowledge_revision_headers(conn, topic_id, limit=settings.knowledge_revision_limit)
    checks = list_check_results(conn, topic_id, limit=per_page, offset=offset)
    total_prompt_tokens, total_completion_tokens = sum_check_tokens(conn, topic_id)
    # AUG-038: metadata only — the template never renders raw_content, so this
    # path stops hydrating it (list_articles_for_topic stays for exports/analysis).
    articles = list_article_headers_for_topic(conn, topic_id, limit=per_page)
    article_count = count_articles_for_topic(conn, topic_id)

    formatted = format_interval(topic.check_interval_minutes) if topic.check_interval_minutes else ""
    return templates.TemplateResponse(
        request,
        "topic_detail.html",
        {
            "topic": topic,
            "knowledge": knowledge,
            "revisions": revisions,
            "checks": checks,
            "articles": articles,
            "article_count": article_count,
            "page": page,
            "total_pages": total_pages,
            "formatted_interval": formatted,
            "default_interval": settings.check_interval,
            "knowledge_state_max_tokens": settings.knowledge_state_max_tokens,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "global_confidence_threshold": settings.min_confidence_threshold,
            "global_relevance_threshold": settings.min_relevance_threshold,
            **_feed_source_context(conn, topic, topic_id),
        },
    )


@router.post("/topics/{topic_id}/checks/{check_id}/seen", dependencies=[Depends(verify_csrf)])
async def mark_check_seen_handler(
    topic_id: int,
    check_id: int,
    conn: sqlite3.Connection = Depends(get_db_conn, scope="function"),
) -> Response:
    """Acknowledge one displayed check result (TW-AUD-024).

    Fired by the detail page itself once its content has rendered (see the
    hidden ``hx-trigger="load"`` element near Check History) instead of the old
    GET-time mutation. Idempotent and keyed to ``check_id``: a stale page (a
    newer check has since landed) is a no-op rather than acking the wrong row.
    """
    mark_check_seen(conn, topic_id, check_id)
    conn.commit()
    return Response(status_code=204)


@router.get("/topics/{topic_id}/checks/{check_id}/detail", response_class=HTMLResponse)
async def check_detail(
    request: Request,
    topic_id: int,
    check_id: int,
    conn: sqlite3.Connection = Depends(get_db_conn, scope="function"),
):
    """Render the stored novelty findings for one check (AUG-110).

    Lazy-loaded by HTMX on first expand of the Check History "New Info"
    disclosure, mirroring ``topic_knowledge_diff``. The data is already
    persisted and already reaches notifications; this is a read path, not a
    new LLM call. ``reasoning`` is deliberately never rendered — it is the
    model's raw chain-of-thought, not a finding. GET only, so no CSRF
    (web.md).
    """
    check_result = get_check_result(conn, check_id)
    # The topic_id check is not redundant: without it any topic's URL could
    # read any other topic's findings (mirrors topic_knowledge_diff).
    if check_result is None or check_result.topic_id != topic_id:
        raise HTTPException(status_code=404, detail="Check result not found")

    novelty: NoveltyResult | None = None
    if check_result.llm_response:
        try:
            novelty = NoveltyResult.model_validate_json(check_result.llm_response)
        except Exception:
            logger.warning("Could not parse llm_response for check %d", check_id, exc_info=True)

    return templates.TemplateResponse(request, "_check_detail.html", {"novelty": novelty})


@router.get("/topics/{topic_id}/status", response_class=HTMLResponse)
async def topic_status(
    request: Request,
    topic_id: int,
    since: str | None = None,
    conn: sqlite3.Connection = Depends(get_db_conn, scope="function"),
    settings: Settings = Depends(get_settings),
):
    """HTMX partial: knowledge state fragment for polling during research.

    ``since`` carries the status the client rendered its page under. When the
    status has moved on, the response gets ``HX-Refresh: true`` so htmx reloads
    the whole page once — the h1 badge, Actions row and error alert live outside
    ``#status-area`` and a fragment swap alone would leave them stale. htmx
    discards the body on HX-Refresh; non-htmx callers ignore the header.
    """
    topic = get_topic(conn, topic_id)
    if topic is None:
        # Topic deleted mid-research: return a 200 terminal fragment (no polling
        # trigger) so the every-3s HTMX poll swaps it in and stops (OVH-048).
        # Deliberately no HX-Refresh here — a reload would land on a 404 page,
        # the terminal fragment is the better end state.
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

    response = templates.TemplateResponse(
        request,
        "topic_status.html",
        {
            "topic": topic,
            "knowledge": knowledge,
            "knowledge_state_max_tokens": settings.knowledge_state_max_tokens,
        },
    )
    if since is not None and topic.status.value != since:
        response.headers["HX-Refresh"] = "true"
    return response


@router.get("/topics/{topic_id}/knowledge-diff/{revision_id}", response_class=HTMLResponse)
async def topic_knowledge_diff(
    request: Request,
    topic_id: int,
    revision_id: int,
    conn: sqlite3.Connection = Depends(get_db_conn, scope="function"),
):
    """Render the diff between a knowledge revision and the one before it.

    Lazy-loaded by HTMX on first expand, so the detail page never ships every
    revision's summary. Diffs are computed on read rather than stored — the
    summaries are the source of truth. GET only, so no CSRF (web.md).
    """
    revision = get_knowledge_revision(conn, revision_id)
    # The topic_id check is not redundant: without it any topic's URL could read
    # any other topic's revision. It also subsumes the missing-topic 404, since
    # the FK is ON DELETE CASCADE.
    if revision is None or revision.topic_id != topic_id:
        raise HTTPException(status_code=404, detail="Revision not found")

    previous = get_previous_knowledge_revision(conn, topic_id, revision_id)
    # An 'init' revision starts a new lineage (first research, or Re-initialize).
    # Diffing it against the last revision of the OLD lineage would render a
    # wholesale delete+insert as though the model had rewritten its
    # understanding, so treat it as having no predecessor.
    if revision.source is KnowledgeRevisionSource.INIT:
        previous = None

    previous_text = previous.summary_text if previous else ""
    # difflib is CPU-bound — up to ~0.32 s at MAX_DIFF_SEGMENTS on repetitive
    # input — and would otherwise block the event loop (CLAUDE.md).
    segments = await asyncio.to_thread(diff_segments, previous_text, revision.summary_text)
    inserted = sum(1 for segment in segments if segment.kind == "insert")
    deleted = sum(1 for segment in segments if segment.kind == "delete")
    token_delta = revision.token_count - (previous.token_count if previous else 0)

    return templates.TemplateResponse(
        request,
        "_knowledge_diff.html",
        {
            "revision": revision,
            "previous": previous,
            "segments": segments,
            "inserted": inserted,
            "deleted": deleted,
            "token_delta": token_delta,
        },
    )


@router.get("/topics/{topic_id}/feed-source", response_class=HTMLResponse)
async def topic_feed_source(
    request: Request,
    topic_id: int,
    conn: sqlite3.Connection = Depends(get_db_conn, scope="function"),
):
    """HTMX partial: feed-source health fragment, polled every 30s from the detail page.

    Feed health is written by the background scheduler with no client-side event, so the
    detail page polls this fragment to refresh the badges live (mirrors the #status-area
    self-terminating poll, OVH-048). GET only — no CSRF needed (web.md).
    """
    topic = get_topic(conn, topic_id)
    if topic is None:
        # Topic deleted mid-poll: return a 200 terminal fragment (no poll trigger) so the
        # every-30s HTMX poll swaps it in and stops, matching topic_status()'s handling.
        return templates.TemplateResponse(request, "_feed_source.html", {"topic": None})

    return templates.TemplateResponse(
        request,
        "_feed_source.html",
        {"topic": topic, **_feed_source_context(conn, topic, topic_id)},
    )


def _topic_row_context(conn: sqlite3.Connection, topic: Topic, topic_id: int) -> dict:
    """Build the shared ``_topic_row.html`` context (OVH-154).

    Single source for the topic-row partial's data: the topic, its most recent
    check, and a COUNT-based article total. Used by both the check/redirect path
    and toggle-active so the article-count source can no longer drift between
    them. ``just_checked`` is intentionally NOT included here — callers add it
    only where a fresh check just ran (see ``_topic_row_response``); omitting it
    leaves the marker falsy so an unrelated re-render like toggle-active does not
    re-fire a browser notification (OVH-119).
    """
    checks = list_check_results(conn, topic_id, limit=1)
    return {
        "topic": topic,
        "last_check": checks[0] if checks else None,
        "article_count": count_articles_for_topic(conn, topic_id),
    }


def _topic_row_response(
    request: Request,
    conn: sqlite3.Connection,
    topic: Topic,
    topic_id: int,
    *,
    just_checked: bool = False,
    checking: bool = False,
    baseline_check_id: int | None = None,
) -> Response:
    """Render the topic-row partial for HTMX, or redirect to the detail page for a full navigation.

    ``just_checked`` marks the row as the result of a fresh check (emits
    ``data-just-checked`` for the dashboard afterSwap handler) so unrelated
    re-renders like toggle-active don't re-fire a browser notification (OVH-119).

    ``checking`` renders the row polling toward the completion of a just-queued
    background check, keyed to ``baseline_check_id`` — the newest check result
    that existed before the check was queued (AUG-217).
    """
    if not request.headers.get("HX-Request"):
        return RedirectResponse(url=f"/topics/{topic_id}", status_code=303)

    return templates.TemplateResponse(
        request,
        "_topic_row.html",
        {
            **_topic_row_context(conn, topic, topic_id),
            "just_checked": just_checked,
            "checking": checking,
            "baseline_check_id": baseline_check_id,
        },
    )


@router.get("/topics/{topic_id}/row", response_class=HTMLResponse)
async def topic_row(
    request: Request,
    topic_id: int,
    since: str | None = None,
    since_check_id: int | None = None,
    conn: sqlite3.Connection = Depends(get_db_conn, scope="function"),
):
    """HTMX partial: single dashboard row, polled while a topic is new/researching
    or while a just-queued manual check is still in flight.

    ``since`` is the status the row was rendered with. While unchanged this
    returns 204 (htmx skips the swap, leaving checkbox/focus state untouched);
    on a transition the full row is re-rendered once and, being ready/error, no
    longer carries poll attributes — the poll terminates itself.

    ``since_check_id`` is the newest check result that existed when a manual
    check was queued (0 if none). A READY topic's status never changes across
    a manual check, so completion is detected by a newer ``check_results`` row
    appearing rather than by a status transition (AUG-217): while none exists
    and the background task is still running, this returns 204; once one
    exists, the row is re-rendered with ``data-just-checked`` set exactly once.
    If the task is no longer running and still produced no newer row (e.g. it
    crashed before ``check_topic``'s transaction committed), the row is
    rendered as-is so the poll does not spin forever. GET only — no CSRF
    needed (web.md).
    """
    topic = get_topic(conn, topic_id)
    if topic is None:
        # Topic deleted mid-poll: 200 empty body so the outerHTML swap removes
        # the row and the poll dies with it (OVH-048).
        return HTMLResponse("")

    if since_check_id is not None:
        latest = list_check_results(conn, topic_id, limit=1)
        if latest and latest[0].id != since_check_id:
            return _topic_row_response(request, conn, topic, topic_id, just_checked=True)
        if await _checking_state.is_checking(topic_id):
            return Response(status_code=204)
        return _topic_row_response(request, conn, topic, topic_id)

    if since is not None and topic.status.value == since:
        return Response(status_code=204)

    # just_checked stays False: a status-poll re-render must not re-fire the
    # dashboard's browser notification (OVH-119).
    return _topic_row_response(request, conn, topic, topic_id)


@router.post("/topics/{topic_id}/check", response_class=HTMLResponse, dependencies=[Depends(verify_csrf)])
async def check_topic_handler(
    request: Request,
    topic_id: int,
    background_tasks: BackgroundTasks,
    conn: sqlite3.Connection = Depends(get_db_conn, scope="function"),
    settings: Settings = Depends(get_settings),
):
    """Manual check trigger.

    Enqueues the fetch+LLM pipeline as a background task (it opens its own
    connection) and returns immediately, so the request connection is never
    held across the long awaits. The response renders a checking row that
    polls ``topic_row`` toward completion; HTMX requests get the topic-row
    partial, plain-form submissions redirect to the topic detail page.

    The response cannot claim ``data-just-checked`` here: the background task
    has not run yet, so this still renders the pre-check row. Doing so anyway
    let a stale unseen result re-fire a notification while the real completion
    produced no swap at all (AUG-217) — the marker is set only once
    ``topic_row``'s poll observes a newer check result.
    """
    await _checking_state.clear_stale(600)

    topic = get_topic(conn, topic_id)
    if topic is None:
        raise HTTPException(status_code=404, detail="Topic not found")

    baseline = list_check_results(conn, topic_id, limit=1)
    baseline_check_id = baseline[0].id if baseline else None

    if await _checking_state.is_checking(topic_id):
        # Already checking — return current state without enqueueing a duplicate,
        # still polling toward the in-flight check's completion.
        return _topic_row_response(request, conn, topic, topic_id, checking=True, baseline_check_id=baseline_check_id)

    # Defer the pipeline to a background task with its own connection. The task
    # is the authoritative owner of the per-topic guard: it acquires
    # ``start_check`` at entry (so two near-simultaneous submissions still
    # de-dupe even though both passed the read above) and releases it when done
    # (OVH-033/OVH-096).
    db_path = getattr(request.app.state, "db_path", None)
    background_tasks.add_task(background._run_single_check, topic_id, settings, db_path)

    return _topic_row_response(request, conn, topic, topic_id, checking=True, baseline_check_id=baseline_check_id)


@router.post("/topics/{topic_id}/toggle-active", dependencies=[Depends(verify_csrf)])
async def set_topic_active(
    request: Request,
    topic_id: int,
    conn: sqlite3.Connection = Depends(get_db_conn, scope="function"),
    active: bool = Form(...),
):
    """Set a topic's monitoring state to the submitted value.

    The desired state travels in the request instead of being derived by negating
    the stored row. Negation made the command a toggle, so replaying an identical
    POST — which the built-in HTMX error toast does after an ambiguous network or
    response failure — silently re-enabled the checks, notifications and provider
    spend the user had just turned off. Reapplying the same command is now a no-op
    (AUG-290). The path is unchanged so existing bookmarks and templates keep
    working.
    """
    topic = get_topic(conn, topic_id)
    if topic is None:
        raise HTTPException(status_code=404, detail="Topic not found")

    if topic.is_active != active:
        topic.is_active = active
        update_topic(conn, topic)
        conn.commit()

    # HTMX request from dashboard — return updated row partial. No just_checked:
    # a toggle is not a fresh check, so the marker stays absent (OVH-119/OVH-154).
    if request.headers.get("HX-Request"):
        return templates.TemplateResponse(
            request,
            "_topic_row.html",
            _topic_row_context(conn, topic, topic_id),
        )

    return RedirectResponse(url=f"/topics/{topic_id}", status_code=303)


@router.post("/topics/{topic_id}/init", dependencies=[Depends(verify_csrf)])
async def reinit_topic(
    request: Request,
    topic_id: int,
    background_tasks: BackgroundTasks,
    conn: sqlite3.Connection = Depends(get_db_conn, scope="function"),
    settings: Settings = Depends(get_settings),
):
    """Re-trigger initial research for error recovery.

    Ownership is taken before the status changes, not after. The handler used to
    commit RESEARCHING and then queue a task that tried to claim the in-flight
    guard; if a check of the same topic held it, the task exited silently and the
    topic sat in RESEARCHING until stuck recovery called it an error (AUG-137).
    Both the guard and the durable claim are decided here, so a refusal is
    something the user sees and no status is written for work that never starts.
    """
    topic = get_topic(conn, topic_id)
    if topic is None:
        raise HTTPException(status_code=404, detail="Topic not found")

    assert topic.id is not None
    if not topic.is_active:
        raise HTTPException(status_code=409, detail="Topic is paused. Enable it before re-initializing.")

    # Mirrors the refusal ``initialize_new_topic`` already makes. Without it the
    # claim below is a RESEARCHING -> RESEARCHING self-transition that always
    # wins, so an initializer held by another process — a CLI ``init``, a second
    # container — leaves this handler free to admit a second one. Both then spend
    # on the same init, and whichever finishes first has its knowledge write
    # rolled back by the other's terminal status.
    if topic.status is TopicStatus.RESEARCHING:
        raise HTTPException(status_code=409, detail="This topic is already being initialized.")

    owner = await _checking_state.start_check(topic_id)
    if owner is None:
        raise HTTPException(status_code=409, detail="This topic is busy right now. Try again when it finishes.")

    claimed = claim_topic_for_init(conn, topic_id, topic.status)
    if not claimed:
        await _checking_state.finish_check(topic_id, owner)
        raise HTTPException(status_code=409, detail="This topic is busy right now. Try again when it finishes.")

    # An explicit Retry starts from a clean slate (OVH-098): reset the vestigial
    # init_attempts counter, fenced to the claim this handler just won.
    update_topic_init_status(
        conn,
        topic_id,
        status=TopicStatus.RESEARCHING,
        status_changed_at=datetime.now(UTC),
        error_message=None,
        init_attempts=0,
        expected_status=TopicStatus.RESEARCHING,
    )
    conn.commit()

    db_path = getattr(request.app.state, "db_path", None)
    background_tasks.add_task(background._run_init, topic.id, settings, db_path, owner, claimed=True)

    return RedirectResponse(url=f"/topics/{topic_id}", status_code=303)


@router.post("/topics/{topic_id}/delete", dependencies=[Depends(verify_csrf)])
async def delete_topic_handler(
    topic_id: int,
    conn: sqlite3.Connection = Depends(get_db_conn, scope="function"),
):
    """Delete a topic and redirect to dashboard."""
    delete_topic(conn, topic_id)
    conn.commit()
    return RedirectResponse(url="/", status_code=303)


@router.get("/topics/{topic_id}/edit", response_class=HTMLResponse)
async def topic_edit_form(
    request: Request,
    topic_id: int,
    conn: sqlite3.Connection = Depends(get_db_conn, scope="function"),
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
            # One tag per line, matching the textarea the form now uses. Joining
            # with commas and reparsing on comma split a single OPML folder such
            # as "Policy, Europe" into two tags on any unchanged save (AUG-339).
            "tags_string": "\n".join(topic.tags),
            "global_confidence_threshold": settings.min_confidence_threshold,
            "global_relevance_threshold": settings.min_relevance_threshold,
            "novelty_instruction_max": NOVELTY_INSTRUCTION_MAX_CHARS,
            "exa_enabled": settings.exa.enabled,
        },
    )


@router.post("/topics/{topic_id}/edit", dependencies=[Depends(verify_csrf)])
async def edit_topic_handler(
    request: Request,
    topic_id: int,
    conn: sqlite3.Connection = Depends(get_db_conn, scope="function"),
    settings: Settings = Depends(get_settings),
    name: str = Form(...),
    description: str = Form(...),
    feed_urls: str = Form(""),
    feed_mode: str = Form("auto"),
    check_interval: str = Form(""),
    tags: str = Form(""),
    confidence_threshold: str = Form(""),
    relevance_threshold: str = Form(""),
    novelty_instruction: str = Form(""),
    importance_threshold: str = Form(""),
):
    """Update an existing topic's name, description, feed URLs, and feed mode."""
    topic = get_topic(conn, topic_id)
    if topic is None:
        raise HTTPException(status_code=404, detail="Topic not found")

    from app.interval import format_interval

    mode, urls, parsed_interval, errors = await validate_topic_form(feed_mode, feed_urls, check_interval)
    name = parse_topic_name(name, errors)
    tag_list = normalize_tags(tags.splitlines())
    conf_threshold = parse_threshold(confidence_threshold, "Confidence threshold", errors)
    rel_threshold = parse_threshold(relevance_threshold, "Relevance threshold", errors)
    instruction = parse_novelty_instruction(novelty_instruction, errors)
    imp_threshold = parse_importance(importance_threshold, errors)

    # Guard only a CONVERSION into EXA while Exa is disabled: block turning a working
    # AUTO/MANUAL topic into a non-fetching EXA one. An already-EXA topic edits freely
    # (it degrades gracefully and is surfaced via the check/init failure path).
    if mode == FeedMode.EXA and topic.feed_mode != FeedMode.EXA and not settings.exa.enabled:
        errors.append("Exa search is not enabled. Configure an Exa API key in Settings first.")

    def _render_errors() -> HTMLResponse:
        # Reuse the already-parsed interval (no re-parse) for the schedule preview.
        formatted = format_interval(parsed_interval) if parsed_interval else ""
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
                "interval_preview": formatted,
                "tags": tags,
                "confidence_threshold": confidence_threshold,
                "relevance_threshold": relevance_threshold,
                "novelty_instruction": novelty_instruction,
                "importance_threshold": importance_threshold,
                "default_interval": settings.check_interval,
                "global_confidence_threshold": settings.min_confidence_threshold,
                "global_relevance_threshold": settings.min_relevance_threshold,
                "novelty_instruction_max": NOVELTY_INSTRUCTION_MAX_CHARS,
                "exa_enabled": settings.exa.enabled,
            },
            status_code=422,
        )

    if errors:
        return _render_errors()

    # Renaming onto another topic's name hit the UNIQUE constraint and reached the
    # global 500 handler, throwing the submitted form away. Creation already
    # translates this into form feedback; the edit path now matches it (AUG-147).
    clash = get_topic_by_name(conn, name)
    if clash is not None and clash.id != topic_id:
        errors.append("A topic with that name already exists")
        return _render_errors()

    topic.name = name
    topic.description = description
    topic.feed_urls = urls
    topic.feed_mode = mode
    topic.check_interval_minutes = parsed_interval
    topic.tags = tag_list
    topic.confidence_threshold = conf_threshold
    topic.relevance_threshold = rel_threshold
    topic.novelty_instruction = instruction
    topic.importance_threshold = imp_threshold
    try:
        # Configuration columns only: this snapshot is older than the DNS
        # validation above, so writing its lifecycle fields back would undo any
        # status transition that landed during that await (AUG-022).
        update_topic_config(conn, topic)
        conn.commit()
    except sqlite3.IntegrityError:
        # Defense-in-depth against a name race between the check above and the
        # UPDATE, mirroring create_topic_handler.
        conn.rollback()
        errors.append("A topic with that name already exists")
        return _render_errors()

    return RedirectResponse(url=f"/topics/{topic_id}", status_code=303)


@router.post("/topics/bulk-delete", dependencies=[Depends(verify_csrf)])
async def bulk_delete_handler(
    request: Request,
    conn: sqlite3.Connection = Depends(get_db_conn, scope="function"),
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
    conn: sqlite3.Connection = Depends(get_db_conn, scope="function"),
    settings: Settings = Depends(get_settings),
):
    """Trigger checks for multiple topics."""
    form = await request.form()
    topic_ids = form.getlist("topic_ids")
    db_path = getattr(request.app.state, "db_path", None)
    # Dedup so a duplicated checkbox id (crafted form or double-submit) cannot
    # queue the same topic's check twice in one request (OVH-166). Preserve the
    # first-seen order; the per-topic guard in _run_single_check would skip a
    # same-process duplicate anyway, but dropping it here avoids the redundant
    # sequential background task entirely.
    queued: set[int] = set()
    for tid in topic_ids:
        try:
            topic_id = int(str(tid))
        except (TypeError, ValueError) as exc:
            logger.warning("Failed to queue check for topic %s: %s", tid, exc)
            continue
        if topic_id in queued:
            continue
        try:
            topic = get_topic(conn, topic_id)
            if topic and topic.id is not None and topic.status == TopicStatus.READY:
                background_tasks.add_task(background._run_single_check, topic.id, settings, db_path)
                queued.add(topic_id)
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
    owner = _checking_state.start_check_all()
    if owner is not None:
        db_path = getattr(request.app.state, "db_path", None)
        background_tasks.add_task(background._run_check_all, settings, db_path, owner)
    return RedirectResponse(url="/", status_code=303)


@router.post("/topics/{topic_id}/checks/{check_id}/notify", dependencies=[Depends(verify_csrf)])
async def force_notify(
    request: Request,
    topic_id: int,
    check_id: int,
    conn: sqlite3.Connection = Depends(get_db_conn, scope="function"),
    settings: Settings = Depends(get_settings),
):
    """Re-send a specific check result through EVERY configured channel.

    The automatic path sends Apprise notifications AND webhooks; this manual
    recovery action used to call only Apprise, so a webhook-only configuration was
    told "Delivery failed" without its one channel ever being attempted (AUG-109).

    The resend goes through the same durable delivery intents as an automatic one,
    so a target that fails here is retried by the drain rather than lost.
    """
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

    if not settings.notifications.urls and not settings.notifications.webhook_urls:
        return HTMLResponse(
            '<span style="color: var(--pico-del-color, red);">No delivery target configured</span>',
            status_code=400,
        )

    db_path = getattr(request.app.state, "db_path", None)
    # Every write below is a short, await-free block, so reusing the request
    # connection when the app names no explicit path is safe — and it is the only
    # way to reach the same database the request is already reading.
    own_conn = None if db_path is not None else conn
    try:
        novelty = NoveltyResult.model_validate_json(check_result.llm_response)
        title, body = format_notification(topic.name, novelty)
        intents = build_notification_intents(title, body, settings, topic_id, check_result_id=check_id)
        webhook_intents = build_webhook_intents(topic.name, novelty, settings, topic_id, check_id)

        # Persist the intents and commit BEFORE any send, exactly as the automatic
        # path does: a resend that dies mid-flight is still owed, not lost.
        with short_conn(own_conn, db_path) as intent_conn:
            create_notification_intents(intent_conn, intents)
            create_webhook_intents(intent_conn, webhook_intents)
            intent_conn.commit()

        deliveries = await deliver_notification_intents(intents, settings, db_path, own_conn)
        webhooks_sent = await deliver_webhook_intents(webhook_intents, settings, db_path, own_conn)

        parts: list[str] = []
        if intents:
            parts.append(f"Apprise {sum(1 for d in deliveries if d.ok)}/{len(intents)}")
        if webhook_intents:
            parts.append(f"webhooks {webhooks_sent}/{len(webhook_intents)}")
        summary = ", ".join(parts)

        all_ok = all(d.ok for d in deliveries) and webhooks_sent == len(webhook_intents)
        if all_ok:
            return HTMLResponse(f'<span style="color: var(--pico-ins-color, green);">Sent! ({summary})</span>')
        return HTMLResponse(
            f'<span style="color: var(--pico-del-color, red);">Delivery failed ({summary}); queued for retry</span>'
        )
    except Exception as exc:
        logger.warning("Force notify failed for check %d", check_id, exc_info=True)
        from markupsafe import escape

        return HTMLResponse(f'<span style="color: var(--pico-del-color, red);">Error: {escape(str(exc))}</span>')
