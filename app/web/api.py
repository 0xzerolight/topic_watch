"""JSON API endpoints for Topic Watch (v1).

Read-only API for scripting and monitoring, plus one mutation endpoint
to trigger topic checks. Reuses existing CRUD functions and Pydantic models.
"""

import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from app.checker import check_topic
from app.config import Settings
from app.crud import (
    count_check_results,
    get_knowledge_state,
    get_topic,
    list_check_results,
    list_topics,
    max_check_result_id,
)
from app.database import get_db
from app.models import KnowledgeState, Topic, TopicStatus, normalize_tag
from app.web.csrf import verify_csrf
from app.web.dependencies import get_db_conn, get_settings
from app.web.state import _checking_state

router = APIRouter(prefix="/api/v1", tags=["api"])


class CheckTriggerResponse(BaseModel):
    """Outcome of a triggered check.

    The pipeline is deliberately fail-safe: an unreachable source, a failed
    analysis and a failed delivery all record themselves on the CheckResult and
    return HTTP 200 with ``has_new_info`` false. Returning only status,
    ``has_new_info`` and the row id therefore gave automation the same shape for
    a broken run and a clean quiet one (AUG-203). The recorded outcome fields
    are carried here so a script can tell them apart without a second request.
    """

    status: str
    check_result_id: int | None
    has_new_info: bool
    articles_found: int
    articles_new: int
    stage_error: str | None
    notification_sent: bool
    notification_error: str | None
    notify_disposition: str | None


@router.get("/topics")
async def api_list_topics(
    active: bool | None = None,
    tag: str | None = None,
    conn: sqlite3.Connection = Depends(get_db_conn, scope="function"),
) -> list[Topic]:
    """List all topics with optional filters.

    ``active`` is a tri-state filter: ``true`` returns only active topics,
    ``false`` returns only inactive topics, and omitting it returns all topics.

    ``tag`` is canonicalized the same way stored tags are, so a query differing
    only in Unicode form or spacing still matches (AUG-338).
    """
    if tag is not None:
        tag = normalize_tag(tag)
    return list_topics(conn, is_active=active, tag=tag)


@router.get("/topics/{topic_id}")
async def api_get_topic(
    topic_id: int,
    conn: sqlite3.Connection = Depends(get_db_conn, scope="function"),
) -> dict:
    """Get a single topic with its knowledge state."""
    topic = get_topic(conn, topic_id)
    if topic is None:
        raise HTTPException(status_code=404, detail="Topic not found")
    knowledge = get_knowledge_state(conn, topic_id)
    return {"topic": topic, "knowledge": knowledge}


@router.get("/topics/{topic_id}/checks")
async def api_list_checks(
    topic_id: int,
    page: int = 1,
    per_page: int = 20,
    cutoff_id: int | None = None,
    conn: sqlite3.Connection = Depends(get_db_conn, scope="function"),
) -> dict:
    """Get check history for a topic with pagination.

    ``cutoff_id`` pins the traversal to the check set as of the first page: it
    is minted from the current max id when omitted and returned in the
    response so callers carry it on later requests. Without it, a check
    committing between page requests shifts every row underneath the
    OFFSET-based traversal — page 2 can repeat page 1's last row or skip a
    tail row, and totals can change mid-browse (AUG-314).
    """
    topic = get_topic(conn, topic_id)
    if topic is None:
        raise HTTPException(status_code=404, detail="Topic not found")

    per_page = max(1, min(per_page, 100))
    if cutoff_id is None:
        cutoff_id = max_check_result_id(conn, topic_id)
    total = count_check_results(conn, topic_id, cutoff_id=cutoff_id)
    pages = max(1, (total + per_page - 1) // per_page)
    # Clamp into range before deriving the offset: an out-of-range page returned an
    # empty list indistinguishable from "no history", and a very large one produced
    # an OFFSET too big for SQLite to bind (TW-AUD-023).
    page = min(max(1, page), pages)
    offset = (page - 1) * per_page

    checks = list_check_results(conn, topic_id, limit=per_page, offset=offset, cutoff_id=cutoff_id)

    return {
        "checks": checks,
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": pages,
        "cutoff_id": cutoff_id,
    }


@router.get("/topics/{topic_id}/knowledge")
async def api_get_knowledge(
    topic_id: int,
    conn: sqlite3.Connection = Depends(get_db_conn, scope="function"),
) -> KnowledgeState:
    """Get the current knowledge state for a topic."""
    topic = get_topic(conn, topic_id)
    if topic is None:
        raise HTTPException(status_code=404, detail="Topic not found")
    knowledge = get_knowledge_state(conn, topic_id)
    if knowledge is None:
        raise HTTPException(status_code=404, detail="No knowledge state for this topic")
    return knowledge


@router.post("/topics/{topic_id}/check", dependencies=[Depends(verify_csrf)])
async def api_trigger_check(
    request: Request,
    topic_id: int,
    settings: Settings = Depends(get_settings),
) -> CheckTriggerResponse:
    """Trigger a check for a specific topic.

    Runs synchronously and may take several seconds; returns the check result.
    Honors the same per-topic in-flight guard the web/UI path uses: a second
    concurrent check of the same topic (two API POSTs, or an API call racing a
    UI 'Check now') returns 409 instead of launching a duplicate pipeline that
    would double-spend the LLM and double-notify (OVH-019).

    AUG-202: this handler takes no request-scoped connection. Passing one into
    ``check_topic`` pinned it for the whole pipeline — feed, content, LLM and
    notification awaits — which is minutes of held request and SQLite resources
    on the one path the web remediation had left behind. The precondition reads
    below use a short connection and ``check_topic`` opens its own per phase.
    """
    db_path = getattr(request.app.state, "db_path", None)
    with get_db(db_path) as conn:
        topic = get_topic(conn, topic_id)
    if topic is None:
        raise HTTPException(status_code=404, detail="Topic not found")
    if topic.status != TopicStatus.READY:
        raise HTTPException(status_code=409, detail=f"Topic is not ready (status: {topic.status.value})")

    # Clear entries from a crashed prior run (mirrors the UI handler) before
    # claiming the guard, so a stale slot can never wedge the endpoint.
    await _checking_state.clear_stale(600)
    owner = await _checking_state.start_check(topic_id)
    if owner is None:
        raise HTTPException(status_code=409, detail="A check for this topic is already in progress")
    try:
        result = await check_topic(topic, settings, db_path=db_path, guard=False)
    finally:
        await _checking_state.finish_check(topic_id, owner)
    return CheckTriggerResponse(
        status="checked",
        check_result_id=result.id,
        has_new_info=result.has_new_info,
        articles_found=result.articles_found,
        articles_new=result.articles_new,
        stage_error=result.stage_error,
        notification_sent=result.notification_sent,
        notification_error=result.notification_error,
        notify_disposition=result.notify_disposition,
    )
