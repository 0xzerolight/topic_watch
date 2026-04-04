"""JSON API endpoints for Topic Watch (v1).

Read-only API for scripting and monitoring, plus one mutation endpoint
to trigger topic checks. Reuses existing CRUD functions and Pydantic models.
"""

import sqlite3

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request

from app.checker import check_topic
from app.config import Settings
from app.crud import (
    count_check_results,
    get_knowledge_state,
    get_topic,
    list_check_results,
    list_topics,
)
from app.models import KnowledgeState, Topic, TopicStatus
from app.web.csrf import verify_csrf
from app.web.dependencies import get_db_conn, get_settings

router = APIRouter(prefix="/api/v1", tags=["api"])


@router.get("/topics")
async def api_list_topics(
    active: bool | None = None,
    tag: str | None = None,
    conn: sqlite3.Connection = Depends(get_db_conn),
) -> list[Topic]:
    """List all topics with optional filters."""
    return list_topics(conn, active_only=active is True if active is not None else False, tag=tag)


@router.get("/topics/{topic_id}")
async def api_get_topic(
    topic_id: int,
    conn: sqlite3.Connection = Depends(get_db_conn),
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
    conn: sqlite3.Connection = Depends(get_db_conn),
) -> dict:
    """Get check history for a topic with pagination."""
    topic = get_topic(conn, topic_id)
    if topic is None:
        raise HTTPException(status_code=404, detail="Topic not found")

    per_page = max(1, min(per_page, 100))
    page = max(1, page)
    offset = (page - 1) * per_page

    checks = list_check_results(conn, topic_id, limit=per_page, offset=offset)
    total = count_check_results(conn, topic_id)

    return {
        "checks": checks,
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": (total + per_page - 1) // per_page if per_page else 0,
    }


@router.get("/topics/{topic_id}/knowledge")
async def api_get_knowledge(
    topic_id: int,
    conn: sqlite3.Connection = Depends(get_db_conn),
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
    topic_id: int,
    request: Request,
    background_tasks: BackgroundTasks,
    conn: sqlite3.Connection = Depends(get_db_conn),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Trigger a check for a specific topic."""
    topic = get_topic(conn, topic_id)
    if topic is None:
        raise HTTPException(status_code=404, detail="Topic not found")
    if topic.status != TopicStatus.READY:
        raise HTTPException(status_code=409, detail=f"Topic is not ready (status: {topic.status.value})")

    result = await check_topic(topic, conn, settings)
    return {"status": "checked", "has_new_info": result.has_new_info, "check_result_id": result.id}
