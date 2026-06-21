"""Data export routes: per-topic JSON/CSV and bulk topics JSON.

Colocates the download endpoints (OVH-155) so the filename slug + CSV
formula-injection guards live in one place, separate from topic CRUD/HTMX in
topics.py and from OPML import/export in opml.py.
"""

import csv
import io
import json
import re
import sqlite3
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.crud import (
    get_knowledge_state,
    get_topic,
    list_articles_for_topic,
    list_check_results,
    list_topics,
)
from app.web.dependencies import get_db_conn

router = APIRouter()

# Upper bound on rows pulled into memory for a single-topic export, so a large
# article/check history can't materialise an unbounded result set (OVH-051).
# Comfortably above any single-user volume; index-backed (m014) so the LIMIT is
# index-ordered, not a full sort.
_EXPORT_ROW_CAP = 10000


def _slug_for_filename(name: str) -> str:
    """ASCII-only slug for a download filename, never empty/degenerate (OVH-167).

    Lowercases, maps spaces to underscores, drops anything outside ``[a-z0-9_-]``,
    then collapses/strips underscores. A fully non-ASCII name (e.g. CJK/Cyrillic)
    would otherwise slug to "" or a bare "_", yielding a degenerate filename like
    ``checks_5_.csv``; in that case fall back to ``"topic"`` so the Content-
    Disposition hint stays sensible. This affects only the suggested download
    filename, not any filesystem path.
    """
    slug = re.sub(r"[^a-z0-9_-]", "", name.replace(" ", "_").lower())
    slug = re.sub(r"_+", "_", slug).strip("_-")
    return slug or "topic"


# Leading characters a spreadsheet (Excel/Sheets/LibreOffice) may interpret as
# the start of a formula when a CSV cell is opened, enabling CSV/formula
# injection (CWE-1236, OVH-168).
_CSV_FORMULA_TRIGGERS = ("=", "+", "-", "@", "\t", "\r")


def _csv_safe(value: object) -> object:
    """Neutralize a CSV cell against spreadsheet formula injection (OVH-168).

    If the stringified cell begins with a formula-trigger character, prefix it
    with a single quote so a spreadsheet treats it as literal text instead of a
    formula. Non-string values (ints/bools/None) are returned unchanged — only a
    string that actually starts with a trigger is rewritten, so well-formed data
    is untouched. Under the single-user threat model the realistic vector is a
    malicious upstream provider error string echoed into ``notification_error``.
    """
    if not isinstance(value, str):
        return value
    if value and value[0] in _CSV_FORMULA_TRIGGERS:
        return "'" + value
    return value


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
    safe_name = _slug_for_filename(topic.name)
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
        # Neutralize every cell against spreadsheet formula injection (OVH-168);
        # only the free-text fields can carry attacker-influenced content, but
        # the guard is a cheap no-op on the numeric/boolean cells.
        writer.writerow(
            [
                _csv_safe(check.id),
                _csv_safe(check.topic_id),
                _csv_safe(check.checked_at.isoformat()),
                _csv_safe(check.articles_found),
                _csv_safe(check.articles_new),
                # OVH-111: emit 0/1, not Python "True"/"False" — consistent with
                # the on-disk INTEGER and the JSON export. int() flows through
                # _csv_safe unchanged (non-strings are returned as-is).
                _csv_safe(int(check.has_new_info)),
                _csv_safe(int(check.notification_sent)),
                _csv_safe(check.notification_error or ""),
                _csv_safe(check.stage_error or ""),
            ]
        )

    content = output.getvalue()
    safe_name = _slug_for_filename(topic.name)
    filename = f"checks_{topic_id}_{safe_name}.csv"

    return StreamingResponse(
        iter([content]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
