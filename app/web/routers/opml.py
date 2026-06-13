"""OPML import/export and bulk topic export routes."""

import json
import sqlite3
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse, StreamingResponse

from app.config import Settings
from app.crud import list_topics
from app.models import FeedMode, Topic, TopicStatus
from app.web.csrf import verify_csrf
from app.web.dependencies import get_db_conn, get_settings

router = APIRouter()


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


@router.get("/export/opml")
async def export_opml_handler(
    conn: sqlite3.Connection = Depends(get_db_conn),
):
    """Export all topics as OPML XML."""
    from app.opml import export_opml

    topics = list_topics(conn)
    topic_dicts = [{"name": t.name, "feed_urls": t.feed_urls, "tags": t.tags} for t in topics]
    xml_content = export_opml(topic_dicts)

    return StreamingResponse(
        iter([xml_content]),
        media_type="application/xml",
        headers={"Content-Disposition": 'attachment; filename="topic_watch_export.opml"'},
    )


@router.post("/import/opml", dependencies=[Depends(verify_csrf)])
async def import_opml_handler(
    request: Request,
    conn: sqlite3.Connection = Depends(get_db_conn),
    settings: Settings = Depends(get_settings),
):
    """Import topics from an OPML file."""
    import asyncio

    from fastapi import UploadFile

    from app.crud import create_topic, get_all_feed_urls
    from app.opml import parse_opml

    form = await request.form()
    opml_file = form.get("opml_file")
    if not isinstance(opml_file, UploadFile) or opml_file.filename == "":
        return RedirectResponse(url="/?error=No+file+selected", status_code=303)

    # Read file with 1MB size cap
    content_bytes = await opml_file.read(1024 * 1024 + 1)
    if len(content_bytes) > 1024 * 1024:
        return RedirectResponse(url="/?error=File+too+large+(max+1MB)", status_code=303)

    try:
        content = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return RedirectResponse(url="/?error=Invalid+file+encoding+(must+be+UTF-8)", status_code=303)

    existing_urls = get_all_feed_urls(conn)

    # Run OPML parsing (includes SSRF validation with DNS lookups) in a thread
    result = await asyncio.to_thread(parse_opml, content, existing_urls)

    if result.warnings and not result.topics:
        warning_msg = result.warnings[0][:200]
        return RedirectResponse(url=f"/?error={warning_msg}", status_code=303)

    # Create topics with NEW status
    created = 0
    skipped_name_dupes = 0
    for topic_data in result.topics:
        # Check for name collision
        existing = conn.execute("SELECT 1 FROM topics WHERE name = ?", (topic_data["name"],)).fetchone()
        if existing:
            skipped_name_dupes += 1
            continue

        default_interval = settings.check_interval_minutes
        topic = Topic(
            name=topic_data["name"],
            description=f"News monitoring for {topic_data['name']}",
            feed_urls=topic_data["feed_urls"],
            feed_mode=FeedMode.MANUAL,
            status=TopicStatus.NEW,
            check_interval_minutes=default_interval,
            tags=topic_data.get("tags", []),
        )
        create_topic(conn, topic)
        created += 1

    conn.commit()

    # Build summary message
    parts = [f"Imported {created} topic(s)"]
    total_skipped = result.skipped_dupes + skipped_name_dupes
    if total_skipped:
        parts.append(f"skipped {total_skipped} duplicate(s)")
    if result.skipped_invalid:
        parts.append(f"skipped {result.skipped_invalid} invalid URL(s)")
    if created > 0:
        parts.append("topics will initialize gradually (~1/min)")
    msg = ", ".join(parts) + "."

    return RedirectResponse(url=f"/?msg={msg}", status_code=303)
