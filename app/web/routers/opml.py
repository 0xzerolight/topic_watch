"""OPML import/export routes.

Per-topic JSON/CSV and the bulk topics-JSON export live in ``exports.py``
(OVH-155); this module keeps the OPML-specific import/export.
"""

import sqlite3
from typing import TYPE_CHECKING
from urllib.parse import quote

from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse, StreamingResponse

from app.config import Settings
from app.crud import list_topics
from app.models import FeedMode, Topic, TopicStatus
from app.web.csrf import verify_csrf
from app.web.dependencies import get_db_conn, get_settings

if TYPE_CHECKING:
    from app.opml import OPMLResult

router = APIRouter()


def _import_failure_message(result: "OPMLResult") -> str:
    """Describe a failed import from its counts alone.

    Validation warnings quote the rejected feed URL verbatim, and that URL can
    carry userinfo or a signed query token. Copying one into the Location header
    put it in browser history, in synced history and in any access log along the
    way (AUG-206), so the message is rebuilt from the structured counts instead.
    """
    reasons = []
    if result.skipped_invalid:
        reasons.append(f"{result.skipped_invalid} feed URL(s) rejected")
    if result.skipped_dupes:
        reasons.append(f"{result.skipped_dupes} duplicate feed(s)")
    if result.skipped_name_dupes:
        reasons.append(f"{result.skipped_name_dupes} name collision(s)")
    if not reasons:
        return "No topics imported: the file contained no usable feeds."
    return "No topics imported: " + ", ".join(reasons) + "."


@router.get("/export/opml")
async def export_opml_handler(
    conn: sqlite3.Connection = Depends(get_db_conn, scope="function"),
):
    """Export all topics as OPML XML.

    OPML can only carry a stored feed URL, so AUTO and Exa topics have nothing to
    write. They used to vanish from the file with no trace; the count now travels
    with the export so a download is never mistaken for a full backup
    (TW-AUD-026).
    """
    from app.opml import export_opml

    topics = list_topics(conn)
    exportable = [t for t in topics if t.feed_urls]
    topic_dicts = [{"name": t.name, "feed_urls": t.feed_urls, "tags": t.tags} for t in exportable]
    xml_content = export_opml(topic_dicts, omitted_count=len(topics) - len(exportable))

    return StreamingResponse(
        iter([xml_content]),
        media_type="application/xml",
        headers={"Content-Disposition": 'attachment; filename="topic_watch_export.opml"'},
    )


@router.post("/import/opml", dependencies=[Depends(verify_csrf)])
async def import_opml_handler(
    request: Request,
    conn: sqlite3.Connection = Depends(get_db_conn, scope="function"),
    settings: Settings = Depends(get_settings),
):
    """Import topics from an OPML file."""
    import asyncio

    # Starlette's class, not fastapi.UploadFile: request.form() always yields the
    # former, and fastapi.UploadFile is a subclass — so an isinstance check against
    # the FastAPI one rejects every real upload as "no file selected".
    from starlette.datastructures import UploadFile

    from app.crud import create_topic, get_all_feed_urls, get_all_topic_names
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
    existing_names = get_all_topic_names(conn)

    # Run OPML parsing (includes SSRF validation with DNS lookups) in a thread.
    # All dedup (URL + name collision) is resolved inside parse_opml.
    result = await asyncio.to_thread(parse_opml, content, existing_urls, existing_names)

    if result.warnings and not result.topics:
        return RedirectResponse(url=f"/?error={quote(_import_failure_message(result))}", status_code=303)

    # Create topics with NEW status (collisions already filtered by parse_opml).
    created = 0
    for topic_data in result.topics:
        topic = Topic(
            name=topic_data["name"],
            description=f"News monitoring for {topic_data['name']}",
            feed_urls=topic_data["feed_urls"],
            feed_mode=FeedMode.MANUAL,
            status=TopicStatus.NEW,
            # NULL, not the current global value. OPML carries no interval, and
            # writing today's global number froze every imported topic on it: a
            # later change to the global cadence left them behind, looking as
            # though the user had chosen a custom one (TW-AUD-025).
            check_interval_minutes=None,
            tags=topic_data.get("tags", []),
        )
        create_topic(conn, topic)
        created += 1

    conn.commit()

    # Build summary message
    parts = [f"Imported {created} topic(s)"]
    total_skipped = result.skipped_dupes + result.skipped_name_dupes
    if total_skipped:
        parts.append(f"skipped {total_skipped} duplicate(s)")
    if result.skipped_invalid:
        parts.append(f"skipped {result.skipped_invalid} invalid URL(s)")
    if created > 0:
        parts.append("topics will initialize gradually (~1/min)")
    msg = ", ".join(parts) + "."

    return RedirectResponse(url=f"/?msg={msg}", status_code=303)
