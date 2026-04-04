"""OPML import/export for Topic Watch.

Handles parsing OPML files from RSS readers (FreshRSS, Miniflux, Tiny Tiny RSS)
and exporting topics as OPML for backup/migration.
"""

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import UTC, datetime
from urllib.parse import urlparse

from app.url_validation import validate_feed_url

logger = logging.getLogger(__name__)

MAX_IMPORT_TOPICS = 500
MAX_OUTLINE_DEPTH = 10


@dataclass
class OPMLResult:
    """Result of parsing an OPML file."""

    topics: list[dict] = field(default_factory=list)
    skipped_dupes: int = 0
    skipped_invalid: int = 0
    warnings: list[str] = field(default_factory=list)


def _derive_name_from_url(url: str) -> str:
    """Derive a topic name from a feed URL's domain."""
    try:
        parsed = urlparse(url)
        return parsed.hostname or url
    except Exception:
        return url


def _walk_outlines(
    element: ET.Element,
    existing_feed_urls: set[str],
    result: OPMLResult,
    parent_tags: list[str],
    depth: int = 0,
) -> None:
    """Recursively walk OPML outline elements, extracting feed entries."""
    if depth > MAX_OUTLINE_DEPTH:
        return

    for outline in element.findall("outline"):
        xml_url = outline.get("xmlUrl")
        text = outline.get("text") or outline.get("title") or ""

        if xml_url:
            # This is a feed entry
            xml_url = xml_url.strip()
            if not xml_url:
                continue

            # Dedup: skip if URL already exists in any topic
            if xml_url in existing_feed_urls:
                result.skipped_dupes += 1
                continue

            # Also dedup within the same import
            if any(xml_url in t.get("feed_urls", []) for t in result.topics):
                result.skipped_dupes += 1
                continue

            # SSRF validation
            error = validate_feed_url(xml_url)
            if error:
                result.skipped_invalid += 1
                result.warnings.append(error)
                continue

            name = text.strip() if text.strip() else _derive_name_from_url(xml_url)

            result.topics.append(
                {
                    "name": name,
                    "feed_urls": [xml_url],
                    "tags": list(parent_tags),
                }
            )
            # Track this URL as existing for in-import dedup
            existing_feed_urls.add(xml_url)
        else:
            # This is a folder — use its text as a tag for children
            folder_name = text.strip()
            child_tags = parent_tags + [folder_name] if folder_name else parent_tags
            _walk_outlines(outline, existing_feed_urls, result, child_tags, depth + 1)


def parse_opml(content: str, existing_feed_urls: set[str]) -> OPMLResult:
    """Parse OPML content and return extracted feed entries.

    Args:
        content: Raw XML string of the OPML file.
        existing_feed_urls: Set of feed URLs already in the database (for dedup).

    Returns:
        OPMLResult with parsed topics, skip counts, and warnings.
    """
    result = OPMLResult()

    try:
        root = ET.fromstring(content)  # noqa: S314 — entity expansion disabled by default in Python 3.11+ expat; 1MB size cap adds defense-in-depth
    except ET.ParseError as exc:
        result.warnings.append(f"Invalid XML: {exc}")
        return result

    body = root.find("body")
    if body is None:
        result.warnings.append("No <body> element found in OPML file.")
        return result

    # Make a mutable copy so in-import dedup can track new URLs
    working_urls = set(existing_feed_urls)
    _walk_outlines(body, working_urls, result, parent_tags=[], depth=0)

    if not result.topics:
        if result.skipped_dupes == 0 and result.skipped_invalid == 0:
            result.warnings.append("No feeds found in OPML file.")
        return result

    # Cap at MAX_IMPORT_TOPICS, alphabetical sort
    if len(result.topics) > MAX_IMPORT_TOPICS:
        result.topics.sort(key=lambda t: t["name"].lower())
        result.warnings.append(
            f"OPML contains {len(result.topics)} feeds. "
            f"Imported first {MAX_IMPORT_TOPICS} alphabetically. "
            f"Import again to add more (duplicates will be skipped)."
        )
        result.topics = result.topics[:MAX_IMPORT_TOPICS]

    return result


def export_opml(topics: list[dict]) -> str:
    """Export topics as OPML XML string.

    Args:
        topics: List of dicts with 'name', 'feed_urls', and 'tags' keys.
               Typically from [t.model_dump() for t in topic_list].

    Returns:
        Valid OPML 2.0 XML string.
    """
    opml = ET.Element("opml", version="2.0")
    head = ET.SubElement(opml, "head")
    title = ET.SubElement(head, "title")
    title.text = "Topic Watch Export"
    date_created = ET.SubElement(head, "dateCreated")
    date_created.text = datetime.now(UTC).strftime("%a, %d %b %Y %H:%M:%S %z")

    body = ET.SubElement(opml, "body")

    # Group topics by first tag for folder structure
    folders: dict[str, list[dict]] = {}
    no_tag: list[dict] = []

    for topic in topics:
        tags = topic.get("tags", [])
        if tags:
            folder_name = tags[0]
            folders.setdefault(folder_name, []).append(topic)
        else:
            no_tag.append(topic)

    # Add ungrouped topics at root level
    for topic in no_tag:
        for url in topic.get("feed_urls", []):
            ET.SubElement(body, "outline", text=topic["name"], xmlUrl=url, type="rss")

    # Add grouped topics in folders
    for folder_name, folder_topics in sorted(folders.items()):
        folder_el = ET.SubElement(body, "outline", text=folder_name)
        for topic in folder_topics:
            for url in topic.get("feed_urls", []):
                ET.SubElement(folder_el, "outline", text=topic["name"], xmlUrl=url, type="rss")

    return ET.tostring(opml, encoding="unicode", xml_declaration=True)
