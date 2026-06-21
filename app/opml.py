"""OPML import/export for Topic Watch.

Handles parsing OPML files from RSS readers (FreshRSS, Miniflux, Tiny Tiny RSS)
and exporting topics as OPML for backup/migration.
"""

import logging
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from urllib.parse import urlparse

from app.url_validation import validate_feed_url

logger = logging.getLogger(__name__)

MAX_IMPORT_TOPICS = 500
MAX_OUTLINE_DEPTH = 10

# Bound on concurrent feed-URL validations (each does a blocking getaddrinfo).
# Caps both wall-clock time and resolver fan-out for a large import so a handful
# of slow/unresolvable hosts no longer serialize into a multi-minute import
# (OVH-053). ``parse_opml`` runs inside ``asyncio.to_thread`` (worker thread, no
# event loop), so a ThreadPoolExecutor — not asyncio — is the right primitive.
_VALIDATION_CONCURRENCY = 16


@dataclass
class OPMLResult:
    """Result of parsing an OPML file."""

    topics: list[dict] = field(default_factory=list)
    skipped_dupes: int = 0
    skipped_invalid: int = 0
    skipped_name_dupes: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class _Candidate:
    """A raw feed entry from the pure structural walk (pre-validation/dedup)."""

    name: str
    url: str
    tags: list[str]


def _derive_name_from_url(url: str) -> str:
    """Derive a topic name from a feed URL's domain."""
    try:
        parsed = urlparse(url)
        return parsed.hostname or url
    except Exception:
        return url


def _walk_outlines(
    element: ET.Element,
    candidates: list[_Candidate],
    parent_tags: list[str],
    depth: int = 0,
) -> None:
    """Recursively walk OPML outline elements, collecting raw feed candidates.

    Pure structural pass: no DNS / SSRF validation and no cross-import dedup, so
    parse correctness is unit-testable without sockets. It only extracts
    ``(name, url, tags)``. Validation, dedup, and capping happen in ``parse_opml``.
    """
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
            name = text.strip() if text.strip() else _derive_name_from_url(xml_url)
            candidates.append(_Candidate(name=name, url=xml_url, tags=list(parent_tags)))
        else:
            # This is a folder — use its text as a tag for children
            folder_name = text.strip()
            child_tags = parent_tags + [folder_name] if folder_name else parent_tags
            _walk_outlines(outline, candidates, child_tags, depth + 1)


def _validate_urls_concurrently(urls: list[str]) -> dict[str, str | None]:
    """Validate a deduped URL list concurrently, returning ``{url: error|None}``.

    Each URL is validated with :func:`validate_feed_url` (DNS resolution +
    private-address check) in a bounded thread pool so a large import's blocking
    ``getaddrinfo`` calls run in parallel rather than back-to-back (OVH-053).
    The SSRF invariant is preserved: every URL still passes through
    ``validate_feed_url``. A tiny/empty list skips the pool entirely.
    """
    if not urls:
        return {}
    if len(urls) == 1:
        return {urls[0]: validate_feed_url(urls[0])}

    workers = min(_VALIDATION_CONCURRENCY, len(urls))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(validate_feed_url, urls))
    return dict(zip(urls, results, strict=True))


def parse_opml(
    content: str,
    existing_feed_urls: set[str],
    existing_topic_names: set[str] | None = None,
) -> OPMLResult:
    """Parse OPML content and return extracted feed entries.

    Orchestrates a pure structural walk followed by a separate validation/dedup
    pass: walk -> dedup (URL) -> validate (SSRF) -> dedup (name collision) -> cap.
    Network I/O is confined to ``validate_feed_url`` in the second pass.

    Args:
        content: Raw XML string of the OPML file.
        existing_feed_urls: Set of feed URLs already in the database (for dedup).
        existing_topic_names: Set of topic names already in the database. Feeds
            whose name collides with one are skipped and counted in
            ``skipped_name_dupes`` (replaces the router's raw SQL check).

    Returns:
        OPMLResult with parsed topics, skip counts, and warnings.
    """
    result = OPMLResult()
    existing_names = existing_topic_names or set()

    try:
        root = ET.fromstring(content)  # noqa: S314 — entity expansion disabled by default in Python 3.11+ expat; 1MB size cap adds defense-in-depth
    except ET.ParseError as exc:
        result.warnings.append(f"Invalid XML: {exc}")
        return result

    body = root.find("body")
    if body is None:
        result.warnings.append("No <body> element found in OPML file.")
        return result

    # 1. Pure structural walk (no network, no dedup).
    candidates: list[_Candidate] = []
    _walk_outlines(body, candidates, parent_tags=[], depth=0)

    # 2a. URL dedup pass (no network). Drop candidates whose URL already exists
    # in the DB or appeared earlier in this import, so validation never resolves
    # a URL twice. ``seen_urls`` is consumed in order to preserve intra-import
    # dedup semantics (first occurrence wins).
    seen_urls = set(existing_feed_urls)
    survivors: list[_Candidate] = []
    for candidate in candidates:
        if candidate.url in seen_urls:
            result.skipped_dupes += 1
            continue
        survivors.append(candidate)
        seen_urls.add(candidate.url)

    # 2b. Concurrent SSRF validation of the deduped URL set — the only network
    # step. Each URL still flows through ``validate_feed_url`` (DNS + private-IP
    # check), but bounded concurrency caps wall-clock + resolver fan-out so slow
    # hosts don't serialize (OVH-053). Resolve each unique URL exactly once.
    errors_by_url = _validate_urls_concurrently([c.url for c in survivors])

    # 2c. Apply pass (no network): consume validation results in document order,
    # preserving the original merge / name-collision accounting.
    name_dupes_seen: set[str] = set()
    for candidate in survivors:
        error = errors_by_url.get(candidate.url)
        if error:
            result.skipped_invalid += 1
            result.warnings.append(error)
            continue

        # Merge feeds that share a topic name so a multi-feed topic round-trips
        # intact (export writes one <outline> per feed_url, all sharing the name).
        existing_topic = next((t for t in result.topics if t["name"] == candidate.name), None)
        if existing_topic is not None:
            existing_topic["feed_urls"].append(candidate.url)
            continue

        # Name collision with an existing DB topic — skip (counted once per name).
        if candidate.name in existing_names:
            if candidate.name not in name_dupes_seen:
                name_dupes_seen.add(candidate.name)
                result.skipped_name_dupes += 1
            continue

        result.topics.append(
            {
                "name": candidate.name,
                "feed_urls": [candidate.url],
                "tags": list(candidate.tags),
            }
        )

    if not result.topics:
        if result.skipped_dupes == 0 and result.skipped_invalid == 0 and result.skipped_name_dupes == 0:
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
