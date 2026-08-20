"""OPML import/export for Topic Watch.

Handles parsing OPML files from RSS readers (FreshRSS, Miniflux, Tiny Tiny RSS)
and exporting topics as OPML for backup/migration.
"""

import json
import logging
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from urllib.parse import urlparse

from app.models import normalize_tags
from app.url_validation import validate_feed_url

logger = logging.getLogger(__name__)

MAX_IMPORT_TOPICS = 500
MAX_OUTLINE_DEPTH = 10

# Feed URLs one imported topic may accumulate. Merging used to be keyed on
# display text alone, so an arbitrary number of same-named third-party feeds
# could land in one topic and fan out to that many concurrent fetches per check
# (AUG-204).
MAX_FEEDS_PER_TOPIC = 20

# Longest imported topic name kept. The name comes from an attribute a third
# party wrote and is persisted, rendered in every list, and used as a search
# query; an unbounded one wrecks the table it lands in (AUG-330).
MAX_TOPIC_NAME_CHARS = 120

# Attributes Topic Watch writes on its own export. Their presence — and only
# their presence — makes two outlines part of the same topic, so third-party
# display text is never treated as stable identity (AUG-204). ``TOPIC_ATTR``
# names the owning topic; ``TAGS_ATTR`` carries the full tag list as JSON, which
# the folder structure alone cannot express (TW-AUD-026).
TOPIC_ATTR = "topicWatchTopic"
TAGS_ATTR = "topicWatchTags"

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
    group: str | None = None
    """Owning Topic Watch topic, from ``TOPIC_ATTR``. ``None`` for third-party
    OPML, whose outlines are never merged with each other."""


def _derive_name_from_url(url: str) -> str:
    """Derive a topic name from a feed URL's domain."""
    try:
        parsed = urlparse(url)
        return parsed.hostname or url
    except Exception:
        return url


def _parse_tags_attr(raw: str | None) -> list[str] | None:
    """Read the JSON tag list Topic Watch's own export writes, or ``None``."""
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(value, list):
        return None
    return normalize_tags(str(tag) for tag in value)


def _disambiguate(name: str, url: str, taken: set[str]) -> str:
    """Make ``name`` unique within this import by appending the feed's host."""
    host = _derive_name_from_url(url)
    candidate = f"{name} ({host})"[:MAX_TOPIC_NAME_CHARS]
    suffix = 2
    while candidate in taken:
        candidate = f"{name} ({host} {suffix})"[:MAX_TOPIC_NAME_CHARS]
        suffix += 1
    return candidate


def _walk_outlines(
    element: ET.Element,
    candidates: list[_Candidate],
    parent_tags: list[str],
    depth: int = 0,
) -> None:
    """Recursively walk OPML outline elements, collecting raw feed candidates.

    Pure structural pass: no DNS / SSRF validation and no cross-import dedup, so
    parse correctness is unit-testable without sockets. It only extracts
    ``(name, url, tags, group)``. Validation, dedup, and capping happen in
    ``parse_opml``.
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
            own_tags = _parse_tags_attr(outline.get(TAGS_ATTR))
            candidates.append(
                _Candidate(
                    name=name[:MAX_TOPIC_NAME_CHARS],
                    url=xml_url,
                    tags=own_tags if own_tags is not None else normalize_tags(parent_tags),
                    group=outline.get(TOPIC_ATTR),
                )
            )
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
    by_group: dict[str, dict] = {}
    taken_names: set[str] = set()
    disambiguated = 0
    for candidate in survivors:
        error = errors_by_url.get(candidate.url)
        if error:
            result.skipped_invalid += 1
            result.warnings.append(error)
            continue

        # Merge only within one Topic Watch group, so a multi-feed topic still
        # round-trips through our own export while two unrelated third-party feeds
        # that happen to share display text stay two topics (AUG-204).
        group_topic = by_group.get(candidate.group) if candidate.group else None
        if group_topic is not None:
            if len(group_topic["feed_urls"]) < MAX_FEEDS_PER_TOPIC:
                group_topic["feed_urls"].append(candidate.url)
            else:
                result.skipped_dupes += 1
            continue

        # Name collision with an existing DB topic — skip (counted once per name).
        if candidate.name in existing_names:
            if candidate.name not in name_dupes_seen:
                name_dupes_seen.add(candidate.name)
                result.skipped_name_dupes += 1
            continue

        # Two unmerged feeds sharing a name would collide on the UNIQUE topic
        # name, so make the second one distinguishable rather than dropping it.
        name = candidate.name
        if name in taken_names:
            name = _disambiguate(name, candidate.url, taken_names)
            disambiguated += 1
        taken_names.add(name)

        topic = {"name": name, "feed_urls": [candidate.url], "tags": list(candidate.tags)}
        result.topics.append(topic)
        if candidate.group:
            by_group[candidate.group] = topic

    if disambiguated:
        result.warnings.append(
            f"{disambiguated} feed(s) shared a name with another feed in this file "
            f"and were imported as separate topics with the source host appended."
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


def _feed_outline(parent: ET.Element, topic: dict, url: str) -> None:
    """Write one feed outline, carrying its topic identity and full tag list.

    ``text``/``xmlUrl``/``type`` are what any RSS reader consumes. The two extra
    attributes are Topic Watch's own round-trip extension: they say which topic
    the outline belongs to (so re-import merges exactly the feeds that were one
    topic, and nothing else — AUG-204) and carry every tag, which the one-folder
    -per-topic structure cannot express (TW-AUD-026).
    """
    attrs = {
        "text": topic["name"],
        "xmlUrl": url,
        "type": "rss",
        TOPIC_ATTR: topic["name"],
    }
    tags = topic.get("tags", [])
    if tags:
        attrs[TAGS_ATTR] = json.dumps(tags, ensure_ascii=False)
    ET.SubElement(parent, "outline", attrs)


def export_opml(topics: list[dict], omitted_count: int = 0) -> str:
    """Export topics as OPML XML string.

    Args:
        topics: List of dicts with 'name', 'feed_urls', and 'tags' keys.
               Typically from [t.model_dump() for t in topic_list].
        omitted_count: Topics the caller could not represent because they have
               no stored feed URLs (AUTO and Exa sources). Recorded as a comment
               so the file never passes for a complete backup (TW-AUD-026).

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
    if omitted_count:
        body.append(
            ET.Comment(
                f" {omitted_count} topic(s) omitted: no stored feed URLs. Automatic and "
                f"Exa search sources have no feed to write here — use the JSON export "
                f"for a complete backup. "
            )
        )

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
            _feed_outline(body, topic, url)

    # Add grouped topics in folders
    for folder_name, folder_topics in sorted(folders.items()):
        folder_el = ET.SubElement(body, "outline", text=folder_name)
        for topic in folder_topics:
            for url in topic.get("feed_urls", []):
                _feed_outline(folder_el, topic, url)

    return ET.tostring(opml, encoding="unicode", xml_declaration=True)
