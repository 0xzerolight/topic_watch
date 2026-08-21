"""Exa AI search source.

Queries the Exa ``/search`` API (https://exa.ai) and maps results directly to
``FeedEntry``, bypassing feedparser. Structurally modeled on
``webhooks.send_webhook``: scheme allowlist -> offloaded SSRF check ->
``follow_redirects=False`` client -> typed httpx handling -> ``redact_url``
logging -> never raises. Exa returns page text, carried through as prefetched
``FeedEntry.content`` so the pipeline skips a second content fetch.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import httpx

from app.log_redaction import redact_url
from app.models import FeedMode
from app.scraping.source import (
    Deadline,
    FeedEntry,
    FeedHealthCallback,
    FeedHealthOutcome,
    FeedResponse,
    FetchStatus,
    SourceDeadlineExceeded,
    SourceIdentity,
    SourceRequest,
    bounded,
    register_source,
)
from app.url_validation import is_private_url

if TYPE_CHECKING:
    from app.config import ExaSettings
    from app.models import Topic

logger = logging.getLogger(__name__)

_DEFAULT_EXA_BASE_URL = "https://api.exa.ai"
_EXA_TEXT_MAX_CHARS = 5000

EXA_SOURCE = SourceIdentity(name="exa")
"""Exa returns publisher URLs directly, so no async URL resolution is needed."""


def _report(
    health_callback: FeedHealthCallback | None,
    endpoint: str,
    status: FetchStatus,
    error_msg: str | None = None,
) -> None:
    """Record this attempt's verdict. Exa has no conditional-GET validators."""
    if health_callback:
        health_callback(FeedHealthOutcome(endpoint, status, error_msg))


def _failed(
    endpoint: str,
    health_callback: FeedHealthCallback | None,
    error_msg: str,
    status: FetchStatus = FetchStatus.FAILED,
) -> FeedResponse:
    """One failed Exa attempt, recorded and counted."""
    _report(health_callback, endpoint, status, error_msg)
    return FeedResponse.from_source(EXA_SOURCE, feeds_total=1, feeds_failed=1)


def _map_exa_result(raw: dict[str, Any]) -> FeedEntry | None:
    """Map one Exa result to a ``FeedEntry``, or ``None`` if unusable.

    Requires a non-empty http(s) ``url`` and ``title``. ``publishedDate`` is
    heterogeneous in real data (Z-suffixed aware, date-only naive, ``null``, or
    even non-string), so normalize to a tz-aware UTC datetime or ``None`` — a
    naive datetime mixed with an aware sibling would make ``_select_candidates``
    raise ``TypeError`` (mirrors ``rss._parse_feed_date``, always ``tz=UTC``).
    """
    url = (raw.get("url") or "").strip()
    title = (raw.get("title") or "").strip()
    if not url or not title:
        return None
    # Match the RSS scheme guard: never store a non-http(s) url (OVH-014).
    if urlparse(url).scheme not in ("http", "https"):
        return None

    published: datetime | None = None
    raw_date = raw.get("publishedDate")
    if isinstance(raw_date, str):
        try:
            published = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            published = None
    if published is not None and published.tzinfo is None:
        published = published.replace(tzinfo=UTC)

    # AUG-308: prefetched text counts only when it is a string with something in
    # it. Whitespace-only text is truthy and would short-circuit the publisher
    # fetch into storing a blank body, and a non-string value would fail
    # FeedEntry validation and discard an otherwise usable url/title row.
    raw_text = raw.get("text")
    text = raw_text.strip() if isinstance(raw_text, str) else ""
    # summary stays "" (that field means "RSS summary"); Exa's full text rides on
    # ``content`` as the single prefetched channel that short-circuits extraction.
    return FeedEntry(
        title=title,
        url=url,
        published=published,
        summary="",
        source_feed=EXA_SOURCE.name,
        content=(text or None),
    )


def _exa_results(data: object) -> list[Any]:
    """The result rows of an Exa envelope, or ``ValueError`` if it is not one.

    Exa's contract is a JSON object with a list-valued ``results``. Anything else
    — a top level that is not an object, ``results: null``, a string or a dict —
    is schema drift, and accepting it produced either a raised ``TypeError`` or a
    silent zero-entry "success" that cleared the source's failure state (AUG-174).
    """
    if not isinstance(data, dict):
        raise ValueError(f"Exa response was {type(data).__name__}, expected a JSON object")
    results = data.get("results", [])
    if not isinstance(results, list):
        raise ValueError(f"Exa 'results' was {type(results).__name__}, expected a list")
    return results


def _map_exa_results(results: list[Any], topic_name: str) -> list[FeedEntry]:
    """Map the usable rows of a result list, isolating each one.

    Per-result isolation mirrors RSS (OVH-024): one bad row never zeroes a batch
    that also carried good ones.
    """
    entries: list[FeedEntry] = []
    for raw in results:
        try:
            entry = _map_exa_result(raw)
        except Exception:
            logger.warning("Skipping malformed Exa result for topic '%s'", topic_name, exc_info=True)
            continue
        if entry is not None:
            entries.append(entry)
    return entries


async def fetch_exa_entries(
    topic: Topic,
    exa_settings: ExaSettings,
    *,
    max_results: int,
    timeout: float,
    client: httpx.AsyncClient | None = None,
    health_callback: FeedHealthCallback | None = None,
    deadline: Deadline | None = None,
) -> FeedResponse:
    """Query the Exa ``/search`` API for ``topic`` and return a ``FeedResponse``.

    Never raises: any failure logs a warning and returns a ``FeedResponse`` whose
    counters reflect the outcome, so the check pipeline degrades gracefully.

    ``health_callback`` records the outcome for the Feed Health dashboard keyed on
    the effective endpoint. It is invoked once per attempted fetch (success or
    failure) but NOT on the disabled/no-key early return, where nothing is
    attempted (mirrors ``all_sources_failed`` semantics). Exa has no
    conditional-GET validators, so etag/last_modified are always ``None``.

    ``deadline`` bounds the search request against the topic's whole source
    budget (TW-AUD-018); running out is recorded like any other failed fetch.
    """
    budget = deadline if deadline is not None else Deadline.after()
    if not exa_settings.enabled or not exa_settings.api_key:
        # Nothing attempted (no HTTP). feeds_total=0 keeps _log_feed_coverage from
        # reporting an "all sources failed" line for a self-inflicted disabled state.
        logger.warning("Exa source requested for topic '%s' but Exa is disabled or has no API key", topic.name)
        return FeedResponse.from_source(EXA_SOURCE, feeds_total=0, feeds_failed=0)

    endpoint = f"{(exa_settings.base_url or _DEFAULT_EXA_BASE_URL).rstrip('/')}/search"

    # base_url is user-configurable, so validate the effective endpoint (SSRF).
    try:
        if urlparse(endpoint).scheme not in ("http", "https"):
            logger.warning("Blocked Exa request to non-http(s) endpoint: %s", redact_url(endpoint))
            return _failed(endpoint, health_callback, "Non-http(s) endpoint")
        if await asyncio.to_thread(is_private_url, endpoint):
            logger.warning("Blocked Exa request to private/reserved endpoint: %s", redact_url(endpoint))
            return _failed(endpoint, health_callback, "Private/reserved endpoint")
    except Exception:
        logger.warning("Blocked Exa request to malformed endpoint: %s", redact_url(endpoint), exc_info=True)
        return _failed(endpoint, health_callback, "Malformed endpoint")

    query = f"{topic.name} {topic.description}".strip()
    body: dict[str, Any] = {
        "query": query,
        "numResults": max_results,
        "type": "auto",
        "category": "news",
        "contents": {"text": {"maxCharacters": _EXA_TEXT_MAX_CHARS}},
    }
    headers = {"x-api-key": exa_settings.api_key}

    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(timeout=timeout, follow_redirects=False)
    assert client is not None
    try:
        response = await bounded(budget, "the Exa search", client.post(endpoint, json=body, headers=headers))
        response.raise_for_status()
        data = response.json()
        # Envelope handling belongs INSIDE this boundary. Reading `results` after it
        # meant a 200 whose `results` was null raised straight out of a function
        # whose whole contract is that it never raises, taking the check down
        # instead of recording one failed source (AUG-174).
        results = _exa_results(data)
        entries = _map_exa_results(results, topic.name)
    except SourceDeadlineExceeded as exc:
        logger.warning("Exa request out of budget for topic '%s': %s", topic.name, exc)
        return _failed(endpoint, health_callback, str(exc), FetchStatus.ABORTED)
    except httpx.TimeoutException:
        logger.warning("Exa request timed out for topic '%s'", topic.name)
        return _failed(endpoint, health_callback, "Request timed out")
    except httpx.HTTPStatusError as exc:
        logger.warning("Exa returned HTTP %d for topic '%s'", exc.response.status_code, topic.name)
        return _failed(endpoint, health_callback, f"HTTP {exc.response.status_code}")
    except Exception as exc:
        # NetworkError, JSON-decode (ValueError), malformed envelope, and any other
        # failure: never raise.
        logger.warning("Exa request failed for topic '%s'", topic.name, exc_info=True)
        return _failed(endpoint, health_callback, f"{type(exc).__name__}: {exc}")
    finally:
        if owns_client:
            await client.aclose()

    if results and not entries:
        # Exa answered with rows and not one of them was usable. That is a protocol
        # failure wearing the shape of quiet news: recorded as success it reset feed
        # health and fed the silence heartbeat a healthy-empty check while
        # monitoring received nothing (AUG-307). A genuinely empty list stays healthy.
        logger.warning("Exa returned %d results for topic '%s', none usable", len(results), topic.name)
        return _failed(endpoint, health_callback, f"Exa returned {len(results)} results, none usable")

    _report(health_callback, endpoint, FetchStatus.OK if entries else FetchStatus.EMPTY)
    return FeedResponse.from_source(EXA_SOURCE, entries=entries, feeds_total=1, feeds_failed=0)


async def fetch_exa_source(topic: Topic, request: SourceRequest) -> FeedResponse:
    """Registry adapter: EXA-mode topics fetch through the Exa search API.

    Exa is the only source with a hard configuration prerequisite, so the
    missing-settings case is answered here rather than by a branch in the
    dispatcher. Nothing is attempted, hence ``feeds_total=0``.
    """
    if request.exa_settings is None:
        logger.warning("Topic '%s' uses Exa mode but no Exa settings were supplied", topic.name)
        return FeedResponse.from_source(EXA_SOURCE, feeds_total=0, feeds_failed=0)
    return await fetch_exa_entries(
        topic,
        request.exa_settings,
        max_results=request.max_results,
        timeout=request.timeout,
        health_callback=request.health_callback,
        deadline=request.deadline,
    )


register_source(FeedMode.EXA, fetch_exa_source)
