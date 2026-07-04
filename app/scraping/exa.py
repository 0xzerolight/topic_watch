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
from app.scraping.rss import FeedEntry, FeedResponse
from app.url_validation import is_private_url

if TYPE_CHECKING:
    from app.config import ExaSettings
    from app.models import Topic

logger = logging.getLogger(__name__)

_DEFAULT_EXA_BASE_URL = "https://api.exa.ai"
_EXA_TEXT_MAX_CHARS = 5000


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

    text = raw.get("text") or ""
    # summary stays "" (that field means "RSS summary"); Exa's full text rides on
    # ``content`` as the single prefetched channel that short-circuits extraction.
    return FeedEntry(
        title=title,
        url=url,
        published=published,
        summary="",
        source_feed="exa",
        content=(text or None),
    )


async def fetch_exa_entries(
    topic: Topic,
    exa_settings: ExaSettings,
    *,
    max_results: int,
    timeout: float,
    client: httpx.AsyncClient | None = None,
) -> FeedResponse:
    """Query the Exa ``/search`` API for ``topic`` and return a ``FeedResponse``.

    Never raises: any failure logs a warning and returns a ``FeedResponse`` whose
    counters reflect the outcome, so the check pipeline degrades gracefully.
    """
    if not exa_settings.enabled or not exa_settings.api_key:
        # Nothing attempted (no HTTP). feeds_total=0 keeps _log_feed_coverage from
        # reporting an "all sources failed" line for a self-inflicted disabled state.
        logger.warning("Exa source requested for topic '%s' but Exa is disabled or has no API key", topic.name)
        return FeedResponse(provider_name="exa", feeds_total=0, feeds_failed=0)

    endpoint = f"{(exa_settings.base_url or _DEFAULT_EXA_BASE_URL).rstrip('/')}/search"

    # base_url is user-configurable, so validate the effective endpoint (SSRF).
    try:
        if urlparse(endpoint).scheme not in ("http", "https"):
            logger.warning("Blocked Exa request to non-http(s) endpoint: %s", redact_url(endpoint))
            return FeedResponse(provider_name="exa", feeds_total=1, feeds_failed=1)
        if await asyncio.to_thread(is_private_url, endpoint):
            logger.warning("Blocked Exa request to private/reserved endpoint: %s", redact_url(endpoint))
            return FeedResponse(provider_name="exa", feeds_total=1, feeds_failed=1)
    except Exception:
        logger.warning("Blocked Exa request to malformed endpoint: %s", redact_url(endpoint), exc_info=True)
        return FeedResponse(provider_name="exa", feeds_total=1, feeds_failed=1)

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
        response = await client.post(endpoint, json=body, headers=headers)
        response.raise_for_status()
        data = response.json()
    except httpx.TimeoutException:
        logger.warning("Exa request timed out for topic '%s'", topic.name)
        return FeedResponse(provider_name="exa", feeds_total=1, feeds_failed=1)
    except httpx.HTTPStatusError as exc:
        logger.warning("Exa returned HTTP %d for topic '%s'", exc.response.status_code, topic.name)
        return FeedResponse(provider_name="exa", feeds_total=1, feeds_failed=1)
    except Exception:
        # NetworkError, JSON-decode (ValueError), and any other failure: never raise.
        logger.warning("Exa request failed for topic '%s'", topic.name, exc_info=True)
        return FeedResponse(provider_name="exa", feeds_total=1, feeds_failed=1)
    finally:
        if owns_client:
            await client.aclose()

    entries: list[FeedEntry] = []
    results = data.get("results", []) if isinstance(data, dict) else []
    for raw in results:
        # Per-result isolation (mirrors RSS OVH-024): one bad result never zeroes the batch.
        try:
            entry = _map_exa_result(raw)
        except Exception:
            logger.warning("Skipping malformed Exa result for topic '%s'", topic.name, exc_info=True)
            continue
        if entry is not None:
            entries.append(entry)

    return FeedResponse(
        entries=entries,
        provider_name="exa",
        needs_url_resolution=False,
        feeds_total=1,
        feeds_failed=0,
    )
