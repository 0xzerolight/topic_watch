"""Article content extraction.

Fetches HTML pages via httpx and extracts main article text
using trafilatura. Falls back to RSS summary when extraction fails.
"""

import asyncio
import logging
from typing import cast

import httpx
import trafilatura

from app.log_redaction import redact_url
from app.url_validation import is_private_url, safe_get

logger = logging.getLogger(__name__)

_ARTICLE_FETCH_TIMEOUT = 20.0
_DEFAULT_MAX_CONTENT_LENGTH = 5000


def _truncate(text: str, max_length: int) -> str:
    """Truncate text at a word boundary, appending '...' if needed."""
    if len(text) <= max_length:
        return text
    truncated = text[:max_length]
    # Find last space to break at word boundary
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]
    return truncated + "..."


async def _fetch_html(
    url: str,
    client: httpx.AsyncClient | None = None,
    timeout: float = _ARTICLE_FETCH_TIMEOUT,
) -> str | None:
    """Fetch a URL and return HTML text, or None on error."""
    if await asyncio.to_thread(is_private_url, url):
        logger.warning("Blocked article fetch to private/reserved URL: %s", redact_url(url))
        return None
    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(timeout=timeout, follow_redirects=False)
    assert client is not None
    max_attempts = 2
    try:
        for attempt in range(max_attempts):
            try:
                response = await safe_get(client, url)
                response.raise_for_status()
                return response.text
            except (httpx.TimeoutException, httpx.HTTPStatusError) as exc:
                is_server_error = isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code >= 500
                is_timeout = isinstance(exc, httpx.TimeoutException)
                if (is_timeout or is_server_error) and attempt < max_attempts - 1:
                    logger.debug("Article fetch attempt %d failed for %s, retrying", attempt + 1, url)
                    await asyncio.sleep(2)
                    continue
                logger.warning("Failed to fetch article: %s", url, exc_info=True)
                return None
            except Exception:
                logger.warning("Failed to fetch article: %s", url, exc_info=True)
                return None
        return None  # pragma: no cover
    finally:
        if owns_client:
            await client.aclose()


async def extract_article_content(
    url: str,
    fallback_summary: str = "",
    client: httpx.AsyncClient | None = None,
    max_content_length: int = _DEFAULT_MAX_CONTENT_LENGTH,
    timeout: float = _ARTICLE_FETCH_TIMEOUT,
    prefetched: str | None = None,
) -> str:
    """Extract article text from a URL, falling back to the RSS summary.

    ``prefetched`` short-circuits the network fetch when the source already provides
    full text (e.g. Exa search): a non-empty value is truncated and returned directly.
    An empty/None ``prefetched`` falls through to the normal fetch → summary path.
    """
    if prefetched:
        return _truncate(prefetched, max_content_length)

    html = await _fetch_html(url, client, timeout=timeout)
    if html:
        # OVH-115: parse the DOM once and reuse the tree across both extraction
        # passes instead of re-parsing the raw HTML each time (verified
        # output-identical at trafilatura 2.0.0). load_html returns None on
        # unparseable input, in which case both passes fall through to the
        # summary as before.
        tree = trafilatura.load_html(html)
        extracted = cast(str | None, trafilatura.extract(tree, favor_precision=True))
        if extracted:
            return _truncate(extracted, max_content_length)
        # Second attempt: favor recall over precision for JS-heavy or complex sites
        extracted = cast(str | None, trafilatura.extract(tree, favor_recall=True))
        if extracted:
            return _truncate(extracted, max_content_length)
        # OVH-045: HTML fetched but both passes extracted nothing (JS-heavy,
        # paywall, anti-bot). Surface the summary-degradation so an operator can
        # tell the LLM is reasoning over the short RSS summary, not full text.
        logger.info("trafilatura extracted nothing for %s, using RSS summary", url)
    else:
        logger.debug("No HTML fetched for %s, using RSS summary", url)

    return _truncate(fallback_summary, max_content_length)
