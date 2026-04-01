"""Article content extraction.

Fetches HTML pages via httpx and extracts main article text
using trafilatura. Falls back to RSS summary when extraction fails.
"""

import asyncio
import logging
from typing import cast

import httpx
import trafilatura

from app.url_validation import is_private_url

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
    if is_private_url(url):
        logger.warning("Blocked article fetch to private/reserved URL: %s", url)
        return None
    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(timeout=timeout, follow_redirects=True)
    assert client is not None
    max_attempts = 2
    try:
        for attempt in range(max_attempts):
            try:
                response = await client.get(url)
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
) -> str:
    """Extract article text from a URL, falling back to the RSS summary."""
    html = await _fetch_html(url, client, timeout=timeout)
    if html:
        extracted = cast(str | None, trafilatura.extract(html, favor_precision=True))
        if extracted:
            return _truncate(extracted, max_content_length)

    return _truncate(fallback_summary, max_content_length)
