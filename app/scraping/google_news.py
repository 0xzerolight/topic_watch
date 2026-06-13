"""Google News redirect URL resolution.

Google News RSS feeds use opaque redirect URLs (news.google.com/rss/articles/...)
that don't HTTP-redirect to the actual article. This module resolves them to real
article URLs using Google's internal batchexecute endpoint.

The technique: fetch the article page to extract a signature and timestamp,
then POST to batchexecute with those params to get the decoded URL.
"""

import asyncio
import json
import logging
import re
from urllib.parse import quote, urlparse

import httpx

from app.url_validation import safe_get, safe_send

logger = logging.getLogger(__name__)

# Cookie to bypass GDPR consent page on news.google.com.
# Without this, European IPs get redirected to consent.google.com
# and the article page (with the needed data attributes) is never served.
_CONSENT_COOKIE = {
    "SOCS": "CAISNQgDEitib3FfaWRlbnRpdHlmcm9udGVuZHVpc2VydmVyXzIwMjMwODI5LjA3X3AxGgJlbiACGgYIgJa_pwY",
}

_SIG_RE = re.compile(r'data-n-a-sg="([^"]+)"')
_TS_RE = re.compile(r'data-n-a-ts="([^"]+)"')

_RESOLVE_TIMEOUT = 10.0
_REQUEST_DELAY = 0.5  # seconds between resolution requests to avoid rate limiting
_BATCHEXECUTE_URL = "https://news.google.com/_/DotsSplashUi/data/batchexecute"
_USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"


def is_google_news_url(url: str) -> bool:
    """Check if a URL is a Google News redirect that needs resolution."""
    return "news.google.com/" in url and ("/articles/" in url or "/read/" in url)


def _extract_article_id(url: str) -> str | None:
    """Extract the base64 article ID from a Google News URL."""
    parsed = urlparse(url)
    parts = parsed.path.split("/")
    if len(parts) > 1 and parts[-2] in ("articles", "read"):
        return parts[-1]
    return None


class _RateLimitedError(Exception):
    """Raised internally when Google News returns HTTP 429.

    Distinguishes a genuine rate-limit (where retrying further URLs is futile)
    from an ordinary per-URL resolution failure (where the batch should continue).
    """


async def _get_decoding_params(
    article_id: str,
    client: httpx.AsyncClient,
) -> tuple[str, str] | None:
    """Fetch signature and timestamp from the Google News article page.

    Returns (signature, timestamp) tuple or None on failure.

    Raises:
        _RateLimitedError: if Google returns HTTP 429.
    """
    for path_prefix in ("articles", "rss/articles"):
        url = f"https://news.google.com/{path_prefix}/{article_id}"
        try:
            response = await safe_get(client, url)
            if response.status_code == 429:
                logger.warning("Google News rate limited (429) fetching decoding params")
                raise _RateLimitedError
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.debug("Failed to fetch Google News article page (%s): %s", path_prefix, exc)
            continue

        sig_match = _SIG_RE.search(response.text)
        ts_match = _TS_RE.search(response.text)
        if sig_match and ts_match:
            return sig_match.group(1), ts_match.group(1)

    return None


async def _decode_url(
    article_id: str,
    signature: str,
    timestamp: str,
    client: httpx.AsyncClient,
) -> str | None:
    """Call Google's batchexecute endpoint to decode the article URL."""
    payload_inner = [
        "Fbv4je",
        json.dumps(
            [
                "garturlreq",
                [
                    ["X", "X", ["X", "X"], None, None, 1, 1, "US:en", None, 1, None, None, None, None, None, 0, 1],
                    "X",
                    "X",
                    1,
                    [1, 1, 1],
                    1,
                    1,
                    None,
                    0,
                    0,
                    None,
                    0,
                ],
                article_id,
                timestamp,
                signature,
            ]
        ),
    ]
    form_data = f"f.req={quote(json.dumps([[payload_inner]]))}"

    try:
        request = client.build_request(
            "POST",
            _BATCHEXECUTE_URL,
            content=form_data,
            headers={"Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"},
        )
        response = await safe_send(client, request)
        if response.status_code == 429:
            logger.warning("Google News rate limited (429) during URL decode")
            raise _RateLimitedError
        response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.debug("batchexecute request failed: %s", exc)
        return None

    try:
        # Response format: )]}'\n\n<json data>
        parts = response.text.split("\n\n", 1)
        if len(parts) < 2:
            return None
        parsed = json.loads(parts[1])[:-2]
        decoded_url: str = json.loads(parsed[0][2])[1]
        if decoded_url and isinstance(decoded_url, str) and decoded_url.startswith("http"):
            return decoded_url
    except (json.JSONDecodeError, IndexError, TypeError, KeyError) as exc:
        logger.debug("Failed to parse batchexecute response: %s", exc)

    return None


async def resolve_google_news_url(
    url: str,
    client: httpx.AsyncClient,
) -> str:
    """Resolve a single Google News redirect URL to the actual article URL.

    Returns the resolved URL, or the original URL if resolution fails
    (including when rate-limited).
    """
    try:
        return await _resolve_or_raise(url, client)
    except _RateLimitedError:
        return url


async def _resolve_or_raise(
    url: str,
    client: httpx.AsyncClient,
) -> str:
    """Resolve a single URL, propagating ``_RateLimitedError`` on HTTP 429.

    Returns the resolved URL, or the original URL on ordinary (non-rate-limit)
    failure. The batch resolver uses the raised ``_RateLimitedError`` to decide
    whether to abort the remaining URLs.
    """
    article_id = _extract_article_id(url)
    if not article_id:
        return url

    params = await _get_decoding_params(article_id, client)
    if not params:
        return url

    signature, timestamp = params
    decoded = await _decode_url(article_id, signature, timestamp, client)
    if decoded:
        logger.debug("Resolved Google News URL: %s -> %s", url[:60], decoded[:80])
        return decoded

    return url


async def resolve_google_news_urls(
    urls: list[str],
    timeout: float = _RESOLVE_TIMEOUT,
    request_delay: float = _REQUEST_DELAY,
) -> dict[str, str]:
    """Batch-resolve Google News redirect URLs to actual article URLs.

    Resolves URLs sequentially with a delay between requests to avoid
    triggering Google's rate limiter. A single URL failing to resolve does NOT
    abort the batch — the resolver continues and resolves as many as possible.
    Only a genuine HTTP 429 rate-limit aborts the remaining URLs, since
    continuing would just be throttled too.

    Args:
        urls: List of URLs (may include non-Google News URLs, which are skipped).
        timeout: HTTP timeout for resolution requests.
        request_delay: Delay in seconds between resolution requests.

    Returns:
        Dict mapping original Google News URLs to resolved URLs.
        Only contains entries for URLs that were successfully resolved.
    """
    google_urls = [u for u in urls if is_google_news_url(u)]
    if not google_urls:
        return {}

    resolved: dict[str, str] = {}
    failures = 0

    async with httpx.AsyncClient(
        cookies=_CONSENT_COOKIE,
        headers={"User-Agent": _USER_AGENT},
        timeout=timeout,
        follow_redirects=False,
    ) as client:
        for i, url in enumerate(google_urls):
            if i > 0:
                await asyncio.sleep(request_delay)

            try:
                result = await _resolve_or_raise(url, client)
            except _RateLimitedError:
                logger.warning(
                    "Google News rate-limited (429); aborting remaining %d URL(s)",
                    len(google_urls) - i - 1,
                )
                break

            if result == url:
                # Ordinary resolution failure for this URL — log and keep going
                # so one bad URL doesn't drop the rest of the batch.
                failures += 1
                logger.debug("Could not resolve Google News URL, continuing: %s", url[:80])
            else:
                resolved[url] = result

    if failures:
        logger.info("Google News batch: %d URL(s) could not be resolved", failures)

    if resolved:
        logger.info("Resolved %d/%d Google News URLs to actual article URLs", len(resolved), len(google_urls))
    elif google_urls:
        logger.warning("Failed to resolve any of %d Google News URLs", len(google_urls))

    return resolved
