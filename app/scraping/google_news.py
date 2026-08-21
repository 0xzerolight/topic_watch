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
import secrets
import time
from typing import Any
from urllib.parse import quote, urlparse

import httpx

from app.scraping.source import Deadline, SourceDeadlineExceeded, bounded, host_matches, url_hostname
from app.url_validation import safe_get, safe_send

logger = logging.getLogger(__name__)

GOOGLE_NEWS_HOST = "news.google.com"
"""The only host whose entry links are opaque Google News redirects."""

# Cookie to bypass GDPR consent page on news.google.com.
# Without this, European IPs get redirected to consent.google.com
# and the article page (with the needed data attributes) is never served.
_CONSENT_COOKIE = {
    "SOCS": "CAISNQgDEitib3FfaWRlbnRpdHlmcm9udGVuZHVpc2VydmVyXzIwMjMwODI5LjA3X3AxGgJlbiACGgYIgJa_pwY",
}

_SIG_RE = re.compile(r'data-n-a-sg="([^"]+)"')
_TS_RE = re.compile(r'data-n-a-ts="([^"]+)"')

_RESOLVE_TIMEOUT = 10.0
_REQUEST_DELAY = 0.5  # max jittered delay before each request, to avoid rate limiting
# Small bounded fan-out for resolution. Caps concurrent requests to stay under
# Google's throttle threshold while removing the strict O(N) serialization
# (OVH-056). A genuine 429 still aborts the remaining (unstarted) work.
_RESOLVE_CONCURRENCY = 3
_BATCHEXECUTE_URL = "https://news.google.com/_/DotsSplashUi/data/batchexecute"
_USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
_RPC_ID = "Fbv4je"
"""The batchexecute RPC whose record carries the decoded URL."""

_RATE_LIMIT_COOLDOWN_SECONDS = 300.0
_MAX_RATE_LIMIT_COOLDOWN_SECONDS = 3600.0

_rate_limited_until = 0.0
"""Monotonic deadline until which Google has told us to stop (AUG-305).

Process-wide, because the throttle is: the semaphore and abort flag used to be
allocated per call, so the default three concurrent topic checks each got their
own three-wide gate, and a 429 seen by one batch could not stop the next batch
from resolving against the same provider. Monotonic, so a clock adjustment
cannot lift the cooldown early or strand it (wave-A clock policy)."""

_resolver_gate: tuple[object, asyncio.Semaphore] | None = None


def _shared_gate() -> asyncio.Semaphore:
    """The process-wide concurrency gate for Google resolution.

    Rebuilt when the running loop changes, because an ``asyncio.Semaphore``
    binds to the first loop that awaits it; every process that matters runs one
    loop, so in practice this is created once and shared by every batch.
    """
    global _resolver_gate
    loop = asyncio.get_running_loop()
    if _resolver_gate is None or _resolver_gate[0] is not loop:
        _resolver_gate = (loop, asyncio.Semaphore(_RESOLVE_CONCURRENCY))
    return _resolver_gate[1]


def _rate_limit_cooldown(retry_after: float | None) -> float:
    """Seconds to stay away, honouring a usable ``Retry-After`` within bounds."""
    if retry_after is None or retry_after <= 0:
        return _RATE_LIMIT_COOLDOWN_SECONDS
    return min(retry_after, _MAX_RATE_LIMIT_COOLDOWN_SECONDS)


def _enter_rate_limit_cooldown(retry_after: float | None) -> None:
    global _rate_limited_until
    _rate_limited_until = max(_rate_limited_until, time.monotonic() + _rate_limit_cooldown(retry_after))


def _parse_retry_after(response: httpx.Response) -> float | None:
    """The ``Retry-After`` delay in seconds, when the header carries a usable one."""
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return float(raw.strip())
    except ValueError:
        return None  # HTTP-date form: fall back to the default cooldown


def is_google_news_url(url: str) -> bool:
    """Check if a URL is a Google News redirect that needs resolution.

    Identity is the parsed hostname plus Google's documented redirect paths
    (TW-AUD-031). The previous raw-URL substring test matched any host that
    merely mentioned ``news.google.com/`` in its path or query, which routed
    arbitrary feed URLs into Google's resolver, and missed nothing a host check
    does not already cover.
    """
    return host_matches(url_hostname(url), GOOGLE_NEWS_HOST) and _extract_article_id(url) is not None


def _extract_article_id(url: str) -> str | None:
    """Extract the base64 article ID from a Google News URL."""
    parsed = urlparse(url)
    parts = parsed.path.split("/")
    if len(parts) > 1 and parts[-2] in ("articles", "read") and parts[-1]:
        return parts[-1]
    return None


class _RateLimitedError(Exception):
    """Raised internally when Google News returns HTTP 429.

    Distinguishes a genuine rate-limit (where retrying further URLs is futile)
    from an ordinary per-URL resolution failure (where the batch should continue).
    Carries the ``Retry-After`` delay when Google supplied a usable one.
    """

    def __init__(self, retry_after: float | None = None) -> None:
        super().__init__("Google News rate limited (429)")
        self.retry_after = retry_after


class _DecoderBrokeError(Exception):
    """Raised internally when a 200 batchexecute body fails to parse structurally.

    Google's internal batchexecute response shape changes from time to time. When
    it does, the decode step throws on a *successful* HTTP response — a structural
    decoder break, not an ordinary per-URL miss (where params/decoded are simply
    absent). Distinguishing the two lets the batch resolver emit a semantic
    'decoder broke vs URLs unresolvable' label so a whole-format change is not
    masked as sporadic misses at INFO level (OVH-134).
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
                raise _RateLimitedError(_parse_retry_after(response))
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
            raise _RateLimitedError(_parse_retry_after(response))
        response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.debug("batchexecute request failed: %s", exc)
        return None

    payload = _rpc_payload(response.text)
    if payload is None:
        # The record is there and says Google has no URL for this article — an
        # ordinary miss. Feeding that null to json.loads raised TypeError and
        # dressed every normal no-result frame up as a format change (AUG-310).
        return None

    try:
        decoded_url: str = json.loads(payload)[1]
    except (json.JSONDecodeError, IndexError, TypeError, KeyError) as exc:
        # A record whose payload no longer matches the expected shape: the decoder
        # broke (Google changed the format), distinct from an unresolvable URL.
        logger.debug("Failed to parse batchexecute payload: %s", exc)
        raise _DecoderBrokeError(str(exc)) from exc

    if decoded_url and isinstance(decoded_url, str) and decoded_url.startswith("http"):
        return decoded_url
    return None


def _rpc_payload(body: str) -> Any | None:
    """The decode RPC's payload from a batchexecute body, or ``None`` for a miss.

    Separates "the response is not the shape we know" from "the response says
    there is no URL". Only the first is a decoder break: a missing JSON body, a
    non-list container or no record for our RPC id means Google changed the
    format, and that warrants the format-change warning. A present record with a
    null payload is Google's normal way of saying it could not resolve this one.
    """
    # Response format: )]}'\n\n<json data>
    parts = body.split("\n\n", 1)
    if len(parts) < 2:
        raise _DecoderBrokeError("batchexecute response had no JSON body")
    try:
        frames = json.loads(parts[1])
    except json.JSONDecodeError as exc:
        logger.debug("Failed to parse batchexecute response: %s", exc)
        raise _DecoderBrokeError(str(exc)) from exc
    if not isinstance(frames, list):
        raise _DecoderBrokeError(f"batchexecute body was {type(frames).__name__}, expected a list")
    for frame in frames:
        if isinstance(frame, list) and len(frame) > 2 and _RPC_ID in frame[:2]:
            return frame[2]
    raise _DecoderBrokeError(f"no {_RPC_ID} record in the batchexecute response")


async def resolve_google_news_url(
    url: str,
    client: httpx.AsyncClient,
) -> str:
    """Resolve a single Google News redirect URL to the actual article URL.

    Returns the resolved URL, or the original URL if resolution fails
    (including when rate-limited or the decoder broke).
    """
    try:
        return await _resolve_or_raise(url, client)
    except (_RateLimitedError, _DecoderBrokeError):
        return url


async def _resolve_or_raise(
    url: str,
    client: httpx.AsyncClient,
) -> str:
    """Resolve a single URL, propagating control signals to the batch resolver.

    Returns the resolved URL, or the original URL on ordinary (non-rate-limit)
    failure. Propagates ``_RateLimitedError`` on HTTP 429 (the batch aborts) and
    ``_DecoderBrokeError`` when a 200 response failed to parse structurally (the
    batch labels it a decoder break, distinct from an unresolvable URL, OVH-134).
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
    deadline: Deadline | None = None,
) -> dict[str, str]:
    """Batch-resolve Google News redirect URLs to actual article URLs.

    Resolves under the process-wide bounded gate (``_RESOLVE_CONCURRENCY``) via
    ``asyncio.gather`` instead of strict O(N) serialization, with a jittered
    pre-request throttle (0..``request_delay``) replacing the fixed inter-request
    sleep — the same average request rate at a fraction of the wall-clock latency
    (OVH-056). A single URL failing to resolve does NOT abort the batch; the
    resolver resolves as many as possible.

    The gate and the 429 cooldown are shared by every concurrent batch (AUG-305).
    Allocating them per call gave each of the three concurrent topic checks its
    own three-wide gate — nine in flight against one provider — and left a 429
    observed by one batch unable to stop another from starting. A 429 now puts
    the whole process into a bounded cooldown (honouring ``Retry-After``), so
    later batches short-circuit instead of piling on; tasks already in flight may
    finish, and their unresolved entries simply keep their redirect URL.

    Args:
        urls: List of URLs (may include non-Google News URLs, which are skipped).
        timeout: HTTP timeout for resolution requests.
        request_delay: Upper bound (seconds) on the jittered per-request throttle.
        deadline: The topic's source budget. Resolution is a best-effort
            enrichment — unresolved entries keep their redirect URL — so an
            expired budget short-circuits the remaining URLs exactly like a 429
            does, rather than failing the batch (TW-AUD-018).

    Returns:
        Dict mapping original Google News URLs to resolved URLs.
        Only contains entries for URLs that were successfully resolved.
    """
    # One resolution per distinct URL, in first-seen order: the result is keyed by
    # URL anyway, so a repeated entry used to buy the same answer twice — spending
    # one of the three slots a unique URL needed and counting itself as a miss in
    # the batch metrics (AUG-311).
    google_urls = list(dict.fromkeys(u for u in urls if is_google_news_url(u)))
    if not google_urls:
        return {}

    budget = deadline if deadline is not None else Deadline.after()
    resolved: dict[str, str] = {}
    failures = 0
    decoder_breaks = 0
    out_of_budget = False
    if time.monotonic() < _rate_limited_until:
        logger.info("Google News is rate-limiting; %d URL(s) keep their redirect", len(google_urls))
        return {}

    semaphore = _shared_gate()
    # Shared 429 abort flag: once Google rate-limits, every not-yet-started task
    # short-circuits rather than piling on more throttled requests.
    aborted = asyncio.Event()

    async with httpx.AsyncClient(
        cookies=_CONSENT_COOKIE,
        headers={"User-Agent": _USER_AGENT},
        timeout=timeout,
        follow_redirects=False,
    ) as client:

        async def _resolve_one(url: str) -> tuple[str, str | None, bool]:
            """Return (url, resolved_url, decoder_broke).

            ``resolved_url`` is None on failure/abort/429; ``decoder_broke`` is
            True only when a 200 batchexecute body failed to parse structurally
            (OVH-134).
            """
            nonlocal out_of_budget
            async with semaphore:
                if aborted.is_set() or budget.expired():
                    return url, None, False
                # Jittered throttle: 0..request_delay before each request keeps the
                # average rate near the old fixed delay without the strict stall,
                # and never outlasts the budget it is spending.
                if request_delay > 0:
                    await asyncio.sleep(budget.slice(secrets.SystemRandom().uniform(0, request_delay)))
                try:
                    result = await bounded(budget, "Google News resolution", _resolve_or_raise(url, client))
                except _RateLimitedError as exc:
                    aborted.set()
                    _enter_rate_limit_cooldown(exc.retry_after)
                    return url, None, False
                except _DecoderBrokeError:
                    return url, None, True
                except SourceDeadlineExceeded:
                    out_of_budget = True
                    return url, None, False
            return url, (None if result == url else result), False

        outcomes = await asyncio.gather(*(_resolve_one(u) for u in google_urls))

    for url, result, decoder_broke in outcomes:
        if decoder_broke:
            decoder_breaks += 1
        if result is None:
            failures += 1
            logger.debug("Could not resolve Google News URL: %s", url[:80])
        else:
            resolved[url] = result

    if aborted.is_set():
        logger.warning("Google News rate-limited (429); aborted remaining resolutions")

    if out_of_budget:
        logger.warning("Google News resolution ran out of budget; %d URL(s) keep their redirect", failures)

    # A structural decoder break (Google changed the batchexecute response shape)
    # is a different failure mode than an unresolvable URL: it tends to affect
    # every URL and means stored articles keep their unfetchable redirect links.
    # Surface the partial case at WARNING so it is not masked as sporadic misses
    # (the total-break case already WARNs at 'Failed to resolve any', OVH-134).
    if decoder_breaks:
        logger.warning(
            "Google News batchexecute decoder broke for %d/%d URL(s) — response format may have changed",
            decoder_breaks,
            len(google_urls),
        )

    if failures:
        logger.info("Google News batch: %d URL(s) could not be resolved", failures)

    if resolved:
        logger.info("Resolved %d/%d Google News URLs to actual article URLs", len(resolved), len(google_urls))
    elif google_urls:
        logger.warning("Failed to resolve any of %d Google News URLs", len(google_urls))

    return resolved
