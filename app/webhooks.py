"""Custom webhook delivery for Topic Watch.

Sends structured JSON payloads to arbitrary HTTP endpoints when new information
is found, complementing the Apprise notifications.

Delivery is intent-based: a per-target row is committed inside the check's
durable transaction before any POST is attempted, and the send is a separate
claim -> POST -> apply cycle that the live path and the retry drain share
(TW-AUD-004). Nothing here holds a SQLite connection across a network await.
"""

import asyncio
import logging
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx

from app.analysis.llm import NoveltyResult
from app.config import Settings
from app.crud import (
    abandon_expired_webhooks,
    apply_webhook_outcome,
    claim_webhook_intent,
    list_due_webhook_intents,
    release_stale_webhook_claims,
)
from app.database import short_conn
from app.log_redaction import redact_url
from app.models import PendingWebhook, next_attempt_at, to_db_utc
from app.url_validation import ResolverSaturatedError, _classify_url, _Destination
from app.web.state import live_claim, live_claim_tokens

logger = logging.getLogger(__name__)

_WEBHOOK_TIMEOUT = 10.0

# OVH-139: bound how many queued deliveries the retry drain sends at once.
# The live path already fans out with asyncio.gather; the retry drain previously
# ran strictly one-at-a-time, so a backlog of K failures cost K x up-to-timeout
# seconds at the start of every cycle, delaying due checks. A small cap mirrors
# the live path while staying gentle on endpoints.
_RETRY_DRAIN_CONCURRENCY = 5

# AUG-027: rows per drain. Bounds how long queued deliveries can hold a scheduler
# cycle before due-topic work starts; the rest wait for the next tick.
_RETRY_DRAIN_LIMIT = 20

# Single-flight guard: serializes webhook drains within this process so two
# overlapping drains (scheduler tick vs. a UI/CLI check-all) cannot both walk
# the queue at once. The cross-process case is covered by the atomic per-row
# claim below. (OVH-017)
_retry_lock = asyncio.Lock()

# Claims older than this are treated as stale (a drainer crashed mid-send, or its
# POST timed out with an unknown outcome) and re-armed. Comfortably exceeds the
# per-item send timeout so an in-flight send is never stolen. Measured against
# the stored wall-clock claim stamp; a clock jump can therefore re-arm a live
# claim early, which is harmless because the apply is fenced by claim_token
# rather than by elapsed time (AUG-277).
_CLAIM_STALE_AFTER_SECONDS = 600.0

# HTTP statuses that will not change on a retry: the payload or the address is
# wrong, so three more identical POSTs only add noise (AUG-324).
_TERMINAL_STATUSES = frozenset({400, 401, 403, 404, 410, 413, 422})

# Ceiling on a Retry-After the receiver asked for. A header is a hint, not a
# lease: without a bound, one bad value parks a delivery indefinitely.
_MAX_RETRY_AFTER_SECONDS = 3600.0


@dataclass(frozen=True)
class WebhookOutcome:
    """What one webhook POST actually did.

    A bare bool erased everything the retry scheduler needs: a permanent 400 was
    indistinguishable from a transient 503, and a receiver's stated recovery time
    was discarded, so a 429 could burn its whole retry budget before the endpoint
    was ready to answer (AUG-324).
    """

    ok: bool
    status: int | None = None
    retryable: bool = True
    retry_after_s: float | None = None
    error: str | None = None


def _parse_retry_after(value: str | None) -> float | None:
    """Parse a ``Retry-After`` header: delta-seconds or HTTP-date."""
    if not value:
        return None
    raw = value.strip()
    try:
        return max(0.0, min(float(int(raw)), _MAX_RETRY_AFTER_SECONDS))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    delta = (when - datetime.now(UTC)).total_seconds()
    return max(0.0, min(delta, _MAX_RETRY_AFTER_SECONDS))


def _build_webhook_payload(topic_name: str, novelty_result: NoveltyResult) -> dict:
    """Build the JSON payload for a webhook POST."""
    return {
        "topic": topic_name,
        "reasoning": novelty_result.reasoning,
        "summary": novelty_result.summary or "",
        "key_facts": novelty_result.key_facts,
        "source_urls": novelty_result.source_urls,
        "confidence": novelty_result.confidence,
        "relevance": novelty_result.relevance,
        "importance": novelty_result.importance,
        "timestamp": datetime.now(UTC).isoformat(),
    }


async def send_webhook(url: str, payload: dict, timeout: float = _WEBHOOK_TIMEOUT) -> WebhookOutcome:
    """POST a JSON payload to a webhook URL. Never raises.

    ``timeout`` is a TOTAL wall-clock deadline covering validation and the whole
    POST, not just each individual network operation (AUG-247). httpx's timeout
    is per-operation inactivity, so an endpoint trickling sub-timeout chunks kept
    the send — and the gather and check cycle around it — alive indefinitely,
    long enough for the ten-minute stale-claim window to expire and permit a
    duplicate send. The granular httpx phase timeouts are kept underneath it.

    SSRF note: _classify_url performs blocking DNS resolution, so it is
    offloaded to a worker thread to avoid stalling the event loop. A
    DNS-rebinding TOCTOU window between this check and the POST is a
    pre-existing, architectural limitation shared by all outbound fetches.

    A non-http(s) scheme and a URL that will not parse are terminal — neither
    becomes valid on the next attempt. So is a host that RESOLVED to a private
    or reserved address: that verdict does not change on a retry. A host the
    resolver could not answer for at all is the one blocked outcome that stays
    retryable, because it is the absence of a verdict rather than one.
    """
    try:
        async with asyncio.timeout(timeout):
            return await _send_webhook_within_deadline(url, payload, timeout)
    except TimeoutError:
        logger.warning("Webhook exceeded its %.1fs deadline for %s", timeout, redact_url(url))
        # Retryable: a slow endpoint is a transient condition, not a rejected
        # payload. The outcome is unknown, so the retry may duplicate — which is
        # the honest trade a non-idempotent transport forces (TW-AUD-004).
        return WebhookOutcome(ok=False, error="deadline exceeded")


async def _send_webhook_within_deadline(url: str, payload: dict, timeout: float) -> WebhookOutcome:
    """Validate and POST one webhook. Caller owns the total deadline."""
    # Validate the URL BEFORE the POST. A malformed URL (e.g. an unbracketed or
    # otherwise broken IPv6 literal) makes urlparse / _classify_url raise
    # ValueError, which would violate the documented "Never raises" contract, so
    # any validation error counts as blocked (OVH-131).
    try:
        # Scheme allowlist BEFORE the POST (OVH-141). _classify_url() reports
        # PUBLIC for schemes with no netloc (file://, gopher://, ftp://), so
        # without this explicit check the first hop would rely solely on httpx
        # raising UnsupportedProtocol — a weaker backstop than the per-hop
        # redirect checks.
        if urlparse(url).scheme not in ("http", "https"):
            logger.warning("Blocked webhook to non-http(s) URL: %s", redact_url(url))
            return WebhookOutcome(ok=False, retryable=False, error="non-http(s) URL")

        try:
            destination = await asyncio.to_thread(_classify_url, url)
        except ResolverSaturatedError:
            # No lookup ran, so nothing is known about the host: blocked, and
            # retryable for the same reason an unresolvable one is.
            destination = _Destination.UNRESOLVABLE

        if destination is _Destination.PRIVATE:
            # Terminal: the host RESOLVED to a private/reserved address. That
            # verdict does not change on a retry, so spending the budget on it
            # only delays the abandonment the ledger will record anyway.
            logger.warning("Blocked webhook to private/reserved URL: %s", redact_url(url))
            return WebhookOutcome(ok=False, retryable=False, error="private/reserved URL")
        if destination is not _Destination.PUBLIC:
            # Retryable: a host DNS could not answer for — a resolver restart, a
            # container that started before its DNS, a lookup that hit the
            # resolve timeout. The send is still blocked; only the finality is gone.
            logger.warning("Blocked webhook to unresolvable URL: %s", redact_url(url))
            return WebhookOutcome(ok=False, error="unresolvable URL")
    except Exception:
        logger.warning("Blocked webhook to malformed URL: %s", redact_url(url), exc_info=True)
        return WebhookOutcome(ok=False, retryable=False, error="malformed URL")

    try:
        # follow_redirects=False (httpx default, made explicit) so a 3xx to a
        # private address can't bypass the _classify_url check above. The
        # per-phase timeouts stay under the caller's total deadline; connect gets
        # a tighter share so a black-holed host fails fast rather than spending
        # the whole budget before a single byte moves.
        phase_timeout = httpx.Timeout(timeout, connect=min(timeout, 5.0))
        async with httpx.AsyncClient(timeout=phase_timeout, follow_redirects=False) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            logger.info("Webhook delivered to %s (status %d)", redact_url(url), response.status_code)
            return WebhookOutcome(ok=True, status=response.status_code)
    except httpx.TimeoutException:
        logger.warning("Webhook timeout for %s", redact_url(url))
        return WebhookOutcome(ok=False, error="timeout")
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        retryable = status not in _TERMINAL_STATUSES
        retry_after = _parse_retry_after(exc.response.headers.get("Retry-After")) if retryable else None
        logger.warning(
            "Webhook HTTP %d for %s (%s)",
            status,
            redact_url(url),
            "retryable" if retryable else "permanent",
        )
        return WebhookOutcome(
            ok=False,
            status=status,
            retryable=retryable,
            retry_after_s=retry_after,
            error=f"HTTP {status}",
        )
    except Exception as exc:
        logger.warning("Webhook error for %s", redact_url(url), exc_info=True)
        return WebhookOutcome(ok=False, error=type(exc).__name__)


def build_webhook_intents(
    topic_name: str,
    novelty_result: NoveltyResult,
    settings: Settings,
    topic_id: int,
    check_result_id: int | None = None,
) -> list[PendingWebhook]:
    """One unsaved delivery intent per configured webhook target. Pure.

    Pure so the caller can build the intents with nothing open and hand them to
    the durable transaction, which inserts them alongside the CheckResult that
    justifies them.
    """
    payload = _build_webhook_payload(topic_name, novelty_result)
    return [
        PendingWebhook(topic_id=topic_id, check_result_id=check_result_id, url=url, payload=payload)
        for url in settings.notifications.webhook_urls
    ]


async def _deliver_one_intent(
    intent: PendingWebhook,
    db_path: Path | None,
    conn: sqlite3.Connection | None = None,
) -> bool:
    """Claim, POST, apply — for one intent. Returns True when it was delivered.

    Each database interaction is its own short connection with its own commit, so
    a sibling's failure can never roll back what this one already applied, and no
    connection is open across the POST.
    """
    intent_id = intent.id
    if intent_id is None:
        return False

    claim_token = secrets.token_hex(8)
    now_iso = to_db_utc(datetime.now(UTC))
    with short_conn(conn, db_path) as claim_conn:
        won = claim_webhook_intent(claim_conn, intent_id, claim_token, now_iso)
        claim_conn.commit()
    if not won:
        logger.debug("Webhook intent id=%d not claimable (claimed, exhausted or not due); skipping", intent_id)
        return False

    # Registered while the POST is in flight so the wall-clock stale release
    # cannot re-arm a live claim after a forward clock step and deliver the same
    # payload twice (AUG-277).
    with live_claim(claim_token):
        return await _send_claimed_webhook_intent(intent, intent_id, claim_token, db_path, conn)


async def _send_claimed_webhook_intent(
    intent: PendingWebhook,
    intent_id: int,
    claim_token: str,
    db_path: Path | None,
    conn: sqlite3.Connection | None,
) -> bool:
    """POST an already-claimed intent and apply its outcome (caller holds the live claim)."""
    outcome: WebhookOutcome | None = None
    try:
        outcome = await send_webhook(intent.url, intent.payload)

        due = (
            None
            if outcome.ok or not outcome.retryable
            else next_attempt_at(intent.retry_count, hint_s=outcome.retry_after_s)
        )
        with short_conn(conn, db_path) as apply_conn:
            applied = apply_webhook_outcome(
                apply_conn,
                intent_id,
                claim_token,
                sent=outcome.ok,
                error=outcome.error,
                next_attempt_at=due,
                terminal=not outcome.retryable,
            )
            apply_conn.commit()
        if not applied:
            logger.warning("Late apply for webhook intent id=%d ignored: the claim is no longer ours", intent_id)
        elif not outcome.ok and not outcome.retryable:
            logger.warning(
                "Abandoning webhook intent id=%d without retry (topic_id=%s url=%s reason=%s)",
                intent_id,
                intent.topic_id,
                redact_url(intent.url),
                outcome.error,
            )
        return outcome.ok
    except Exception as exc:
        # An escaping exception must still land an outcome. Without this the row
        # stays 'sending' for good: retry_count never moves so it is never
        # abandoned, retention prunes only terminal rows, the queue view hides it,
        # and every later drain POSTs it again. ``sent`` carries whatever the POST
        # already told us, so a payload the endpoint accepted before the apply hit
        # a locked database is recorded as delivered instead of being sent twice;
        # the claim fence makes it a no-op if that first apply landed after all.
        logger.warning("Webhook intent id=%d raised during delivery", intent_id, exc_info=True)
        sent = outcome is not None and outcome.ok
        try:
            with short_conn(conn, db_path) as apply_conn:
                apply_webhook_outcome(
                    apply_conn,
                    intent_id,
                    claim_token,
                    sent=sent,
                    error=type(exc).__name__,
                    next_attempt_at=None if sent else next_attempt_at(intent.retry_count),
                )
                apply_conn.commit()
        except Exception:
            # The database is what failed; leave the claim to the stale release
            # rather than raising into the gather, which records nothing at all.
            logger.warning("Could not record the outcome for webhook intent id=%d", intent_id, exc_info=True)
        return sent


async def deliver_webhook_intents(
    intents: list[PendingWebhook],
    settings: Settings,
    db_path: Path | None,
    conn: sqlite3.Connection | None = None,
) -> int:
    """Deliver a batch of already-persisted webhook intents. Returns the count sent.

    ``return_exceptions=True`` and every result inspected (AUG-263): the default
    fail-fast gather propagated the first claim/apply error while its siblings kept
    running, unwinding the drain's lock and the scheduler's job around still-live
    children. Ownership is held until every child has settled.
    """
    if not intents:
        return 0

    semaphore = asyncio.Semaphore(_RETRY_DRAIN_CONCURRENCY)

    async def _process(intent: PendingWebhook) -> bool:
        async with semaphore:
            return await _deliver_one_intent(intent, db_path, conn)

    results = await asyncio.gather(*(_process(intent) for intent in intents), return_exceptions=True)

    delivered = 0
    for intent, result in zip(intents, results, strict=True):
        if isinstance(result, BaseException):
            logger.warning("Webhook intent id=%s failed unexpectedly", intent.id, exc_info=(type(result), result, None))
            continue
        if result:
            delivered += 1

    if delivered < len(intents):
        logger.warning("Webhooks: %d/%d delivered", delivered, len(intents))
    else:
        logger.info("Webhooks: all %d delivered", delivered)
    return delivered


async def retry_pending_webhooks(
    conn: sqlite3.Connection | None = None,
    settings: Settings | None = None,
    *,
    db_path: Path | None = None,
) -> None:
    """Drain the webhook delivery queue: due intents get one more attempt.

    Same claim/send/apply cycle as the live path — a queued intent and a
    just-created one are the same kind of thing, so they take the same code path.

    Args:
        conn: Optional existing connection (back-compat; callers that already
            own a connection may pass it). Reused but still committed per item
            and never held across a send.
        settings: Application settings (required).
        db_path: Database path used to open short-lived connections when no
            ``conn`` is provided.
    """
    if settings is None:
        raise ValueError("settings is required")

    # Single-flight: only one drain runs at a time in this process. A second
    # caller skips rather than walking the same queue concurrently (OVH-017).
    if _retry_lock.locked():
        logger.debug("Webhook retry already in progress; skipping overlapping drain")
        return

    async with _retry_lock:
        await _drain_webhook_intents(conn, settings, db_path)


async def _drain_webhook_intents(
    conn: sqlite3.Connection | None,
    settings: Settings,
    db_path: Path | None,
) -> None:
    """Drain the webhook queue once (caller holds ``_retry_lock``)."""
    now = datetime.now(UTC)
    stale_cutoff = to_db_utc(now - timedelta(seconds=_CLAIM_STALE_AFTER_SECONDS))
    with short_conn(conn, db_path) as snapshot:
        released = release_stale_webhook_claims(snapshot, stale_cutoff, live_claim_tokens())
        if released:
            logger.warning("Re-armed %d stale webhook claim(s)", released)
        for item in abandon_expired_webhooks(snapshot):
            # One WARNING per permanently-dropped delivery so an abandoned
            # webhook is observable: identify it by topic/check ids and the
            # redacted destination (never the secret-bearing full URL) (OVH-040).
            logger.warning(
                "Abandoning webhook after max retries (topic_id=%s check_result_id=%s url=%s created_at=%s)",
                item.topic_id,
                item.check_result_id,
                redact_url(item.url),
                item.created_at.isoformat(),
            )
        snapshot.commit()
        pending = list_due_webhook_intents(snapshot, to_db_utc(now), _RETRY_DRAIN_LIMIT)

    if not pending:
        return

    logger.info("Retrying %d pending webhook(s)", len(pending))
    await deliver_webhook_intents(pending, settings, db_path, conn)
