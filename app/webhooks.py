"""Custom webhook delivery for Topic Watch.

Sends structured JSON payloads to arbitrary HTTP endpoints when
new information is found, complementing the Apprise notifications.
"""

import asyncio
import logging
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

import httpx

from app.analysis.llm import NoveltyResult
from app.config import Settings
from app.crud import (
    claim_pending_webhook,
    create_pending_webhook,
    delete_expired_webhooks,
    delete_pending_webhook,
    increment_webhook_retry,
    list_pending_webhooks,
    release_stale_webhook_claims,
)
from app.database import short_conn
from app.log_redaction import redact_url
from app.models import PendingWebhook
from app.url_validation import is_private_url

logger = logging.getLogger(__name__)

_WEBHOOK_TIMEOUT = 10.0

# OVH-139: bound how many queued deliveries the retry drain sends at once.
# The live path (send_webhooks) already fans out with asyncio.gather; the retry
# drain previously ran strictly one-at-a-time, so a backlog of K failures cost
# K x up-to-timeout seconds at the start of every cycle, delaying due checks.
# A small cap mirrors the live path while staying gentle on endpoints.
_RETRY_DRAIN_CONCURRENCY = 5

# Single-flight guard: serializes webhook drains within this process so two
# overlapping drains (scheduler tick vs. a UI/CLI check-all) cannot both walk
# the queue at once. The cross-process case is covered by the atomic per-row
# claim (claimed_at) below. (OVH-017)
_retry_lock = asyncio.Lock()

# Claims older than this are treated as stale (a drainer crashed mid-send) and
# released so the row can be re-claimed. Comfortably exceeds the per-item send
# timeout so an in-flight send is never stolen.
_CLAIM_STALE_AFTER = timedelta(minutes=10)


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
        "timestamp": datetime.now(UTC).isoformat(),
    }


async def send_webhook(url: str, payload: dict, timeout: float = _WEBHOOK_TIMEOUT) -> bool:
    """POST a JSON payload to a webhook URL.

    Returns True on success (2xx response), False on failure.
    Never raises — all errors are caught and logged.

    SSRF note: is_private_url performs blocking DNS resolution, so it is
    offloaded to a worker thread to avoid stalling the event loop. A
    DNS-rebinding TOCTOU window between this check and the POST is a
    pre-existing, architectural limitation shared by all outbound fetches.
    """
    # Validate the URL BEFORE the POST. A malformed URL (e.g. an unbracketed or
    # otherwise broken IPv6 literal) makes urlparse / is_private_url raise
    # ValueError, which would violate the documented "Never raises" contract —
    # both callers rely on it (send_webhooks' gather and retry_pending_webhooks'
    # try/except), so a leaked exception silently re-queues an unparseable URL
    # with no specific log. Treat any validation error as "blocked" (OVH-131).
    try:
        # Scheme allowlist BEFORE the POST (OVH-141). is_private_url() returns
        # False for schemes with no netloc (file://, gopher://, ftp://), so
        # without this explicit check the first hop would rely solely on httpx
        # raising UnsupportedProtocol — a weaker backstop than the per-hop
        # redirect checks.
        if urlparse(url).scheme not in ("http", "https"):
            logger.warning("Blocked webhook to non-http(s) URL: %s", redact_url(url))
            return False

        if await asyncio.to_thread(is_private_url, url):
            logger.warning("Blocked webhook to private/reserved URL: %s", redact_url(url))
            return False
    except Exception:
        logger.warning("Blocked webhook to malformed URL: %s", redact_url(url), exc_info=True)
        return False

    try:
        # follow_redirects=False (httpx default, made explicit) so a 3xx to a
        # private address can't bypass the is_private_url check above.
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            logger.info("Webhook delivered to %s (status %d)", redact_url(url), response.status_code)
            return True
    except httpx.TimeoutException:
        logger.warning("Webhook timeout for %s", redact_url(url))
        return False
    except httpx.HTTPStatusError as exc:
        logger.warning("Webhook HTTP %d for %s", exc.response.status_code, redact_url(url))
        return False
    except Exception:
        logger.warning("Webhook error for %s", redact_url(url), exc_info=True)
        return False


async def send_webhooks(
    topic_name: str,
    novelty_result: NoveltyResult,
    settings: Settings,
    conn: sqlite3.Connection | None = None,
    topic_id: int | None = None,
    check_result_id: int | None = None,
) -> int:
    """Send webhook notifications to all configured webhook URLs.

    Args:
        topic_name: The topic name.
        novelty_result: The novelty analysis result.
        settings: Application settings.
        conn: Optional DB connection. When given together with topic_id,
            failed deliveries are enqueued to pending_webhooks for retry.
        topic_id: Topic id used when enqueuing failed deliveries.
        check_result_id: Optional originating check result id (for traceability).

    Returns:
        Number of successfully delivered webhooks.
    """
    webhook_urls = settings.notifications.webhook_urls
    if not webhook_urls:
        return 0

    payload = _build_webhook_payload(topic_name, novelty_result)

    tasks = [send_webhook(url, payload) for url in webhook_urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    success_count = 0
    can_queue = conn is not None and topic_id is not None
    queued = False
    for url, result in zip(webhook_urls, results, strict=True):
        if isinstance(result, bool) and result:
            success_count += 1
        elif can_queue:
            # Persist the failed delivery so a later cycle can retry it
            # instead of dropping it (fire-and-forget loses failures).
            try:
                assert conn is not None and topic_id is not None
                create_pending_webhook(conn, topic_id, url, payload, check_result_id)
                queued = True
            except Exception:
                logger.warning("Failed to enqueue webhook for retry (url=%s)", redact_url(url), exc_info=True)

    if queued:
        assert conn is not None
        conn.commit()

    if success_count < len(webhook_urls):
        logger.warning(
            "Webhooks: %d/%d delivered for topic '%s'",
            success_count,
            len(webhook_urls),
            topic_name,
        )
    else:
        logger.info(
            "Webhooks: all %d delivered for topic '%s'",
            success_count,
            topic_name,
        )

    return success_count


async def retry_pending_webhooks(
    conn: sqlite3.Connection | None = None,
    settings: Settings | None = None,
    *,
    db_path: Path | None = None,
) -> None:
    """Retry any pending webhook deliveries from previous check cycles.

    Mirrors retry_pending_notifications: successful deliveries are deleted,
    failures get their retry count incremented, and deliveries exceeding
    max_retries are dropped.

    Connection handling: no sqlite connection is held across the network
    sends. The pending rows are snapshotted under a short connection, the
    POSTs run with no open connection, and results are applied with a commit
    *per item* so a mid-loop crash never rolls back already-applied
    delete/increment operations (which would let a permanently-failing URL be
    retried unbounded across restarts).

    Args:
        conn: Optional existing connection (back-compat; callers that already
            own a connection may pass it). When given, it is reused but still
            committed per item and never held across a send.
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
        await _drain_pending_webhooks(conn, db_path)


async def _drain_pending_webhooks(
    conn: sqlite3.Connection | None,
    db_path: Path | None,
) -> None:
    """Drain the webhook retry queue once (caller holds ``_retry_lock``)."""
    # --- Phase 1: snapshot pending rows under a short-lived connection. ---
    stale_cutoff = (datetime.now(UTC) - _CLAIM_STALE_AFTER).isoformat()
    with short_conn(conn, db_path) as snapshot:
        released = release_stale_webhook_claims(snapshot, stale_cutoff)
        if released:
            logger.warning("Released %d stale webhook claim(s)", released)
        abandoned = delete_expired_webhooks(snapshot)
        for item in abandoned:
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
        if abandoned:
            logger.warning("Deleted %d expired pending webhook(s)", len(abandoned))
        snapshot.commit()
        pending = list_pending_webhooks(snapshot)

    if not pending:
        return

    logger.info("Retrying %d pending webhook(s)", len(pending))

    # --- Phase 2: claim, send with NO connection held, then apply per item. ---
    # OVH-139: process items with bounded concurrency (mirrors the live path's
    # bounded gather) instead of strict K x timeout serialization. The 1.6
    # invariants are preserved: each row is still claimed atomically exactly
    # once (only the winning drainer sends, no double-delivery), and each item
    # is applied + committed on its own short connection. The claim and apply
    # blocks contain no await points, so on a shared ``conn`` they never
    # interleave mid-transaction.
    semaphore = asyncio.Semaphore(_RETRY_DRAIN_CONCURRENCY)

    async def _process(webhook: PendingWebhook) -> None:
        webhook_id = webhook.id
        assert webhook_id is not None
        async with semaphore:
            # Atomically claim this row. A concurrent (cross-process) drainer
            # that already claimed it returns rowcount 0 here, so we skip — only
            # the winner sends, preventing double-delivery (OVH-017).
            claimed_at = datetime.now(UTC).isoformat()
            with short_conn(conn, db_path) as claim_conn:
                won = claim_pending_webhook(claim_conn, webhook_id, claimed_at)
                claim_conn.commit()
            if not won:
                logger.debug("Webhook id=%d already claimed by another drain; skipping", webhook_id)
                return

            try:
                sent = await send_webhook(webhook.url, webhook.payload)
            except Exception:
                sent = False
                logger.warning("Retry error for webhook id=%d", webhook_id, exc_info=True)

            # Apply this single result and commit immediately so another item's
            # failure can't roll back what was already applied. On failure,
            # increment_webhook_retry also clears the claim so the next cycle can
            # re-claim and retry.
            with short_conn(conn, db_path) as apply_conn:
                if sent:
                    delete_pending_webhook(apply_conn, webhook_id)
                    logger.info("Retry succeeded for webhook id=%d", webhook_id)
                else:
                    increment_webhook_retry(apply_conn, webhook_id)
                    logger.warning("Retry failed for webhook id=%d", webhook_id)
                apply_conn.commit()

    await asyncio.gather(*(_process(webhook) for webhook in pending))
