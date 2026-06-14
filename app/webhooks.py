"""Custom webhook delivery for Topic Watch.

Sends structured JSON payloads to arbitrary HTTP endpoints when
new information is found, complementing the Apprise notifications.
"""

import asyncio
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import httpx

from app.analysis.llm import NoveltyResult
from app.config import Settings
from app.crud import (
    create_pending_webhook,
    delete_expired_webhooks,
    delete_pending_webhook,
    increment_webhook_retry,
    list_pending_webhooks,
)
from app.database import short_conn
from app.url_validation import is_private_url

logger = logging.getLogger(__name__)

_WEBHOOK_TIMEOUT = 10.0


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
    if await asyncio.to_thread(is_private_url, url):
        logger.warning("Blocked webhook to private/reserved URL: %s", url)
        return False

    try:
        # follow_redirects=False (httpx default, made explicit) so a 3xx to a
        # private address can't bypass the is_private_url check above.
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            logger.info("Webhook delivered to %s (status %d)", url, response.status_code)
            return True
    except httpx.TimeoutException:
        logger.warning("Webhook timeout for %s", url)
        return False
    except httpx.HTTPStatusError as exc:
        logger.warning("Webhook HTTP %d for %s", exc.response.status_code, url)
        return False
    except Exception:
        logger.warning("Webhook error for %s", url, exc_info=True)
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
                logger.warning("Failed to enqueue webhook for retry (url=%s)", url, exc_info=True)

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

    # --- Phase 1: snapshot pending rows under a short-lived connection. ---
    with short_conn(conn, db_path) as snapshot:
        expired = delete_expired_webhooks(snapshot)
        if expired:
            logger.warning("Deleted %d expired pending webhook(s)", expired)
        snapshot.commit()
        pending = list_pending_webhooks(snapshot)

    if not pending:
        return

    logger.info("Retrying %d pending webhook(s)", len(pending))

    # --- Phase 2: send with NO connection held, then apply per item. ---
    for webhook in pending:
        webhook_id = webhook["id"]
        try:
            sent = await send_webhook(webhook["url"], webhook["payload"])
        except Exception:
            sent = False
            logger.warning("Retry error for webhook id=%d", webhook_id, exc_info=True)

        # Apply this single result and commit immediately so a later item's
        # crash can't roll back what was already applied.
        with short_conn(conn, db_path) as apply_conn:
            if sent:
                delete_pending_webhook(apply_conn, webhook_id)
                logger.info("Retry succeeded for webhook id=%d", webhook_id)
            else:
                increment_webhook_retry(apply_conn, webhook_id)
                logger.warning("Retry failed for webhook id=%d", webhook_id)
            apply_conn.commit()
