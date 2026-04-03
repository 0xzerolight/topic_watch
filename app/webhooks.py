"""Custom webhook delivery for Topic Watch.

Sends structured JSON payloads to arbitrary HTTP endpoints when
new information is found, complementing the Apprise notifications.
"""

import asyncio
import logging
from datetime import UTC, datetime

import httpx

from app.analysis.llm import NoveltyResult
from app.config import Settings

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
        "timestamp": datetime.now(UTC).isoformat(),
    }


async def send_webhook(url: str, payload: dict, timeout: float = _WEBHOOK_TIMEOUT) -> bool:
    """POST a JSON payload to a webhook URL.

    Returns True on success (2xx response), False on failure.
    Never raises — all errors are caught and logged.
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
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


async def send_webhooks(topic_name: str, novelty_result: NoveltyResult, settings: Settings) -> int:
    """Send webhook notifications to all configured webhook URLs.

    Args:
        topic_name: The topic name.
        novelty_result: The novelty analysis result.
        settings: Application settings.

    Returns:
        Number of successfully delivered webhooks.
    """
    webhook_urls = settings.notifications.webhook_urls
    if not webhook_urls:
        return 0

    payload = _build_webhook_payload(topic_name, novelty_result)

    tasks = [send_webhook(url, payload) for url in webhook_urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    success_count = sum(1 for r in results if isinstance(r, bool) and r)

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
