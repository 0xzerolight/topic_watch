"""Notification delivery via Apprise.

Thin wrapper around the Apprise library. All notification URLs
come from the application settings (Apprise URL format).
"""

import asyncio
import logging

import apprise

from app.analysis.llm import NoveltyResult
from app.config import Settings

logger = logging.getLogger(__name__)


def format_notification(topic_name: str, novelty_result: NoveltyResult) -> tuple[str, str]:
    """Format a NoveltyResult into a notification title and body.

    Args:
        topic_name: The name of the topic.
        novelty_result: A NoveltyResult with has_new_info=True.

    Returns:
        Tuple of (title, body) strings.
    """
    title = f"Topic Watch: {topic_name}"

    parts: list[str] = []
    if novelty_result.summary:
        parts.append(novelty_result.summary)

    if novelty_result.key_facts:
        parts.append("")
        parts.append("Key facts:")
        for fact in novelty_result.key_facts:
            parts.append(f"  - {fact}")

    if novelty_result.source_urls:
        parts.append("")
        parts.append("Sources:")
        for url in novelty_result.source_urls:
            parts.append(f"  {url}")

    confidence_pct = int(novelty_result.confidence * 100)
    parts.append("")
    parts.append(f"Confidence: {confidence_pct}%")

    body = "\n".join(parts)
    return title, body


def _send_notification_sync(title: str, body: str, settings: Settings) -> bool:
    """Send a notification synchronously (blocks on I/O).

    Use send_notification() for the async wrapper.
    """
    urls = settings.notifications.urls
    if not urls:
        logger.debug("No notification URLs configured, skipping notification")
        return False

    ap = apprise.Apprise()
    for url in urls:
        ap.add(url)

    try:
        result = ap.notify(title=title, body=body)
        if result:
            logger.info("Notification sent: %s", title)
        else:
            logger.warning("Notification delivery failed for: %s", title)
        return bool(result)
    except Exception:
        logger.warning("Notification error for: %s", title, exc_info=True)
        return False


async def send_notification(title: str, body: str, settings: Settings) -> bool:
    """Send a notification without blocking the async event loop.

    Wraps the synchronous Apprise call in a thread executor.
    """
    return await asyncio.to_thread(_send_notification_sync, title, body, settings)
