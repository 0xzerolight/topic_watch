"""Notification delivery via Apprise.

Thin wrapper around the Apprise library. All notification URLs
come from the application settings (Apprise URL format).
"""

import asyncio
import logging

import apprise

from app.analysis.llm import NoveltyResult
from app.config import Settings

# Single canonical URL-redaction helper (fold-in): notification logging uses the
# most-complete app.log_redaction.redact_url, which strips userinfo/query and also
# drops long (likely-secret) path segments — strictly stronger than the old
# scheme+host form against secret leakage. Re-exported so existing
# ``from app.notifications import redact_url`` call sites keep working.
from app.log_redaction import redact_url as redact_url
from app.models import NotificationDelivery

logger = logging.getLogger(__name__)

# Literal placeholder tokens that appear ONLY in documentation/example URLs
# (config.example.yml, README, the setup UI). A real notification URL carries
# concrete credentials and never these words, so an unedited example is dropped
# instead of silently delivering — e.g. the shipped ``ntfy://your-topic-name``
# would otherwise POST to the public ntfy.sh topic "your-topic-name". Kept
# deliberately narrow (whole placeholder tokens, case-insensitive) so it never
# drops a real URL (OVH: example-URL leak guard).
_PLACEHOLDER_URL_MARKERS = (
    "your-topic-name",
    "your_ntfy_topic",
    "webhook_id",
    "webhook_token",
    "bot_token",
    "chat_id",
    "token_a",
    "token_b",
    "token_c",
    "user_key",
    "api_token",
    "your-api-key",
)


def _is_placeholder_url(url: str) -> bool:
    """True if ``url`` is an unedited documentation/example placeholder."""
    lowered = url.lower()
    return any(marker in lowered for marker in _PLACEHOLDER_URL_MARKERS)


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
    relevance_pct = int(novelty_result.relevance * 100)
    parts.append("")
    parts.append(f"Confidence: {confidence_pct}%")
    parts.append(f"Relevance: {relevance_pct}%")
    parts.append(f"Importance: {novelty_result.importance}/5")

    body = "\n".join(parts)
    return title, body


def _deliver_one(title: str, body: str, url: str) -> NotificationDelivery:
    """Deliver to a single notification URL with its own Apprise instance.

    One instance per URL means a failure (down channel, invalid URL) is
    attributable to that URL alone and can be re-queued individually, instead
    of collapsing the whole batch to one bool (OVH-027/OVH-039). Never raises.
    """
    if _is_placeholder_url(url):
        # An unedited example URL (e.g. the shipped ntfy://your-topic-name) would
        # deliver to a real public target. Drop it rather than leak (OVH guard).
        logger.warning("Skipping placeholder/example notification URL: %s", redact_url(url))
        return NotificationDelivery(url=url, ok=False, error="placeholder notification URL")

    ap = apprise.Apprise()
    if not ap.add(url):
        # OVH-027: a typo'd/unsupported URL is dropped by Apprise at add() time.
        # Surface it instead of silently succeeding on the other channels.
        logger.warning("Skipping invalid notification URL: %s", redact_url(url))
        return NotificationDelivery(url=url, ok=False, error="invalid notification URL")

    try:
        ok = bool(ap.notify(title=title, body=body))
        if ok:
            logger.info("Notification sent to %s: %s", redact_url(url), title)
            return NotificationDelivery(url=url, ok=True)
        logger.warning("Notification delivery failed for %s: %s", redact_url(url), title)
        return NotificationDelivery(url=url, ok=False, error="delivery failed")
    except Exception as exc:
        logger.warning("Notification error for %s: %s", redact_url(url), title, exc_info=True)
        return NotificationDelivery(url=url, ok=False, error=str(exc))


def _deliver_per_url_sync(title: str, body: str, urls: list[str]) -> list[NotificationDelivery]:
    """Deliver to each URL independently (blocks on I/O).

    Use send_notification_per_url() for the async wrapper.
    """
    return [_deliver_one(title, body, url) for url in urls]


async def send_notification_per_url(
    title: str,
    body: str,
    settings: Settings,
    *,
    url: str | None = None,
) -> list[NotificationDelivery]:
    """Deliver a notification per-URL, returning one result per target.

    Each URL gets its own Apprise instance and a per-URL outcome, so a partial
    failure (one channel down) is attributable and re-queueable on its own
    rather than re-sending the whole batch (OVH-039). Invalid URLs are reported
    as failed deliveries rather than silently dropped (OVH-027).

    Args:
        title: Notification title.
        body: Notification body.
        settings: Application settings (provides the configured URLs).
        url: When given, deliver to only this single URL (the retry-drain path,
            where each pending row carries one already-failed target). When
            None, deliver to every configured URL.

    Returns:
        One NotificationDelivery per attempted URL (empty if none configured).
        Never raises — a timeout yields a single failed delivery per target.

    Timeout semantics (OVH-116): ``wait_for`` bounds only the *awaiting coroutine*,
    so on timeout the scheduler is freed and never blocked on a hung send. It does
    NOT cancel the underlying ``to_thread`` worker — a thread cannot be cancelled —
    so ``_deliver_per_url_sync`` keeps running until Apprise's socket I/O returns,
    and that executor slot is reclaimed only then, not at the timeout. At
    single-user scale (default ~32-slot pool) slot pressure is implausible, but a
    hung send does occupy a worker past its deadline.
    """
    urls = [url] if url is not None else list(settings.notifications.urls)
    if not urls:
        logger.debug("No notification URLs configured, skipping notification")
        return []

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_deliver_per_url_sync, title, body, urls),
            timeout=settings.apprise_timeout_seconds,
        )
    except TimeoutError:
        logger.warning(
            "Notification timed out after %ss: %s",
            settings.apprise_timeout_seconds,
            title,
        )
        return [NotificationDelivery(url=u, ok=False, error="timed out") for u in urls]


async def send_notification(title: str, body: str, settings: Settings, *, url: str | None = None) -> bool:
    """Send a notification without blocking the async event loop.

    Delivers per-URL (see send_notification_per_url) and collapses to a single
    bool for the boolean callers/tests: True only if every attempted target
    delivered. A partial failure returns False so the caller can re-queue, but
    callers needing per-URL granularity (to re-queue only the failed targets)
    should use send_notification_per_url directly. Never raises.

    Args:
        url: When given, send to only this single URL (retry-drain per-row path).
    """
    results = await send_notification_per_url(title, body, settings, url=url)
    if not results:
        return False
    return all(r.ok for r in results)
