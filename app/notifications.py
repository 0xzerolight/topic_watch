"""Notification delivery via Apprise.

Thin wrapper around the Apprise library. All notification URLs
come from the application settings (Apprise URL format).
"""

import asyncio
import logging
from urllib.parse import urlparse, urlunparse

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
from app.url_validation import validate_outbound_url

logger = logging.getLogger(__name__)

# Apprise's own plugin constructors log rejected URL components — including the
# raw offending value — at WARNING/ERROR through the "apprise" logger, and that
# happens INSIDE ap.add()/ap.notify(), before this module gets a chance to
# report anything through redact_url(). Even with Apprise's own CWE-312
# masking (on by default), a scheme where the secret sits in the URL's
# authority rather than its userinfo — ``pover://user@APPTOKEN``,
# ``tgram://BADTOKEN/chat_id`` — still reaches that log verbatim (AUG-268).
# Every failure mode this module cares about is already reported through its
# own logger with redaction (see _deliver_one below), so Apprise's internal
# diagnostics are silenced entirely rather than filtered message-by-message —
# a filter would have to track every plugin's message shape and go stale.
logging.getLogger("apprise").propagate = False
logging.getLogger("apprise").addHandler(logging.NullHandler())

# Payload bounds, in characters, applied when the message is built (AUG-323).
# Apprise plugins carry per-service title/body limits and upstream's default
# overflow policy is to hand the oversized payload to the service anyway, so the
# ceiling has to be ours. The values sit under the tightest limits among the
# services the README recommends (Discord's 2000-character message body is the
# binding one; the rest allow more), and are constants rather than settings: a
# transport bound is not a preference the user should have to discover.
NOTIFICATION_TITLE_CHAR_LIMIT = 200
NOTIFICATION_BODY_CHAR_LIMIT = 1800
_TRUNCATION_SUFFIX = "..."
_OMISSION_LINE = "[trimmed to fit the notification channel]"

# Apprise schemes that are a *generic* HTTP request rather than a named service:
# the URL alone picks the host, method, headers and payload, which is the same
# capability ``notifications.webhook_urls`` offers — and that one has always run
# through the SSRF gate. Left ungated, ``json://127.0.0.1:9000/`` was a way to
# aim an attacker-shaped POST at an internal service (AUG-004).
#
# Deliberately scheme-based, not host-based: many Apprise plugins repurpose the
# URL's host field as a token (``discord://<webhook_id>/<token>`` parses to
# ``host="<webhook_id>"``), so resolving every plugin's ``.host`` would fail
# closed on services that have no host at all. Named self-hosted services
# (ntfy, Gotify, Matrix, SMTP) keep working on a LAN address unchanged.
_GENERIC_HTTP_SCHEMES: dict[str, str] = {
    "json": "http",
    "jsons": "https",
    "form": "http",
    "forms": "https",
    "xml": "http",
    "xmls": "https",
}


def _generic_http_target(url: str) -> str | None:
    """The plain http(s) URL a generic Apprise notifier would POST to, or ``None``."""
    try:
        parsed = urlparse(url)
        scheme = _GENERIC_HTTP_SCHEMES.get(parsed.scheme.lower())
        if scheme is None:
            return None
        return urlunparse(parsed._replace(scheme=scheme))
    except Exception:
        # An unparseable URL is not a generic-HTTP target we can validate; Apprise
        # rejects it at add() time and it is reported as an invalid URL there.
        return None


# Literal placeholder tokens that appear ONLY in documentation/example URLs
# (config.example.yml, README, the setup UI). A real notification URL carries
# concrete credentials and never these words, so an unedited example is dropped
# instead of silently delivering — e.g. the shipped ``ntfy://your-topic-name``
# would otherwise POST to the public ntfy.sh topic "your-topic-name".
_PLACEHOLDER_URL_MARKERS = frozenset(
    {
        "your-topic-name",
        "your-topic",
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
    }
)


def _url_components(url: str) -> list[str]:
    """Split an Apprise URL into the components a placeholder can occupy.

    Userinfo, host and each path segment — the places the shipped examples put
    their fake credentials. Query values are deliberately excluded: Apprise
    query parameters are options, not identity.
    """
    parsed = urlparse(url)
    components: list[str] = []
    userinfo, _, hostport = (parsed.netloc or "").rpartition("@")
    if userinfo:
        components.extend(userinfo.split(":"))
    host = hostport.rsplit(":", 1)[0] if hostport.count(":") == 1 else hostport
    if host:
        components.append(host)
    components.extend(segment for segment in (parsed.path or "").split("/") if segment)
    return components


def _is_placeholder_url(url: str) -> bool:
    """True if ``url`` is an unedited documentation/example placeholder.

    A marker has to be a WHOLE component of the URL, not a substring of one
    (AUG-245): ``ntfy://api_token-alerts`` is a perfectly good ntfy topic that
    substring matching refused forever, retrying until the target was abandoned.
    Parsing failures fall back to "not a placeholder" — the invalid-URL guard in
    ``_deliver_one`` catches anything Apprise itself cannot use.
    """
    try:
        components = _url_components(url)
    except ValueError:
        return False
    return any(component.lower() in _PLACEHOLDER_URL_MARKERS for component in components)


def _fit(text: str, limit: int) -> str:
    """Trim ``text`` to ``limit`` characters, marking the cut."""
    if len(text) <= limit:
        return text
    if limit <= len(_TRUNCATION_SUFFIX):
        return text[:limit]
    return text[: limit - len(_TRUNCATION_SUFFIX)] + _TRUNCATION_SUFFIX


def format_notification(topic_name: str, novelty_result: NoveltyResult) -> tuple[str, str]:
    """Format a NoveltyResult into a bounded notification title and body.

    The LLM contract puts no length bound on the summary, the fact list or the
    source list, so a verbose-but-valid result could exceed a channel's title or
    body limit — and the channel's own answer to that is either a rejection (which
    the retry drain then replays unchanged until the alert is abandoned) or a
    prefix truncation that eats the score footer first (AUG-323).

    The projection is therefore built to a budget here, once, before the intent
    is persisted, so every attempt sends the same accepted shape:

    * the score footer is reserved first — it is the part prefix truncation loses
      and the part that makes the alert readable at a glance;
    * the summary is trimmed with an explicit marker if it alone overruns;
    * facts and source URLs are added whole while they fit. A URL is never cut in
      half: it is dropped instead, since a truncated link is worse than a missing
      one;
    * anything dropped is announced by a marker line, so a short message is never
      mistaken for the whole story.

    Args:
        topic_name: The name of the topic.
        novelty_result: A NoveltyResult with has_new_info=True.

    Returns:
        Tuple of (title, body) strings, each within its channel-limit constant.
    """
    title = _fit(f"Topic Watch: {topic_name}", NOTIFICATION_TITLE_CHAR_LIMIT)

    confidence_pct = int(novelty_result.confidence * 100)
    relevance_pct = int(novelty_result.relevance * 100)
    footer = [
        "",
        f"Confidence: {confidence_pct}%",
        f"Relevance: {relevance_pct}%",
        f"Importance: {novelty_result.importance}/5",
    ]
    # Reserve the footer and the omission marker (each plus the newline that
    # joins it to what precedes it) so adding either can never push the finished
    # body past the limit.
    reserved = len("\n".join(footer)) + 1 + len(_OMISSION_LINE) + 2
    budget = max(NOTIFICATION_BODY_CHAR_LIMIT - reserved, 0)

    parts: list[str] = []
    used = 0
    omitted = False

    if novelty_result.summary:
        summary = _fit(novelty_result.summary, budget)
        omitted = summary != novelty_result.summary
        parts.append(summary)
        used += len(summary)

    def _add_block(header: str, lines: list[str]) -> None:
        nonlocal used, omitted
        if not lines:
            return
        cost = len(header) + 2  # the blank separator line plus the header
        kept: list[str] = []
        for line in lines:
            if used + cost + len(line) + 1 > budget:
                omitted = True
                continue
            kept.append(line)
            cost += len(line) + 1
        if not kept:
            return
        parts.extend(["", header, *kept])
        used += cost

    _add_block("Key facts:", [f"  - {fact}" for fact in novelty_result.key_facts])
    _add_block("Sources:", [f"  {url}" for url in novelty_result.source_urls])

    if omitted:
        parts.extend(["", _OMISSION_LINE])
    parts.extend(footer)

    return title, "\n".join(parts)


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

    target = _generic_http_target(url)
    if target is not None:
        try:
            validate_outbound_url(target, purpose="Notification target")
        except ValueError as exc:
            logger.warning("Blocked notification to %s: %s", redact_url(url), exc)
            return NotificationDelivery(url=url, ok=False, error="blocked notification target")

    try:
        # add() parses the URL, so it RAISES on inputs Apprise cannot make sense
        # of — an ordinary password containing '[' reads as a broken IPv6 literal.
        # It belongs inside the guard: the delivery state machine leans on the
        # "Never raises" contract, and an escaping exception leaves the intent
        # claimed with no outcome ever recorded (TW-AUD-004).
        ap = apprise.Apprise()
        if not ap.add(url):
            # OVH-027: a typo'd/unsupported URL is dropped by Apprise at add() time.
            # Surface it instead of silently succeeding on the other channels.
            logger.warning("Skipping invalid notification URL: %s", redact_url(url))
            return NotificationDelivery(url=url, ok=False, error="invalid notification URL")

        ok = bool(ap.notify(title=title, body=body))
        if ok:
            logger.info("Notification sent to %s: %s", redact_url(url), title)
            return NotificationDelivery(url=url, ok=True)
        logger.warning("Notification delivery failed for %s: %s", redact_url(url), title)
        return NotificationDelivery(url=url, ok=False, error="delivery failed")
    except Exception as exc:
        logger.warning("Notification error for %s: %s", redact_url(url), title, exc_info=True)
        return NotificationDelivery(url=url, ok=False, error=str(exc))


async def send_single_notification(
    title: str,
    body: str,
    url: str,
    timeout_s: float,
) -> NotificationDelivery:
    """Deliver to ONE target under its OWN deadline. Never raises.

    One deadline per target is the whole point (AUG-071). A single ``wait_for``
    around a worker that sent every URL in sequence meant one stalled channel
    fabricated a failure for the channels that had already delivered, and the
    pipeline then queued retries for messages the user had received.

    ``wait_for`` bounds the awaiting coroutine, not the thread: a thread cannot be
    cancelled, so an expired send keeps running until Apprise's socket I/O
    returns. That is why a timeout is reported as ``timed_out`` rather than
    ``ok=False`` — the outcome is genuinely unknown, and the caller must leave the
    delivery intent claimed rather than schedule a retry that could duplicate it.
    """
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_deliver_one, title, body, url),
            timeout=timeout_s,
        )
    except TimeoutError:
        logger.warning(
            "Notification to %s timed out after %ss (outcome unknown): %s",
            redact_url(url),
            timeout_s,
            title,
        )
        return NotificationDelivery(url=url, ok=False, error="timed out", timed_out=True)


async def send_notification_per_url(
    title: str,
    body: str,
    settings: Settings,
    *,
    url: str | None = None,
) -> list[NotificationDelivery]:
    """Deliver a notification per-URL, returning one result per target.

    Each URL gets its own Apprise instance, its own deadline and its own outcome,
    so a partial failure (one channel down, one channel hung) is attributable and
    re-queueable on its own rather than collapsing the whole batch (OVH-039,
    AUG-071). Invalid URLs are reported as failed deliveries rather than silently
    dropped (OVH-027).

    Args:
        title: Notification title.
        body: Notification body.
        settings: Application settings (provides the configured URLs).
        url: When given, deliver to only this single URL (the retry-drain path,
            where each pending row carries one already-failed target). When
            None, deliver to every configured URL.

    Returns:
        One NotificationDelivery per attempted URL (empty if none configured).
        Never raises.
    """
    urls = [url] if url is not None else list(settings.notifications.urls)
    if not urls:
        logger.debug("No notification URLs configured, skipping notification")
        return []

    return list(
        await asyncio.gather(
            *(send_single_notification(title, body, u, settings.apprise_timeout_seconds) for u in urls)
        )
    )


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
