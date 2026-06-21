"""Shared form validation for topic create/edit handlers."""

import asyncio

from app.models import FeedMode
from app.url_validation import validate_feed_urls


async def validate_topic_form(
    feed_mode: str, feed_urls: str, check_interval: str
) -> tuple[FeedMode, list[str], int | None, list[str]]:
    """Validate and parse the shared topic-form fields.

    Returns ``(mode, urls, parsed_interval, errors)`` where ``urls`` is the
    parsed feed-URL list (empty for AUTO mode) and ``errors`` aggregates feed
    and interval validation messages. Mirrors the logic previously duplicated
    in the create and edit handlers.

    Feed-URL validation resolves DNS (blocking ``getaddrinfo``); it is offloaded
    via ``asyncio.to_thread`` so it never stalls the single-worker event loop
    that the scheduler tick shares (OVH-054). Cheap parsing/interval checks stay
    inline. AUTO mode has no manual feeds, so no thread is scheduled.
    """
    from app.interval import parse_interval

    mode = FeedMode.AUTO if feed_mode == "auto" else FeedMode.MANUAL

    urls: list[str] = []
    errors: list[str] = []
    if mode == FeedMode.MANUAL:
        urls = [u.strip() for u in feed_urls.strip().splitlines() if u.strip()]
        if urls:
            errors = await asyncio.to_thread(validate_feed_urls, urls)

    parsed_interval: int | None = None
    if check_interval.strip():
        try:
            parsed_interval = parse_interval(check_interval)
        except ValueError as e:
            errors.append(str(e))

    return mode, urls, parsed_interval, errors


def parse_threshold(value: str, label: str, errors: list[str]) -> float | None:
    """Parse an optional 0.0-1.0 threshold field.

    Blank input returns ``None`` (inherit the global threshold). Non-numeric or
    out-of-range input appends a message to ``errors`` and returns ``None``.
    """
    text = value.strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        errors.append(f"{label} must be a number between 0.0 and 1.0")
        return None
    if not 0.0 <= parsed <= 1.0:
        errors.append(f"{label} must be between 0.0 and 1.0")
        return None
    return parsed
