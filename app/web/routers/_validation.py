"""Shared form validation for topic create/edit handlers."""

from app.models import FeedMode
from app.url_validation import validate_feed_urls


def validate_topic_form(
    feed_mode: str, feed_urls: str, check_interval: str
) -> tuple[FeedMode, list[str], int | None, list[str]]:
    """Validate and parse the shared topic-form fields.

    Returns ``(mode, urls, parsed_interval, errors)`` where ``urls`` is the
    parsed feed-URL list (empty for AUTO mode) and ``errors`` aggregates feed
    and interval validation messages. Mirrors the logic previously duplicated
    in the create and edit handlers.
    """
    from app.interval import parse_interval

    mode = FeedMode.AUTO if feed_mode == "auto" else FeedMode.MANUAL

    urls: list[str] = []
    errors: list[str] = []
    if mode == FeedMode.MANUAL:
        urls = [u.strip() for u in feed_urls.strip().splitlines() if u.strip()]
        errors = validate_feed_urls(urls)

    parsed_interval: int | None = None
    if check_interval.strip():
        try:
            parsed_interval = parse_interval(check_interval)
        except ValueError as e:
            errors.append(str(e))

    return mode, urls, parsed_interval, errors
