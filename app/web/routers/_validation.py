"""Shared form validation for topic and settings handlers."""

import asyncio

from pydantic import ValidationError

from app.models import NOVELTY_INSTRUCTION_MAX_CHARS, FeedMode
from app.url_validation import validate_feed_urls


def normalize_base_url(raw: str) -> str | None:
    """Normalize a submitted LLM base URL (OVH-153): blank -> None, else trimmed.

    An explicitly-set base_url is honored for every provider (OVH-104 reversal),
    so this no longer drops it for cloud providers — it only collapses blank
    input to ``None`` so the setup pre-flight check and the persisted model agree.
    """
    return raw.strip() or None


def format_validation_errors(exc: ValidationError) -> list[str]:
    """Render a Pydantic ``ValidationError`` into user-facing field messages.

    Shared by the setup and settings handlers (OVH-153) so both surface the same
    ``<field path>: <message>`` form on a 422 re-render.
    """
    return [f"{' → '.join(str(loc) for loc in e['loc'])}: {e['msg']}" for e in exc.errors()]


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

    # Strict three-way map: an unknown mode is an error, not a silent MANUAL fallback
    # (which would build a never-fetching EXA-as-MANUAL topic with empty feed_urls).
    mode_map = {"auto": FeedMode.AUTO, "manual": FeedMode.MANUAL, "exa": FeedMode.EXA}
    mode = mode_map.get(feed_mode, FeedMode.AUTO)

    urls: list[str] = []
    errors: list[str] = []
    if feed_mode not in mode_map:
        errors.append(f"Invalid feed mode: {feed_mode!r}")
    # AUTO and EXA carry no manual feed URLs; only MANUAL validates them.
    if mode == FeedMode.MANUAL:
        urls = [u.strip() for u in feed_urls.strip().splitlines() if u.strip()]
        if urls:
            errors.extend(await asyncio.to_thread(validate_feed_urls, urls))
        else:
            # An empty MANUAL list used to be accepted and persisted, producing a
            # topic with no source at all that failed its first initialization
            # with a source-unavailable error. Fail here instead (AUG-097).
            errors.append("Manual mode needs at least one feed URL.")

    parsed_interval: int | None = None
    if check_interval.strip():
        try:
            parsed_interval = parse_interval(check_interval)
        except ValueError as e:
            errors.append(str(e))

    return mode, urls, parsed_interval, errors


def parse_topic_name(value: str, errors: list[str]) -> str:
    """Trim a submitted topic name and reject a blank one.

    The browser's ``required`` attribute and the UNIQUE constraint both accept
    whitespace-only input, and ``""`` and ``" "`` are distinct to SQLite — so the
    form could create several visually unnamed topics, each with a meaningless
    source query (AUG-150). Trimming is part of the fix: the stored name is the
    one shown, so surrounding whitespace must not survive into identity.
    """
    text = value.strip()
    if not text:
        errors.append("Topic name is required.")
    return text


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


def parse_importance(value: str, errors: list[str]) -> int | None:
    """Parse the optional per-topic importance notify threshold.

    Blank input returns ``None`` (no suppression — notify on any importance).
    Non-integer or out-of-range input appends a message to ``errors`` and
    returns ``None``. Single fixed label by design, unlike ``parse_threshold``
    which serves two fields.
    """
    text = value.strip()
    if not text:
        return None
    try:
        parsed = int(text)
    except ValueError:
        errors.append("Importance threshold must be a whole number between 1 and 5")
        return None
    if not 1 <= parsed <= 5:
        errors.append("Importance threshold must be between 1 and 5")
        return None
    return parsed


def parse_novelty_instruction(value: str, errors: list[str]) -> str | None:
    """Parse the optional per-topic novelty instruction.

    Blank input returns ``None`` (no topic-specific criteria). Input longer than
    ``NOVELTY_INSTRUCTION_MAX_CHARS`` appends a message to ``errors`` and returns
    ``None`` — this form boundary is the cap's enforcement point (the template
    ``maxlength`` is only a client-side hint).
    """
    text = value.strip()
    if not text:
        return None
    if len(text) > NOVELTY_INSTRUCTION_MAX_CHARS:
        errors.append(f"Novelty instruction must be at most {NOVELTY_INSTRUCTION_MAX_CHARS} characters")
        return None
    return text
