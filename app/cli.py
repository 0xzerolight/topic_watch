"""CLI entrypoint for Topic Watch.

Provides manual access to the check pipeline, topic initialization,
and topic listing. Run as: python -m app.cli <command>

Concurrency constraint (OVH-097): the per-topic and whole-cycle in-flight
guards (``app.web.state._checking_state``) are process-local. A CLI invocation
gets its own fresh, empty guard state, so ``check``/``check-all``/``init`` do
NOT coordinate with a running server's scheduler or UI. Running them against the
SAME database as a live server can double-check a topic, double-spend the LLM,
and emit duplicate notifications — directly against the novelty-only promise.
Run CLI commands only when the server (and its scheduler) is stopped, or against
a separate/offline database.
"""

import argparse
import asyncio
import logging
import os
import platform
import re
import sqlite3
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

from app.config import Settings, is_api_key_env_sourced, is_exa_key_env_sourced, load_settings, resolve_db_path
from app.crud import (
    get_knowledge_state,
    get_topic,
    get_topic_by_name,
    list_all_feed_health,
    list_topics,
)
from app.database import get_db, get_schema_version, init_db
from app.log_redaction import redact_url
from app.logging_config import setup_logging
from app.models import CheckResult, TopicStatus, is_internal_failure, is_source_failure

logger = logging.getLogger(__name__)

# Width of the Name column in `list`.
_NAME_COLUMN = 30

# Bound on the per-item detail lists `doctor` and `check-all` print, so one
# imported OPML file with hundreds of dead feeds can no longer flood a terminal
# or an issue form and bury the summary above it (AUG-332).
_MAX_LISTED_ITEMS = 10

# Final per-line cap for a redacted feed URL in the `doctor` report.
_FEED_URL_WIDTH = 100

# Concurrency restriction, shown in `--help` for the whole CLI and for each
# command that runs the pipeline. It used to live only in this module's
# docstring, which no user ever sees (AUG-077).
CONCURRENCY_WARNING = (
    "Run 'check', 'check-all' and 'init' only while the server (and its scheduler) "
    "is stopped, or against a separate database. The in-flight guards are "
    "process-local, so a CLI run against a live server's database can double-check "
    "a topic, double-spend the LLM and send duplicate notifications."
)

# Characters a terminal acts on rather than draws: C0/C1 controls (line breaks,
# ESC), the bidirectional formatting set, zero-width marks and the BOM. A topic
# name arrives from an OPML file a third party wrote (app/opml.py keeps whatever
# the attribute contained), so any of these can forge a row, reverse the visible
# order of a line, or hide text outright (AUG-330).
_TERMINAL_UNSAFE = re.compile(
    "["
    "\x00-\x1f\x7f-\x9f"  # C0 and C1 controls: line breaks, ESC, CSI
    "؜"  # Arabic letter mark
    "​-‏"  # zero-width space/joiners, LRM, RLM
    "  "  # line and paragraph separators
    "‪-‮"  # bidi embeddings and overrides
    "⁠-⁤"  # word joiner and invisible operators
    "⁦-⁩"  # bidi isolates
    "﻿"  # BOM / zero-width no-break space
    "]"
)


def display_width(text: str) -> int:
    """Columns ``text`` occupies in a monospaced terminal.

    East Asian wide and fullwidth characters take two cells; combining marks
    take none. Without this a CJK name silently doubles its column and shears
    the table.
    """
    width = 0
    for char in text:
        if unicodedata.combining(char):
            continue
        width += 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
    return width


def terminal_safe(text: str, *, width: int | None) -> str:
    """Render untrusted text as a single, bounded, literally-displayed line.

    Control and directional-formatting characters become visible ``\\uXXXX``
    escapes; ordinary Unicode (accents, CJK, emoji) is left alone. When ``width``
    is given the result is truncated to that many terminal columns.
    """
    cleaned = _TERMINAL_UNSAFE.sub(lambda match: f"\\u{ord(match.group()):04x}", text)
    if width is None or display_width(cleaned) <= width:
        return cleaned

    kept: list[str] = []
    used = 0
    for char in cleaned:
        char_width = (
            0 if unicodedata.combining(char) else (2 if unicodedata.east_asian_width(char) in ("W", "F") else 1)
        )
        if used + char_width > width - 1:
            break
        kept.append(char)
        used += char_width
    return "".join(kept) + "…"


def _open_database(settings: Settings) -> Path:
    """Resolve the CONFIGURED database, create/migrate it, and return its path.

    Every command threads this path into ``get_db`` and into the pipeline.
    Letting them fall through to ``db_path=None`` meant a supported non-default
    ``db_path`` was honored by the server but ignored by the CLI, which then
    created and operated on an empty default database (TW-AUD-027).
    """
    db_path = resolve_db_path(settings)
    init_db(db_path)
    return db_path


def summarize_check(result: CheckResult) -> tuple[str, bool]:
    """Render one finished check as ``(summary, failed)``.

    ``has_new_info`` alone cannot tell a clean quiet check from a check that
    never saw its sources, failed inside the pipeline, could not be analyzed, or
    whose alert was never delivered — every one of those leaves ``has_new_info``
    false. Printing them all as "no change" and exiting 0 let a cron job report a
    broken monitor as healthy (AUG-075). ``failed`` drives a non-zero exit.
    """
    label = "NEW INFO" if result.has_new_info else "no change"
    problems: list[str] = []

    stage = result.stage_error
    if stage:
        if stage.startswith("skipped:"):
            # Nothing ran at all, so there is no outcome to qualify.
            return stage, True
        if is_source_failure(stage):
            label = "SOURCE FAILURE"
        elif is_internal_failure(stage):
            label = "PIPELINE FAILURE"
        problems.append(stage)
    if result.notification_error:
        problems.append(f"delivery failed: {result.notification_error}")

    if problems:
        return f"{label} — {'; '.join(problems)}", True
    return label, False


async def _cmd_check(topic_name: str) -> None:
    """Check a single topic for new information."""
    from app.checker import check_topic

    settings = load_settings()
    db_path = _open_database(settings)

    with get_db(db_path) as conn:
        topic = get_topic_by_name(conn, topic_name)
    if topic is None:
        logger.error("Topic not found: '%s'", topic_name)
        sys.exit(1)

    # No connection is held across the pipeline: check_topic opens its own per
    # phase (AUG-136).
    result = await check_topic(topic, settings, db_path=db_path)
    summary, failed = summarize_check(result)

    print(f"Check complete for '{terminal_safe(topic_name, width=None)}':")
    print(f"  Articles found: {result.articles_found}")
    print(f"  New articles: {result.articles_new}")
    print(f"  New info: {result.has_new_info}")
    print(f"  Notification sent: {result.notification_sent}")
    print(f"  Outcome: {summary}")
    if failed:
        sys.exit(1)


async def _cmd_check_all() -> None:
    """Check every active, ready topic whose interval has elapsed."""
    from app.checker import check_all_topics

    settings = load_settings()
    db_path = _open_database(settings)

    results = await check_all_topics(settings, db_path)

    with get_db(db_path) as conn:
        active_ready = [t for t in list_topics(conn, is_active=True) if t.status == TopicStatus.READY]

    checked_ids = {r.topic_id for r in results}
    names = {t.id: t.name for t in active_ready}
    # The command checks DUE topics, never every active one. Saying so — and
    # naming the topics it left alone — is the difference between a manual
    # refresh the user can trust and a silently partial run (AUG-076).
    skipped = [t for t in active_ready if t.id not in checked_ids]

    print(f"Check cycle complete: {len(results)} due topic(s) checked")
    failed = 0
    for r in results:
        summary, is_failed = summarize_check(r)
        failed += is_failed
        name = names.get(r.topic_id) or f"id={r.topic_id}"
        print(f"  {terminal_safe(name, width=_NAME_COLUMN)}: {summary}")

    if skipped:
        print(f"Not checked: {len(skipped)} active topic(s) not due yet")
        for topic in skipped[:_MAX_LISTED_ITEMS]:
            print(f"  {terminal_safe(topic.name, width=_NAME_COLUMN)}")
        if len(skipped) > _MAX_LISTED_ITEMS:
            print(f"  ... and {len(skipped) - _MAX_LISTED_ITEMS} more")

    if failed:
        print(f"{failed} of {len(results)} check(s) failed")
        sys.exit(1)


async def _cmd_init(topic_name: str) -> None:
    """Run initial knowledge research for a topic.

    Delegates to ``checker.initialize_new_topic`` — the same initializer the web
    layer and the scheduler use. The CLI previously carried its own copy of that
    workflow, which claimed the topic itself, passed the fetch only two of the
    eight settings the canonical path threads through, and re-implemented the
    transition commit. Its only real job is turning the outcome into text and an
    exit code (TW-AUD-027).
    """
    from app.checker import initialize_new_topic

    settings = load_settings()
    db_path = _open_database(settings)

    with get_db(db_path) as conn:
        topic = get_topic_by_name(conn, topic_name)

    if topic is None:
        logger.error("Topic not found: '%s'", topic_name)
        sys.exit(1)
    if topic.status == TopicStatus.RESEARCHING:
        # Another init (scheduler gradual init or a second CLI run) already
        # holds the RESEARCHING claim — cooperate and bail (OVH-018).
        logger.error(
            "Topic '%s' is already being researched; skipping concurrent init.",
            topic_name,
        )
        sys.exit(1)

    topic_id = topic.id
    assert topic_id is not None
    verb = "Re-initializing" if topic.status == TopicStatus.READY else "Initializing"
    print(f"{verb} knowledge for '{terminal_safe(topic_name, width=None)}'...")

    await initialize_new_topic(topic, settings, db_path=db_path)

    with get_db(db_path) as conn:
        final = get_topic(conn, topic_id)
        knowledge = get_knowledge_state(conn, topic_id) if final is not None else None

    if final is None:
        logger.error("Topic '%s' disappeared during initialization", topic_name)
        sys.exit(1)

    if final.status == TopicStatus.READY:
        if knowledge is not None:
            print(f"  Knowledge state built ({knowledge.token_count} tokens)")
        print(f"  Topic '{terminal_safe(topic_name, width=None)}' is now READY")
        return

    if final.status == TopicStatus.NEW:
        # A re-init that found no fresh articles is not a failure: the topic keeps
        # waiting for a later cycle.
        print("  No new articles this run — topic stays NEW and will retry.")
        return

    reason = final.error_message or "unknown error"
    logger.error("Initialization failed for '%s': %s", topic_name, reason)
    print(f"  Initialization failed: {reason}")
    sys.exit(1)


def _in_docker() -> bool:
    """Best-effort container detection for the diagnostic report."""
    return os.path.exists("/.dockerenv")


# Config keys that carry secrets (or are rendered specially); never dumped raw.
_SECRET_CONFIG_KEYS = frozenset({"api_key", "base_url", "urls", "webhook_urls"})


def _url_scheme(url: str) -> str | None:
    """Scheme of a configured URL, or ``None`` when it cannot be parsed.

    Notification settings accept arbitrary strings, and some of them (an
    unmatched IPv6 bracket, say) make ``urlparse`` raise. That raise happened
    outside ``doctor``'s ``load_settings`` handler, so one malformed entry
    defeated the very command meant to diagnose it (AUG-328).
    """
    try:
        return urlparse(url).scheme or "?"
    except ValueError:
        return None


def _render_config(settings: Settings) -> list[str]:
    """Build secret-safe configuration lines for ``doctor``.

    Secrets are never emitted: ``api_key`` is shown as a boolean, ``base_url``
    is redacted to scheme+host, and notification/webhook URLs are reduced to
    per-scheme counts (no host or path). All remaining settings are dumped by
    key via a *denylist* (``_SECRET_CONFIG_KEYS``), so a newly-added setting is
    surfaced automatically rather than silently dropped from bug reports.
    """
    lines: list[str] = []
    llm = settings.llm
    lines.append(f"  llm.model: {llm.model or '(unset)'}")
    key_state = "set" if llm.api_key else "not set"
    if is_api_key_env_sourced():
        key_state += " (from env)"
    lines.append(f"  llm.api_key: {key_state}")
    if llm.base_url:
        lines.append(f"  llm.base_url: {redact_url(llm.base_url)}")

    exa = settings.exa
    lines.append(f"  exa.enabled: {exa.enabled}")
    exa_key_state = "set" if exa.api_key else "not set"
    if is_exa_key_env_sourced():
        exa_key_state += " (from env)"
    lines.append(f"  exa.api_key: {exa_key_state}")
    if exa.base_url:
        lines.append(f"  exa.base_url: {redact_url(exa.base_url)}")

    for label, urls in (
        ("notifications.urls", settings.notifications.urls),
        ("notifications.webhook_urls", settings.notifications.webhook_urls),
    ):
        if urls:
            schemes = [_url_scheme(u) for u in urls]
            counts = Counter(s for s in schemes if s is not None)
            invalid = sum(1 for s in schemes if s is None)
            parts = [f"{scheme} x{n}" for scheme, n in sorted(counts.items())]
            if invalid:
                parts.append(f"{invalid} unparseable")
            lines.append(f"  {label}: {len(urls)} ({', '.join(parts)})")
        else:
            lines.append(f"  {label}: none")

    lines.append(f"  is_configured: {'yes' if settings.is_configured() else 'no'}")

    dumped = settings.model_dump()
    for key, value in dumped.items():
        if key in ("llm", "notifications", "exa"):
            continue  # sub-models rendered above
        if isinstance(value, (str, int, float, bool)):
            lines.append(f"  {key}: {value}")
    # Nested llm.* scalars beyond the three handled above (forward-compatible).
    for key, value in (dumped.get("llm") or {}).items():
        if key in _SECRET_CONFIG_KEYS or key == "model":
            continue
        if isinstance(value, (str, int, float, bool)):
            lines.append(f"  llm.{key}: {value}")
    return lines


def _render_topics(conn: sqlite3.Connection) -> list[str]:
    """Per-status topic counts; degrades to 'unavailable' if the table is absent."""
    try:
        topics = list_topics(conn)
    except sqlite3.Error:
        return ["  topics: unavailable"]
    counts = Counter(t.status.value for t in topics)
    summary = ", ".join(f"{status.value} {counts.get(status.value, 0)}" for status in TopicStatus)
    return [f"  topics: {summary}"]


def _render_feeds(conn: sqlite3.Connection) -> list[str]:
    """Feed-health summary with redacted failing-feed URLs; degrades cleanly."""
    try:
        feeds = list_all_feed_health(conn)
    except sqlite3.Error:
        return ["  feeds: unavailable"]
    failing = [f for f in feeds if f.consecutive_failures > 0]
    ok = len(feeds) - len(failing)
    lines = [f"  feeds: {ok} OK / {len(failing)} failing"]
    # A bounded sample plus an omitted count: an OPML import admits 500 topics at
    # a time and feed-health rows are never pruned, so the unbounded loop this
    # replaces could bury the summary under hundreds of URLs and push far more
    # feed metadata into a pasted bug report than troubleshooting needs (AUG-332).
    for feed in failing[:_MAX_LISTED_ITEMS]:
        url = terminal_safe(redact_url(feed.feed_url), width=_FEED_URL_WIDTH)
        lines.append(f"    failing: {url} (x{feed.consecutive_failures})")
    if len(failing) > _MAX_LISTED_ITEMS:
        lines.append(f"    ... and {len(failing) - _MAX_LISTED_ITEMS} more failing feed(s)")
    return lines


def _render_database(settings: Settings) -> list[str]:
    """Build read-only database diagnostics.

    Opens the DB ``mode=ro`` and never creates or migrates it (no ``get_db`` /
    ``init_db``). Reading an *existing* WAL database via ``mode=ro`` may create
    transient ``-wal`` / ``-shm`` sidecars — acceptable and unavoidable for a
    live-correct read — but this never creates the primary ``.db`` or its parent
    directory, nor mutates existing content.
    """
    db_path = resolve_db_path(settings)
    lines = [f"database: {db_path}"]
    if not db_path.exists():
        lines.append("  unavailable (file not found)")
        return lines
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        lines.append(f"  schema: {get_schema_version(conn)}")
        lines.extend(_render_topics(conn))
        lines.extend(_render_feeds(conn))
    except sqlite3.Error as exc:
        lines.append(f"  unavailable ({exc})")
    finally:
        if conn is not None:
            conn.close()
    return lines


def _cmd_doctor() -> None:
    """Print a secret-safe diagnostic report for bug reports.

    Read-only with respect to the primary database: never calls
    ``init_db`` / ``get_db`` / ``run_migrations`` (each would ``mkdir`` ``data/``
    and create a WAL ``.db``). Safe to run against a live server.
    """
    from app import __version__

    print(f"version: {__version__}")
    print(f"python: {platform.python_version()}")
    print(f"os: {platform.platform()}")
    print(f"deployment: {'docker' if _in_docker() else 'local'}")

    try:
        settings = load_settings()
    except Exception as exc:  # noqa: BLE001 - diagnostics must never crash
        print(f"configuration: unavailable ({exc})")
        return

    print("configuration:")
    for line in _render_config(settings):
        print(line)
    for line in _render_database(settings):
        print(line)


def _cmd_list() -> None:
    """List all topics with their status."""
    db_path = _open_database(load_settings())

    with get_db(db_path) as conn:
        topics = list_topics(conn)

    if not topics:
        print("No topics configured.")
        return

    print(f"{'Name':<30} {'Status':<15} {'Active':<8} {'Interval':<10}")
    print("-" * 63)
    for topic in topics:
        active = "yes" if topic.is_active else "no"
        if topic.check_interval_minutes:
            from app.interval import format_interval

            interval = format_interval(topic.check_interval_minutes)
        else:
            interval = "default"
        name = terminal_safe(topic.name, width=_NAME_COLUMN)
        padding = " " * (_NAME_COLUMN - display_width(name))
        print(f"{name}{padding} {topic.status.value:<15} {active:<8} {interval:<10}")


def main() -> None:
    """CLI entrypoint."""
    from app import __version__

    parser = argparse.ArgumentParser(
        prog="topic-watch",
        description=f"Topic Watch — AI-powered news monitoring\n\n{CONCURRENCY_WARNING}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"topic-watch {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # check <topic_name>
    check_parser = subparsers.add_parser(
        "check",
        help="Check a single topic for new information",
        description=CONCURRENCY_WARNING,
    )
    check_parser.add_argument("topic_name", help="Name of the topic to check")

    # check-all
    subparsers.add_parser(
        "check-all",
        # The command runs one scheduler cycle: active READY topics whose interval
        # has elapsed. It never forced every active topic through the pipeline, and
        # saying it did made a partial run look complete (AUG-076).
        help="Check every active topic that is due for a check",
        description=(
            "Checks every active topic whose check interval has elapsed, and lists the "
            f"active topics it left alone because they are not due yet.\n\n{CONCURRENCY_WARNING}"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # init <topic_name>
    init_parser = subparsers.add_parser(
        "init",
        help="Run initial knowledge research for a topic",
        description=CONCURRENCY_WARNING,
    )
    init_parser.add_argument("topic_name", help="Name of the topic to initialize")

    # list
    subparsers.add_parser("list", help="List all topics and their status")

    # doctor
    subparsers.add_parser("doctor", help="Print a secret-safe diagnostic report for bug reports")

    args = parser.parse_args()
    setup_logging()

    if args.command == "check":
        asyncio.run(_cmd_check(args.topic_name))
    elif args.command == "check-all":
        asyncio.run(_cmd_check_all())
    elif args.command == "init":
        asyncio.run(_cmd_init(args.topic_name))
    elif args.command == "list":
        _cmd_list()
    elif args.command == "doctor":
        _cmd_doctor()


if __name__ == "__main__":
    main()
