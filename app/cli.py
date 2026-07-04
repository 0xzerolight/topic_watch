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
import sqlite3
import sys
from collections import Counter
from datetime import UTC, datetime
from urllib.parse import urlparse

from app.config import Settings, is_api_key_env_sourced, is_exa_key_env_sourced, load_settings, resolve_db_path
from app.crud import (
    get_topic_by_name,
    list_all_feed_health,
    list_topics,
    mark_articles_processed,
    update_topic,
)
from app.database import get_db, get_schema_version, init_db
from app.log_redaction import redact_url
from app.logging_config import setup_logging
from app.models import Topic, TopicStatus

logger = logging.getLogger(__name__)


async def _cmd_check(topic_name: str) -> None:
    """Check a single topic for new information."""
    from app.checker import check_topic

    settings = load_settings()
    init_db()

    with get_db() as conn:
        topic = get_topic_by_name(conn, topic_name)
        if topic is None:
            logger.error("Topic not found: '%s'", topic_name)
            sys.exit(1)

        result = await check_topic(topic, conn, settings)

    print(f"Check complete for '{topic_name}':")
    print(f"  Articles found: {result.articles_found}")
    print(f"  New info: {result.has_new_info}")
    print(f"  Notification sent: {result.notification_sent}")


async def _cmd_check_all() -> None:
    """Check all active, ready topics."""
    from app.checker import check_all_topics

    settings = load_settings()
    init_db()

    results = await check_all_topics(settings)

    print(f"Check cycle complete: {len(results)} topics checked")
    for r in results:
        status = "NEW INFO" if r.has_new_info else "no change"
        print(f"  Topic {r.topic_id}: {status}")


async def _cmd_init(topic_name: str) -> None:
    """Run initial knowledge research for a topic.

    Claims the topic (NEW/READY → RESEARCHING) before the long fetch+LLM work,
    builds the initial knowledge state, and transitions it to READY (or ERROR).
    """
    settings = load_settings()
    init_db()

    # Track an exit code to apply AFTER the get_db() context closes. Calling
    # sys.exit() inside the `with` raises SystemExit, which get_db treats as a
    # failure and rolls back — discarding the ERROR-status write the operator/UI
    # rely on (OVH-002). Set the error state, commit, flag the failure, and exit
    # once the connection has committed and closed.
    failed = False

    with get_db() as conn:
        topic = get_topic_by_name(conn, topic_name)
        if topic is None:
            logger.error("Topic not found: '%s'", topic_name)
            failed = True
        elif topic.status == TopicStatus.RESEARCHING:
            # Another init (scheduler gradual init or a second CLI run) already
            # holds the RESEARCHING claim — cooperate and bail (OVH-018).
            logger.error(
                "Topic '%s' is already being researched; skipping concurrent init.",
                topic_name,
            )
            failed = True
        else:
            assert topic.id is not None
            if topic.status == TopicStatus.READY:
                print(f"Re-initializing knowledge for '{topic_name}'...")
            else:
                print(f"Initializing knowledge for '{topic_name}'...")

            # Claim the topic before the long fetch+LLM work so the scheduler's
            # gradual init (and a second CLI init) are excluded by the same
            # RESEARCHING status guard (OVH-018). Commit so the claim is durable
            # and visible to concurrent connections immediately.
            topic.status = TopicStatus.RESEARCHING
            topic.status_changed_at = datetime.now(UTC)
            topic.error_message = None
            update_topic(conn, topic)
            conn.commit()

            failed = await _run_init(topic, topic_name, conn, settings)

    if failed:
        sys.exit(1)


async def _run_init(topic: Topic, topic_name: str, conn: sqlite3.Connection, settings: Settings) -> bool:
    """Fetch + build knowledge for a claimed (RESEARCHING) topic.

    Persists a terminal ERROR or READY status, committing explicitly so the
    write survives even though the caller exits non-zero on failure (OVH-002).
    Returns True if the init failed (caller should exit 1).
    """
    from app.analysis.knowledge import initialize_knowledge
    from app.scraping import all_sources_failed, fetch_new_articles_for_topic

    # Fetch articles
    try:
        fetch_result = await fetch_new_articles_for_topic(
            topic, conn, max_articles=settings.max_articles_per_check, exa_settings=settings.exa
        )
        articles = fetch_result.articles
    except Exception:
        logger.error("Failed to fetch articles for '%s'", topic_name, exc_info=True)
        topic.status = TopicStatus.ERROR
        topic.status_changed_at = datetime.now(UTC)
        topic.error_message = "Failed to fetch articles during initialization"
        update_topic(conn, topic)
        conn.commit()
        return True

    if not articles:
        # Mode-agnostic message: EXA topics have no feed URLs, so don't tell the operator
        # to "check the feed URLs" — a total source failure (bad key / feeds down) reads
        # differently from a genuinely empty result.
        sources_down = all_sources_failed(fetch_result.feeds_total, fetch_result.feeds_failed)
        logger.error("No articles found for '%s' (check feed sources / credentials; see logs).", topic_name)
        topic.status = TopicStatus.ERROR
        topic.status_changed_at = datetime.now(UTC)
        topic.error_message = (
            "All feed source(s) failed during initialization (check credentials/connectivity; see logs)"
            if sources_down
            else "No articles found during initialization"
        )
        update_topic(conn, topic)
        conn.commit()
        return True

    print(f"  Fetched {len(articles)} articles")

    # Build initial knowledge state (create_knowledge_state uses INSERT OR REPLACE,
    # so re-init of READY topics works without a separate delete step)
    assert topic.id is not None
    try:
        write_result = await initialize_knowledge(topic, articles, conn, settings)
    except Exception:
        logger.error(
            "Knowledge initialization failed for '%s'",
            topic_name,
            exc_info=True,
        )
        topic.status = TopicStatus.ERROR
        topic.status_changed_at = datetime.now(UTC)
        topic.error_message = "LLM failed during knowledge initialization"
        update_topic(conn, topic)
        conn.commit()
        return True

    # Mark articles as processed
    article_ids = [a.id for a in articles if a.id is not None]
    if article_ids:
        mark_articles_processed(conn, article_ids)

    # Transition to READY
    topic.status = TopicStatus.READY
    topic.status_changed_at = datetime.now(UTC)
    topic.error_message = None
    update_topic(conn, topic)
    conn.commit()

    print(f"  Knowledge state built ({write_result.state.token_count} tokens)")
    print(f"  Topic '{topic_name}' is now READY")
    return False


def _in_docker() -> bool:
    """Best-effort container detection for the diagnostic report."""
    return os.path.exists("/.dockerenv")


# Config keys that carry secrets (or are rendered specially); never dumped raw.
_SECRET_CONFIG_KEYS = frozenset({"api_key", "base_url", "urls", "webhook_urls"})


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
            counts = Counter(urlparse(u).scheme or "?" for u in urls)
            summary = ", ".join(f"{scheme} x{n}" for scheme, n in sorted(counts.items()))
            lines.append(f"  {label}: {len(urls)} ({summary})")
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
    for feed in failing:
        lines.append(f"    failing: {redact_url(feed.feed_url)} (x{feed.consecutive_failures})")
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
    init_db()

    with get_db() as conn:
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
        print(f"{topic.name:<30} {topic.status.value:<15} {active:<8} {interval:<10}")


def main() -> None:
    """CLI entrypoint."""
    from app import __version__

    parser = argparse.ArgumentParser(
        prog="topic-watch",
        description="Topic Watch — AI-powered news monitoring",
    )
    parser.add_argument("--version", action="version", version=f"topic-watch {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # check <topic_name>
    check_parser = subparsers.add_parser("check", help="Check a single topic for new information")
    check_parser.add_argument("topic_name", help="Name of the topic to check")

    # check-all
    subparsers.add_parser("check-all", help="Check all active topics for new information")

    # init <topic_name>
    init_parser = subparsers.add_parser("init", help="Run initial knowledge research for a topic")
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
