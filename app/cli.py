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
import sqlite3
import sys
from datetime import UTC, datetime

from app.config import Settings, load_settings
from app.crud import (
    get_topic_by_name,
    list_topics,
    mark_articles_processed,
    update_topic,
)
from app.database import get_db, init_db
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
    from app.scraping import fetch_new_articles_for_topic

    # Fetch articles
    try:
        fetch_result = await fetch_new_articles_for_topic(topic, conn, max_articles=settings.max_articles_per_check)
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
        logger.error("No articles found for '%s'. Check the feed URLs.", topic_name)
        topic.status = TopicStatus.ERROR
        topic.status_changed_at = datetime.now(UTC)
        topic.error_message = "No articles found during initialization"
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


if __name__ == "__main__":
    main()
