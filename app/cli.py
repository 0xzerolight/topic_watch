"""CLI entrypoint for Topic Watch.

Provides manual access to the check pipeline, topic initialization,
and topic listing. Run as: python -m app.cli <command>
"""

import argparse
import asyncio
import logging
import sys

from app.config import load_settings
from app.crud import (
    get_topic_by_name,
    list_topics,
    mark_articles_processed,
    update_topic,
)
from app.database import get_db, init_db
from app.models import TopicStatus

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    """Configure logging for CLI usage."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


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

    with get_db() as conn:
        results = await check_all_topics(conn, settings)

    print(f"Check cycle complete: {len(results)} topics checked")
    for r in results:
        status = "NEW INFO" if r.has_new_info else "no change"
        print(f"  Topic {r.topic_id}: {status}")


async def _cmd_init(topic_name: str) -> None:
    """Run initial knowledge research for a topic.

    Fetches articles, builds initial knowledge state, and transitions
    the topic from RESEARCHING to READY status.
    """
    from app.analysis.knowledge import initialize_knowledge
    from app.scraping import fetch_new_articles_for_topic

    settings = load_settings()
    init_db()

    with get_db() as conn:
        topic = get_topic_by_name(conn, topic_name)
        if topic is None:
            logger.error("Topic not found: '%s'", topic_name)
            sys.exit(1)

        if topic.status == TopicStatus.READY:
            print(f"Re-initializing knowledge for '{topic_name}'...")
        else:
            print(f"Initializing knowledge for '{topic_name}'...")

        # Fetch articles
        try:
            fetch_result = await fetch_new_articles_for_topic(topic, conn, max_articles=settings.max_articles_per_check)
            articles = fetch_result.articles
        except Exception:
            logger.error("Failed to fetch articles for '%s'", topic_name, exc_info=True)
            topic.status = TopicStatus.ERROR
            topic.error_message = "Failed to fetch articles during initialization"
            update_topic(conn, topic)
            sys.exit(1)

        if not articles:
            logger.error("No articles found for '%s'. Check the feed URLs.", topic_name)
            topic.status = TopicStatus.ERROR
            topic.error_message = "No articles found during initialization"
            update_topic(conn, topic)
            sys.exit(1)

        print(f"  Fetched {len(articles)} articles")

        # Build initial knowledge state (create_knowledge_state uses INSERT OR REPLACE,
        # so re-init of READY topics works without a separate delete step)
        assert topic.id is not None
        try:
            state = await initialize_knowledge(topic, articles, conn, settings)
        except Exception:
            logger.error(
                "Knowledge initialization failed for '%s'",
                topic_name,
                exc_info=True,
            )
            topic.status = TopicStatus.ERROR
            topic.error_message = "LLM failed during knowledge initialization"
            update_topic(conn, topic)
            sys.exit(1)

        # Mark articles as processed
        article_ids = [a.id for a in articles if a.id is not None]
        if article_ids:
            mark_articles_processed(conn, article_ids)

        # Transition to READY
        topic.status = TopicStatus.READY
        topic.error_message = None
        update_topic(conn, topic)

        print(f"  Knowledge state built ({state.token_count} tokens)")
        print(f"  Topic '{topic_name}' is now READY")


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
        interval = f"{topic.check_interval_minutes}m" if topic.check_interval_minutes else "default"
        print(f"{topic.name:<30} {topic.status.value:<15} {active:<8} {interval:<10}")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="topic-watch",
        description="Topic Watch — AI-powered news monitoring",
    )
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
    _setup_logging()

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
