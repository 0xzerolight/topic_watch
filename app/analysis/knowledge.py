"""Knowledge state management: initialization and updates.

Orchestrates LLM calls to build and maintain the rolling knowledge
summary for each topic, with database persistence.
"""

import logging
import re
import sqlite3
from datetime import UTC, datetime

from app.analysis.llm import (
    NoveltyResult,
    count_tokens,
    generate_initial_knowledge,
    generate_knowledge_update,
)
from app.config import Settings
from app.crud import create_knowledge_state, get_knowledge_state, update_knowledge_state
from app.models import Article, KnowledgeState, Topic

logger = logging.getLogger(__name__)


def _truncate_to_budget(text: str, max_tokens: int, model: str) -> tuple[str, int]:
    """Truncate text by removing trailing sentences until it fits the token budget."""
    token_count = count_tokens(text, model)
    if token_count <= max_tokens:
        return text, token_count

    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) <= 1:
        return text, token_count

    while len(sentences) > 1:
        sentences.pop()
        truncated = " ".join(sentences)
        token_count = count_tokens(truncated, model)
        if token_count <= max_tokens:
            return truncated, token_count

    # Down to one sentence — return as-is
    final = sentences[0]
    return final, count_tokens(final, model)


async def initialize_knowledge(
    topic: Topic,
    articles: list[Article],
    conn: sqlite3.Connection,
    settings: Settings,
) -> KnowledgeState:
    """Build an initial knowledge state from articles and store it.

    Raises on LLM failure — the caller should set the topic status to 'error'.
    """
    if topic.id is None:
        raise ValueError("Topic must have an ID")

    result = await generate_initial_knowledge(articles, topic, settings)

    if not result.sufficient_data:
        logger.warning(
            "Insufficient data for topic '%s' (confidence=%.2f): %s",
            topic.name,
            result.confidence,
            result.updated_summary,
        )
        # Still store it — the summary explains what's missing, which is useful
        # context for the next check cycle. The topic still transitions to READY.

    if result.token_count > settings.knowledge_state_max_tokens:
        logger.warning(
            "Knowledge state for topic '%s' exceeds token budget (%d > %d)",
            topic.name,
            result.token_count,
            settings.knowledge_state_max_tokens,
        )
        result.updated_summary, result.token_count = _truncate_to_budget(
            result.updated_summary,
            settings.knowledge_state_max_tokens,
            settings.llm.model,
        )
        logger.info(
            "Truncated knowledge state for topic '%s' to %d tokens",
            topic.name,
            result.token_count,
        )

    state = KnowledgeState(
        topic_id=topic.id,
        summary_text=result.updated_summary,
        token_count=result.token_count,
    )
    created = create_knowledge_state(conn, state)
    conn.commit()

    logger.info(
        "Initialized knowledge for topic '%s' (%d tokens)",
        topic.name,
        result.token_count,
    )
    return created


async def update_knowledge(
    topic: Topic,
    novelty_result: NoveltyResult,
    conn: sqlite3.Connection,
    settings: Settings,
) -> KnowledgeState:
    """Update the knowledge state with new findings and persist.

    Raises ValueError if no existing knowledge state is found.
    Raises on LLM failure — the caller should handle gracefully.
    """
    if topic.id is None:
        raise ValueError("Topic must have an ID")

    current = get_knowledge_state(conn, topic.id)
    if current is None:
        raise ValueError(f"No knowledge state found for topic '{topic.name}' (id={topic.id})")

    result = await generate_knowledge_update(current.summary_text, novelty_result, topic, settings)

    if not result.sufficient_data:
        logger.warning(
            "Knowledge update for topic '%s' had insufficient data, preserving existing state",
            topic.name,
        )
        return current

    if result.token_count > settings.knowledge_state_max_tokens:
        logger.warning(
            "Knowledge state for topic '%s' exceeds token budget (%d > %d)",
            topic.name,
            result.token_count,
            settings.knowledge_state_max_tokens,
        )
        result.updated_summary, result.token_count = _truncate_to_budget(
            result.updated_summary,
            settings.knowledge_state_max_tokens,
            settings.llm.model,
        )
        logger.info(
            "Truncated knowledge state for topic '%s' to %d tokens",
            topic.name,
            result.token_count,
        )

    current.summary_text = result.updated_summary
    current.token_count = result.token_count
    current.updated_at = datetime.now(UTC)

    update_knowledge_state(conn, current)
    conn.commit()

    logger.info(
        "Updated knowledge for topic '%s' (%d tokens)",
        topic.name,
        result.token_count,
    )
    return current
