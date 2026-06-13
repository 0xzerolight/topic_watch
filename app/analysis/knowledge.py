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
    compress_knowledge_summary,
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


async def compress_knowledge(
    text: str,
    topic: Topic,
    settings: Settings,
) -> tuple[str, int]:
    """Fit a knowledge summary into the token budget without losing facts.

    Calls the LLM to condense ``text`` to within ``knowledge_state_max_tokens``,
    preserving all distinct facts (unlike trailing-sentence truncation, which
    silently drops them and causes spurious re-detection downstream).

    Degrades gracefully: if the LLM compression fails — or produces output that
    still exceeds the budget — it falls back to lossy ``_truncate_to_budget``
    rather than raising, so an over-budget update never crashes the pipeline.

    Returns:
        ``(summary_text, token_count)`` guaranteed to fit the budget. The
        ``token_count`` is recomputed with the authoritative ``count_tokens``.
    """
    max_tokens = settings.knowledge_state_max_tokens
    model = settings.llm.model
    try:
        result = await compress_knowledge_summary(text, topic, settings)
    except Exception:
        logger.warning(
            "Knowledge compression failed for topic '%s'; falling back to truncation",
            topic.name,
            exc_info=True,
        )
        return _truncate_to_budget(text, max_tokens, model)

    # Recompute authoritatively — never trust a self-reported count.
    compressed = result.compressed_summary
    token_count = count_tokens(compressed, model)
    if token_count > max_tokens:
        # Compression undershot the budget — truncate what it produced rather
        # than persist an over-budget state.
        logger.warning(
            "Compressed knowledge for topic '%s' still over budget (%d > %d); truncating",
            topic.name,
            token_count,
            max_tokens,
        )
        return _truncate_to_budget(compressed, max_tokens, model)

    logger.info(
        "Compressed knowledge for topic '%s' to %d tokens",
        topic.name,
        token_count,
    )
    return compressed, token_count


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
            "Knowledge state for topic '%s' exceeds token budget (%d > %d), compressing",
            topic.name,
            result.token_count,
            settings.knowledge_state_max_tokens,
        )
        result.updated_summary, result.token_count = await compress_knowledge(
            result.updated_summary,
            topic,
            settings,
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
            "Knowledge state for topic '%s' exceeds token budget (%d > %d), compressing",
            topic.name,
            result.token_count,
            settings.knowledge_state_max_tokens,
        )
        result.updated_summary, result.token_count = await compress_knowledge(
            result.updated_summary,
            topic,
            settings,
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
