"""Knowledge state management: initialization and updates.

Orchestrates LLM calls to build and maintain the rolling knowledge
summary for each topic, with database persistence.
"""

import logging
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime

from app.analysis.llm import (
    NoveltyResult,
    TokenUsage,
    compress_knowledge_summary,
    count_tokens,
    generate_initial_knowledge,
    generate_knowledge_update,
)
from app.config import Settings
from app.crud import create_knowledge_state, get_knowledge_state, update_knowledge_state
from app.models import Article, KnowledgeState, Topic

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeWriteResult:
    """Outcome of an init/update knowledge write, with cost + sufficiency signals.

    Fields:
        state: The persisted (or, for an insufficient update, the preserved
            existing) ``KnowledgeState``.
        usage: ``TokenUsage`` consumed by the LLM call (prompt/completion tokens;
            both 0 if the provider omitted usage).
        sufficient_data: The LLM's ``sufficient_data`` verdict. For init, ``False``
            means "insufficient data, worth retrying" (the state was still stored
            with an explanation); ``True`` means good knowledge was built. For
            update, ``False`` means the existing state was preserved unchanged.
    """

    state: KnowledgeState
    usage: TokenUsage = field(default_factory=TokenUsage)
    sufficient_data: bool = True


def _truncate_to_budget(text: str, max_tokens: int, model: str) -> tuple[str, int]:
    """Truncate text by keeping leading sentences until it fits the token budget.

    Keeps the largest leading prefix of sentences whose token count fits
    ``max_tokens`` (dropping trailing sentences), falling back to the first
    sentence as-is when even that overflows. Identical semantics to a one-at-a-
    time trailing-drop loop, but the kept-sentence count is located by binary
    search so the model-aware ``count_tokens`` runs O(log n) times instead of
    O(n) (OVH-049).
    """
    token_count = count_tokens(text, model)
    if token_count <= max_tokens:
        return text, token_count

    sentences = re.split(r"(?<=[.!?])\s+", text)
    n = len(sentences)
    if n <= 1:
        return text, token_count

    def count_first(k: int) -> int:
        return count_tokens(" ".join(sentences[:k]), model)

    # The full text (k == n) is already known to overflow, so search [1, n-1]
    # for the largest k whose leading prefix fits. ``best`` tracks the latest
    # fitting prefix and its authoritative token count.
    lo, hi = 1, n - 1
    best_k = 0
    best_count = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        mid_count = count_first(mid)
        if mid_count <= max_tokens:
            best_k = mid
            best_count = mid_count
            lo = mid + 1
        else:
            hi = mid - 1

    if best_k:
        return " ".join(sentences[:best_k]), best_count

    # Even the first sentence overflows — return it as-is (lossy but never empty).
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
        ``(summary_text, token_count)`` fitting the budget in the normal case.
        The ``token_count`` is recomputed with the authoritative ``count_tokens``.

        Overflow caveat (OVH-164): the fallback ``_truncate_to_budget`` keeps the
        first sentence intact rather than ever returning empty text. So when the
        fallback fires AND that single leading sentence alone exceeds
        ``max_tokens`` (a single mega-sentence with no boundaries to truncate at),
        the returned ``token_count`` may be > the budget. Persisting it is the
        deliberate lesser evil — losing the only sentence would drop all facts —
        but callers must not assume the result always fits.
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
) -> KnowledgeWriteResult:
    """Build an initial knowledge state from articles and store it.

    Raises on LLM failure — the caller should set the topic status to 'error'.

    Returns a ``KnowledgeWriteResult``. ``result.sufficient_data`` is the clean,
    string-free signal the caller can branch on:

    * raises (hard error) -> set topic status to 'error'
    * ``sufficient_data is False`` -> insufficient data, worth retrying in a later
      cycle (the explanatory summary was still persisted)
    * ``sufficient_data is True`` -> good knowledge was built and stored
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
    return KnowledgeWriteResult(
        state=created,
        usage=TokenUsage(result.prompt_tokens, result.completion_tokens),
        sufficient_data=result.sufficient_data,
    )


async def update_knowledge(
    topic: Topic,
    novelty_result: NoveltyResult,
    conn: sqlite3.Connection,
    settings: Settings,
) -> KnowledgeWriteResult:
    """Update the knowledge state with new findings and persist.

    Raises ValueError if no existing knowledge state is found.
    Raises on LLM failure — the caller should handle gracefully.

    Returns a ``KnowledgeWriteResult`` carrying the persisted (or preserved)
    state, the LLM ``usage``, and ``sufficient_data`` (``False`` means the LLM
    found the new findings too vague to merge, so the existing state was kept).
    """
    if topic.id is None:
        raise ValueError("Topic must have an ID")

    current = get_knowledge_state(conn, topic.id)
    if current is None:
        raise ValueError(f"No knowledge state found for topic '{topic.name}' (id={topic.id})")

    result = await generate_knowledge_update(current.summary_text, novelty_result, topic, settings)
    usage = TokenUsage(result.prompt_tokens, result.completion_tokens)

    if not result.sufficient_data:
        logger.warning(
            "Knowledge update for topic '%s' had insufficient data, preserving existing state",
            topic.name,
        )
        return KnowledgeWriteResult(state=current, usage=usage, sufficient_data=False)

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
    return KnowledgeWriteResult(state=current, usage=usage, sufficient_data=True)
