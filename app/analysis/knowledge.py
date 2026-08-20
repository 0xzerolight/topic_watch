"""Knowledge state management: initialization and updates.

Orchestrates LLM calls to build and maintain the rolling knowledge
summary for each topic, with database persistence.
"""

import logging
import re
import sqlite3
from dataclasses import dataclass, field

from app.analysis.llm import (
    KnowledgeStateUpdate,
    NoveltyResult,
    TokenUsage,
    compress_knowledge_summary,
    count_tokens,
    generate_initial_knowledge,
    generate_knowledge_update,
)
from app.config import Settings
from app.crud import (
    create_knowledge_revision,
    create_knowledge_state,
    prune_knowledge_revisions,
    update_knowledge_state_cas,
)
from app.models import Article, KnowledgeRevision, KnowledgeRevisionSource, KnowledgeState, Topic

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeUpdatePlan:
    """A knowledge summary the LLM produced but nobody has persisted yet.

    Knowledge generation is a multi-second LLM round-trip, so it runs with no
    database connection open; the resulting plan is a plain value that the
    caller's single durable transaction applies later, alongside the article
    disposition and the CheckResult. Splitting the LLM call from the write is
    what lets those become one commit instead of three independent ones.

    Fields:
        summary_text: The summary to persist (already fitted to the token budget).
        token_count: Its authoritative ``count_tokens`` value.
        usage: ``TokenUsage`` for the generation call plus any compression
            round-trip (both 0 if the provider omitted usage).
        sufficient_data: The LLM's verdict. On the update path ``False`` means the
            findings were too vague to merge, so ``summary_text`` is the unchanged
            current summary and the caller must NOT apply the plan. On the init
            path ``False`` means thin/off-topic articles; the explanatory summary
            is still stored as the baseline and the topic still goes READY.
    """

    summary_text: str
    token_count: int
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
) -> tuple[str, int, TokenUsage]:
    """Fit a knowledge summary into the token budget without losing facts.

    Calls the LLM to condense ``text`` to within ``knowledge_state_max_tokens``,
    preserving all distinct facts (unlike trailing-sentence truncation, which
    silently drops them and causes spurious re-detection downstream).

    Degrades gracefully: if the LLM compression fails — or produces output that
    still exceeds the budget — it falls back to lossy ``_truncate_to_budget``
    rather than raising, so an over-budget update never crashes the pipeline.

    Returns:
        ``(summary_text, token_count, usage)`` fitting the budget in the normal
        case. ``usage`` is the compression round-trip's ``TokenUsage`` so its
        cost flows into the per-check totals instead of vanishing (OVH-129); it
        is ``TokenUsage()`` (zero) on the truncation-fallback path, where no LLM
        round-trip succeeded.

        ``token_count`` is the authoritative ``count_tokens`` value. On the
        success path it reuses the count ``compress_knowledge_summary`` already
        computed over the same string (itself ``count_tokens`` output, not an
        LLM-claimed number), avoiding a redundant model-aware re-tokenization
        (OVH-135).

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
        text_out, count_out = _truncate_to_budget(text, max_tokens, model)
        return text_out, count_out, TokenUsage()

    usage = TokenUsage(result.prompt_tokens, result.completion_tokens)
    compressed = result.compressed_summary
    # Reuse the count compress_knowledge_summary already computed (count_tokens
    # output over this exact string), not a redundant recount (OVH-135).
    token_count = result.token_count
    if token_count > max_tokens:
        # Compression undershot the budget — truncate what it produced rather
        # than persist an over-budget state.
        logger.warning(
            "Compressed knowledge for topic '%s' still over budget (%d > %d); truncating",
            topic.name,
            token_count,
            max_tokens,
        )
        text_out, count_out = _truncate_to_budget(compressed, max_tokens, model)
        return text_out, count_out, usage

    logger.info(
        "Compressed knowledge for topic '%s' to %d tokens",
        topic.name,
        token_count,
    )
    return compressed, token_count, usage


async def _compress_if_over_budget(
    result: KnowledgeStateUpdate,
    topic: Topic,
    settings: Settings,
) -> TokenUsage:
    """Compress ``result`` in place if it exceeds the knowledge token budget (OVH-177).

    Shared by ``initialize_knowledge`` and ``update_knowledge``, which previously
    duplicated this block verbatim. When over budget, logs a warning, runs the
    compression round-trip, and writes the fitted summary + count back onto
    ``result``. Returns the compression's ``TokenUsage`` so the caller can fold its
    cost into the per-check totals (OVH-129); returns ``TokenUsage()`` (zero) when
    the result already fits and no round-trip ran.
    """
    if result.token_count <= settings.knowledge_state_max_tokens:
        return TokenUsage()
    logger.warning(
        "Knowledge state for topic '%s' exceeds token budget (%d > %d), compressing",
        topic.name,
        result.token_count,
        settings.knowledge_state_max_tokens,
    )
    result.updated_summary, result.token_count, compress_usage = await compress_knowledge(
        result.updated_summary,
        topic,
        settings,
    )
    return compress_usage


async def prepare_initial_knowledge(
    topic: Topic,
    articles: list[Article],
    settings: Settings,
) -> KnowledgeUpdatePlan:
    """Build an initial knowledge summary from articles. Writes nothing.

    Raises on LLM failure — the caller sets the topic status to 'error'.
    ``sufficient_data is False`` means thin/off-topic articles; the explanatory
    summary is still worth storing as the baseline and the topic still goes READY,
    so the caller applies the plan either way.
    """
    result = await generate_initial_knowledge(articles, topic, settings)
    # Track total LLM cost: the generation call plus any compression round-trip
    # that fires below (OVH-129).
    usage = TokenUsage(result.prompt_tokens, result.completion_tokens)

    if not result.sufficient_data:
        logger.warning(
            "Insufficient data for topic '%s' (confidence=%.2f): %s",
            topic.name,
            result.confidence,
            result.updated_summary,
        )

    compress_usage = await _compress_if_over_budget(result, topic, settings)
    return KnowledgeUpdatePlan(
        summary_text=result.updated_summary,
        token_count=result.token_count,
        usage=TokenUsage(
            usage.prompt_tokens + compress_usage.prompt_tokens,
            usage.completion_tokens + compress_usage.completion_tokens,
        ),
        sufficient_data=result.sufficient_data,
    )


async def prepare_knowledge_update(
    topic: Topic,
    novelty_result: NoveltyResult,
    current_summary: str,
    settings: Settings,
) -> KnowledgeUpdatePlan:
    """Merge new findings into ``current_summary``. Writes nothing.

    Raises on LLM failure — the caller records that distinctly and leaves the
    stored knowledge untouched. ``sufficient_data is False`` means the findings
    were too vague to merge: the returned plan carries the unchanged
    ``current_summary`` and the caller must not apply it.
    """
    result = await generate_knowledge_update(current_summary, novelty_result, topic, settings)
    usage = TokenUsage(result.prompt_tokens, result.completion_tokens)

    if not result.sufficient_data:
        logger.warning(
            "Knowledge update for topic '%s' had insufficient data, preserving existing state",
            topic.name,
        )
        return KnowledgeUpdatePlan(
            summary_text=current_summary,
            token_count=0,
            usage=usage,
            sufficient_data=False,
        )

    # Fold the compression round-trip's cost into the reported usage (OVH-129).
    compress_usage = await _compress_if_over_budget(result, topic, settings)
    return KnowledgeUpdatePlan(
        summary_text=result.updated_summary,
        token_count=result.token_count,
        usage=TokenUsage(
            usage.prompt_tokens + compress_usage.prompt_tokens,
            usage.completion_tokens + compress_usage.completion_tokens,
        ),
        sufficient_data=True,
    )


def apply_knowledge_update(
    conn: sqlite3.Connection,
    topic_id: int,
    plan: KnowledgeUpdatePlan,
    *,
    expected_version: int,
    source: KnowledgeRevisionSource,
    settings: Settings,
    change_note: str | None = None,
) -> bool:
    """Persist a prepared plan plus its revision. Does NOT commit.

    Returns ``False`` when the knowledge state moved since ``expected_version``
    was snapshotted — another check finished first, so this plan was built on a
    summary that is no longer current and applying it would lose the winner's
    update. The caller aborts its transaction rather than recording a success.

    The revision append and prune run inside the caller's transaction, not after
    it. That is the point of the design: knowledge state, revision history,
    article disposition and the CheckResult become one commit, so a failure at
    any step leaves none of them — never a knowledge write the check never
    recorded, or a recorded check whose knowledge was rolled back (OVH-009).
    """
    applied = update_knowledge_state_cas(
        conn,
        topic_id,
        summary_text=plan.summary_text,
        token_count=plan.token_count,
        expected_version=expected_version,
    )
    if not applied:
        # Either the row moved (a real conflict) or there is no row yet (first
        # init). Distinguishing them takes one SELECT under the write lock the
        # CAS already holds, so no third writer can slip between the two.
        exists = conn.execute("SELECT 1 FROM knowledge_states WHERE topic_id = ?", (topic_id,)).fetchone()
        if exists:
            logger.warning(
                "Knowledge state for topic_id=%d moved since version %d; discarding this update",
                topic_id,
                expected_version,
            )
            return False
        create_knowledge_state(
            conn,
            KnowledgeState(
                topic_id=topic_id,
                summary_text=plan.summary_text,
                token_count=plan.token_count,
                version=1,
            ),
        )

    create_knowledge_revision(
        conn,
        KnowledgeRevision(
            topic_id=topic_id,
            summary_text=plan.summary_text,
            token_count=plan.token_count,
            source=source,
            change_note=change_note,
        ),
    )
    prune_knowledge_revisions(conn, topic_id, settings.knowledge_revision_limit)
    return True
