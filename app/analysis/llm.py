"""LLM wrapper for novelty detection and knowledge state generation.

Uses Instructor + LiteLLM for structured output with automatic
validation retry. All LLM calls go through this module.
"""

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any, cast

import instructor
import litellm
from instructor.core import InstructorRetryException
from pydantic import BaseModel, Field
from tenacity import AsyncRetrying, retry_if_not_exception_type, stop_after_attempt

from app.analysis.prompts import (
    build_knowledge_compress_messages,
    build_knowledge_init_messages,
    build_knowledge_update_messages,
    build_novelty_messages,
)
from app.config import Settings, is_cloud_provider
from app.models import Article, Topic

logger = logging.getLogger(__name__)


# --- Token usage ---


@dataclass(frozen=True)
class TokenUsage:
    """Per-call LLM token consumption, extracted from the raw completion.

    Both fields default to 0 when usage is unavailable (some providers omit it,
    or the call short-circuited to a safe default before any LLM round-trip).
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0


def _extract_usage(completion: Any) -> TokenUsage:
    """Pull prompt/completion token counts off a raw litellm completion.

    Returns ``TokenUsage(0, 0)`` if the completion has no usable usage block
    (missing attribute, None, or non-integer values) so callers never crash on
    provider-specific shapes.
    """
    usage = getattr(completion, "usage", None)
    if usage is None:
        return TokenUsage()

    def _coerce(value: Any) -> int:
        try:
            return int(value) if value is not None else 0
        except (TypeError, ValueError):
            return 0

    # litellm.Usage supports both attribute and mapping access depending on provider.
    prompt = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    if prompt is None and isinstance(usage, dict):
        prompt = usage.get("prompt_tokens")
    if completion_tokens is None and isinstance(usage, dict):
        completion_tokens = usage.get("completion_tokens")
    return TokenUsage(prompt_tokens=_coerce(prompt), completion_tokens=_coerce(completion_tokens))


# --- Response models (structured output) ---


class NoveltyResult(BaseModel):
    """LLM response for novelty detection.

    ``prompt_tokens`` / ``completion_tokens`` are NOT filled by the LLM — they
    default to 0 and are populated from the raw completion's usage after the
    call (0 on the safe-default error path or when the provider omits usage).
    """

    reasoning: str = Field(default="", description="Brief chain-of-thought: what you compared, why you decided.")
    has_new_info: bool
    # Consumed by the knowledge-update prompt's "New Findings to Incorporate"
    # block, so the model must populate it whenever there is new info (OVH-026).
    summary: str | None = Field(
        default=None,
        description=(
            "A one-to-two sentence neutral summary of the new development. "
            "Required when has_new_info is true; null only when has_new_info is false."
        ),
    )
    key_facts: list[str] = []
    source_urls: list[str] = []
    confidence: float = Field(ge=0.0, le=1.0)
    relevance: float = Field(
        ge=0.0,
        le=1.0,
        default=0.0,
        description="How relevant the new information is to the topic description (0=off-topic, 1=exactly what user asked)",
    )
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Set ONLY on the fail-safe error path (LLM call failed). Lets the caller
    # distinguish a genuine analysis failure from a clean "nothing new" result
    # without making analyze_articles raise (settled decision #3). None on
    # every successful call, including a legitimate has_new_info=False. The
    # description instructs the model not to populate it; analyze_articles also
    # force-resets it on the success path (belt-and-suspenders).
    error: str | None = Field(
        default=None,
        description="Internal error channel; the model must always leave this null.",
    )


class KnowledgeStateUpdate(BaseModel):
    """LLM response for knowledge state init/update.

    ``prompt_tokens`` / ``completion_tokens`` are populated from the raw
    completion's usage after the call (not filled by the LLM); they default to
    0 when the provider omits usage.
    """

    sufficient_data: bool = Field(
        description="False if the articles lack enough relevant information to build a useful summary."
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="How confident you are in the accuracy of this summary based on source articles."
    )
    updated_summary: str = Field(
        description="The knowledge summary. If sufficient_data is false, explain what information was missing."
    )
    token_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class CompressedKnowledge(BaseModel):
    """LLM response for knowledge-state compression."""

    compressed_summary: str = Field(
        description="The condensed knowledge summary: same facts, less verbosity, within the token budget."
    )
    token_count: int = 0


# --- Helpers ---


def _summarize_exc(exc: BaseException, *, limit: int = 200) -> str:
    """One-line, length-bounded summary of an exception for stored error fields."""
    summary = f"{type(exc).__name__}: {exc}".replace("\n", " ").strip()
    return summary[:limit]


def _instructor_retries(max_retries: int) -> AsyncRetrying:
    """Build instructor's per-call retry policy.

    Instructor's ``max_retries`` governs structured-output *validation* retries
    (re-prompting when the LLM's response fails Pydantic validation). We keep
    those, but explicitly exclude ``RateLimitError`` from instructor's retry set
    so a 429 propagates *bare* to ``_call_with_rate_limit_retry``, which owns the
    rate-limit backoff. Otherwise instructor would swallow the 429 inside an
    ``InstructorRetryException`` and immediately re-fire it ``max_retries`` times
    with zero delay — hammering the throttled provider and hiding the rate limit
    from operators (OVH-008).
    """
    return AsyncRetrying(
        stop=stop_after_attempt(max_retries + 1),
        retry=retry_if_not_exception_type(litellm.RateLimitError),
    )


def _unwrap_rate_limit(exc: BaseException) -> litellm.RateLimitError | None:
    """Return the underlying ``RateLimitError`` if ``exc`` represents a 429.

    Belt-and-suspenders for the rate-limit backoff: a bare ``RateLimitError`` is
    returned as-is, and an ``InstructorRetryException`` is inspected (its first
    arg and its ``failed_attempts``) for an underlying ``RateLimitError`` in case
    a provider/instructor path still wraps it despite ``_instructor_retries``.
    """
    if isinstance(exc, litellm.RateLimitError):
        return exc
    if isinstance(exc, InstructorRetryException):
        for arg in exc.args:
            if isinstance(arg, litellm.RateLimitError):
                return arg
        for attempt in exc.failed_attempts or []:
            attempt_exc = getattr(attempt, "exception", None)
            if isinstance(attempt_exc, litellm.RateLimitError):
                return attempt_exc
    return None


_client: instructor.AsyncInstructor | None = None


def _get_client(settings: Settings) -> instructor.AsyncInstructor:
    """Return a cached async instructor-patched litellm client.

    The client wraps ``litellm.acompletion``, which is stateless — model, key,
    base_url, etc. are passed per call — so a single instance is reused across
    all calls instead of being rebuilt each time. ``settings`` is accepted for
    call-site symmetry and to keep ``_get_client`` the patch seam used by tests.
    """
    global _client
    if _client is None:
        _client = cast(instructor.AsyncInstructor, instructor.from_litellm(litellm.acompletion))
    return _client


def _effective_base_url(settings: Settings) -> str | None:
    """Return base_url only for non-cloud providers (safety net for misconfigured configs)."""
    if settings.llm.base_url and not is_cloud_provider(settings.llm.model):
        return settings.llm.base_url
    return None


def count_tokens(text: str, model: str) -> int:
    """Count tokens using litellm's model-aware tokenizer.

    Falls back to len(text) // 4 if the tokenizer fails.
    """
    try:
        return litellm.token_counter(model=model, text=text)  # type: ignore[no-any-return]
    except Exception:
        logger.debug("Token counting failed for model %s, using fallback", model)
        return len(text) // 4


async def _call_with_rate_limit_retry(
    call_func: Any,
    max_retries: int = 3,
    base_delay: float = 5.0,
    backoff_multiplier: float = 3.0,
) -> Any:
    """Wrap an async LLM call with exponential backoff on rate limit errors.

    On a rate-limit error, waits ``base_delay * (backoff_multiplier ** attempt)``
    seconds and retries up to ``max_retries`` times. Rate limits are detected via
    ``_unwrap_rate_limit`` so this fires whether the call raises a bare
    ``RateLimitError`` (the expected path now that instructor is told not to
    retry 429s — see ``_instructor_retries``) or an ``InstructorRetryException``
    that still wraps one. All other exceptions are re-raised immediately.
    """
    last_exc: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            return await call_func()
        except Exception as exc:
            rate_limit = _unwrap_rate_limit(exc)
            if rate_limit is None:
                raise
            last_exc = exc
            if attempt >= max_retries:
                break
            delay = base_delay * (backoff_multiplier**attempt)
            logger.warning(
                "Rate limit hit (attempt %d/%d), retrying in %.0fs",
                attempt + 1,
                max_retries,
                delay,
            )
            await asyncio.sleep(delay)
    assert last_exc is not None
    raise last_exc


# --- key_facts restatement filtering ---

# A key_fact is dropped only when it is a CLEAR restatement of the existing
# knowledge summary: either its normalized text appears verbatim inside the
# normalized summary, or a long *contiguous* word-sequence (n-gram) of the fact
# appears verbatim in the summary. Phrase-level matching (not bag-of-words set
# overlap) is required so a short genuinely-new fact whose words merely scatter
# across a long summary is never silently dropped. Conservative by design.
#
# Restatement requires the longest fact word-sequence shared contiguously with
# the summary to cover at least this fraction of the fact's content words...
_RESTATEMENT_PHRASE_OVERLAP_THRESHOLD = 0.8
# ...and the fact must have at least this many content words. Shorter facts are
# never auto-dropped, since coincidental phrase matches are too easy on them.
_RESTATEMENT_MIN_FACT_WORDS = 4
_WORD_RE = re.compile(r"\w+")


def _normalize_for_match(text: str) -> str:
    """Lowercase and collapse whitespace for substring comparison."""
    return " ".join(text.lower().split())


def _content_words(text: str) -> list[str]:
    """Extract ordered, lowercased word tokens (multiplicity preserved)."""
    return [m.group(0) for m in _WORD_RE.finditer(text.lower())]


def _longest_contiguous_run(fact_words: list[str], summary_words: list[str]) -> int:
    """Length of the longest contiguous run of ``fact_words`` that appears, in
    order and adjacent, anywhere within ``summary_words``.

    Classic DP for longest common substring over token sequences.
    """
    if not fact_words or not summary_words:
        return 0
    prev = [0] * (len(summary_words) + 1)
    best = 0
    for fw in fact_words:
        curr = [0] * (len(summary_words) + 1)
        for j, sw in enumerate(summary_words, start=1):
            if fw == sw:
                curr[j] = prev[j - 1] + 1
                if curr[j] > best:
                    best = curr[j]
        prev = curr
    return best


def _is_restatement(fact: str, knowledge_summary: str) -> bool:
    """True if ``fact`` clearly restates content already in the summary.

    Two conservative, phrase-level signals:
    * normalized-substring: the fact's normalized text appears verbatim within
      the normalized knowledge summary; or
    * contiguous-phrase: the fact has ≥ ``_RESTATEMENT_MIN_FACT_WORDS`` content
      words AND its longest contiguous word-sequence shared with the summary
      covers ≥ ``_RESTATEMENT_PHRASE_OVERLAP_THRESHOLD`` of the fact's words.

    Set-overlap is deliberately NOT used: scattered, non-contiguous word matches
    must never drop a short, genuinely-new fact.

    An empty summary or empty fact is never a restatement.
    """
    if not knowledge_summary.strip() or not fact.strip():
        return False

    norm_fact = _normalize_for_match(fact)
    norm_summary = _normalize_for_match(knowledge_summary)
    if norm_fact and norm_fact in norm_summary:
        return True

    fact_words = _content_words(fact)
    if len(fact_words) < _RESTATEMENT_MIN_FACT_WORDS:
        return False
    summary_words = _content_words(knowledge_summary)
    run = _longest_contiguous_run(fact_words, summary_words)
    return run / len(fact_words) >= _RESTATEMENT_PHRASE_OVERLAP_THRESHOLD


def _filter_restated_key_facts(key_facts: list[str], knowledge_summary: str) -> list[str]:
    """Drop key_facts that clearly restate the current knowledge summary.

    Kept conservative: only removes clear restatements. If every fact is filtered
    the caller keeps ``has_new_info`` as-is with an empty ``key_facts`` (the
    summary still conveys the novelty).
    """
    return [fact for fact in key_facts if not _is_restatement(fact, knowledge_summary)]


# --- source_urls subset guard (prompt-injection output validation) ---


def _filter_source_urls(source_urls: list[str], articles: list[Article]) -> list[str]:
    """Keep only LLM-returned source_urls that match an input article URL.

    A successful-but-manipulated completion can emit an attacker-chosen
    source_url (e.g. a phishing link injected via feed text) that still passes
    schema validation and would otherwise flow into notifications/webhooks
    (OVH-058). Cross-checking against the input set drops any smuggled URL while
    preserving order and de-duplicating. Comparison is on the exact URL string,
    matching how the URLs were presented to the model.
    """
    allowed = {article.url for article in articles}
    seen: set[str] = set()
    kept: list[str] = []
    for url in source_urls:
        if url in allowed and url not in seen:
            kept.append(url)
            seen.add(url)
    return kept


# --- Public API ---


async def analyze_articles(
    articles: list[Article],
    knowledge_summary: str,
    topic: Topic,
    settings: Settings,
) -> NoveltyResult:
    """Analyze articles for novelty against the current knowledge state.

    Returns a safe default (has_new_info=False) on any LLM error
    to prevent spurious notifications. On success, ``prompt_tokens`` /
    ``completion_tokens`` are populated from the raw completion's usage, and
    ``key_facts`` that merely restate the knowledge summary are dropped.
    """

    async def _do_call() -> tuple[NoveltyResult, Any]:
        client = _get_client(settings)
        messages = build_novelty_messages(articles, knowledge_summary, topic)
        return await client.chat.completions.create_with_completion(  # type: ignore[no-any-return]
            model=settings.llm.model,
            response_model=NoveltyResult,
            messages=messages,  # type: ignore[arg-type]
            max_retries=_instructor_retries(settings.llm_max_retries),
            api_key=settings.llm.api_key,
            api_base=_effective_base_url(settings),
            timeout=settings.llm_analysis_timeout,
            temperature=settings.llm_temperature,
        )

    try:
        result, completion = await _call_with_rate_limit_retry(_do_call, max_retries=settings.llm_max_retries)
    except Exception as exc:
        logger.warning("LLM analysis failed for topic '%s'", topic.name, exc_info=True)
        return NoveltyResult(has_new_info=False, confidence=0.0, error=_summarize_exc(exc))

    novelty: NoveltyResult = result
    # ``error`` is in the LLM's structured-output schema, so a model can populate
    # it on a clean run. Force it None here so ONLY the except-branch above ever
    # sets it; otherwise the checker mis-stamps a healthy run as analysis_failed.
    novelty.error = None
    usage = _extract_usage(completion)
    novelty.prompt_tokens = usage.prompt_tokens
    novelty.completion_tokens = usage.completion_tokens
    novelty.key_facts = _filter_restated_key_facts(novelty.key_facts, knowledge_summary)
    # Drop any source_url not in the input set so an injected completion cannot
    # smuggle an attacker-chosen URL into notifications/webhooks (OVH-058).
    novelty.source_urls = _filter_source_urls(novelty.source_urls, articles)
    return novelty


async def generate_initial_knowledge(
    articles: list[Article],
    topic: Topic,
    settings: Settings,
) -> KnowledgeStateUpdate:
    """Generate an initial knowledge state from articles.

    Raises on failure — knowledge initialization is critical.
    """

    async def _do_call() -> tuple[KnowledgeStateUpdate, Any]:
        client = _get_client(settings)
        messages = build_knowledge_init_messages(articles, topic, settings.knowledge_state_max_tokens)
        return await client.chat.completions.create_with_completion(  # type: ignore[no-any-return]
            model=settings.llm.model,
            response_model=KnowledgeStateUpdate,
            messages=messages,  # type: ignore[arg-type]
            max_retries=_instructor_retries(settings.llm_max_retries),
            api_key=settings.llm.api_key,
            api_base=_effective_base_url(settings),
            timeout=settings.llm_knowledge_timeout,
            temperature=settings.llm_temperature,
        )

    raw_result, completion = await _call_with_rate_limit_retry(_do_call, max_retries=settings.llm_max_retries)
    result: KnowledgeStateUpdate = raw_result
    result.token_count = count_tokens(result.updated_summary, settings.llm.model)
    usage = _extract_usage(completion)
    result.prompt_tokens = usage.prompt_tokens
    result.completion_tokens = usage.completion_tokens
    return result


async def compress_knowledge_summary(
    current_summary: str,
    topic: Topic,
    settings: Settings,
) -> CompressedKnowledge:
    """Compress an over-budget knowledge summary while preserving its facts.

    Raises on failure — the caller decides how to degrade (e.g. fall back to
    lossy truncation). The returned ``token_count`` is recomputed authoritatively.
    """

    async def _do_call() -> CompressedKnowledge:
        client = _get_client(settings)
        messages = build_knowledge_compress_messages(
            current_summary=current_summary,
            topic=topic,
            max_tokens=settings.knowledge_state_max_tokens,
        )
        return await client.chat.completions.create(  # type: ignore[no-any-return]
            model=settings.llm.model,
            response_model=CompressedKnowledge,
            messages=messages,  # type: ignore[arg-type]
            max_retries=_instructor_retries(settings.llm_max_retries),
            api_key=settings.llm.api_key,
            api_base=_effective_base_url(settings),
            timeout=settings.llm_knowledge_timeout,
            temperature=settings.llm_temperature,
        )

    result: CompressedKnowledge = await _call_with_rate_limit_retry(_do_call, max_retries=settings.llm_max_retries)
    result.token_count = count_tokens(result.compressed_summary, settings.llm.model)
    return result


async def generate_knowledge_update(
    current_summary: str,
    novelty_result: NoveltyResult,
    topic: Topic,
    settings: Settings,
) -> KnowledgeStateUpdate:
    """Update the knowledge state with new findings.

    Raises on failure — knowledge updates are critical.
    """

    async def _do_call() -> tuple[KnowledgeStateUpdate, Any]:
        client = _get_client(settings)
        messages = build_knowledge_update_messages(
            current_summary=current_summary,
            novelty_summary=novelty_result.summary or "",
            key_facts=novelty_result.key_facts,
            topic=topic,
            max_tokens=settings.knowledge_state_max_tokens,
        )
        return await client.chat.completions.create_with_completion(  # type: ignore[no-any-return]
            model=settings.llm.model,
            response_model=KnowledgeStateUpdate,
            messages=messages,  # type: ignore[arg-type]
            max_retries=_instructor_retries(settings.llm_max_retries),
            api_key=settings.llm.api_key,
            api_base=_effective_base_url(settings),
            timeout=settings.llm_knowledge_timeout,
            temperature=settings.llm_temperature,
        )

    raw_result, completion = await _call_with_rate_limit_retry(_do_call, max_retries=settings.llm_max_retries)
    result: KnowledgeStateUpdate = raw_result
    result.token_count = count_tokens(result.updated_summary, settings.llm.model)
    usage = _extract_usage(completion)
    result.prompt_tokens = usage.prompt_tokens
    result.completion_tokens = usage.completion_tokens
    return result
