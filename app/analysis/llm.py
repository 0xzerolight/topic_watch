"""LLM wrapper for novelty detection and knowledge state generation.

Uses Instructor + LiteLLM for structured output with automatic
validation retry. All LLM calls go through this module.
"""

import asyncio
import logging
from typing import Any, cast

import instructor
import litellm
from pydantic import BaseModel, Field

from app.analysis.prompts import (
    build_knowledge_init_messages,
    build_knowledge_update_messages,
    build_novelty_messages,
)
from app.config import Settings, is_cloud_provider
from app.models import Article, Topic

logger = logging.getLogger(__name__)


# --- Response models (structured output) ---


class NoveltyResult(BaseModel):
    """LLM response for novelty detection."""

    has_new_info: bool
    summary: str | None = None
    key_facts: list[str] = []
    source_urls: list[str] = []
    confidence: float = Field(ge=0.0, le=1.0)


class KnowledgeStateUpdate(BaseModel):
    """LLM response for knowledge state init/update."""

    updated_summary: str
    token_count: int


# --- Helpers ---


def _get_client(settings: Settings) -> instructor.AsyncInstructor:
    """Create an async instructor-patched litellm client."""
    return cast(instructor.AsyncInstructor, instructor.from_litellm(litellm.acompletion))


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

    On RateLimitError, waits base_delay * (backoff_multiplier ** attempt) seconds
    and retries up to max_retries times. All other exceptions are re-raised immediately.
    """
    last_exc: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            return await call_func()
        except litellm.RateLimitError as exc:
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
        except Exception:
            raise
    assert last_exc is not None
    raise last_exc


# --- Public API ---


async def analyze_articles(
    articles: list[Article],
    knowledge_summary: str,
    topic: Topic,
    settings: Settings,
) -> NoveltyResult:
    """Analyze articles for novelty against the current knowledge state.

    Returns a safe default (has_new_info=False) on any LLM error
    to prevent spurious notifications.
    """

    async def _do_call() -> NoveltyResult:
        client = _get_client(settings)
        messages = build_novelty_messages(articles, knowledge_summary, topic)
        return await client.chat.completions.create(  # type: ignore[no-any-return]
            model=settings.llm.model,
            response_model=NoveltyResult,
            messages=messages,  # type: ignore[arg-type]
            max_retries=settings.llm_max_retries,
            api_key=settings.llm.api_key,
            api_base=_effective_base_url(settings),
            timeout=settings.llm_analysis_timeout,
        )

    try:
        result: NoveltyResult = await _call_with_rate_limit_retry(_do_call)
        return result
    except Exception:
        logger.warning("LLM analysis failed for topic '%s'", topic.name, exc_info=True)
        return NoveltyResult(has_new_info=False, confidence=0.0)


async def generate_initial_knowledge(
    articles: list[Article],
    topic: Topic,
    settings: Settings,
) -> KnowledgeStateUpdate:
    """Generate an initial knowledge state from articles.

    Raises on failure — knowledge initialization is critical.
    """

    async def _do_call() -> KnowledgeStateUpdate:
        client = _get_client(settings)
        messages = build_knowledge_init_messages(articles, topic, settings.knowledge_state_max_tokens)
        return await client.chat.completions.create(  # type: ignore[no-any-return]
            model=settings.llm.model,
            response_model=KnowledgeStateUpdate,
            messages=messages,  # type: ignore[arg-type]
            max_retries=settings.llm_max_retries,
            api_key=settings.llm.api_key,
            api_base=_effective_base_url(settings),
            timeout=settings.llm_knowledge_timeout,
        )

    result: KnowledgeStateUpdate = await _call_with_rate_limit_retry(_do_call)
    result.token_count = count_tokens(result.updated_summary, settings.llm.model)
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

    async def _do_call() -> KnowledgeStateUpdate:
        client = _get_client(settings)
        messages = build_knowledge_update_messages(
            current_summary=current_summary,
            novelty_summary=novelty_result.summary or "",
            key_facts=novelty_result.key_facts,
            topic=topic,
            max_tokens=settings.knowledge_state_max_tokens,
        )
        return await client.chat.completions.create(  # type: ignore[no-any-return]
            model=settings.llm.model,
            response_model=KnowledgeStateUpdate,
            messages=messages,  # type: ignore[arg-type]
            max_retries=settings.llm_max_retries,
            api_key=settings.llm.api_key,
            api_base=_effective_base_url(settings),
            timeout=settings.llm_knowledge_timeout,
        )

    result: KnowledgeStateUpdate = await _call_with_rate_limit_retry(_do_call)
    result.token_count = count_tokens(result.updated_summary, settings.llm.model)
    return result
