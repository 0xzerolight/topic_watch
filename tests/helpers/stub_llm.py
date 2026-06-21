"""Context manager stubbing the LLM boundary so no live calls happen.

The real seam (see ``tests/test_analysis.py``) is ``app.analysis.llm._get_client``,
which returns an instructor-patched async client. Every public function in
``app.analysis.llm`` calls ``client.chat.completions.create(...)`` with a
``response_model`` of either ``NoveltyResult`` or ``KnowledgeStateUpdate``.

``stub_llm_boundary`` patches ``_get_client`` to return a mock whose
``create_with_completion`` (and legacy ``create``) dispatches on that
``response_model``, returning the canned result you supply. Because the patch is
at ``_get_client``, the production message-building, token recomputation, usage
extraction, and persistence code all run for real — only the network call is
replaced.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

from app.analysis.llm import CompressedKnowledge, KnowledgeStateUpdate, NoveltyResult


class _StubUsage:
    """Minimal stand-in for litellm's usage block (prompt/completion tokens)."""

    def __init__(self, prompt_tokens: int = 12, completion_tokens: int = 8) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _StubCompletion:
    """Minimal stand-in for a raw litellm completion exposing ``.usage``."""

    def __init__(self, usage: _StubUsage | None = None) -> None:
        self.usage = usage if usage is not None else _StubUsage()


def make_knowledge_update(
    summary: str = "Canned knowledge summary.",
    *,
    sufficient_data: bool = True,
    confidence: float = 0.9,
) -> KnowledgeStateUpdate:
    """Build a canned KnowledgeStateUpdate (token_count is recomputed by prod code)."""
    return KnowledgeStateUpdate(
        sufficient_data=sufficient_data,
        confidence=confidence,
        updated_summary=summary,
        token_count=0,
    )


def make_compressed_knowledge(
    summary: str = "Canned compressed knowledge.",
) -> CompressedKnowledge:
    """Build a canned CompressedKnowledge (token_count is recomputed by prod code)."""
    return CompressedKnowledge(compressed_summary=summary, token_count=0)


def make_novelty_result(
    *,
    has_new_info: bool = True,
    summary: str | None = "Canned novelty summary.",
    key_facts: list[str] | None = None,
    source_urls: list[str] | None = None,
    confidence: float = 0.95,
    relevance: float = 0.9,
) -> NoveltyResult:
    """Build a canned NoveltyResult with notification-worthy defaults."""
    return NoveltyResult(
        has_new_info=has_new_info,
        summary=summary,
        key_facts=key_facts or [],
        source_urls=source_urls or [],
        confidence=confidence,
        relevance=relevance,
    )


@contextmanager
def stub_llm_boundary(
    *,
    novelty: NoveltyResult | None = None,
    knowledge_init: KnowledgeStateUpdate | None = None,
    knowledge_update: KnowledgeStateUpdate | None = None,
    compressed: CompressedKnowledge | None = None,
) -> Iterator[AsyncMock]:
    """Patch ``app.analysis.llm._get_client`` so no live LLM call happens.

    The returned mock client's ``chat.completions.create`` inspects the
    ``response_model`` kwarg and returns:

    * ``novelty`` for ``response_model is NoveltyResult``
    * ``knowledge_init`` (falling back to ``knowledge_update``) for the FIRST
      ``KnowledgeStateUpdate`` call, and ``knowledge_update`` thereafter — so a
      single ``with`` block can serve both ``initialize_new_topic`` and a
      later ``check_topic`` knowledge-update without re-patching.
    * ``compressed`` for ``response_model is CompressedKnowledge`` — so a smoke
      case that drives the over-budget compression branch (small budget) stays
      offline instead of AssertionError-ing (OVH-162). Defaults to a canned
      ``CompressedKnowledge`` when not supplied.

    Yields:
        The ``AsyncMock`` standing in for ``client.chat.completions.create``,
        so callers can assert on call args / counts.
    """
    novelty_result = novelty if novelty is not None else make_novelty_result()
    init_result = knowledge_init if knowledge_init is not None else make_knowledge_update()
    update_result = knowledge_update if knowledge_update is not None else make_knowledge_update()
    compressed_result = compressed if compressed is not None else make_compressed_knowledge()

    state = {"knowledge_calls": 0}

    def _dispatch(kwargs: dict) -> object:
        response_model = kwargs.get("response_model")
        if response_model is NoveltyResult:
            return novelty_result
        if response_model is CompressedKnowledge:
            return compressed_result
        if response_model is KnowledgeStateUpdate:
            state["knowledge_calls"] += 1
            if state["knowledge_calls"] == 1:
                return init_result
            return update_result
        raise AssertionError(f"Unexpected response_model in stubbed LLM call: {response_model!r}")

    async def _create(*_args: object, **kwargs: object) -> object:
        return _dispatch(kwargs)

    async def _create_with_completion(*_args: object, **kwargs: object) -> tuple[object, object]:
        return _dispatch(kwargs), _StubCompletion()

    mock_create = AsyncMock(side_effect=_create_with_completion)
    mock_completions = MagicMock()
    # compress_knowledge_summary still uses .create; everything else uses
    # create_with_completion. Wire both so the whole pipeline stays offline.
    mock_completions.create = AsyncMock(side_effect=_create)
    mock_completions.create_with_completion = mock_create
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions
    mock_client = MagicMock()
    mock_client.chat = mock_chat

    with patch("app.analysis.llm._get_client", return_value=mock_client):
        yield mock_create
