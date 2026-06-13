"""Context manager stubbing the LLM boundary so no live calls happen.

The real seam (see ``tests/test_analysis.py``) is ``app.analysis.llm._get_client``,
which returns an instructor-patched async client. Every public function in
``app.analysis.llm`` calls ``client.chat.completions.create(...)`` with a
``response_model`` of either ``NoveltyResult`` or ``KnowledgeStateUpdate``.

``stub_llm_boundary`` patches ``_get_client`` to return a mock whose ``create``
dispatches on that ``response_model``, returning the canned result you supply.
Because the patch is at ``_get_client``, the production message-building, token
recomputation, and persistence code all run for real — only the network call is
replaced.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

from app.analysis.llm import KnowledgeStateUpdate, NoveltyResult


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
) -> Iterator[AsyncMock]:
    """Patch ``app.analysis.llm._get_client`` so no live LLM call happens.

    The returned mock client's ``chat.completions.create`` inspects the
    ``response_model`` kwarg and returns:

    * ``novelty`` for ``response_model is NoveltyResult``
    * ``knowledge_init`` (falling back to ``knowledge_update``) for the FIRST
      ``KnowledgeStateUpdate`` call, and ``knowledge_update`` thereafter — so a
      single ``with`` block can serve both ``initialize_new_topic`` and a
      later ``check_topic`` knowledge-update without re-patching.

    Yields:
        The ``AsyncMock`` standing in for ``client.chat.completions.create``,
        so callers can assert on call args / counts.
    """
    novelty_result = novelty if novelty is not None else make_novelty_result()
    init_result = knowledge_init if knowledge_init is not None else make_knowledge_update()
    update_result = knowledge_update if knowledge_update is not None else make_knowledge_update()

    state = {"knowledge_calls": 0}

    async def _create(*_args: object, **kwargs: object) -> object:
        response_model = kwargs.get("response_model")
        if response_model is NoveltyResult:
            return novelty_result
        if response_model is KnowledgeStateUpdate:
            state["knowledge_calls"] += 1
            if state["knowledge_calls"] == 1:
                return init_result
            return update_result
        raise AssertionError(f"Unexpected response_model in stubbed LLM call: {response_model!r}")

    mock_create = AsyncMock(side_effect=_create)
    mock_completions = MagicMock()
    mock_completions.create = mock_create
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions
    mock_client = MagicMock()
    mock_client.chat = mock_chat

    with patch("app.analysis.llm._get_client", return_value=mock_client):
        yield mock_create
