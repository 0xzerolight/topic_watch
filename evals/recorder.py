"""Recording pass-through proxy around the LLM client — the missing observability.

Wraps the real instructor client so every ``create_with_completion`` call is
captured (exact prompt, response_model, raw parsed result, token usage) before
being returned to the caller unchanged. Installed via the same patch seam the
test stub uses (``app.analysis.llm._get_client``), so all the production
message-building, validation, and post-filtering run for real — only the wire is
observed.

The recorder deep-copies the parsed model at capture time: ``analyze_articles``
mutates its result in place (filtering ``key_facts`` / ``source_urls``), so the
snapshot is what preserves the RAW model for raw-vs-final divergence inspection.

The ``api_key`` passed per call is deliberately never captured.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import instructor
import litellm
from pydantic import BaseModel

from app.analysis.llm import TokenUsage, _extract_usage


@dataclass
class CallRecord:
    """One captured LLM round-trip (raw, before any caller post-processing)."""

    response_model: type[BaseModel] | None
    messages: list[dict[str, Any]]
    model: str | None
    temperature: float | None
    parsed: BaseModel
    usage: TokenUsage = field(default_factory=TokenUsage)


class _Completions:
    """Stands in for ``client.chat.completions``; records then delegates."""

    def __init__(self, inner: Any, records: list[CallRecord]) -> None:
        self._inner = inner
        self._records = records

    async def create_with_completion(self, **kwargs: Any) -> tuple[Any, Any]:
        parsed, completion = await self._inner.chat.completions.create_with_completion(**kwargs)
        self._records.append(
            CallRecord(
                response_model=kwargs.get("response_model"),
                messages=copy.deepcopy(kwargs.get("messages") or []),
                model=kwargs.get("model"),
                temperature=kwargs.get("temperature"),
                parsed=parsed.model_copy(deep=True) if isinstance(parsed, BaseModel) else parsed,
                usage=_extract_usage(completion),
            )
        )
        return parsed, completion


class _Chat:
    def __init__(self, inner: Any, records: list[CallRecord]) -> None:
        self.completions = _Completions(inner, records)


class _RecordingProxy:
    """Minimal stand-in for an instructor client exposing ``.chat.completions``.

    Only ``create_with_completion`` is wrapped — every production LLM function
    calls it (app/analysis/llm.py); none use bare ``.create``.
    """

    def __init__(self, inner: Any, records: list[CallRecord]) -> None:
        self.chat = _Chat(inner, records)


@contextmanager
def recording_client(*, inner: Any = None) -> Iterator[list[CallRecord]]:
    """Patch ``app.analysis.llm._get_client`` to a recording proxy.

    ``inner`` defaults to the real instructor client
    (``instructor.from_litellm(litellm.acompletion)``); offline tests inject a
    mock so no live call ever fires. Yields the growing list of ``CallRecord``s.
    """
    if inner is None:
        inner = instructor.from_litellm(litellm.acompletion)
    records: list[CallRecord] = []
    proxy = _RecordingProxy(inner, records)
    with patch("app.analysis.llm._get_client", return_value=proxy):
        yield records
