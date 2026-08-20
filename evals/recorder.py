"""Recording pass-through proxy around the LLM client — the missing observability.

Wraps the real instructor client so every ``create_with_completion`` call is
captured (exact prompt, response_model, mode, raw parsed result, token usage)
before being returned to the caller unchanged. Installed via the same patch
seam the test stub uses (``app.analysis.llm._get_client``), so all the
production message-building, validation, and post-filtering run for real —
only the wire is observed.

Production's structured-output fallback (``_create_structured`` in
app/analysis/llm.py) bakes the mode (TOOLS -> JSON -> MD_JSON) into a DISTINCT
client per mode via ``_get_client(settings, mode)``, because the mode decides
how the request is shaped (forced tool_choice vs response_format vs plain
prompting). This proxy is mode-aware for the same reason: one constant proxy
for every mode would route every fallback attempt through the same inner
client, silently skipping the mode switch a real retry makes (AUG-294). Every
attempt is recorded — including ones a provider rejects — so a fallback that
never happens because the recorder faked success is visible, not silent.

The recorder deep-copies the parsed model at capture time: ``analyze_articles``
mutates its result in place (filtering ``key_facts`` / ``source_urls``), so the
snapshot is what preserves the RAW model for raw-vs-final divergence inspection.

The ``api_key`` passed per call is deliberately never captured; if it leaks
into a raised error's message, that message is scrubbed before recording.
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
    """One captured LLM round-trip (raw, before any caller post-processing).

    ``parsed`` is ``None`` and ``error`` is set when the inner call raised —
    the attempt is still recorded, just without a result.
    """

    response_model: type[BaseModel] | None
    messages: list[dict[str, Any]]
    model: str | None
    temperature: float | None
    mode: instructor.Mode | None
    parsed: BaseModel | None
    error: str | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)


def _sanitize_error(exc: BaseException, api_key: Any) -> str:
    """Render ``exc`` for the record, scrubbing a literal ``api_key`` if a
    provider error message happened to echo it back."""
    text = f"{type(exc).__name__}: {exc}"
    if isinstance(api_key, str) and api_key:
        text = text.replace(api_key, "***")
    return text


class _Completions:
    """Stands in for ``client.chat.completions``; records then delegates."""

    def __init__(self, inner: Any, records: list[CallRecord], mode: instructor.Mode) -> None:
        self._inner = inner
        self._records = records
        self._mode = mode

    async def create_with_completion(self, **kwargs: Any) -> tuple[Any, Any]:
        try:
            parsed, completion = await self._inner.chat.completions.create_with_completion(**kwargs)
        except Exception as exc:
            self._records.append(
                CallRecord(
                    response_model=kwargs.get("response_model"),
                    messages=copy.deepcopy(kwargs.get("messages") or []),
                    model=kwargs.get("model"),
                    temperature=kwargs.get("temperature"),
                    mode=self._mode,
                    parsed=None,
                    error=_sanitize_error(exc, kwargs.get("api_key")),
                )
            )
            raise
        self._records.append(
            CallRecord(
                response_model=kwargs.get("response_model"),
                messages=copy.deepcopy(kwargs.get("messages") or []),
                model=kwargs.get("model"),
                temperature=kwargs.get("temperature"),
                mode=self._mode,
                parsed=parsed.model_copy(deep=True) if isinstance(parsed, BaseModel) else parsed,
                usage=_extract_usage(completion),
            )
        )
        return parsed, completion


class _Chat:
    def __init__(self, inner: Any, records: list[CallRecord], mode: instructor.Mode) -> None:
        self.completions = _Completions(inner, records, mode)


class _RecordingProxy:
    """Minimal stand-in for an instructor client exposing ``.chat.completions``.

    Only ``create_with_completion`` is wrapped — every production LLM function
    calls it (app/analysis/llm.py); none use bare ``.create``.
    """

    def __init__(self, inner: Any, records: list[CallRecord], mode: instructor.Mode) -> None:
        self.chat = _Chat(inner, records, mode)


@contextmanager
def recording_client(*, inner: Any | dict[instructor.Mode, Any] | None = None) -> Iterator[list[CallRecord]]:
    """Patch ``app.analysis.llm._get_client`` with a mode-aware recording proxy.

    One proxy is built per distinct ``mode`` actually requested, matching
    production's own lazy per-mode client cache. ``inner`` selects what each
    proxy wraps:

    - ``None`` (default): a real ``instructor.from_litellm(..., mode=mode)``
      client per mode — the production shape, used by ``scenario``/``live``.
    - a single mock: reused for every mode (offline tests that don't exercise
      mode-specific fallback behavior).
    - a ``{mode: mock}`` dict: a distinct mock per mode (offline tests of the
      fallback sequence itself).

    Yields the growing list of ``CallRecord``s, one per attempt — successful
    or not.
    """
    records: list[CallRecord] = []
    proxies: dict[instructor.Mode, _RecordingProxy] = {}

    def _inner_for(mode: instructor.Mode) -> Any:
        if inner is None:
            return instructor.from_litellm(litellm.acompletion, mode=mode)
        if isinstance(inner, dict):
            return inner[mode]
        return inner

    def _get_client_side_effect(_settings: Any, mode: instructor.Mode = instructor.Mode.TOOLS) -> _RecordingProxy:
        if mode not in proxies:
            proxies[mode] = _RecordingProxy(_inner_for(mode), records, mode)
        return proxies[mode]

    with patch("app.analysis.llm._get_client", side_effect=_get_client_side_effect):
        yield records
