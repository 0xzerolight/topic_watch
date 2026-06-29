"""Dispatch the four LLM stages with recording, and build a RunArtifact.

``KIND_DISPATCH`` is the single registry of the four kinds. It is NOT a uniform
tuple: the builders and llm functions have heterogeneous signatures (notably
``knowledge_update`` hands ``generate_knowledge_update`` a ``NoveltyResult``
object on the run side, while ``build_knowledge_update_messages`` takes the
summary + key_facts separately on the dry-run side). Each kind therefore maps to
a small adapter of two closures — ``run`` (await the real llm fn) and ``build``
(produce the messages for ``--dry-run``) — that encapsulate the per-kind argument
mapping. Both ``run_scenario`` and ``--dry-run`` share the registry but not a
single call signature.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from app.analysis.llm import (
    NoveltyResult,
    analyze_articles,
    compress_knowledge_summary,
    generate_initial_knowledge,
    generate_knowledge_update,
)
from app.analysis.prompts import (
    build_knowledge_compress_messages,
    build_knowledge_init_messages,
    build_knowledge_update_messages,
    build_novelty_messages,
)
from app.config import Settings
from app.models import Article, Topic
from app.scraping.rss import compute_article_hash
from evals.recorder import CallRecord, recording_client
from evals.scenario import (
    CapturedCall,
    Expectation,
    ExpectCheck,
    RunArtifact,
    Scenario,
)

RunFn = Callable[[Scenario, Settings], Awaitable[BaseModel]]
BuildFn = Callable[[Scenario, Settings], list[Any]]


# --- scenario -> app objects ---


def _topic(sc: Scenario) -> Topic:
    return Topic(
        name=sc.topic.name,
        description=sc.topic.description,
        confidence_threshold=sc.topic.confidence_threshold,
        relevance_threshold=sc.topic.relevance_threshold,
    )


def _articles(sc: Scenario) -> list[Article]:
    return [
        Article(
            topic_id=1,  # synthetic: no DB on the offline kinds, so FK/uniqueness are moot
            title=a.title,
            url=a.url,
            content_hash=compute_article_hash(a.url, a.title),
            raw_content=a.content,
            source_feed=a.source_feed,
            published_at=a.published,
        )
        for a in sc.articles
    ]


def _update_novelty(sc: Scenario) -> NoveltyResult:
    """Reconstruct the NoveltyResult that the knowledge-update stage consumes."""
    return NoveltyResult(
        has_new_info=True,
        summary=sc.novelty_summary or "",
        key_facts=sc.key_facts,
        confidence=1.0,
    )


# --- the registry ---


@dataclass(frozen=True)
class _KindAdapter:
    run: RunFn
    build: BuildFn


KIND_DISPATCH: dict[str, _KindAdapter] = {
    "novelty": _KindAdapter(
        run=lambda sc, s: analyze_articles(_articles(sc), sc.knowledge_summary, _topic(sc), s),
        build=lambda sc, s: build_novelty_messages(_articles(sc), sc.knowledge_summary, _topic(sc)),
    ),
    "knowledge_init": _KindAdapter(
        run=lambda sc, s: generate_initial_knowledge(_articles(sc), _topic(sc), s),
        build=lambda sc, s: build_knowledge_init_messages(_articles(sc), _topic(sc), s.knowledge_state_max_tokens),
    ),
    "knowledge_update": _KindAdapter(
        run=lambda sc, s: generate_knowledge_update(sc.knowledge_summary, _update_novelty(sc), _topic(sc), s),
        build=lambda sc, s: build_knowledge_update_messages(
            sc.knowledge_summary, sc.novelty_summary or "", sc.key_facts, _topic(sc), s.knowledge_state_max_tokens
        ),
    ),
    "compress": _KindAdapter(
        run=lambda sc, s: compress_knowledge_summary(sc.knowledge_summary, _topic(sc), s),
        build=lambda sc, s: build_knowledge_compress_messages(
            current_summary=sc.knowledge_summary, topic=_topic(sc), max_tokens=s.knowledge_state_max_tokens
        ),
    ),
}


# --- expectations (soft) ---


def _result_text(result: BaseModel) -> str:
    """The human-readable summary text, whichever field the kind exposes."""
    for attr in ("summary", "updated_summary", "compressed_summary"):
        val = getattr(result, attr, None)
        if val:
            return str(val)
    return ""


def _evaluate_expect(expect: Expectation, result: BaseModel) -> list[ExpectCheck]:
    checks: list[ExpectCheck] = []

    def add(check: str, ok: bool, detail: str) -> None:
        checks.append(ExpectCheck(check=check, ok=ok, detail=detail))

    if expect.has_new_info is not None:
        actual = getattr(result, "has_new_info", None)
        add("has_new_info", actual == expect.has_new_info, f"expected {expect.has_new_info}, got {actual}")
    conf = float(getattr(result, "confidence", 0.0) or 0.0)
    if expect.min_confidence is not None:
        add("min_confidence", conf >= expect.min_confidence, f"{conf} >= {expect.min_confidence}")
    if expect.max_confidence is not None:
        add("max_confidence", conf <= expect.max_confidence, f"{conf} <= {expect.max_confidence}")
    if expect.min_relevance is not None:
        rel = float(getattr(result, "relevance", 0.0) or 0.0)
        add("min_relevance", rel >= expect.min_relevance, f"{rel} >= {expect.min_relevance}")
    if expect.summary_contains is not None:
        needle = expect.summary_contains.lower()
        add("summary_contains", needle in _result_text(result).lower(), f"{expect.summary_contains!r} in summary")
    if expect.sufficient_data is not None:
        actual = getattr(result, "sufficient_data", None)
        add("sufficient_data", actual == expect.sufficient_data, f"expected {expect.sufficient_data}, got {actual}")
    return checks


# --- artifact assembly ---


def _to_captured(record: CallRecord) -> CapturedCall:
    return CapturedCall(
        response_model=record.response_model.__name__ if record.response_model else "unknown",
        messages=record.messages,
        raw_parsed=record.parsed.model_dump(mode="json") if isinstance(record.parsed, BaseModel) else {},
        prompt_tokens=record.usage.prompt_tokens,
        completion_tokens=record.usage.completion_tokens,
    )


def build_artifact(
    scenario: Scenario,
    settings: Settings,
    result: BaseModel,
    records: list[CallRecord],
    *,
    created_at: str | None = None,
) -> RunArtifact:
    """Assemble a RunArtifact from a stage result and its captured calls."""
    return RunArtifact(
        name=scenario.name,
        kind=scenario.kind,
        model=settings.llm.model,
        temperature=settings.llm_temperature,
        created_at=created_at or datetime.now(UTC).isoformat(),
        calls=[_to_captured(r) for r in records],
        final=result.model_dump(mode="json"),
        final_error=getattr(result, "error", None),
        expect_results=_evaluate_expect(scenario.expect, result) if scenario.expect else [],
        scenario=scenario,
    )


async def run_scenario(
    scenario: Scenario,
    settings: Settings,
    *,
    inner: Any = None,
    created_at: str | None = None,
) -> RunArtifact:
    """Run one scenario against the (real) LLM with recording; return a RunArtifact.

    ``inner`` is the recorder's inner client — None uses the real one; offline
    tests inject a mock. No DB/HTTP for any of the four offline kinds.
    """
    adapter = KIND_DISPATCH[scenario.kind]
    with recording_client(inner=inner) as records:
        result = await adapter.run(scenario, settings)
    return build_artifact(scenario, settings, result, records, created_at=created_at)
