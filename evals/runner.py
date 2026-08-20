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

import re
import sqlite3
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from app import __version__ as _app_version
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
from app.crud import create_topic, get_knowledge_state, get_topic_by_name
from app.database import get_connection, init_db
from app.models import Article, Topic
from app.scraping import fetch_new_articles_for_topic
from app.scraping.rss import compute_article_hash
from evals.recorder import CallRecord, recording_client
from evals.scenario import (
    CapturedCall,
    Expectation,
    ExpectCheck,
    RunArtifact,
    Scenario,
    ScenarioArticle,
    ScenarioTopic,
    dump_scenario,
)


class LiveError(RuntimeError):
    """Raised when a live run cannot proceed (topic missing, prod DB unreadable)."""


RunFn = Callable[[Scenario, Settings], Awaitable[BaseModel]]
BuildFn = Callable[[Scenario, Settings], list[Any]]


# --- scenario -> app objects ---


def _topic(sc: Scenario) -> Topic:
    return Topic(
        name=sc.topic.name,
        description=sc.topic.description,
        confidence_threshold=sc.topic.confidence_threshold,
        relevance_threshold=sc.topic.relevance_threshold,
        novelty_instruction=sc.topic.novelty_instruction,
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


def _evaluate_expect(expect: Expectation, result: BaseModel | None, error: str | None) -> list[ExpectCheck]:
    """Evaluate soft expectation checks against a run's result.

    A run that produced no trustworthy result (``error`` set, e.g. a
    swallowed novelty LLM failure or a caught knowledge-stage exception)
    collapses to a single failing "execution" check instead of the normal
    per-field comparisons — otherwise a safe-default value could coincidentally
    satisfy an expectation and render as MATCH, hiding a provider/harness
    failure behind a green result (AUG-296).
    """
    if error is not None or result is None:
        return [ExpectCheck(check="execution", ok=False, detail=f"run failed, expectations not evaluated: {error}")]
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
    if expect.min_importance is not None:
        imp = int(getattr(result, "importance", 0) or 0)
        add("min_importance", imp >= expect.min_importance, f"{imp} >= {expect.min_importance}")
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
        mode=record.mode.value if record.mode is not None else None,
        error=record.error,
        prompt_tokens=record.usage.prompt_tokens,
        completion_tokens=record.usage.completion_tokens,
    )


def build_artifact(
    scenario: Scenario,
    settings: Settings,
    result: BaseModel | None,
    records: list[CallRecord],
    *,
    error: str | None = None,
    created_at: str | None = None,
) -> RunArtifact:
    """Assemble a RunArtifact from a stage result (or a caught run error) and
    its captured calls. ``error`` takes precedence over a result's own
    ``error`` attribute (e.g. NoveltyResult.error) — both funnel into the
    same ``final_error`` outcome field so callers have one place to check for
    a failed run.
    """
    final_error = error if error is not None else getattr(result, "error", None)
    return RunArtifact(
        code_version=_app_version,
        name=scenario.name,
        kind=scenario.kind,
        model=settings.llm.model,
        temperature=settings.llm_temperature,
        created_at=created_at or datetime.now(UTC).isoformat(),
        calls=[_to_captured(r) for r in records],
        final=result.model_dump(mode="json") if result is not None else None,
        final_error=final_error,
        expect_results=_evaluate_expect(scenario.expect, result, final_error) if scenario.expect else [],
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

    novelty's ``analyze_articles`` never raises (production fail-safe
    invariant) — its failure surfaces as ``result.error``. The other three
    kinds DO raise on failure (production invariant: knowledge init/update are
    critical); the harness catches that here so a harder failure still
    finalizes an artifact as evidence instead of vanishing into an uncaught
    traceback with nothing saved (AUG-296). Either path funnels into
    ``final_error`` via ``build_artifact``.
    """
    adapter = KIND_DISPATCH[scenario.kind]
    result: BaseModel | None = None
    error: str | None = None
    with recording_client(inner=inner) as records:
        try:
            result = await adapter.run(scenario, settings)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    return build_artifact(scenario, settings, result, records, error=error, created_at=created_at)


# --- live run (real fetch, prod read-only, scratch-DB isolation) ---


def _slug(name: str) -> str:
    """Filename-safe slug for naming live RunArtifacts / freeze files."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "live"


def _open_readonly(db_path: str | Path) -> sqlite3.Connection:
    """Open the production DB read-only so any write raises (structural safety).

    A mode=ro open of a WAL database can fail if a -wal is uncommitted under a
    busy server, or on a read-only mount; surface that as a clear LiveError.
    """
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        conn.execute("SELECT 1").fetchone()  # force the -shm map now, fail loudly here
    except sqlite3.OperationalError as exc:
        raise LiveError(
            f"Could not open {db_path} read-only ({exc}). Run `live` only against an "
            "idle database — stop the server first."
        ) from exc
    return conn


def _scenario_from_live(topic: Topic, summary: str, articles: list[Article], kind: str) -> Scenario:
    """Build a reproducible Scenario from a live fetch (basis for --freeze)."""
    return Scenario(
        kind=kind,  # type: ignore[arg-type]
        topic=ScenarioTopic(
            name=topic.name,
            description=topic.description,
            confidence_threshold=topic.confidence_threshold,
            relevance_threshold=topic.relevance_threshold,
            novelty_instruction=topic.novelty_instruction,
        ),
        knowledge_summary=summary,
        articles=[
            ScenarioArticle(
                title=a.title,
                url=a.url,
                content=a.raw_content or "",
                published=a.published_at,
                source_feed=a.source_feed,
            )
            for a in articles
        ],
        name=_slug(topic.name),
    )


# Kinds a live fetch cannot build a faithful scenario for. knowledge_update
# needs a real NoveltyResult (novelty_summary + key_facts) to update FROM;
# nothing in a live fetch produces one, so run_live used to fabricate
# has_new_info=True with a blank summary — an input production never sends,
# evaluated at the cost of a billed call (AUG-047).
_LIVE_UNSUPPORTED_KINDS = frozenset({"knowledge_update"})


async def run_live(
    topic_name: str,
    settings: Settings,
    *,
    kind: str = "novelty",
    inner: Any = None,
    created_at: str | None = None,
    freeze_path: str | Path | None = None,
    prod_db_path: str | Path | None = None,
) -> RunArtifact:
    """Fetch a topic's feeds live and run the LLM stage, without touching prod data.

    The production DB is opened read-only (topic + knowledge load only); all feed
    fetch bookkeeping (articles, feed_health, dedup) happens in a throwaway
    scratch DB in a tempdir. ``inner`` is the recorder's inner client (None ->
    real; tests inject a mock).

    ``kind="knowledge_update"`` is rejected outright (see
    ``_LIVE_UNSUPPORTED_KINDS``): run ``live --kind novelty`` (optionally
    ``--freeze``) first, then hand-author a ``scenario`` YAML with the real
    ``novelty_summary``/``key_facts`` for the update stage.
    """
    if kind in _LIVE_UNSUPPORTED_KINDS:
        raise LiveError(
            f"live --kind {kind} is not supported: a live fetch has no real novelty result to update "
            "from, so it would fabricate one and evaluate input production never sends. Run `live "
            "--kind novelty` first, then hand-author a `scenario` YAML with the real "
            "novelty_summary/key_facts."
        )
    prod = prod_db_path if prod_db_path is not None else settings.db_path
    ro = _open_readonly(prod)
    try:
        topic = get_topic_by_name(ro, topic_name)
        if topic is None or topic.id is None:
            raise LiveError(f"Topic not found: {topic_name!r}")
        knowledge = get_knowledge_state(ro, topic.id)
        summary = knowledge.summary_text if knowledge else ""
    finally:
        ro.close()

    with tempfile.TemporaryDirectory(prefix="evals-scratch-") as tmp:
        scratch_path = Path(tmp) / "scratch.db"
        init_db(scratch_path)
        scratch = get_connection(scratch_path)
        try:
            scratch_topic = create_topic(scratch, topic)  # mutates topic.id -> scratch rowid
            scratch.commit()
            fetch_result = await fetch_new_articles_for_topic(
                scratch_topic,
                scratch,
                max_articles=settings.max_articles_per_check,
                feed_fetch_timeout=settings.feed_fetch_timeout,
                article_fetch_timeout=settings.article_fetch_timeout,
                feed_max_retries=settings.feed_max_retries,
                concurrency=settings.content_fetch_concurrency,
                exa_settings=settings.exa,
            )
            articles = fetch_result.articles
        finally:
            scratch.close()

    scenario = _scenario_from_live(scratch_topic, summary, articles, kind)
    if freeze_path is not None:
        dump_scenario(scenario, Path(freeze_path))
    return await run_scenario(scenario, settings, inner=inner, created_at=created_at)
