"""Offline tests for the evals harness (no live LLM calls).

The LLM-network guarantee rests on THIS file's seams, not on conftest: the
autouse ``_stub_dns_resolution`` fixture only blocks SSRF DNS, not LLM calls.
Every test here either injects a mock inner client into ``recording_client`` or
patches ``instructor.from_litellm`` to raise, so an accidental real build is a
test failure rather than a billed network round-trip.
"""

from __future__ import annotations

import textwrap
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.analysis import llm as llm_mod
from app.analysis.llm import CompressedKnowledge, KnowledgeStateUpdate, NoveltyResult
from app.config import LLMSettings, Settings
from tests.helpers.stub_llm import _StubCompletion, _StubUsage


def _novelty(**kw: object) -> NoveltyResult:
    base: dict[str, object] = {"has_new_info": True, "summary": "s", "confidence": 0.9}
    base.update(kw)
    return NoveltyResult(**base)  # type: ignore[arg-type]


def _settings() -> Settings:
    """An offline-safe Settings (no file read, no real key)."""
    return Settings(llm=LLMSettings(model="openai/gpt-4o-mini", api_key="test-key-not-real"))


def _mock_inner(
    *,
    novelty: NoveltyResult | None = None,
    knowledge: KnowledgeStateUpdate | None = None,
    compressed: CompressedKnowledge | None = None,
) -> MagicMock:
    """A mock inner client that dispatches canned results on response_model."""

    async def _cwc(**kwargs: object) -> tuple[object, object]:
        rm = kwargs.get("response_model")
        if rm is NoveltyResult:
            parsed: object = novelty
        elif rm is CompressedKnowledge:
            parsed = compressed
        else:
            parsed = knowledge
        return parsed, _StubCompletion()

    inner = MagicMock()
    inner.chat.completions.create_with_completion = AsyncMock(side_effect=_cwc)
    return inner


# --- recorder ---


async def test_recording_client_captures_messages_model_response_model_and_usage() -> None:
    from evals.recorder import recording_client

    parsed = _novelty(key_facts=["a"])
    completion = _StubCompletion(_StubUsage(prompt_tokens=11, completion_tokens=7))
    inner = MagicMock()
    inner.chat.completions.create_with_completion = AsyncMock(return_value=(parsed, completion))

    with recording_client(inner=inner) as records:
        client = llm_mod._get_client(MagicMock())  # patched to return the recording proxy
        result, comp = await client.chat.completions.create_with_completion(
            model="some/model",
            response_model=NoveltyResult,
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.2,
            api_key="SUPER_SECRET_KEY",
        )

    assert result is parsed and comp is completion  # passthrough, unchanged
    assert len(records) == 1
    rec = records[0]
    assert rec.response_model is NoveltyResult
    assert rec.messages == [{"role": "user", "content": "hi"}]
    assert rec.model == "some/model"
    assert rec.temperature == 0.2
    assert rec.usage.prompt_tokens == 11
    assert rec.usage.completion_tokens == 7
    # The api_key must never be captured anywhere on the record.
    assert "SUPER_SECRET_KEY" not in repr(rec)


async def test_recorded_parsed_is_snapshot_immune_to_later_mutation() -> None:
    """analyze_articles mutates the parsed result in place (filters key_facts).

    The record must hold the RAW parsed state so raw-vs-final divergence is
    visible, so it deep-copies the parsed model at capture time.
    """
    from evals.recorder import recording_client

    parsed = _novelty(key_facts=["x", "y"], source_urls=["http://kept"])
    inner = MagicMock()
    inner.chat.completions.create_with_completion = AsyncMock(return_value=(parsed, _StubCompletion()))

    with recording_client(inner=inner) as records:
        client = llm_mod._get_client(MagicMock())
        result, _ = await client.chat.completions.create_with_completion(
            model="m", response_model=NoveltyResult, messages=[], temperature=0.2
        )
        # Simulate analyze_articles' post-call mutation of the same object.
        result.key_facts = []
        result.source_urls = []

    assert records[0].parsed.key_facts == ["x", "y"]
    assert records[0].parsed.source_urls == ["http://kept"]


def test_recording_client_builds_real_inner_when_none_injected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Omitting the mock inner builds the real client — proving the no-live-call
    guarantee is load-bearing. With from_litellm patched to raise, the default
    path raises; the injected path does not.
    """
    import evals.recorder as recorder

    def _boom(*_a: object, **_k: object) -> object:
        raise AssertionError("real client build attempted")

    monkeypatch.setattr(recorder.instructor, "from_litellm", _boom)

    with pytest.raises(AssertionError, match="real client build attempted"), recorder.recording_client():
        pass

    # Injecting an inner avoids the real build entirely.
    with recorder.recording_client(inner=MagicMock()):
        pass


# --- scenario + RunArtifact ---


def test_scenario_yaml_round_trip_preserves_published(tmp_path) -> None:
    from evals.scenario import (
        Scenario,
        ScenarioArticle,
        ScenarioTopic,
        dump_scenario,
        load_scenario,
    )

    sc = Scenario(
        kind="novelty",
        topic=ScenarioTopic(name="Acme", description="track acme", confidence_threshold=0.7),
        knowledge_summary="known state",
        articles=[
            ScenarioArticle(
                title="t1",
                url="http://a",
                content="body one",
                published=datetime(2025, 1, 15, 12, 0, tzinfo=UTC),
                source_feed="http://feed",
            )
        ],
        name="myscen",
    )
    p = tmp_path / "myscen.yml"
    dump_scenario(sc, p)
    loaded = load_scenario(p)

    assert loaded.kind == "novelty"
    assert loaded.topic.name == "Acme"
    assert loaded.topic.confidence_threshold == 0.7
    assert loaded.knowledge_summary == "known state"
    assert len(loaded.articles) == 1
    assert loaded.articles[0].content == "body one"
    assert loaded.articles[0].published == datetime(2025, 1, 15, 12, 0, tzinfo=UTC)
    assert loaded.name == "myscen"  # derived from filename stem


def test_load_scenario_parses_handauthored_yaml(tmp_path) -> None:
    from evals.scenario import load_scenario

    p = tmp_path / "dup_event.yml"
    p.write_text(
        textwrap.dedent(
            """
            kind: novelty
            topic:
              name: "Acme"
              description: "track acme funding"
            knowledge_summary: "Acme raised a $5M Series A."
            articles:
              - title: "dup"
                url: "http://a"
                content: "Acme closed a $5M round."
                published: "2025-01-15T12:00:00Z"
                source_feed: "http://feed"
            expect:
              has_new_info: false
              min_confidence: 0.6
              summary_contains: "no new"
            """
        )
    )
    sc = load_scenario(p)

    assert sc.name == "dup_event"
    assert sc.kind == "novelty"
    assert sc.expect is not None
    assert sc.expect.has_new_info is False
    assert sc.expect.min_confidence == 0.6
    assert sc.expect.summary_contains == "no new"
    assert sc.articles[0].published is not None
    assert sc.articles[0].published.year == 2025


def test_run_artifact_save_load_round_trip(tmp_path) -> None:
    from evals.scenario import (
        CapturedCall,
        RunArtifact,
        Scenario,
        ScenarioTopic,
        load_run,
        save_run,
    )

    art = RunArtifact(
        name="s",
        kind="novelty",
        model="some/model",
        temperature=0.2,
        created_at="2025-01-01T00:00:00+00:00",
        calls=[
            CapturedCall(
                response_model="NoveltyResult",
                messages=[{"role": "user", "content": "hi"}],
                raw_parsed={"has_new_info": True, "key_facts": ["x"]},
                prompt_tokens=5,
                completion_tokens=3,
            )
        ],
        final={"has_new_info": True, "key_facts": []},
        final_error=None,
        scenario=Scenario(topic=ScenarioTopic(name="T", description="d")),
    )
    runs = tmp_path / "runs"
    path = save_run(art, runs)

    assert path.exists()
    assert path.parent == runs
    loaded = load_run(path)
    assert loaded.name == "s"
    assert loaded.model == "some/model"
    assert loaded.calls[0].raw_parsed == {"has_new_info": True, "key_facts": ["x"]}
    assert loaded.final == {"has_new_info": True, "key_facts": []}
    assert loaded.scenario.topic.name == "T"


# --- runner: run_scenario ---


@pytest.mark.parametrize("kind", ["novelty", "knowledge_init", "knowledge_update", "compress"])
async def test_run_scenario_dispatches_all_kinds(kind: str) -> None:
    from evals.runner import run_scenario
    from evals.scenario import Scenario, ScenarioArticle, ScenarioTopic

    inner = _mock_inner(
        novelty=_novelty(),
        knowledge=KnowledgeStateUpdate(sufficient_data=True, confidence=0.9, updated_summary="us"),
        compressed=CompressedKnowledge(compressed_summary="cs"),
    )
    sc = Scenario(
        kind=kind,  # type: ignore[arg-type]
        topic=ScenarioTopic(name="T", description="d"),
        knowledge_summary="known state here",
        novelty_summary="a new finding",
        key_facts=["kf one"],
        articles=[ScenarioArticle(title="a", url="http://x", content="body", source_feed="http://f")],
        name="s",
    )
    art = await run_scenario(sc, _settings(), inner=inner)

    assert art.kind == kind
    assert len(art.calls) == 1
    assert art.calls[0].messages  # real built prompt captured
    assert art.final is not None
    assert art.model == "openai/gpt-4o-mini"
    assert art.temperature == 0.2


async def test_run_scenario_novelty_evaluates_expectations() -> None:
    from evals.runner import run_scenario
    from evals.scenario import Expectation, Scenario, ScenarioArticle, ScenarioTopic

    inner = _mock_inner(
        novelty=_novelty(has_new_info=False, summary="nothing new here", confidence=0.85, relevance=0.6)
    )
    sc = Scenario(
        kind="novelty",
        topic=ScenarioTopic(name="T", description="d"),
        knowledge_summary="ks",
        articles=[ScenarioArticle(title="a", url="http://x", content="c", source_feed="http://f")],
        expect=Expectation(has_new_info=False, min_confidence=0.7, summary_contains="nothing"),
        name="s",
    )
    art = await run_scenario(sc, _settings(), inner=inner)

    oks = {c.check: c.ok for c in art.expect_results}
    assert oks["has_new_info"] is True
    assert oks["min_confidence"] is True  # 0.85 >= 0.7
    assert oks["summary_contains"] is True


async def test_run_scenario_expectation_mismatch_is_reported_not_raised() -> None:
    from evals.runner import run_scenario
    from evals.scenario import Expectation, Scenario, ScenarioArticle, ScenarioTopic

    inner = _mock_inner(novelty=_novelty(has_new_info=True, summary="big news", confidence=0.4))
    sc = Scenario(
        kind="novelty",
        topic=ScenarioTopic(name="T", description="d"),
        articles=[ScenarioArticle(title="a", url="http://x", content="c", source_feed="http://f")],
        expect=Expectation(has_new_info=False, min_confidence=0.7),
        name="s",
    )
    art = await run_scenario(sc, _settings(), inner=inner)  # must not raise

    oks = {c.check: c.ok for c in art.expect_results}
    assert oks["has_new_info"] is False  # expected False, got True
    assert oks["min_confidence"] is False  # 0.4 < 0.7


async def test_run_scenario_captures_raw_vs_final_divergence() -> None:
    """The recorder snapshots the raw parsed result; the final reflects
    analyze_articles' post-filtering. A smuggled source_url is dropped from final
    but visible in the raw capture."""
    from evals.runner import run_scenario
    from evals.scenario import Scenario, ScenarioArticle, ScenarioTopic

    inner = _mock_inner(novelty=_novelty(source_urls=["http://evil", "http://x"]))
    sc = Scenario(
        kind="novelty",
        topic=ScenarioTopic(name="T", description="d"),
        articles=[ScenarioArticle(title="a", url="http://x", content="c", source_feed="http://f")],
        name="s",
    )
    art = await run_scenario(sc, _settings(), inner=inner)

    assert art.calls[0].raw_parsed["source_urls"] == ["http://evil", "http://x"]
    assert art.final is not None
    assert art.final["source_urls"] == ["http://x"]  # injected URL filtered out


async def test_run_scenario_surfaces_swallowed_llm_error() -> None:
    """analyze_articles swallows LLM failures into NoveltyResult.error; the
    artifact must surface it so a failure isn't mistaken for 'nothing new'."""
    from evals.runner import run_scenario
    from evals.scenario import Scenario, ScenarioArticle, ScenarioTopic

    inner = MagicMock()
    inner.chat.completions.create_with_completion = AsyncMock(side_effect=RuntimeError("boom"))
    sc = Scenario(
        kind="novelty",
        topic=ScenarioTopic(name="T", description="d"),
        articles=[ScenarioArticle(title="a", url="http://x", content="c", source_feed="http://f")],
        name="s",
    )
    art = await run_scenario(sc, _settings(), inner=inner)  # safe default, no raise

    assert art.final is not None
    assert art.final["has_new_info"] is False
    assert art.final_error is not None
    assert "boom" in art.final_error


# --- runner: run_live (prod read-only + scratch isolation) ---


def test_open_readonly_blocks_writes(tmp_path) -> None:
    import sqlite3

    from app.database import init_db
    from evals.runner import _open_readonly

    db = tmp_path / "prod.db"
    init_db(db)
    ro = _open_readonly(db)
    try:
        with pytest.raises(sqlite3.OperationalError):
            ro.execute(
                "INSERT INTO topics (name, description, feed_urls, feed_mode, created_at, "
                "is_active, status, init_attempts) VALUES "
                "('x', 'y', '[]', 'auto', '2025-01-01T00:00:00+00:00', 1, 'ready', 0)"
            )
    finally:
        ro.close()


def test_open_readonly_raises_live_error_on_missing_db(tmp_path) -> None:
    from evals.runner import LiveError, _open_readonly

    with pytest.raises(LiveError):
        _open_readonly(tmp_path / "does-not-exist.db")


async def test_run_live_uses_scratch_topic_and_reads_prod_readonly(tmp_path, monkeypatch) -> None:
    import evals.runner as runner
    from app.crud import create_topic
    from app.database import get_connection, init_db
    from app.models import Article, Topic, TopicStatus
    from app.scraping import FetchResult

    # Prod DB: a filler topic (id=1) then the target (id=2) so prod id != scratch id.
    prod = tmp_path / "prod.db"
    init_db(prod)
    conn = get_connection(prod)
    create_topic(conn, Topic(name="filler", description="f", feed_urls=[]))
    target = create_topic(
        conn,
        Topic(name="Acme", description="track acme", feed_urls=["http://feed"], status=TopicStatus.READY),
    )
    conn.commit()
    conn.close()
    prod_target_id = target.id
    assert prod_target_id == 2

    captured: dict[str, object] = {}

    async def fake_fetch(topic: Topic, conn, **_kw: object) -> FetchResult:
        captured["topic_id"] = topic.id
        art = Article(
            topic_id=topic.id,  # type: ignore[arg-type]
            title="fetched",
            url="http://x",
            content_hash="h",
            source_feed="http://feed",
            raw_content="live body",
        )
        return FetchResult(articles=[art], total_feed_entries=1)

    monkeypatch.setattr(runner, "fetch_new_articles_for_topic", fake_fetch)

    art = await runner.run_live(
        "Acme", _settings(), kind="novelty", inner=_mock_inner(novelty=_novelty()), prod_db_path=prod
    )

    # fetch ran against the SCRATCH topic (fresh id=1), not the prod-loaded id=2.
    assert captured["topic_id"] == 1
    assert captured["topic_id"] != prod_target_id
    assert len(art.calls) == 1
    assert art.scenario.articles[0].title == "fetched"


async def test_run_live_freeze_writes_replayable_scenario(tmp_path, monkeypatch) -> None:
    import evals.runner as runner
    from app.crud import create_topic
    from app.database import get_connection, init_db
    from app.models import Article, Topic, TopicStatus
    from app.scraping import FetchResult
    from evals.scenario import load_scenario

    prod = tmp_path / "prod.db"
    init_db(prod)
    conn = get_connection(prod)
    create_topic(
        conn,
        Topic(name="Acme Corp", description="track acme", feed_urls=["http://feed"], status=TopicStatus.READY),
    )
    conn.commit()
    conn.close()

    async def fake_fetch(topic: Topic, conn, **_kw: object) -> FetchResult:
        art = Article(
            topic_id=topic.id,  # type: ignore[arg-type]
            title="fetched",
            url="http://x",
            content_hash="h",
            source_feed="http://feed",
            raw_content="live body",
        )
        return FetchResult(articles=[art], total_feed_entries=1)

    monkeypatch.setattr(runner, "fetch_new_articles_for_topic", fake_fetch)

    freeze = tmp_path / "frozen.yml"
    await runner.run_live(
        "Acme Corp",
        _settings(),
        inner=_mock_inner(novelty=_novelty()),
        prod_db_path=prod,
        freeze_path=freeze,
    )

    assert freeze.exists()
    sc = load_scenario(freeze)
    assert sc.topic.name == "Acme Corp"
    assert sc.articles[0].title == "fetched"
    assert sc.articles[0].content == "live body"
