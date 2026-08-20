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
    """Omitting the mock inner builds a real per-mode client the first time
    ``_get_client`` is actually invoked — proving the no-live-call guarantee is
    load-bearing. With from_litellm patched to raise, the default path raises;
    the injected path does not. (Client construction is now lazy per mode, to
    match production's own per-mode client cache, so this must call
    ``_get_client`` rather than exercise nothing inside the context.)
    """
    import evals.recorder as recorder

    def _boom(*_a: object, **_k: object) -> object:
        raise AssertionError("real client build attempted")

    monkeypatch.setattr(recorder.instructor, "from_litellm", _boom)

    with pytest.raises(AssertionError, match="real client build attempted"), recorder.recording_client():
        llm_mod._get_client(MagicMock())

    # Injecting an inner avoids the real build entirely.
    with recorder.recording_client(inner=MagicMock()):
        llm_mod._get_client(MagicMock())


async def test_recording_client_calls_land_on_the_mode_specific_inner() -> None:
    """AUG-294: production bakes the structured-output mode into a DISTINCT
    client at build time (TOOLS/JSON/MD_JSON), because that mode decides how
    the request is shaped. A recorder that returns one constant proxy for
    every ``_get_client(settings, mode)`` call would route every fallback
    attempt through the same client, silently skipping the mode switch a real
    retry makes. Injecting a ``{mode: mock}`` dict must dispatch each mode to
    its own mock, and each captured record must carry the mode it used.
    """
    import instructor

    from evals.recorder import recording_client

    tools_inner = MagicMock()
    tools_inner.chat.completions.create_with_completion = AsyncMock(return_value=(_novelty(), _StubCompletion()))
    json_inner = MagicMock()
    json_inner.chat.completions.create_with_completion = AsyncMock(
        return_value=(_novelty(summary="from json mode"), _StubCompletion())
    )

    with recording_client(inner={instructor.Mode.TOOLS: tools_inner, instructor.Mode.JSON: json_inner}) as records:
        tools_client = llm_mod._get_client(MagicMock(), instructor.Mode.TOOLS)
        await tools_client.chat.completions.create_with_completion(model="m", response_model=NoveltyResult, messages=[])
        json_client = llm_mod._get_client(MagicMock(), instructor.Mode.JSON)
        result, _ = await json_client.chat.completions.create_with_completion(
            model="m", response_model=NoveltyResult, messages=[]
        )

    assert tools_inner.chat.completions.create_with_completion.await_count == 1
    assert json_inner.chat.completions.create_with_completion.await_count == 1
    assert result.summary == "from json mode"
    assert [r.mode for r in records] == [instructor.Mode.TOOLS, instructor.Mode.JSON]


async def test_recording_client_records_a_rejected_attempt_before_reraising() -> None:
    """A structured-output attempt that a provider rejects (e.g. a forced
    tool_choice a mode-fallback retries past) must not vanish — the recorder
    only appended after success, hiding every attempt but the last (AUG-294).
    """
    from evals.recorder import recording_client

    inner = MagicMock()
    inner.chat.completions.create_with_completion = AsyncMock(side_effect=RuntimeError("rejected: SUPER_SECRET_KEY"))

    with recording_client(inner=inner) as records:
        client = llm_mod._get_client(MagicMock())
        with pytest.raises(RuntimeError, match="rejected"):
            await client.chat.completions.create_with_completion(
                model="m",
                response_model=NoveltyResult,
                messages=[{"role": "user", "content": "hi"}],
                api_key="SUPER_SECRET_KEY",
            )

    assert len(records) == 1
    rec = records[0]
    assert rec.parsed is None
    assert rec.error is not None
    assert "RuntimeError" in rec.error
    assert "SUPER_SECRET_KEY" not in rec.error  # the leaked key in the message is sanitized


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


def _minimal_artifact(name: str = "s"):
    from evals.scenario import RunArtifact, Scenario, ScenarioTopic

    return RunArtifact(
        name=name,
        kind="novelty",
        scenario=Scenario(topic=ScenarioTopic(name="T", description="d")),
    )


def test_save_run_uses_restrictive_permissions(tmp_path) -> None:
    """AUG-297: artifacts embed raw article bodies, prompts, and topic
    instructions; an ambient umask must not make them world/group readable
    on a shared self-hosting machine."""
    import stat

    from evals.scenario import save_run

    runs = tmp_path / "runs"
    path = save_run(_minimal_artifact(), runs)

    dir_mode = stat.S_IMODE(runs.stat().st_mode)
    file_mode = stat.S_IMODE(path.stat().st_mode)
    assert dir_mode == 0o700
    assert file_mode == 0o600


def test_dump_scenario_uses_restrictive_permissions(tmp_path) -> None:
    """The same private writer applies to frozen scenarios (--freeze), which
    also carry article bodies and topic instructions (AUG-297)."""
    import stat

    from evals.scenario import Scenario, ScenarioTopic, dump_scenario

    p = tmp_path / "frozen.yml"
    dump_scenario(Scenario(topic=ScenarioTopic(name="T", description="d")), p)

    assert stat.S_IMODE(p.stat().st_mode) == 0o600


def test_save_run_rejects_path_traversal_via_scenario_name(tmp_path) -> None:
    """AUG-298: Scenario.name is attacker/author-controlled (a hand-authored
    YAML `name:` field), and must never let a run escape runs_dir."""
    from evals.scenario import save_run

    runs = tmp_path / "runs"
    path = save_run(_minimal_artifact(name="../../etc/passwd"), runs)

    assert path.resolve().parent == runs.resolve()
    assert ".." not in path.name


def test_save_run_slugifies_absolute_scenario_name(tmp_path) -> None:
    from evals.scenario import save_run

    runs = tmp_path / "runs"
    path = save_run(_minimal_artifact(name="/etc/passwd"), runs)

    assert path.resolve().parent == runs.resolve()


def test_save_run_prunes_old_artifacts_beyond_cap(tmp_path, monkeypatch) -> None:
    """AUG-298: unbounded artifact storage — cap saved runs so an unattended
    habit of eval invocations does not grow runs_dir without bound."""
    import evals.scenario as scenario_mod

    monkeypatch.setattr(scenario_mod, "_MAX_SAVED_RUNS", 3)
    runs = tmp_path / "runs"
    for i in range(5):
        scenario_mod.save_run(_minimal_artifact(name=f"s{i}"), runs)

    assert len(list(runs.glob("*.json"))) == 3


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


async def test_run_scenario_novelty_evaluates_min_importance() -> None:
    from evals.runner import run_scenario
    from evals.scenario import Expectation, Scenario, ScenarioArticle, ScenarioTopic

    inner = _mock_inner(novelty=_novelty(importance=2))
    sc = Scenario(
        kind="novelty",
        topic=ScenarioTopic(name="T", description="d"),
        articles=[ScenarioArticle(title="a", url="http://x", content="c", source_feed="http://f")],
        expect=Expectation(min_importance=4),
        name="s",
    )
    art = await run_scenario(sc, _settings(), inner=inner)

    oks = {c.check: c.ok for c in art.expect_results}
    assert oks["min_importance"] is False  # 2 < 4


async def test_scenario_novelty_instruction_reaches_the_prompt() -> None:
    """A scenario's novelty instruction must land in the built prompt.

    Without it a frozen scenario replays against a different prompt than
    production sent, so an instruction regression is invisible here.
    """
    from evals.runner import run_scenario
    from evals.scenario import Scenario, ScenarioArticle, ScenarioTopic

    inner = _mock_inner(novelty=_novelty())
    sc = Scenario(
        kind="novelty",
        topic=ScenarioTopic(name="T", description="d", novelty_instruction="Official announcements only."),
        articles=[ScenarioArticle(title="a", url="http://x", content="c", source_feed="http://f")],
        name="s",
    )
    art = await run_scenario(sc, _settings(), inner=inner)

    prompt = "\n".join(str(m.get("content", "")) for m in art.calls[0].messages)
    assert "Official announcements only." in prompt


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


async def test_run_scenario_error_skips_expectation_match() -> None:
    """AUG-296: a swallowed LLM failure must not be reportable as MATCH.

    The safe default is has_new_info=False, which numerically satisfies an
    `expect: {has_new_info: false}` — but that "match" would be a lie: the
    model never actually judged anything. An errored run must collapse to a
    single failing execution check instead of the normal per-field checks.
    """
    from evals.runner import run_scenario
    from evals.scenario import Expectation, Scenario, ScenarioArticle, ScenarioTopic

    inner = MagicMock()
    inner.chat.completions.create_with_completion = AsyncMock(side_effect=RuntimeError("provider down"))
    sc = Scenario(
        kind="novelty",
        topic=ScenarioTopic(name="T", description="d"),
        articles=[ScenarioArticle(title="a", url="http://x", content="c", source_feed="http://f")],
        expect=Expectation(has_new_info=False),
        name="s",
    )
    art = await run_scenario(sc, _settings(), inner=inner)

    assert art.final_error is not None
    assert len(art.expect_results) == 1
    check = art.expect_results[0]
    assert check.ok is False
    assert check.check != "has_new_info"  # not evaluated as a normal field match


async def test_run_scenario_raising_kind_returns_evidence_instead_of_raising() -> None:
    """knowledge_init/update/compress raise on failure (production invariant), but
    the harness must still finalize an artifact as evidence (AUG-296) rather than
    losing the run entirely to an uncaught traceback."""
    from evals.runner import run_scenario
    from evals.scenario import Scenario, ScenarioArticle, ScenarioTopic

    inner = MagicMock()
    inner.chat.completions.create_with_completion = AsyncMock(side_effect=RuntimeError("provider down"))
    sc = Scenario(
        kind="knowledge_init",
        topic=ScenarioTopic(name="T", description="d"),
        articles=[ScenarioArticle(title="a", url="http://x", content="c", source_feed="http://f")],
        name="s",
    )
    art = await run_scenario(sc, _settings(), inner=inner)  # must not raise

    assert art.final is None
    assert art.final_error is not None
    assert "provider down" in art.final_error


# --- __main__: exit codes (AUG-296) ---


def test_exit_code_zero_for_clean_match() -> None:
    from evals.__main__ import _exit_code
    from evals.scenario import ExpectCheck, RunArtifact, Scenario, ScenarioTopic

    art = RunArtifact(
        name="s",
        kind="novelty",
        final={"has_new_info": False},
        expect_results=[ExpectCheck(check="has_new_info", ok=True, detail="")],
        scenario=Scenario(topic=ScenarioTopic(name="T", description="d")),
    )
    assert _exit_code(art, strict=False) == 0
    assert _exit_code(art, strict=True) == 0


def test_exit_code_nonzero_for_final_error_even_without_strict() -> None:
    from evals.__main__ import _exit_code
    from evals.scenario import ExpectCheck, RunArtifact, Scenario, ScenarioTopic

    art = RunArtifact(
        name="s",
        kind="novelty",
        final={"has_new_info": False},
        final_error="RuntimeError: boom",
        expect_results=[ExpectCheck(check="execution", ok=False, detail="")],
        scenario=Scenario(topic=ScenarioTopic(name="T", description="d")),
    )
    assert _exit_code(art, strict=False) == 1
    assert _exit_code(art, strict=True) == 1


def test_exit_code_mismatch_nonzero_only_when_strict() -> None:
    from evals.__main__ import _exit_code
    from evals.scenario import ExpectCheck, RunArtifact, Scenario, ScenarioTopic

    art = RunArtifact(
        name="s",
        kind="novelty",
        final={"has_new_info": True},
        expect_results=[ExpectCheck(check="has_new_info", ok=False, detail="")],
        scenario=Scenario(topic=ScenarioTopic(name="T", description="d")),
    )
    assert _exit_code(art, strict=False) == 0
    assert _exit_code(art, strict=True) == 1


def test_exit_code_replay_diff_nonzero_only_when_strict() -> None:
    from evals.__main__ import _exit_code
    from evals.scenario import RunArtifact, Scenario, ScenarioTopic

    art = RunArtifact(
        name="s",
        kind="novelty",
        final={"has_new_info": True},
        scenario=Scenario(topic=ScenarioTopic(name="T", description="d")),
    )
    assert _exit_code(art, strict=False, diff=["final.has_new_info: False -> True"]) == 0
    assert _exit_code(art, strict=True, diff=["final.has_new_info: False -> True"]) == 1
    assert _exit_code(art, strict=True, diff=[]) == 0


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


async def test_run_live_rejects_knowledge_update_kind(tmp_path, monkeypatch) -> None:
    """AUG-047: a live fetch has no real NoveltyResult to update from — the
    live scenario builder never populates novelty_summary/key_facts, so
    `live --kind knowledge_update` used to fabricate has_new_info=True with a
    blank summary and evaluate input production never sends, while consuming
    a billed call. It must be rejected before any prod read or LLM call, even
    though a matching topic genuinely exists in prod."""
    import evals.runner as runner
    from app.crud import create_topic
    from app.database import get_connection, init_db
    from app.models import Topic, TopicStatus
    from evals.runner import LiveError

    prod = tmp_path / "prod.db"
    init_db(prod)
    conn = get_connection(prod)
    create_topic(
        conn,
        Topic(name="Acme", description="track acme", feed_urls=["http://feed"], status=TopicStatus.READY),
    )
    conn.commit()
    conn.close()

    async def fail_fetch(*_a: object, **_kw: object) -> object:
        raise AssertionError("fetch attempted despite unsupported kind")

    monkeypatch.setattr(runner, "fetch_new_articles_for_topic", fail_fetch)

    inner = MagicMock()
    inner.chat.completions.create_with_completion = AsyncMock(side_effect=AssertionError("billed call attempted"))

    with pytest.raises(LiveError):
        await runner.run_live("Acme", _settings(), kind="knowledge_update", inner=inner, prod_db_path=prod)

    inner.chat.completions.create_with_completion.assert_not_awaited()


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


async def test_run_live_passes_exa_settings_to_fetch(tmp_path, monkeypatch) -> None:
    """AUG-046: production passes settings.exa into fetch_new_articles_for_topic
    (app/checker.py); run_live omitted it, so the Exa branch always received
    None and an Exa-mode live/frozen eval used an empty article corpus."""
    import evals.runner as runner
    from app.crud import create_topic
    from app.database import get_connection, init_db
    from app.models import Topic, TopicStatus
    from app.scraping import FetchResult

    prod = tmp_path / "prod.db"
    init_db(prod)
    conn = get_connection(prod)
    create_topic(
        conn,
        Topic(name="Acme", description="track acme", feed_urls=["http://feed"], status=TopicStatus.READY),
    )
    conn.commit()
    conn.close()

    captured_kwargs: dict[str, object] = {}

    async def fake_fetch(topic: Topic, conn, **kwargs: object) -> FetchResult:
        captured_kwargs.update(kwargs)
        return FetchResult(articles=[], total_feed_entries=0)

    monkeypatch.setattr(runner, "fetch_new_articles_for_topic", fake_fetch)

    settings = _settings()
    await runner.run_live("Acme", settings, kind="novelty", inner=_mock_inner(novelty=_novelty()), prod_db_path=prod)

    assert captured_kwargs.get("exa_settings") is settings.exa


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
        Topic(
            name="Acme Corp",
            description="track acme",
            feed_urls=["http://feed"],
            status=TopicStatus.READY,
            novelty_instruction="Official announcements only.",
        ),
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
    # The frozen scenario must carry every prompt-affecting topic field, or a replay
    # builds a different prompt than the live run did.
    assert sc.topic.novelty_instruction == "Official announcements only."
    assert sc.articles[0].title == "fetched"
    assert sc.articles[0].content == "live body"


# --- __main__: rendering, nonce normalization, replay diff ---


def test_normalize_nonce_collapses_fence_hex() -> None:
    from evals.__main__ import normalize_nonce

    a = "--- BEGIN UNTRUSTED ARTICLE CONTENT a1b2c3d4e5f60718\nbody\n--- END UNTRUSTED ARTICLE CONTENT a1b2c3d4e5f60718"
    b = "--- BEGIN UNTRUSTED ARTICLE CONTENT 99aa88bb77cc66dd\nbody\n--- END UNTRUSTED ARTICLE CONTENT 99aa88bb77cc66dd"
    assert normalize_nonce(a) == normalize_nonce(b)
    assert "<nonce>" in normalize_nonce(a)


def _artifact_with_nonce(final: dict, nonce: str):
    from evals.scenario import CapturedCall, RunArtifact, Scenario, ScenarioTopic

    fenced = f"--- BEGIN UNTRUSTED ARTICLE CONTENT {nonce}\nx\n--- END UNTRUSTED ARTICLE CONTENT {nonce}"
    return RunArtifact(
        name="s",
        kind="novelty",
        final=final,
        calls=[
            CapturedCall(
                response_model="NoveltyResult",
                messages=[{"role": "user", "content": fenced}],
                raw_parsed={},
            )
        ],
        scenario=Scenario(topic=ScenarioTopic(name="T", description="d")),
    )


def test_diff_runs_ignores_nonce_and_reports_final_change() -> None:
    from evals.__main__ import diff_runs

    a = _artifact_with_nonce({"has_new_info": True, "confidence": 0.9}, "aaaa1111bbbb2222")
    a2 = _artifact_with_nonce({"has_new_info": True, "confidence": 0.9}, "cccc3333dddd4444")
    assert diff_runs(a, a2) == []  # only the per-call nonce differs -> no spurious diff

    b = _artifact_with_nonce({"has_new_info": False, "confidence": 0.9}, "cccc3333dddd4444")
    diff = diff_runs(a, b)
    assert any("has_new_info" in line for line in diff)


def test_diff_runs_detects_material_changes_behind_an_unchanged_final() -> None:
    """diff_runs must not report equivalence when raw output, model, token usage,
    or expectation verdicts diverge even though `final` is byte-identical — the
    exact silent-regression case AUG-048 describes (e.g. a schema/model/cost
    change hidden behind an unchanged post-filtered result)."""
    from evals.__main__ import diff_runs
    from evals.scenario import CapturedCall, ExpectCheck, RunArtifact, Scenario, ScenarioTopic

    def _art(**overrides: object) -> RunArtifact:
        base: dict[str, object] = dict(
            name="s",
            kind="novelty",
            model="openai/gpt-4o-mini",
            temperature=0.2,
            final={"has_new_info": True, "confidence": 0.9},
            calls=[
                CapturedCall(
                    response_model="NoveltyResult",
                    messages=[{"role": "user", "content": "hi"}],
                    raw_parsed={"has_new_info": True, "source_urls": ["http://safe"]},
                    prompt_tokens=10,
                    completion_tokens=5,
                )
            ],
            expect_results=[ExpectCheck(check="has_new_info", ok=True, detail="")],
            scenario=Scenario(topic=ScenarioTopic(name="T", description="d")),
        )
        base.update(overrides)
        return RunArtifact(**base)  # type: ignore[arg-type]

    old = _art()

    # Raw parsed output became unsafe (a smuggled source_url) though final is unchanged.
    unsafe_raw = _art(
        calls=[
            CapturedCall(
                response_model="NoveltyResult",
                messages=[{"role": "user", "content": "hi"}],
                raw_parsed={"has_new_info": True, "source_urls": ["http://safe", "http://evil"]},
                prompt_tokens=10,
                completion_tokens=5,
            )
        ]
    )
    assert diff_runs(old, unsafe_raw) != []

    # Model changed.
    assert diff_runs(old, _art(model="openai/gpt-4o")) != []

    # Token usage spiked.
    diff = diff_runs(
        old,
        _art(
            calls=[
                CapturedCall(
                    response_model="NoveltyResult",
                    messages=[{"role": "user", "content": "hi"}],
                    raw_parsed={"has_new_info": True, "source_urls": ["http://safe"]},
                    prompt_tokens=9000,
                    completion_tokens=5,
                )
            ]
        ),
    )
    assert diff != []

    # Expectation verdict flipped even though `final` and raw output match.
    diff = diff_runs(old, _art(expect_results=[ExpectCheck(check="has_new_info", ok=False, detail="")]))
    assert diff != []


def test_render_artifact_surfaces_error_and_kind() -> None:
    from evals.__main__ import render_artifact
    from evals.scenario import CapturedCall, RunArtifact, Scenario, ScenarioTopic

    art = RunArtifact(
        name="s",
        kind="novelty",
        final={"has_new_info": False},
        final_error="RuntimeError: boom",
        calls=[
            CapturedCall(response_model="NoveltyResult", messages=[{"role": "system", "content": "sys"}], raw_parsed={})
        ],
        scenario=Scenario(topic=ScenarioTopic(name="T", description="d")),
    )
    out = render_artifact(art)
    assert "boom" in out
    assert "novelty" in out


async def test_replay_reruns_scenario_and_diffs(tmp_path) -> None:
    from evals.__main__ import replay
    from evals.runner import run_scenario
    from evals.scenario import Scenario, ScenarioArticle, ScenarioTopic, save_run

    sc = Scenario(
        kind="novelty",
        topic=ScenarioTopic(name="T", description="d"),
        articles=[ScenarioArticle(title="a", url="http://x", content="c", source_feed="http://f")],
        name="s",
    )
    old = await run_scenario(sc, _settings(), inner=_mock_inner(novelty=_novelty(has_new_info=True, confidence=0.9)))
    run_path = save_run(old, tmp_path / "runs")

    new, diff = await replay(
        run_path, _settings(), inner=_mock_inner(novelty=_novelty(has_new_info=False, confidence=0.2))
    )
    assert new.final is not None
    assert new.final["has_new_info"] is False
    assert any("has_new_info" in line for line in diff)
