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
from app.analysis.llm import NoveltyResult
from tests.helpers.stub_llm import _StubCompletion, _StubUsage


def _novelty(**kw: object) -> NoveltyResult:
    base: dict[str, object] = {"has_new_info": True, "summary": "s", "confidence": 0.9}
    base.update(kw)
    return NoveltyResult(**base)  # type: ignore[arg-type]


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
