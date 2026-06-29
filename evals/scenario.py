"""Scenario inputs, soft expectations, and the RunArtifact — all serializable.

A ``Scenario`` is a self-contained, reproducible input definition for one LLM
stage (hand-authored, or frozen from a live run). A ``RunArtifact`` is one
recorded execution (inputs + captured prompts/results) saved as JSON so a run can
be replayed against the current prompt/code.

Everything here is plain JSON/YAML-friendly data: the runner converts captured
``CallRecord``s (which hold live pydantic objects) into ``CapturedCall``s (plain
dicts) before building a RunArtifact.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

ScenarioKind = Literal["novelty", "knowledge_init", "knowledge_update", "compress"]


class ScenarioTopic(BaseModel):
    """The topic a scenario runs against (becomes an app ``Topic``)."""

    name: str
    description: str
    confidence_threshold: float | None = None
    relevance_threshold: float | None = None


class ScenarioArticle(BaseModel):
    """One article fed to the LLM stage (becomes an app ``Article``)."""

    title: str
    url: str
    content: str = ""
    published: datetime | None = None  # coerced from ISO strings or YAML timestamps
    source_feed: str = "https://eval.local/feed"


class Expectation(BaseModel):
    """Optional soft checks rendered as MATCH/MISMATCH — never a hard gate."""

    has_new_info: bool | None = None
    min_confidence: float | None = None
    max_confidence: float | None = None
    min_relevance: float | None = None
    summary_contains: str | None = None
    sufficient_data: bool | None = None  # knowledge_init / knowledge_update


class Scenario(BaseModel):
    """A reproducible input definition for one LLM stage."""

    kind: ScenarioKind = "novelty"
    topic: ScenarioTopic
    knowledge_summary: str = ""  # current state for novelty / update / compress
    articles: list[ScenarioArticle] = Field(default_factory=list)
    novelty_summary: str | None = None  # knowledge_update: the new finding
    key_facts: list[str] = Field(default_factory=list)  # knowledge_update
    expect: Expectation | None = None
    # Derived from the file stem on load (not part of the YAML body); used to name
    # the RunArtifact. Excluded from dump_scenario output.
    name: str = "scenario"


class CapturedCall(BaseModel):
    """One recorded LLM round-trip, flattened to plain JSON-friendly data."""

    response_model: str
    messages: list[dict[str, Any]]
    raw_parsed: dict[str, Any]
    prompt_tokens: int = 0
    completion_tokens: int = 0


class ExpectCheck(BaseModel):
    """Outcome of one Expectation field check."""

    check: str
    ok: bool
    detail: str = ""


class RunArtifact(BaseModel):
    """One recorded execution: inputs + captured prompts/results + verdicts."""

    name: str
    kind: str
    model: str | None = None
    temperature: float | None = None
    created_at: str = ""
    calls: list[CapturedCall] = Field(default_factory=list)
    final: dict[str, Any] | None = None  # the function-return result (model_dump)
    final_error: str | None = None  # NoveltyResult.error surfaced (swallowed failure)
    expect_results: list[ExpectCheck] = Field(default_factory=list)
    scenario: Scenario  # the inputs, for replay


def load_scenario(path: Path) -> Scenario:
    """Parse a scenario YAML file; ``name`` is taken from the file stem."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Scenario file {path} must be a YAML mapping, got {type(raw).__name__}")
    name = raw.pop("name", None) or Path(path).stem
    return Scenario(name=name, **raw)


def dump_scenario(scenario: Scenario, path: Path) -> None:
    """Write a scenario to YAML (``name`` excluded; datetimes as ISO strings)."""
    data = scenario.model_dump(mode="json", exclude={"name"}, exclude_none=True)
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


def _safe_stamp(created_at: str) -> str:
    """Filename-safe rendering of an ISO timestamp."""
    return created_at.replace(":", "").replace("+", "").replace(".", "") or "run"


def save_run(artifact: RunArtifact, runs_dir: Path) -> Path:
    """Serialize a RunArtifact to ``<runs_dir>/<name>-<stamp>.json``."""
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{artifact.name}-{_safe_stamp(artifact.created_at)}.json"
    path.write_text(artifact.model_dump_json(indent=2))
    return path


def load_run(path: Path) -> RunArtifact:
    """Load a RunArtifact from its JSON file."""
    return RunArtifact.model_validate_json(Path(path).read_text())
