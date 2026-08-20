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

import contextlib
import os
import re
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

ScenarioKind = Literal["novelty", "knowledge_init", "knowledge_update", "compress"]

# RunArtifact envelope version. Bump when a field is added/removed/repurposed
# in a way that would make an old artifact's replay diff misleading;
# load_run() rejects a file whose version is newer than this harness
# understands rather than silently loading it under current field defaults
# (AUG-295).
_ARTIFACT_SCHEMA_VERSION = 1

# Cap on saved run artifacts per runs_dir — oldest evicted first. Eval runs
# persist full prompts and article bodies with no other retention, so an
# unattended habit of `scenario`/`live`/`replay` invocations would otherwise
# grow the directory without bound (AUG-298). A module constant, not a
# setting: this is a spool bound, not something a user needs to tune.
_MAX_SAVED_RUNS = 200

# Characters allowed unescaped in a saved artifact's filename stem. Dots are
# deliberately excluded (not just slashes) so a "../../etc/passwd"-style name
# cannot leave a literal ".." sequence behind after collapsing separators.
_UNSAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_-]+")


class ScenarioTopic(BaseModel):
    """The topic a scenario runs against (becomes an app ``Topic``).

    Carries every topic field the prompt builders read, so a frozen scenario
    reproduces the exact prompt production would have sent. ``importance_threshold``
    is deliberately absent: it gates notification sends in the checker and never
    reaches an LLM prompt.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    confidence_threshold: float | None = None
    relevance_threshold: float | None = None
    novelty_instruction: str | None = None


class ScenarioArticle(BaseModel):
    """One article fed to the LLM stage (becomes an app ``Article``)."""

    model_config = ConfigDict(extra="forbid")

    title: str
    url: str
    content: str = ""
    published: datetime | None = None  # coerced from ISO strings or YAML timestamps
    source_feed: str = "https://eval.local/feed"


class Expectation(BaseModel):
    """Optional soft checks rendered as MATCH/MISMATCH — never a hard gate."""

    model_config = ConfigDict(extra="forbid")

    has_new_info: bool | None = None
    min_confidence: float | None = None
    max_confidence: float | None = None
    min_relevance: float | None = None
    min_importance: int | None = None
    summary_contains: str | None = None
    sufficient_data: bool | None = None  # knowledge_init / knowledge_update


class Scenario(BaseModel):
    """A reproducible input definition for one LLM stage."""

    # Input-side models forbid unknown keys (AUG-050): a typo'd scenario/topic/
    # article/expectation key would otherwise be silently discarded under
    # Pydantic's default forward-compatible policy, quietly dropping the check
    # its author meant to run before a billed LLM call. RunArtifact and its
    # nested output models stay tolerant — they're read back, not authored.
    model_config = ConfigDict(extra="forbid")

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
    """One recorded LLM round-trip, flattened to plain JSON-friendly data.

    ``mode`` is the structured-output mode (TOOLS/JSON/MD_JSON) the attempt
    used; ``error`` is set instead of ``raw_parsed`` being meaningful when the
    provider rejected this specific attempt (a fallback retry may still have
    produced a later, successful call).
    """

    response_model: str
    messages: list[dict[str, Any]]
    raw_parsed: dict[str, Any]
    mode: str | None = None
    error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


class ExpectCheck(BaseModel):
    """Outcome of one Expectation field check."""

    check: str
    ok: bool
    detail: str = ""


class RunArtifact(BaseModel):
    """One recorded execution: inputs + captured prompts/results + verdicts.

    ``schema_version`` gates ``load_run``; ``run_id`` is this run's stable
    identity; ``replay_parent`` (set by ``evals.__main__.replay``) links a
    replay back to the run_id it replayed, so a chain of replays traces back
    to its origin. ``code_version`` is the app version captured under
    (``build_artifact``), so a replay diff can be attributed to a code change
    vs a live model/provider change (AUG-295).
    """

    schema_version: int = _ARTIFACT_SCHEMA_VERSION
    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    replay_parent: str | None = None
    code_version: str | None = None
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
    """Write a scenario to YAML (``name`` excluded; datetimes as ISO strings).

    Frozen scenarios carry article bodies and topic instructions, so this
    goes through the same private writer as ``save_run`` (AUG-297).
    """
    data = scenario.model_dump(mode="json", exclude={"name"}, exclude_none=True)
    _write_private(Path(path), yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


def _safe_stamp(created_at: str) -> str:
    """Filename-safe rendering of an ISO timestamp."""
    return created_at.replace(":", "").replace("+", "").replace(".", "") or "run"


def _safe_name(name: str) -> str:
    """Filesystem-safe slug for an artifact filename component.

    ``Scenario.name`` can come from an untrusted, hand-authored YAML ``name:``
    field (or the ``--freeze`` topic slug); stripping everything but a narrow
    allowlist removes path separators and ``..`` traversal sequences so a
    scenario can never write outside ``runs_dir`` (AUG-298).
    """
    slug = _UNSAFE_NAME_RE.sub("-", name).strip("-")
    return slug or "run"


def _restrict_dir(path: Path) -> None:
    """Create ``path`` (if needed) and ensure it is private (mode 0700)."""
    path.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):  # best-effort: some filesystems/mounts don't support POSIX modes
        os.chmod(path, 0o700)


def _write_private(path: Path, data: str) -> None:
    """Atomically write ``data`` to ``path`` with mode 0600 (owner-only).

    Eval artifacts and frozen scenarios embed raw article bodies, prompts,
    topic instructions, and parsed LLM responses; ambient umask-derived modes
    would otherwise make them readable by another local account on a shared
    self-hosting machine (AUG-297).
    """
    path = Path(path)
    _restrict_dir(path.parent)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(data)
        os.chmod(tmp_name, 0o600)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _prune_old_runs(runs_dir: Path, *, keep: int) -> None:
    """Delete the oldest saved artifacts beyond ``keep`` (AUG-298)."""
    runs = sorted(runs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for stale in runs[keep:]:
        stale.unlink(missing_ok=True)


def save_run(artifact: RunArtifact, runs_dir: Path) -> Path:
    """Serialize a RunArtifact to ``<runs_dir>/<name>-<stamp>.json``.

    ``runs_dir`` is resolved once and the final path is verified to still be
    contained within it — a defense-in-depth check behind ``_safe_name``'s
    sanitization, since ``artifact.name`` is scenario-supplied (AUG-298).
    """
    runs_dir = Path(runs_dir).resolve()
    _restrict_dir(runs_dir)
    name = _safe_name(artifact.name)
    path = (runs_dir / f"{name}-{_safe_stamp(artifact.created_at)}.json").resolve()
    if runs_dir != path.parent:
        raise ValueError(f"Refusing to save artifact outside runs_dir: {path}")
    _write_private(path, artifact.model_dump_json(indent=2))
    _prune_old_runs(runs_dir, keep=_MAX_SAVED_RUNS)
    return path


def load_run(path: Path) -> RunArtifact:
    """Load a RunArtifact from its JSON file.

    Raises ``ValueError`` if the file's ``schema_version`` is newer than this
    harness understands — silently loading it under current field defaults
    could make a replay diff misleading rather than just wrong (AUG-295).
    """
    art = RunArtifact.model_validate_json(Path(path).read_text())
    if art.schema_version > _ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"{path}: artifact schema_version {art.schema_version} is newer than this harness "
            f"supports ({_ARTIFACT_SCHEMA_VERSION}) — upgrade before replaying"
        )
    return art
