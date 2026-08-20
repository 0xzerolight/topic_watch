"""CLI for the eval harness: ``python -m evals <command>``.

Commands:
* ``scenario <file.yml> [--dry-run]`` — run a controlled scenario against the
  real LLM (or just print the prompt with --dry-run, zero API cost).
* ``live <topic> [--freeze <out.yml>] [--kind novelty]`` — fetch a topic's feeds
  live and run the LLM stage, without touching production data.
* ``replay <run.json>`` — re-run a saved run's inputs against the current
  prompt/code and diff the result (nonce-normalized).

``--strict`` (before the subcommand) additionally exits nonzero on a soft
expectation mismatch or non-empty replay diff. Without it, exit is nonzero
only for a provider/harness failure (a swallowed LLM error or a caught
knowledge-stage exception) — never for a mismatch alone, so a mismatch
remains a diagnostic a human reads rather than a build-breaking assertion.

Also holds console rendering and the replay diff — no separate report module.
"""

from __future__ import annotations

import argparse
import asyncio
import re
from difflib import unified_diff
from pathlib import Path
from typing import Any

from app.config import Settings, load_settings
from app.logging_config import setup_logging
from evals.runner import KIND_DISPATCH, LiveError, run_live, run_scenario
from evals.scenario import RunArtifact, load_run, load_scenario, save_run

_NONCE_RE = re.compile(r"(BEGIN|END) UNTRUSTED ARTICLE CONTENT [0-9a-f]+")
_RULE = "─" * 72


def normalize_nonce(text: str) -> str:
    """Collapse the per-call random fence nonce so identical inputs compare equal.

    ``_format_articles`` embeds a fresh ``secrets.token_hex`` nonce in the
    UNTRUSTED fence markers every build, so the same input yields different bytes
    each time — this normalization keeps replay diffs free of spurious churn.
    """
    return _NONCE_RE.sub(r"\1 UNTRUSTED ARTICLE CONTENT <nonce>", text)


def _messages_text(messages: list[dict[str, Any]]) -> str:
    return "\n".join(f"[{m.get('role')}]\n{m.get('content')}" for m in messages)


def diff_runs(old: RunArtifact, new: RunArtifact) -> list[str]:
    """Field-by-field diff of two runs; messages compared nonce-normalized.

    Covers every semantically relevant artifact field, not just the
    post-filtered ``final`` result: raw parsed output, response model,
    request model/temperature, token usage, and expectation verdicts. Two
    runs can produce byte-identical ``final`` output while their raw output
    turned unsafe, their model/schema changed, cost spiked, or an expectation
    flipped — this must not report equivalence in that case (AUG-048).

    Returns one line per difference; an empty list means equivalent.
    """
    lines: list[str] = []
    if old.model != new.model:
        lines.append(f"model: {old.model!r} -> {new.model!r}")
    if old.temperature != new.temperature:
        lines.append(f"temperature: {old.temperature!r} -> {new.temperature!r}")
    of, nf = old.final or {}, new.final or {}
    for key in sorted(set(of) | set(nf)):
        if of.get(key) != nf.get(key):
            lines.append(f"final.{key}: {of.get(key)!r} -> {nf.get(key)!r}")
    if old.final_error != new.final_error:
        lines.append(f"final_error: {old.final_error!r} -> {new.final_error!r}")
    for i in range(max(len(old.calls), len(new.calls))):
        oc = old.calls[i] if i < len(old.calls) else None
        nc = new.calls[i] if i < len(new.calls) else None
        if oc is None or nc is None:
            lines.append(
                f"calls[{i}]: {'missing' if oc is None else 'present'} -> {'missing' if nc is None else 'present'}"
            )
            continue
        if oc.response_model != nc.response_model:
            lines.append(f"calls[{i}].response_model: {oc.response_model!r} -> {nc.response_model!r}")
        if oc.raw_parsed != nc.raw_parsed:
            lines.append(f"calls[{i}].raw_parsed: {oc.raw_parsed!r} -> {nc.raw_parsed!r}")
        if (oc.prompt_tokens, oc.completion_tokens) != (nc.prompt_tokens, nc.completion_tokens):
            lines.append(
                f"calls[{i}].tokens: prompt {oc.prompt_tokens}->{nc.prompt_tokens}, "
                f"completion {oc.completion_tokens}->{nc.completion_tokens}"
            )
        o = normalize_nonce(_messages_text(oc.messages))
        n = normalize_nonce(_messages_text(nc.messages))
        if o != n:
            lines.append(f"messages[{i}] differ (nonce-normalized):")
            lines.extend(unified_diff(o.splitlines(), n.splitlines(), lineterm="", n=1))
    oe = {c.check: c.ok for c in old.expect_results}
    ne = {c.check: c.ok for c in new.expect_results}
    for key in sorted(set(oe) | set(ne)):
        if oe.get(key) != ne.get(key):
            lines.append(f"expect.{key}: {oe.get(key)!r} -> {ne.get(key)!r}")
    return lines


# --- rendering ---


def _section(title: str, body: str) -> str:
    return f"{_RULE}\n{title}\n{_RULE}\n{body}\n"


def render_messages(messages: list[dict[str, Any]]) -> str:
    return "\n\n".join(f"[{m.get('role')}]\n{m.get('content')}" for m in messages)


def render_artifact(art: RunArtifact) -> str:
    """Human-readable dump: scenario, prompt, raw parsed, final, usage, expect."""
    out: list[str] = []
    out.append(_section("SCENARIO", f"name={art.name}  kind={art.kind}  model={art.model}  temp={art.temperature}"))
    for i, call in enumerate(art.calls):
        out.append(_section(f"PROMPT [call {i}: {call.response_model}]", render_messages(call.messages)))
        out.append(_section(f"RAW PARSED [call {i}]", _pretty(call.raw_parsed)))
        out.append(f"tokens: prompt={call.prompt_tokens} completion={call.completion_tokens}\n")
    out.append(_section("FINAL", _pretty(art.final or {})))
    if art.final_error:
        out.append(f"⚠ LLM ERROR (swallowed to safe default): {art.final_error}\n")
    if art.expect_results:
        rows = "\n".join(f"  {'MATCH' if c.ok else 'MISMATCH':8} {c.check}: {c.detail}" for c in art.expect_results)
        out.append(_section("EXPECT", rows))
    return "\n".join(out)


def _pretty(data: dict[str, Any]) -> str:
    return "\n".join(f"  {k}: {v!r}" for k, v in data.items())


# --- run outcome / exit status (AUG-296) ---


def _exit_code(art: RunArtifact, *, strict: bool, diff: list[str] | None = None) -> int:
    """Map a run's outcome to a process exit code.

    A provider/harness failure (``final_error`` set — a swallowed novelty LLM
    error or a caught knowledge-stage exception) is always nonzero, so
    automation can tell "the model said no" apart from "the call blew up". A
    soft expectation mismatch or non-empty replay diff is nonzero only under
    ``--strict``: those are diagnostic signals for a human, not proof the
    harness itself failed to run.
    """
    if art.final_error is not None:
        return 1
    if strict and (any(not c.ok for c in art.expect_results) or bool(diff)):
        return 1
    return 0


# --- replay ---


async def replay(run_path: Path, settings: Settings, *, inner: Any = None) -> tuple[RunArtifact, list[str]]:
    """Re-run a saved run's inputs against the current prompt/code and diff it."""
    old = load_run(run_path)
    new = await run_scenario(old.scenario, settings, inner=inner)
    return new, diff_runs(old, new)


# --- commands ---


def _default_runs_dir(settings: Settings) -> Path:
    return Path(settings.db_path).parent / "eval" / "runs"


async def _cmd_scenario(file: str, settings: Settings, *, dry_run: bool, runs_dir: Path) -> RunArtifact | None:
    scenario = load_scenario(Path(file))
    if dry_run:
        messages = KIND_DISPATCH[scenario.kind].build(scenario, settings)
        print(_section(f"DRY RUN PROMPT [{scenario.kind}]", render_messages(messages)))
        return None
    art = await run_scenario(scenario, settings)
    path = save_run(art, runs_dir)
    print(render_artifact(art))
    print(f"saved: {path}")
    return art


async def _cmd_live(
    topic_name: str, settings: Settings, *, kind: str, freeze: str | None, runs_dir: Path
) -> RunArtifact:
    art = await run_live(topic_name, settings, kind=kind, freeze_path=freeze)
    path = save_run(art, runs_dir)
    print(render_artifact(art))
    if freeze:
        print(f"frozen scenario: {freeze}")
    print(f"saved: {path}")
    return art


async def _cmd_replay(run: str, settings: Settings, *, runs_dir: Path) -> tuple[RunArtifact, list[str]]:
    new, diff = await replay(Path(run), settings)
    save_run(new, runs_dir)
    print(render_artifact(new))
    print(_section("DIFF vs saved run", "\n".join(diff) if diff else "(no differences)"))
    return new, diff


def main() -> None:
    parser = argparse.ArgumentParser(prog="evals", description="On-demand real-LLM eval harness for topic_watch")
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Also exit nonzero on an expectation mismatch or replay drift "
            "(place before the subcommand, e.g. `evals --strict replay run.json`). "
            "A provider/harness failure is always nonzero regardless of this flag."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("scenario", help="Run a controlled scenario against the real LLM")
    sp.add_argument("file", help="Path to a scenario YAML file")
    sp.add_argument("--dry-run", action="store_true", help="Print the prompt only; no API call")

    lp = sub.add_parser("live", help="Fetch a topic's feeds live and run the LLM stage (prod read-only)")
    lp.add_argument("topic_name", help="Name of an existing topic in the production DB")
    lp.add_argument("--kind", default="novelty", choices=sorted(KIND_DISPATCH), help="LLM stage to run")
    lp.add_argument("--freeze", help="Write the fetched inputs to a reusable scenario YAML")

    rp = sub.add_parser("replay", help="Re-run a saved run against the current prompt and diff")
    rp.add_argument("run", help="Path to a saved RunArtifact JSON file")

    args = parser.parse_args()
    setup_logging()
    settings = load_settings()
    runs_dir = _default_runs_dir(settings)

    exit_code = 0
    try:
        if args.command == "scenario":
            art = asyncio.run(_cmd_scenario(args.file, settings, dry_run=args.dry_run, runs_dir=runs_dir))
            if art is not None:
                exit_code = _exit_code(art, strict=args.strict)
        elif args.command == "live":
            art = asyncio.run(
                _cmd_live(args.topic_name, settings, kind=args.kind, freeze=args.freeze, runs_dir=runs_dir)
            )
            exit_code = _exit_code(art, strict=args.strict)
        elif args.command == "replay":
            new, diff = asyncio.run(_cmd_replay(args.run, settings, runs_dir=runs_dir))
            exit_code = _exit_code(new, strict=args.strict, diff=diff)
    except LiveError as exc:
        raise SystemExit(str(exc)) from exc

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
