"""CLI for the eval harness: ``python -m evals <command>``.

Commands:
* ``scenario <file.yml> [--dry-run]`` — run a controlled scenario against the
  real LLM (or just print the prompt with --dry-run, zero API cost).
* ``live <topic> [--freeze <out.yml>] [--kind novelty]`` — fetch a topic's feeds
  live and run the LLM stage, without touching production data.
* ``replay <run.json>`` — re-run a saved run's inputs against the current
  prompt/code and diff the result (nonce-normalized).

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

    Returns one line per difference; an empty list means equivalent.
    """
    lines: list[str] = []
    of, nf = old.final or {}, new.final or {}
    for key in sorted(set(of) | set(nf)):
        if of.get(key) != nf.get(key):
            lines.append(f"final.{key}: {of.get(key)!r} -> {nf.get(key)!r}")
    if old.final_error != new.final_error:
        lines.append(f"final_error: {old.final_error!r} -> {new.final_error!r}")
    for i in range(max(len(old.calls), len(new.calls))):
        o = normalize_nonce(_messages_text(old.calls[i].messages)) if i < len(old.calls) else ""
        n = normalize_nonce(_messages_text(new.calls[i].messages)) if i < len(new.calls) else ""
        if o != n:
            lines.append(f"messages[{i}] differ (nonce-normalized):")
            lines.extend(unified_diff(o.splitlines(), n.splitlines(), lineterm="", n=1))
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


# --- replay ---


async def replay(run_path: Path, settings: Settings, *, inner: Any = None) -> tuple[RunArtifact, list[str]]:
    """Re-run a saved run's inputs against the current prompt/code and diff it."""
    old = load_run(run_path)
    new = await run_scenario(old.scenario, settings, inner=inner)
    return new, diff_runs(old, new)


# --- commands ---


def _default_runs_dir(settings: Settings) -> Path:
    return Path(settings.db_path).parent / "eval" / "runs"


async def _cmd_scenario(file: str, settings: Settings, *, dry_run: bool, runs_dir: Path) -> None:
    scenario = load_scenario(Path(file))
    if dry_run:
        messages = KIND_DISPATCH[scenario.kind].build(scenario, settings)
        print(_section(f"DRY RUN PROMPT [{scenario.kind}]", render_messages(messages)))
        return
    art = await run_scenario(scenario, settings)
    path = save_run(art, runs_dir)
    print(render_artifact(art))
    print(f"saved: {path}")


async def _cmd_live(topic_name: str, settings: Settings, *, kind: str, freeze: str | None, runs_dir: Path) -> None:
    art = await run_live(topic_name, settings, kind=kind, freeze_path=freeze)
    path = save_run(art, runs_dir)
    print(render_artifact(art))
    if freeze:
        print(f"frozen scenario: {freeze}")
    print(f"saved: {path}")


async def _cmd_replay(run: str, settings: Settings, *, runs_dir: Path) -> None:
    new, diff = await replay(Path(run), settings)
    save_run(new, runs_dir)
    print(render_artifact(new))
    print(_section("DIFF vs saved run", "\n".join(diff) if diff else "(no differences)"))


def main() -> None:
    parser = argparse.ArgumentParser(prog="evals", description="On-demand real-LLM eval harness for topic_watch")
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

    try:
        if args.command == "scenario":
            asyncio.run(_cmd_scenario(args.file, settings, dry_run=args.dry_run, runs_dir=runs_dir))
        elif args.command == "live":
            asyncio.run(_cmd_live(args.topic_name, settings, kind=args.kind, freeze=args.freeze, runs_dir=runs_dir))
        elif args.command == "replay":
            asyncio.run(_cmd_replay(args.run, settings, runs_dir=runs_dir))
    except LiveError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
