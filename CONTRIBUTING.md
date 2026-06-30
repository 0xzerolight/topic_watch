# Contributing to Topic Watch

Bug reports, feature ideas, and PRs are all welcome.

## Setup

```bash
git clone https://github.com/<your-fork>/topic_watch.git
cd topic_watch
python -m venv .venv && source .venv/bin/activate
make dev                              # editable install + dev deps + git hooks
cp config.example.yml data/config.yml
```

`make dev` installs pre-commit hooks (ruff, ruff-format, mypy) that run on every
commit and again before every push. Bypass with `git commit --no-verify` if you
must.

## Workflow

1. Branch: `git checkout -b feat/my-feature`.
2. Make changes; add tests.
3. `make ci` — lint, type-check, tests. Must pass.
4. Commit with [Conventional Commits](#commit-messages).
5. Push and open a PR against `main`.

## Checks

| Command | Does |
|---------|------|
| `make lint` | Ruff lint + mypy |
| `make format` | Ruff format + autofix |
| `make test` | Full suite (85% coverage gate — CI fails below it) |
| `make smoke` | Hermetic end-to-end pipeline tests |
| `make ci` | lint + test — run before every PR |

Tests must not make live API calls — mock all HTTP/LLM using the patterns already
in `tests/`. `make smoke` drives the real pipeline (scraping, dedup, extraction,
knowledge persistence, novelty thresholding, notification queueing) end to end
with only the outer boundaries (HTTP, LLM, delivery) stubbed; run it for any
change touching the pipeline.

### Real-LLM eval harness (on-demand, dev-only)

The suite never calls a real model, so it can't catch bugs that only show up in
real LLM output — wrong novelty calls, odd structured output, bad knowledge
states. The `evals/` package fills that gap: real, billed calls (never in CI),
with full observability (prompt, raw result, post-filter result, tokens) around
the LLM stages and no changes to `app/`.

```bash
python -m evals scenario evals/scenarios/dup_event.yml --dry-run   # print prompt, no call
python -m evals scenario evals/scenarios/dup_event.yml             # run vs real model
python -m evals live "My Topic" --freeze /tmp/frozen.yml           # live fetch → scenario file
python -m evals replay data/eval/runs/<name>-<stamp>.json          # re-run + diff
```

Run artifacts land in `data/eval/runs/` (gitignored). The harness ships dev-only
(not in the wheel); its offline logic is tested in `tests/test_evals.py`.

## Commit Messages

[Conventional Commits](https://www.conventionalcommits.org/):

| Prefix | When |
|--------|------|
| `feat:` | New user-facing feature |
| `fix:` | Bug fix |
| `docs:` | Documentation only |
| `chore:` | Maintenance, deps, tooling |
| `ci:` | CI/CD changes |
| `refactor:` | Code change, no behavior change |
| `test:` | Adding or fixing tests |

```
feat: add Slack notification support
fix: handle empty RSS feed without crashing
```

## Pull Requests

- Target `main`, one logical change per PR.
- Include tests for new features and bug fixes.
- All CI checks must pass (lint, format, type-check, tests).
- Write a clear description of what the PR does and why.

## Reporting

Bugs and feature requests → [GitHub Issues](https://github.com/0xzerolight/topic_watch/issues).
For bugs, include repro steps, expected vs. actual behavior, relevant logs, and the
output of `python -m app.cli doctor` (Docker: `docker compose exec topic-watch python -m app.cli doctor`) —
a secret-safe snapshot of version, runtime, redacted config, schema, and feed health.
