# Contributing to Topic Watch

Thanks for your interest in contributing! This guide covers everything you need to get started.

## Getting Started

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-fork>/topic_watch.git
   cd topic_watch
   ```
3. **Set up the dev environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```
4. **Copy the example config:**
   ```bash
   cp config.example.yml data/config.yml
   ```
5. **(Optional) Install git hooks:**
   ```bash
   make dev
   ```
   This installs git hooks via pre-commit. The same checks (ruff, ruff-format, and mypy) run on every commit and again before every push. You can bypass them with `git commit --no-verify` or `git push --no-verify` if needed.

## Development Workflow

1. Create a branch for your change:
   ```bash
   git checkout -b feat/my-feature
   ```
2. Make your changes.
3. Run tests and linting (see below).
4. Commit your changes using [conventional commits](#commit-messages).
5. Push your branch and open a pull request against `main`.

## Code Style

Formatting and linting are handled by [Ruff](https://docs.astral.sh/ruff/). Before committing:

```bash
ruff check .       # lint
ruff format .      # format
```

If you installed pre-commit hooks, these run automatically on `git commit`. The CI pipeline enforces both checks, so PRs with lint or formatting issues will not be merged.

Type checking uses [mypy](https://mypy-lang.org/):

```bash
mypy app/
```

## Testing

Run the full test suite:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=app --cov-report=term-missing
```

Run the hermetic end-to-end smoke tests:

```bash
make smoke
```

`make smoke` runs `tests/smoke`, which drives the real check pipeline (scraping,
dedup, content extraction, knowledge persistence, novelty thresholding,
notification queueing) with only the outermost boundaries stubbed — HTTP, LLM,
and delivery. It is the integration mitigation for the "no live API calls"
policy: because the unit suite mocks heavily, the smoke layer exercises the code
paths between those mocks end to end. Run it before opening a PR that touches the
pipeline.

**Rules for tests:**

- New features and bug fixes should include tests.
- Tests must not make live API calls. Mock all LLM interactions using the patterns already in `tests/`.
- The minimum coverage threshold is 85%. CI will fail if coverage drops below this.

### Real-LLM eval harness (on-demand)

The automated suite never calls a real model, so it cannot catch bugs that only
show up in real LLM output — wrong novelty calls, odd structured output, bad
knowledge-state builds. The `evals/` package fills that gap for manual,
on-demand use (it makes real, billed API calls, so it never runs in CI). It adds
input control and full observability — captured prompt, raw parsed result,
post-filter result, token usage — around the LLM stages, with no changes to
`app/`.

```bash
# Print the exact prompt a scenario would send — no API call, free:
python -m evals scenario evals/scenarios/dup_event.yml --dry-run

# Run a controlled scenario against the real model (uses data/config.yml):
python -m evals scenario evals/scenarios/dup_event.yml

# Fetch a topic's feeds live and run a stage. The production DB is opened
# read-only; all fetch bookkeeping goes to a throwaway scratch DB. --freeze
# writes the fetched inputs to a reusable scenario file:
python -m evals live "My Topic" --freeze /tmp/frozen.yml

# Re-run a saved run against the current prompt/code and diff the result:
python -m evals replay data/eval/runs/<name>-<stamp>.json
```

Scenario files live in `evals/scenarios/`; run artifacts are written under
`data/eval/runs/` (gitignored). The harness itself is dev-only — it is not in the
shipped wheel — but its offline logic is tested in `tests/test_evals.py` and
type-checked as part of `make ci`.

## Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

| Prefix | When to use |
|--------|-------------|
| `feat:` | New user-facing feature |
| `fix:` | Bug fix |
| `docs:` | Documentation only |
| `chore:` | Maintenance, deps, tooling |
| `ci:` | CI/CD changes |
| `refactor:` | Code change with no behavior change |
| `test:` | Adding or fixing tests |

Examples:
```
feat: add Slack notification support
fix: handle empty RSS feed without crashing
docs: clarify reverse proxy setup in README
```

## Pull Request Guidelines

- Target the `main` branch.
- Include tests for new features or bug fixes.
- Ensure all CI checks pass (lint, format, type check, tests).
- Keep PRs focused. One logical change per PR.
- Write a clear description of what the PR does and why.

## Reporting Bugs

Open an issue on [GitHub Issues](https://github.com/0xzerolight/topic_watch/issues). Include:

- Steps to reproduce
- Expected vs. actual behavior
- Relevant logs or error messages
- Python version and OS

## Suggesting Features

Open an issue on [GitHub Issues](https://github.com/0xzerolight/topic_watch/issues). Describe the use case and why the feature would be useful.
