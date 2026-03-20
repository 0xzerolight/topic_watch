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
5. **(Optional) Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

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

**Rules for tests:**

- New features and bug fixes should include tests.
- Tests must not make live API calls. Mock all LLM interactions using the patterns already in `tests/`.
- The minimum coverage threshold is 60%. CI will fail if coverage drops below this.

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
- Keep PRs focused — one logical change per PR.
- Write a clear description of what the PR does and why.

## Reporting Bugs

Open an issue on [GitHub Issues](https://github.com/0xzerolight/topic_watch/issues). Include:

- Steps to reproduce
- Expected vs. actual behavior
- Relevant logs or error messages
- Python version and OS

## Suggesting Features

Open an issue or start a discussion on [GitHub Discussions](https://github.com/0xzerolight/topic_watch/discussions). Describe the use case and why the feature would be useful.
