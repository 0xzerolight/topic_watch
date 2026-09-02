.PHONY: dev test smoke lint format format-check typecheck verify-evals verify-locks coverage docker docker-run run clean ci lock lock-tools lock-upgrade _require_pip_compile help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

dev: ## Install project in editable mode with dev dependencies
	pip install --require-hashes -r requirements-dev.txt
	pip install --no-deps -e .
	pre-commit install
	pre-commit install --hook-type pre-push

# pip-compile ships with pip-tools, which is deliberately NOT in the dev extras:
# it requires pip/setuptools/wheel, so locking it would hash-pin pip itself into
# the requirements-dev.txt that `make dev` and every CI job install.
# `requirements-build.txt` pins the wheel build backend the Docker image uses; it
# is installed only in the builder stage.
lock-tools: ## Install the pinned pip-compile toolchain (use a throwaway/worktree venv)
	pip install "pip==25.1.1" "pip-tools==7.5.3"

_require_pip_compile:
	@command -v pip-compile >/dev/null 2>&1 || { \
		echo "pip-compile not found — the lock targets need it."; \
		echo ""; \
		echo "  make lock-tools    # installs pip==25.1.1 + pip-tools==7.5.3"; \
		echo ""; \
		echo "Run that in a throwaway or worktree venv: it pins pip to 25.1.1."; \
		echo "pip 26 removed an internal that pip-tools 7.5.3 imports, so newer"; \
		echo "pip fails every pip-compile call with an ImportError."; \
		exit 1; \
	}

lock: _require_pip_compile ## Regenerate pinned requirements lockfiles from pyproject.toml
	pip-compile --strip-extras --generate-hashes --output-file=requirements.txt pyproject.toml
	pip-compile --strip-extras --generate-hashes --extra=dev --output-file=requirements-dev.txt pyproject.toml
	pip-compile --strip-extras --generate-hashes --output-file=requirements-build.txt requirements-build.in

# `lock` keeps every version already pinned in the output files, so it cannot
# pull in a bumped dependency. Use this when Dependabot's group PR fails to
# install: its updater rewrites known pins but never adds newly-introduced
# transitive packages, which breaks `pip install --require-hashes`.
lock-upgrade: _require_pip_compile ## Relock at the newest versions pyproject.toml allows
	pip-compile --upgrade --strip-extras --generate-hashes --output-file=requirements.txt pyproject.toml
	pip-compile --upgrade --strip-extras --generate-hashes --extra=dev --output-file=requirements-dev.txt pyproject.toml
	pip-compile --upgrade --strip-extras --generate-hashes --output-file=requirements-build.txt requirements-build.in

test: ## Run tests with pytest
	pytest --tb=short

smoke: ## Run hermetic end-to-end smoke tests
	pytest tests/smoke -q --no-cov

lint: ## Run Ruff linter and mypy type checker
	ruff check .
	mypy app/ evals/ --ignore-missing-imports

format: ## Format code with Ruff
	ruff format .
	ruff check --fix .

format-check: ## Check formatting without modifying files (matches hosted CI)
	ruff format --check .

typecheck: ## Run mypy type checker
	mypy app/ evals/ --ignore-missing-imports

verify-evals: ## Verify the evals harness package imports cleanly (matches hosted CI)
	python -c "import evals"

verify-locks: ## Confirm requirements.txt pins are a hash-identical subset of requirements-dev.txt (AUG-303)
	@grep -Ev '^\s*#|^\s*$$' requirements.txt | while IFS= read -r line; do \
		grep -qxF "$$line" requirements-dev.txt || { echo "requirements.txt line not found verbatim in requirements-dev.txt: $$line"; exit 1; }; \
	done
	@echo "requirements.txt pins verified as a subset of requirements-dev.txt"

coverage: ## Run tests with detailed coverage report
	pytest --cov=app --cov-report=term-missing --cov-report=html --tb=short

docker: ## Build Docker image via docker compose
	docker compose build

docker-run: ## Build and start the Docker container in background
	docker compose up -d --build

run: ## Start dev server with auto-reload
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

ci: lint format-check verify-evals verify-locks test ## Run all CI checks locally (matches hosted CI gate)

clean: ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
