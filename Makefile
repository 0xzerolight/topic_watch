.PHONY: dev test lint format typecheck coverage docker docker-run run clean ci lock help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

dev: ## Install project in editable mode with dev dependencies
	pip install --require-hashes -r requirements-dev.txt
	pip install --no-deps -e .
	pre-commit install
	pre-commit install --hook-type pre-push

lock: ## Regenerate pinned requirements lockfiles from pyproject.toml
	pip-compile --strip-extras --generate-hashes --output-file=requirements.txt pyproject.toml
	pip-compile --strip-extras --generate-hashes --extra=dev --output-file=requirements-dev.txt pyproject.toml

test: ## Run tests with pytest
	pytest --tb=short

lint: ## Run Ruff linter and mypy type checker
	ruff check .
	mypy app/ --ignore-missing-imports

format: ## Format code with Ruff
	ruff format .
	ruff check --fix .

typecheck: ## Run mypy type checker
	mypy app/ --ignore-missing-imports

coverage: ## Run tests with detailed coverage report
	pytest --cov=app --cov-report=term-missing --cov-report=html --tb=short

docker: ## Build Docker image via docker compose
	docker compose build

docker-run: ## Build and start the Docker container in background
	docker compose up -d --build

run: ## Start dev server with auto-reload
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

ci: lint test ## Run all CI checks locally (lint + test)

clean: ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
