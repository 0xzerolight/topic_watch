.PHONY: dev test lint format typecheck coverage docker run clean help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

dev: ## Install project in editable mode with dev dependencies
	pip install -e ".[dev]"
	pre-commit install

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

run: ## Start dev server with auto-reload
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

clean: ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
