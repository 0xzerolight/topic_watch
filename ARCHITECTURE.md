# Architecture

## System Overview

Topic Watch monitors user-defined topics by fetching articles from RSS feeds, then uses an LLM to determine whether the articles contain genuinely new information compared to what is already known. Notifications are sent only when something novel is found. Silence is the default.

```
                         ┌─────────────────┐
                         │   RSS Feeds     │
                         └────────┬────────┘
                                  │
                                  v
                    ┌─────────────────────────┐
                    │   Scraping Pipeline      │
                    │  (fetch, dedup, extract)  │
                    └────────────┬─────────────┘
                                 │
                                 v
              ┌──────────────────────────────────┐
              │  Knowledge State + New Articles   │
              └───────────────┬──────────────────┘
                              │
                              v
                ┌──────────────────────────┐
                │  LLM Novelty Detection   │
                │  (has_new_info? yes/no)   │
                └─────────┬────────────────┘
                          │
                ┌─────────┴─────────┐
                │                   │
              yes                  no
                │                   │
                v                   v
      ┌──────────────────┐   ┌───────────┐
      │ Update Knowledge │   │  Record   │
      │ Send Notification│   │  (done)   │
      │ Record           │   └───────────┘
      └──────────────────┘
```

**Two entry points trigger checks:**
1. **APScheduler** - a background job ticks every 1 minute, queries which topics are due based on their individual `check_interval_minutes`, and runs the pipeline for each.
2. **Web UI** - users can trigger manual checks via the dashboard, which runs the same pipeline as a FastAPI background task.

## Module Map

All application code lives under `app/`.

### Core Pipeline

| Module | Responsibility |
|--------|---------------|
| `checker.py` | Orchestrates the full check pipeline: fetch → analyze → notify → record. `check_topic()` is the primary entry point. `check_all_topics()` iterates due topics. `retry_pending_notifications()` handles failed deliveries. `initialize_new_topic()` builds initial knowledge for NEW topics. |
| `scheduler.py` | APScheduler 3.x `AsyncIOScheduler`, four jobs: every-minute tick (`_scheduled_check` — runs the check cycle, then initializes one NEW topic), stuck-topic recovery (every 5 min, 15-min timeout), weekly VACUUM (Sun 3 AM), daily article cleanup (4 AM). Only the minute tick has jitter; all jobs coalesce and are single-instance. The check cycle (`_run_check_cycle`) retries pending notifications and webhooks before querying due topics. |

### LLM Analysis

| Module | Responsibility |
|--------|---------------|
| `analysis/llm.py` | LiteLLM + Instructor wrappers. Defines `NoveltyResult` (with `confidence` and `relevance` scores), `KnowledgeStateUpdate`, and `TokenUsage`. Token counting, rate limit backoff with exponential delay. Returns safe default (`has_new_info=False`, `confidence=0.0`) on analysis failure. |
| `analysis/prompts.py` | System and user prompt builders for novelty detection and knowledge init/update/compress. Articles truncated to 1500 chars in prompts. |
| `analysis/knowledge.py` | Knowledge state initialization and updates with DB persistence. Token budget enforcement via summary compression. |
| `analysis/restatement.py` | Pure phrase-matching filter (`filter_restated_key_facts`, re-exported by `llm.py`). Drops a key fact only when it is a clear restatement of the existing knowledge summary (normalized verbatim or long contiguous n-gram match), so already-known facts aren't re-flagged as new. Conservative by design. |
| `analysis/citations.py` | `strip_index_citations()` removes ephemeral `(Article [N])`-style citations from LLM output before it's persisted — they reference one run's article list and cause coherence drift if stored. |

### Scraping

| Module | Responsibility |
|--------|---------------|
| `scraping/__init__.py` | `fetch_new_articles_for_topic()` - orchestrates feed fetch, dedup against DB, cross-topic content reuse, concurrent content extraction (semaphore-limited), and article storage. |
| `scraping/rss.py` | RSS/Atom feed fetching via httpx + feedparser. Converts entries to `FeedEntry` models. Retry on timeouts and 5xx. Feed health callbacks. |
| `scraping/content.py` | Article HTML fetch + trafilatura content extraction. Falls back to RSS summary on failure. Content truncated to 5000 chars at word boundary. |
| `scraping/providers.py` | News search provider definitions. `NewsProvider` Protocol plus `GoogleNewsProvider` / `BingNewsProvider` concrete classes that build keyword-search feed URLs from topic name + description (auto feed mode). |
| `scraping/routing.py` | Health-based provider cascade. Tracks per-provider health in-memory and selects the first healthy provider per cycle (Bing first, Google second). Separate from the per-URL `feed_health` table. |
| `scraping/google_news.py` | Resolves opaque Google News redirect URLs (`news.google.com/rss/articles/...`) to real article URLs via Google's `batchexecute` endpoint. |

### Data Layer

| Module | Responsibility |
|--------|---------------|
| `models.py` | Pydantic models: `Topic`, `Article`, `KnowledgeState`, `CheckResult`, `FeedHealth`, `DashboardStats`, `PendingNotification`, `PendingWebhook`. Enums: `TopicStatus` (new/researching/ready/error), `FeedMode` (auto/manual). Each model has `from_row()` and `to_insert_dict()` for SQLite interop; datetime cells are coerced defensively. |
| `crud.py` | All SQL (parameterized), grouped by model: CRUD, feed-health upserts, notification + webhook retry queues, dashboard aggregation, article retention cleanup, stuck-topic recovery. |
| `database.py` | SQLite connection factory (WAL mode, foreign keys, busy timeout). Schema init (`init_db`). Migration runner (`run_migrations`) — backs up the DB before applying pending migrations. |
| `migrations/` | 19 sequential migrations (`m001`–`m019`) registered in `__init__.py` as `(version, description, up_function)` tuples. Tracked in `schema_version`. Append-only. |
| `interval.py` | Human-readable interval parsing/formatting (`m`/`h`/`d`/`w`/`M`, combined syntax like `"1w 3d 2h"`). Enforces min/max interval bounds. |
| `opml.py` | OPML import/export. Parses feeds from RSS readers (FreshRSS, Miniflux, TT-RSS), validates feed URLs, and exports topics as OPML. |

### Web

The route handlers were split out of `routes.py` into the `web/routers/` package. The HTMX/HTML routes are mounted via an aggregate router; the JSON API lives separately in `web/api.py`.

| Module | Responsibility |
|--------|---------------|
| `web/routes.py` | Backwards-compatible shim. Re-exports `router` from `web/routers/` so existing `from app.web.routes import router` imports still work. No handlers live here anymore. |
| `web/routers/__init__.py` | Aggregate router. Includes the per-domain routers in include-order so static topic paths (`/topics/search`, `/topics/new`) register before the dynamic `/topics/{topic_id}` route. |
| `web/routers/dashboard.py` | Dashboard page, `/health` check, and topic search. Reads the dashboard stats cache. |
| `web/routers/topics.py` | Topic CRUD, detail/articles pages, and manual check + init triggers. |
| `web/routers/exports.py` | Data export endpoints: all-topics JSON (`/export/topics/json`) and per-topic JSON/CSV (`/topics/{id}/export/json`, `/topics/{id}/export/csv`). |
| `web/routers/settings.py` | Setup wizard, settings editor, and notification-test endpoint. Reads/writes config via `load_settings()` / `save_settings_to_yaml()`. |
| `web/routers/feed_health.py` | Global feed-health dashboard and feed-URL validation endpoint (rate-limited). |
| `web/routers/opml.py` | OPML import/export and bulk topic export (JSON). |
| `web/routers/background.py` | Background-task helpers (`_run_init`, check-all) that run after the request connection closes, each opening its own DB connection. Coordinates via the shared `_checking_state`. |
| `web/routers/templates.py` | Shared `Jinja2Templates` instance and template filters (`timeago`, `sanitize_error`, `mask_url`, `confidence_badge`). Filters are module-level for unit testing. |
| `web/routers/_validation.py` | Shared topic-form validation (`validate_topic_form`) used by create and edit handlers. |
| `web/api.py` | JSON API v1 (`/api/v1`). Read-only endpoints (list/get topics, checks, knowledge) plus one CSRF-protected mutation to trigger a check. Reuses CRUD and Pydantic models. |
| `web/state.py` | Process-global web state: `CheckingState` (in-progress check tracking with stale-lock detection), dashboard stats cache, and the in-memory feed-validation rate limiter. |
| `web/csrf.py` | Double-submit cookie CSRF middleware + `verify_csrf` dependency. Sets token cookie on responses, validates POST/PUT/DELETE via `X-CSRF-Token` header (HTMX) or `csrf_token` form field. |
| `web/dependencies.py` | FastAPI dependency injection: `get_db_conn` (per-request connection with auto-commit/rollback), `get_settings` (from `app.state`). |
| `web/setup_middleware.py` | ASGI middleware that redirects all routes to `/setup` while `app.state.setup_required` is set (exempts `/setup`, `/health`, `/static`). |

### Infrastructure

| Module | Responsibility |
|--------|---------------|
| `main.py` | FastAPI app + lifespan. Runs migrations, starts/stops the scheduler, mounts the web routers, JSON API, CSRF + setup-redirect middleware, and static files. |
| `config.py` | Pydantic `BaseSettings` with YAML source. Priority: env > YAML > defaults. `load_settings()` / `save_settings_to_yaml()`; cloud/local provider helpers. |
| `logging_config.py` | Plain text or JSON structured logging. Controlled by `TOPIC_WATCH_LOG_FORMAT` and `TOPIC_WATCH_LOG_LEVEL` env vars. |
| `check_context.py` | Correlation IDs via `contextvars.ContextVar`. `CheckIdFilter` injects check ID into all log records. |
| `url_validation.py` | SSRF protection. Blocks private/reserved IPs (localhost, 10.x, 172.16-31.x, 192.168.x, link-local, CGNAT 100.64.0.0/10, IPv6 ULA). |
| `feed_backoff.py` | `feed_backoff_until()` — stateless exponential backoff for persistently-failing feeds, computed from `feed_health` consecutive failures. Bounded by `feed_backoff_base_minutes` / `feed_backoff_cap_hours`. |
| `notifications.py` | Apprise wrapper. Formats `NoveltyResult` into title/body. Sync Apprise send wrapped in `asyncio.to_thread()`. Re-exports `redact_url` from `log_redaction.py`. |
| `log_redaction.py` | Log-hygiene helper. `redact_url` strips userinfo, query strings, fragments, and long (likely-secret) path segments from notification/webhook URLs, keeping scheme + host + a short path prefix for diagnostics. |
| `webhooks.py` | JSON POST to configured webhook endpoints. Concurrent delivery via `asyncio.gather()`. Failed deliveries are queued in `pending_webhooks` and retried via `retry_pending_webhooks()` at the start of each check cycle. |
| `cli.py` | Argparse CLI: `list`, `check`, `check-all`, `init`. |

### Frontend

- `templates/` - 16 Jinja2 templates. Pico CSS + HTMX base layout; partials for dynamic updates.
- `static/themes.css` - Color themes (Nord, Dracula, Solarized, High Contrast, Tokyo Night).
- `static/components.css` - Component styles (cards, badges, tables) layered on Pico.
- `static/theme.js` - Theme switcher with localStorage persistence.
- `static/notifications.js` - Browser push notification wrapper.
- `static/vendor/` - Vendored Pico CSS + HTMX. No build tooling.

## Key Design Decisions

**SQLite, not Postgres.** Single-user self-hosted tool. SQLite eliminates deployment complexity. WAL mode provides adequate concurrency for web server + background scheduler.

**LiteLLM for provider abstraction.** Users switch between OpenAI, Anthropic, Ollama, Gemini, or any supported provider by changing one config string. No provider-specific code in the app.

**Instructor for structured output.** Pydantic response models (`NoveltyResult`, `KnowledgeStateUpdate`) with automatic validation retry. Eliminates JSON parsing fragility.

**Safe defaults on LLM failure.** `analyze_articles()` returns `has_new_info=False` on any error. Users miss an update rather than get a false alert. Knowledge operations raise because correctness is critical there.

**Knowledge state with token budget.** Rolling summary compressed by sentence-level truncation when exceeding `knowledge_state_max_tokens`. Prevents unbounded context growth.

**Apprise for notifications.** Supports 100+ services via URL format. No need for individual service integrations.

**No built-in authentication.** Deliberate choice for a single-user tool. Remote deployments use a reverse proxy with auth (Authelia, Caddy basicauth, etc.).

**CSRF double-submit cookie.** Stateless CSRF protection compatible with HTMX. Cookie is not httponly (HTMX reads it via JS). SameSite=Lax.

**Content hash dedup.** `SHA256(lowercase(url|title))` uniquely identifies articles. Cross-topic content reuse avoids re-fetching the same article content for overlapping topics.

**Scheduler ticks every minute.** Rather than scheduling one APScheduler job per topic, a single job ticks every minute and queries which topics are due. This avoids complex job lifecycle management when topics are added/removed/edited.

## Data Model

### Tables

| Table | Purpose |
|-------|---------|
| `topics` | Core entity. Name, description, `feed_urls` (JSON array), `feed_mode` (auto/manual), `status`, `is_active`, `status_changed_at`, `check_interval_minutes`, `tags` (JSON array), per-topic `confidence_threshold` / `relevance_threshold` (m011, nullable overrides), `init_attempts` (m013). |
| `articles` | Fetched articles linked to a topic. Deduped by `content_hash` (unique per topic). `source_provider` records the news provider (m009), `published_at` the feed entry's date (m018). `processed` flag tracks analysis completion. |
| `knowledge_states` | One per topic. Rolling LLM-generated summary. `token_count` tracks budget usage. |
| `check_results` | Audit log of every check cycle. Stores articles found/new, `has_new_info`, full LLM response JSON, notification outcome, `prompt_tokens` / `completion_tokens` (m012), and `stage_error` recording which pipeline stage failed (m015). |
| `pending_notifications` | Failed notifications queued for retry. Retried at the start of each check cycle. Deleted after `max_retries`. |
| `pending_webhooks` | Failed webhook deliveries queued for retry (m010). Stores `url`, `payload`, `retry_count`/`max_retries`. Retried at the start of each check cycle; expired entries pruned. |
| `feed_health` | Per-feed-URL health. Consecutive failures, total fetches/failures, last success/error timestamps, and `etag` / `last_modified` for HTTP conditional requests (m019). |
| `schema_version` | Migration tracking. Single `version` column. |

### Topic Lifecycle

```
  ┌──────────────┐               ┌──────────────┐    success    ┌─────────┐
  │     NEW      │──────────────>│ RESEARCHING  │──────────────>│  READY  │
  │ (OPML queue) │  one per tick │ (init phase) │               │ (active)│
  └──────────────┘               └──────┬───────┘               └────┬────┘
                                        │ failure                    │ LLM/knowledge error
                                        v                            v
                                 ┌──────────────┐             ┌──────────────┐
                                 │    ERROR     │             │    ERROR     │
                                 │ (user retry) │             │ (user retry) │
                                 └──────────────┘             └──────────────┘
```

Topics created through the UI start in **RESEARCHING**: articles are fetched and an initial knowledge state is built via the LLM. OPML imports instead create topics in **NEW**; the every-minute scheduler tick promotes one NEW topic at a time through initialization (gradual processing to avoid hammering the LLM API). On success, the topic moves to **READY** and enters the normal check cycle. On failure, it moves to **ERROR** with a user-visible message. Users can retry from the dashboard.

## Request Lifecycle

### Scheduled Check Cycle

1. APScheduler calls `_scheduled_check()` every 1 minute (with jitter), which runs `_run_check_cycle()` then initializes one NEW topic.
2. `retry_pending_notifications()` and `retry_pending_webhooks()` retry any failed deliveries from prior cycles (each manages its own short-lived connections).
3. `get_topics_due_for_check()` finds active READY topics whose last check exceeds their interval. Each topic check uses a fresh, short-lived connection so none is held across the HTTP/LLM awaits.
4. For each due topic, `check_topic()` runs with a unique correlation ID:
   - **Fetch** - `fetch_new_articles_for_topic()`: fetch feeds, dedup against DB, extract content.
   - **Analyze** - `analyze_articles()`: LLM compares articles against knowledge state.
   - **Update** - If `has_new_info`: update knowledge state via LLM.
   - **Notify** - Send notification via Apprise + webhooks. Queue for retry on failure.
   - **Record** - Mark articles processed, create `CheckResult`.
5. Each topic is independent. Errors in one do not affect others.

### Manual Check (Web UI)

1. User clicks "Check Now" on the topic detail page.
2. POST to `/topics/{id}/check` with CSRF token.
3. `CheckingState` ensures only one concurrent check per topic.
4. Check runs as a FastAPI `BackgroundTask` using the same `check_topic()` pipeline.

### Topic Creation

1. User submits topic form (name, description, feed URLs or auto mode).
2. Topic created in DB with `status=RESEARCHING`.
3. Background task: fetch articles → `initialize_knowledge()` via LLM → set `status=READY`.
4. On LLM failure: `status=ERROR` with error message. User can retry.

## Configuration

Settings are managed by Pydantic `BaseSettings` in `app/config.py`.

**Sources (highest priority first):**
1. Environment variables - prefix `TOPIC_WATCH_`, nested keys use `__` (e.g., `TOPIC_WATCH_LLM__API_KEY`)
2. YAML file - `data/config.yml`
3. Field defaults in `Settings` class

On first run, `config.example.yml` is auto-copied to `data/config.yml`.

**Runtime access:**
- Web routes: `request.app.state.settings`
- CLI / scheduler: `load_settings()`
- Settings page: writes back to YAML via `save_settings_to_yaml()`

### Configuration Key Reference

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `llm.model` | string | - | LiteLLM model string (e.g. `openai/gpt-5.4-nano`) |
| `llm.api_key` | string | - | API key for your LLM provider |
| `llm.base_url` | string | - | Base URL for self-hosted providers (Ollama, etc.) |
| `notifications.urls` | list | `[]` | [Apprise](https://github.com/caronc/apprise/wiki) notification URLs |
| `notifications.webhook_urls` | list | `[]` | Webhook endpoints for JSON POST (see [HTTP API](#http-api)) |
| `check_interval` | string | `"6h"` | Default check interval. Units: m, h, d, w, M. Combine: `1w 3d`, `2h 30m`. Min 10m, max 6M. |
| `max_articles_per_check` | int | `10` | Articles to process per check per topic (1-100) |
| `knowledge_state_max_tokens` | int | `2000` | Token budget for knowledge state (500-10,000) |
| `article_retention_days` | int | `90` | Days to keep articles before cleanup (1-3,650) |
| `db_path` | string | `data/topic_watch.db` | SQLite database path (relative or absolute) |
| `feed_fetch_timeout` | float | `15.0` | RSS feed fetch timeout (seconds) |
| `article_fetch_timeout` | float | `20.0` | Article content fetch timeout (seconds) |
| `llm_analysis_timeout` | int | `60` | LLM novelty analysis timeout (seconds) |
| `llm_knowledge_timeout` | int | `120` | LLM knowledge generation timeout (seconds) |
| `apprise_timeout_seconds` | int | `30` | Timeout for a single Apprise notification send (seconds) |
| `web_page_size` | int | `20` | Items per page in the web UI (5-200) |
| `feed_max_retries` | int | `2` | RSS feed fetch retries (1-10) |
| `feed_backoff_base_minutes` | int | `15` | Base backoff delay for a persistently-failing feed (minutes, 1-1,440). Env/YAML only. |
| `feed_backoff_cap_hours` | int | `24` | Max backoff delay for a failing feed (hours, 1-168). Env/YAML only. |
| `content_fetch_concurrency` | int | `3` | Concurrent article content fetches (1-20) |
| `topic_check_concurrency` | int | `3` | Concurrent per-topic checks within one scheduler tick (1-20) |
| `scheduler_misfire_grace_time` | int | `300` | APScheduler misfire grace time (seconds, 30-3,600) |
| `scheduler_jitter_seconds` | int | `30` | Random jitter per scheduler tick (seconds, 0-120) |
| `llm_max_retries` | int | `2` | LLM API call retries (0-10) |
| `llm_temperature` | float | `0.2` | LLM sampling temperature (0.0-2.0, lower = more factual) |
| `min_confidence_threshold` | float | `0.7` | Minimum LLM confidence to send notifications (0.0-1.0) |
| `min_relevance_threshold` | float | `0.5` | Minimum relevance to topic description to send notifications (0.0-1.0) |
| `secure_cookies` | bool | `false` | Set the Secure flag on cookies (enable when TLS terminates at a reverse proxy) |

Environment-only settings (no YAML equivalent):

| Variable | Default | Description |
|----------|---------|-------------|
| `TOPIC_WATCH_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `TOPIC_WATCH_LOG_FORMAT` | `text` | `text` or `json` |

## HTTP API

### JSON API v1

A read-only JSON API lives under `/api/v1`, plus one endpoint to trigger a check. Interactive docs are at `/docs` (OpenAPI/Swagger).

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/topics` | List topics. Optional query params: `active` (bool), `tag` (string) |
| `GET` | `/api/v1/topics/{id}` | One topic plus its knowledge state |
| `GET` | `/api/v1/topics/{id}/checks` | Check history, paginated (`page`, `per_page`; `per_page` capped at 100) |
| `GET` | `/api/v1/topics/{id}/knowledge` | Current knowledge state |
| `POST` | `/api/v1/topics/{id}/check` | Trigger a check. Runs synchronously; requires `X-CSRF-Token`. Returns `409` unless the topic status is `ready` |

The check endpoint returns `{"status": "checked", "has_new_info": <bool>, "check_result_id": <int>}`.

### Data Export & OPML

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/export/topics/json` | All topics as JSON |
| `GET` | `/export/opml` | All topics as OPML XML |
| `GET` | `/topics/{id}/export/json` | Single topic with articles, checks, knowledge state |
| `GET` | `/topics/{id}/export/csv` | Check history as CSV |

Move feeds in and out of RSS readers (FreshRSS, Miniflux, Tiny Tiny RSS) via OPML:

- **Export:** `GET /export/opml` downloads all topics as an OPML file.
- **Import:** `POST /import/opml` accepts an OPML upload (`opml_file` form field, 1 MB max, UTF-8). Imported topics start as `new` and initialize gradually (~1/min). Same-named topics are skipped.

### Webhook Payload

POST a JSON payload to any endpoint when new info is found:

```yaml
notifications:
  webhook_urls:
    - "https://your-server.com/webhook/topic-watch"
```

Payload:

```json
{
  "topic": "Topic Name",
  "reasoning": "Brief explanation of why this was flagged as new...",
  "summary": "...",
  "key_facts": ["...", "..."],
  "source_urls": ["https://..."],
  "confidence": 0.92,
  "relevance": 0.88,
  "timestamp": "2026-04-01T12:00:00+00:00"
}
```

10-second timeout per endpoint, concurrent delivery, failures logged but non-blocking.

## Error Handling

**Fail safe on notifications.** LLM analysis failure returns `has_new_info=False`. Users miss an update rather than receive a false alert.

**Fail loud on knowledge.** Knowledge init/update raises on LLM failure. The topic transitions to ERROR with a user-visible message so the problem is surfaced.

**Independent topic checks.** One topic's failure doesn't affect other topics in the same check cycle.

**Notification retry queue.** Failed deliveries are stored in `pending_notifications`. Retried at the start of each check cycle, up to `max_retries` (default 3), then discarded.

**Feed resilience.** Timeouts and 5xx errors get configurable retries. Feed health is tracked per-URL. Empty feeds are not errors.

**Stuck topic recovery.** Two distinct paths recover RESEARCHING topics to ERROR. At startup `recover_stuck_topics` clears *every* RESEARCHING topic immediately — after a restart the background task is dead, so any such topic is definitively stuck. During runtime the periodic scheduler job (every 5 min) calls `recover_stuck_researching`, which only recovers topics that have been RESEARCHING longer than the 15-minute timeout (via `status_changed_at`).

## Security Model

**No authentication.** Intentional for a single-user self-hosted tool. Remote deployments must use a reverse proxy with external auth (Authelia, Caddy, Nginx).

**CSRF.** Double-submit cookie on all POST/PUT/DELETE endpoints. HTMX sends the token via `X-CSRF-Token` header. Regular forms use a hidden field. Timing-safe HMAC comparison.

**SSRF protection.** `url_validation.py` blocks requests to private/reserved IP ranges (127.x, 10.x, 172.16-31.x, 192.168.x, 169.254.x, CGNAT 100.64.0.0/10, localhost, IPv6 ULA/link-local).

**XSS.** Jinja2 auto-escaping enabled. Error messages sanitized via `sanitize_error` template filter. Notification URLs masked in UI.

**SQL injection.** All queries use parameterized statements throughout `crud.py`.

**Rate limiting.** In-memory rate limiter on feed validation endpoint (10 requests per 60 seconds per IP).

**Docker hardening.** Non-root user (`appuser`), health check, `STOPSIGNAL SIGTERM`, 512 MB memory limit, log rotation.

**Sensitive data.** API keys and notification URLs stored in `data/config.yml` (gitignored). Notification URLs masked in the settings UI display.
