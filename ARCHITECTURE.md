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
1. **APScheduler** — a background job ticks every 1 minute, queries which topics are due based on their individual `check_interval_minutes`, and runs the pipeline for each.
2. **Web UI** — users can trigger manual checks via the dashboard, which runs the same pipeline as a FastAPI background task.

## Module Map

All application code lives under `app/`.

### Core Pipeline

| Module | Responsibility |
|--------|---------------|
| `checker.py` | Orchestrates the full check pipeline: fetch → analyze → notify → record. `check_topic()` is the primary entry point. `check_all_topics()` iterates due topics. `retry_pending_notifications()` handles failed deliveries. |
| `scheduler.py` | APScheduler 3.x `AsyncIOScheduler`. Four jobs: topic checks (every 1 min), stuck topic recovery (every 5 min), article cleanup (daily 4 AM), VACUUM (weekly Sunday 3 AM). All jobs coalesce, single instance, with jitter. |

### LLM Analysis

| Module | Responsibility |
|--------|---------------|
| `analysis/llm.py` | LiteLLM + Instructor wrappers. Defines `NoveltyResult` and `KnowledgeStateUpdate` response models. Rate limit backoff with exponential delay. Returns safe default (`has_new_info=False`) on analysis failure. |
| `analysis/prompts.py` | System and user prompt templates for novelty detection and knowledge management. Articles truncated to 1000 chars in prompts. |
| `analysis/knowledge.py` | Knowledge state initialization and updates. Token budget enforcement via sentence-level truncation. |

### Scraping

| Module | Responsibility |
|--------|---------------|
| `scraping/__init__.py` | `fetch_new_articles_for_topic()` — orchestrates feed fetch, dedup against DB, cross-topic content reuse, concurrent content extraction (semaphore-limited), and article storage. |
| `scraping/rss.py` | RSS/Atom feed fetching via httpx + feedparser. Google News RSS URL builder for auto feed mode. Retry on timeouts and 5xx. Feed health callbacks. |
| `scraping/content.py` | Article HTML fetch + trafilatura content extraction. Falls back to RSS summary on failure. Content truncated to 5000 chars at word boundary. |

### Data Layer

| Module | Responsibility |
|--------|---------------|
| `models.py` | Pydantic models: `Topic`, `Article`, `KnowledgeState`, `CheckResult`, `FeedHealth`, `PendingNotification`. Enums: `TopicStatus`, `FeedMode`. Each model has `from_row()` and `to_insert_dict()` for SQLite interop. |
| `crud.py` | All database operations grouped by model. Topic/Article/KnowledgeState/CheckResult CRUD, feed health upserts, pending notification queue, dashboard aggregation, article retention cleanup, stuck topic recovery. |
| `database.py` | SQLite connection factory (WAL mode, foreign keys, busy timeout). Schema initialization. Migration runner. |
| `migrations/` | 8 sequential migrations registered in `__init__.py` as `(version, description, up_function)` tuples. Tracked in `schema_version` table. |

### Web

| Module | Responsibility |
|--------|---------------|
| `web/routes.py` | All FastAPI endpoints + HTMX partials. Dashboard, topic CRUD, manual check triggering, settings page, data export, feed health. `CheckingState` class for in-progress check tracking with stale lock detection. Rate limiter. Template filters (`timeago`, `sanitize_error`, `mask_url`, `confidence_badge`). |
| `web/csrf.py` | Double-submit cookie CSRF middleware. Sets token cookie on responses, validates on POST/PUT/DELETE via `X-CSRF-Token` header (HTMX) or form field. |
| `web/dependencies.py` | FastAPI dependency injection: `get_db_conn` (per-request connection with auto-commit/rollback), `get_settings` (from `app.state`). |

### Infrastructure

| Module | Responsibility |
|--------|---------------|
| `config.py` | Pydantic `BaseSettings` with YAML source. Priority: env > YAML > defaults. `load_settings()` / `save_settings_to_yaml()`. |
| `logging_config.py` | Plain text or JSON structured logging. Controlled by `TOPIC_WATCH_LOG_FORMAT` and `TOPIC_WATCH_LOG_LEVEL` env vars. |
| `check_context.py` | Correlation IDs via `contextvars.ContextVar`. `CheckIdFilter` injects check ID into all log records. |
| `url_validation.py` | SSRF protection. Blocks private/reserved IPs (localhost, 10.x, 172.16-31.x, 192.168.x, link-local, IPv6 ULA). |
| `notifications.py` | Apprise wrapper. Formats `NoveltyResult` into title/body. Sync Apprise send wrapped in `asyncio.to_thread()`. |
| `webhooks.py` | JSON POST to configured webhook endpoints. Concurrent delivery via `asyncio.gather()`. Fire-and-forget with logging. |
| `cli.py` | Argparse CLI: `list`, `check`, `check-all`, `init`. |

### Frontend

- `templates/` — 12 Jinja2 templates. Base layout with Pico CSS + HTMX. HTMX partials for dynamic updates.
- `static/themes.css` — Custom color themes (Nord, Dracula, Solarized, High Contrast, Tokyo Night).
- `static/theme.js` — Theme switcher with localStorage persistence.
- `static/notifications.js` — Browser push notification wrapper.
- `static/vendor/` — Vendored Pico CSS and HTMX. No build tooling.

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
| `topics` | Core entity. Name, description, `feed_urls` (JSON array), `feed_mode` (auto/manual), `status`, `is_active`, `check_interval_minutes`, `tags` (JSON array). |
| `articles` | Fetched articles linked to a topic. Deduped by `content_hash` (unique per topic). `processed` flag tracks analysis completion. |
| `knowledge_states` | One per topic. Rolling LLM-generated summary. `token_count` tracks budget usage. |
| `check_results` | Audit log of every check cycle. Stores articles found/new, `has_new_info`, full LLM response JSON, notification outcome. |
| `pending_notifications` | Failed notifications queued for retry. Retried at the start of each check cycle. Deleted after `max_retries`. |
| `feed_health` | Per-feed-URL health. Consecutive failures, total fetches/failures, last success/error timestamps. |
| `schema_version` | Migration tracking. Single `version` column. |

### Topic Lifecycle

```
  ┌──────────────┐    success    ┌─────────┐
  │ RESEARCHING  │──────────────>│  READY  │
  │ (init phase) │               │ (active)│
  └──────┬───────┘               └────┬────┘
         │ failure                    │ LLM/knowledge error
         v                           v
  ┌──────────────┐           ┌──────────────┐
  │    ERROR     │<──────────│    ERROR     │
  │ (user retry) │           │ (user retry) │
  └──────────────┘           └──────────────┘
```

New topics start in **RESEARCHING** — articles are fetched and an initial knowledge state is built via the LLM. On success, the topic moves to **READY** and enters the normal check cycle. On failure, it moves to **ERROR** with a user-visible message. Users can retry from the dashboard.

## Request Lifecycle

### Scheduled Check Cycle

1. APScheduler calls `_scheduled_check()` every 1 minute (with jitter).
2. `retry_pending_notifications()` retries any failed deliveries from prior cycles.
3. `get_topics_due_for_check()` finds active READY topics whose last check exceeds their interval.
4. For each due topic, `check_topic()` runs with a unique correlation ID:
   - **Fetch** — `fetch_new_articles_for_topic()`: fetch feeds, dedup against DB, extract content.
   - **Analyze** — `analyze_articles()`: LLM compares articles against knowledge state.
   - **Update** — If `has_new_info`: update knowledge state via LLM.
   - **Notify** — Send notification via Apprise + webhooks. Queue for retry on failure.
   - **Record** — Mark articles processed, create `CheckResult`.
5. Each topic is independent — errors in one do not affect others.

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
1. Environment variables — prefix `TOPIC_WATCH_`, nested keys use `__` (e.g., `TOPIC_WATCH_LLM__API_KEY`)
2. YAML file — `data/config.yml`
3. Field defaults in `Settings` class

On first run, `config.example.yml` is auto-copied to `data/config.yml`.

**Runtime access:**
- Web routes: `request.app.state.settings`
- CLI / scheduler: `load_settings()`
- Settings page: writes back to YAML via `save_settings_to_yaml()`

## Error Handling

**Fail safe on notifications.** LLM analysis failure returns `has_new_info=False`. Users miss an update rather than receive a false alert.

**Fail loud on knowledge.** Knowledge init/update raises on LLM failure. The topic transitions to ERROR with a user-visible message so the problem is surfaced.

**Independent topic checks.** One topic's failure doesn't affect other topics in the same check cycle.

**Notification retry queue.** Failed deliveries are stored in `pending_notifications`. Retried at the start of each check cycle, up to `max_retries` (default 3), then discarded.

**Feed resilience.** Timeouts and 5xx errors get configurable retries. Feed health is tracked per-URL. Empty feeds are not errors.

**Stuck topic recovery.** Topics stuck in RESEARCHING are recovered to ERROR status both on startup (`recover_stuck_topics`) and by a periodic scheduler job (15-minute timeout).

## Security Model

**No authentication.** Intentional for a single-user self-hosted tool. Remote deployments must use a reverse proxy with external auth (Authelia, Caddy, Nginx).

**CSRF.** Double-submit cookie on all POST/PUT/DELETE endpoints. HTMX sends the token via `X-CSRF-Token` header. Regular forms use a hidden field. Timing-safe HMAC comparison.

**SSRF protection.** `url_validation.py` blocks requests to private/reserved IP ranges (127.x, 10.x, 172.16-31.x, 192.168.x, 169.254.x, localhost, IPv6 ULA/link-local).

**XSS.** Jinja2 auto-escaping enabled. Error messages sanitized via `sanitize_error` template filter. Notification URLs masked in UI.

**SQL injection.** All queries use parameterized statements throughout `crud.py`.

**Rate limiting.** In-memory rate limiter on feed validation endpoint (10 requests per 60 seconds per IP).

**Docker hardening.** Non-root user (`appuser`), health check, `STOPSIGNAL SIGTERM`, 512 MB memory limit, log rotation.

**Sensitive data.** API keys and notification URLs stored in `data/config.yml` (gitignored). Notification URLs masked in the settings UI display.
