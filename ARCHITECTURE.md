# Architecture

## System Overview

Topic Watch monitors user-defined topics by fetching articles — from RSS feeds (an auto-generated news search, or manually configured URLs) or, per-topic, the optional Exa AI search API — then uses an LLM to determine whether the articles contain genuinely new information compared to what is already known. Notifications are sent only when something novel clears the topic's confidence, relevance and importance gates. Silence is the default.

```
                    ┌──────────────────────────┐
                    │  RSS Feeds / Exa Search   │
                    └────────────┬─────────────┘
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
                │  LLM Novelty Detection    │
                │  (has_new_info? yes/no)   │
                └─────────┬─────────────────┘
                          │
                ┌─────────┴──────────────────────┐
                │                                │
              yes                               no
                │                                │
                v                                v
   ┌───────────────────────────────┐       ┌───────────┐
   │ Clears confidence/relevance?   │       │  Record   │
   └──────────┬───────────────┬────┘       │  (done)   │
            no│               │yes         └───────────┘
              v               v
       ┌───────────┐   ┌────────────────────────────┐
       │  Record   │   │  Update Knowledge State     │
       │  (done)   │   │  Clears importance?         │
       └───────────┘   │  no  -> record only         │
                        │  yes -> record, THEN notify │
                        └─────────────────────────────┘
```

Record always lands before any notification is attempted — the durable transaction (see Request Lifecycle below) commits knowledge, article state and the `CheckResult` first, and only then are notification/webhook deliveries attempted.

**Four entry points trigger checks, all funneling through `check_topic()` / `check_all_topics()`:**
1. **APScheduler** - a background job ticks every 1 minute, queries which topics are due based on their individual `check_interval_minutes`, and runs the pipeline for each.
2. **Web UI** - users can trigger manual checks via the dashboard/topic detail page, which runs the same pipeline as a FastAPI background task.
3. **JSON API** - `POST /api/v1/topics/{id}/check` runs a check synchronously and returns its outcome.
4. **CLI** - `python -m app.cli check` / `check-all` run the same pipeline from a separate process.

The scheduler, web UI and API share one process, so their in-flight guards (`app.web.state._checking_state`) coordinate with each other. The CLI is a separate process with its own empty guard state at every invocation — it does not coordinate with a running server, and running it against the same database as a live server can double-check a topic and double-notify (see the CLI module's own docstring).

## Module Map

All application code lives under `app/`.

### Core Pipeline

| Module | Responsibility |
|--------|---------------|
| `checker.py` | Orchestrates the full check pipeline in phases that never hold a database connection across a network or LLM call: snapshot (P0) → fetch (P1) → analyze (P2) → generate the knowledge update (P3) → one durable transaction committing knowledge + article state + `CheckResult` + delivery intents (C3) → notification/webhook sends (P4) → record the delivery outcome (C4) → Silence Heartbeat (P5). `check_topic()` is the primary entry point. `check_all_topics()` iterates due topics. `retry_pending_notifications()` drains queued delivery intents; `retry_pending_check_intents()` resumes accepted manual checks a dead process left behind. `initialize_new_topic()` builds initial knowledge for NEW topics through the same snapshot/offline-work/durable-commit shape. |
| `scheduler.py` | APScheduler 3.x `AsyncIOScheduler`, four jobs: every-minute tick (`_scheduled_check` — runs the check cycle, then initializes one NEW topic), stuck-topic recovery (every 5 min, 15-min timeout), weekly VACUUM (Sun 3 AM), daily article + delivery-ledger cleanup (4 AM). Only the minute tick has jitter; all jobs coalesce and are single-instance. One process at a time runs a scheduler: `start_scheduler()` takes an advisory lock on `scheduler.lock` beside the database and returns `None` without one, so extra web workers serve requests only. Per-job outcomes (last success/error, consecutive failures, missed runs) are recorded and reported by `/health`. The check cycle (`_run_check_cycle`) snapshots due topics, then drains pending notification/webhook retries *alongside* the per-topic checks in one `asyncio.gather()` rather than ahead of them, so a retry backlog cannot delay a tick's due-topic work. |

### LLM Analysis

| Module | Responsibility |
|--------|---------------|
| `analysis/llm.py` | LiteLLM + Instructor wrappers. `NoveltyResponse` is the strict live provider contract (`relevance`/`importance` required, no defaults); `analyze_articles` decodes into it and converts to `NoveltyResult`, the permissive internal/stored shape every scoring field defaults on (so pre-existing `llm_response` blobs still re-parse). Also defines `KnowledgeStateUpdate` and `TokenUsage`. Token counting, rate limit backoff with exponential delay. Returns safe default (`has_new_info=False`, `confidence=0.0`) on analysis failure. |
| `analysis/prompts.py` | System and user prompt builders for novelty detection and knowledge init/update/compress. Articles truncated to 1500 chars in prompts. |
| `analysis/knowledge.py` | Knowledge state initialization and updates with DB persistence. Token budget enforcement via summary compression. |
| `analysis/restatement.py` | Pure phrase-matching filter (`filter_restated_key_facts`, re-exported by `llm.py`). Drops a key fact only when it is a clear restatement of the existing knowledge summary (normalized verbatim or long contiguous n-gram match), so already-known facts aren't re-flagged as new. Conservative by design. |
| `analysis/citations.py` | `strip_index_citations()` removes ephemeral `(Article [N])`-style citations from LLM output before it's persisted — they reference one run's article list and cause coherence drift if stored. |
| `analysis/knowledge_diff.py` | Pure `difflib` diffing for the revision timeline. `split_segments()` breaks a knowledge summary into sentence/bullet segments; `diff_segments()` returns `DiffSegment` records (added / removed / unchanged). No DB access, no LLM call — diffs are computed on read and never stored. |

### Scraping

| Module | Responsibility |
|--------|---------------|
| `scraping/__init__.py` | `fetch_new_articles_for_topic()` - orchestrates feed fetch, dedup against DB, cross-topic content reuse, concurrent content extraction (semaphore-limited), and article storage. |
| `scraping/source.py` | Neutral source layer shared by every fetcher (RSS/Exa alike): `FeedEntry`/`FeedResponse` DTOs, `FetchStatus`, article/story identity and dedup helpers, a monotonic per-attempt `Deadline`, and the `register_source`/`fetch_feeds_for_topic` mode-to-fetcher registry. Sources register themselves at import; this module imports no source. |
| `scraping/rss.py` | RSS/Atom feed fetching via httpx + feedparser. Converts entries to `FeedEntry` models. Retry on timeouts and 5xx. Feed health callbacks. |
| `scraping/content.py` | Article HTML fetch + trafilatura content extraction. Falls back to RSS summary on failure. Content truncated to 5000 chars at word boundary. |
| `scraping/providers.py` | News search provider definitions. `NewsProvider` Protocol plus `GoogleNewsProvider` / `BingNewsProvider` concrete classes that build keyword-search feed URLs from topic name + description (auto feed mode). |
| `scraping/routing.py` | Health-based provider cascade. Tracks per-provider health in-memory and selects the first healthy provider per cycle (Bing first, Google second). Separate from the per-URL `feed_health` table. |
| `scraping/exa.py` | Exa AI search source for EXA-mode topics. `fetch_exa_entries()` queries the Exa `/search` API and maps results straight to `FeedEntry`, bypassing feedparser. Same hardening as `webhooks.send_webhook` (scheme allowlist, offloaded SSRF check, no redirects, redacted logging, never raises). Exa returns page text, carried through as prefetched `FeedEntry.content` so the pipeline skips a second fetch. |
| `scraping/google_news.py` | Resolves opaque Google News redirect URLs (`news.google.com/rss/articles/...`) to real article URLs via Google's `batchexecute` endpoint. |

### Data Layer

| Module | Responsibility |
|--------|---------------|
| `models.py` | Pydantic models: `Topic`, `Article`, `KnowledgeState`, `KnowledgeRevision`, `CheckResult`, `FeedHealth`, `DashboardStats`, `PendingNotification`, `PendingWebhook`. Enums: `TopicStatus` (new/researching/ready/error), `FeedMode` (auto/manual/exa), `KnowledgeRevisionSource` (init/update). Each model has `from_row()` and `to_insert_dict()` for SQLite interop; datetime cells are coerced defensively. |
| `crud.py` | All SQL (parameterized), grouped by model: CRUD, feed-health upserts, notification + webhook retry queues, check-intent admission/claim/apply/release, dashboard aggregation, article retention cleanup, stuck-topic recovery. |
| `database.py` | SQLite connection factory (WAL mode, foreign keys, busy timeout). Schema init (`init_db`). `backup_database()` uses the sqlite3 backup API (not a file copy — safe under WAL) and runs `PRAGMA integrity_check`, discarding and raising `BackupVerificationError` on a bad backup. Migration runner (`run_migrations`) backs up the DB first, validates `schema_version` is an exact contiguous prefix of the registry (else `SchemaLedgerError`), then applies each pending migration's `up()` plus its ledger row inside one `BEGIN IMMEDIATE` per version. |
| `migrations/` | 29 sequential migrations (`m001`–`m029`) registered in `__init__.py` as `(version, description, up_function)` tuples. Tracked in `schema_version`. Append-only. |
| `interval.py` | Human-readable interval parsing/formatting (`m`/`h`/`d`/`w`/`M`, combined syntax like `"1w 3d 2h"`). Enforces min/max interval bounds. |
| `opml.py` | OPML import/export. Parses feeds from RSS readers (FreshRSS, Miniflux, TT-RSS), validates feed URLs, and exports topics as OPML. |

### Web

The route handlers live in the `web/routers/` package. The HTMX/HTML routes are mounted via an aggregate router; the JSON API lives separately in `web/api.py`.

| Module | Responsibility |
|--------|---------------|
| `web/routers/__init__.py` | Aggregate router. Includes the per-domain routers in include-order so static topic paths (`/topics/search`, `/topics/new`) register before the dynamic `/topics/{topic_id}` route. |
| `web/routers/dashboard.py` | Dashboard page, `/health` check (liveness plus a `scheduler` block reporting background-monitoring freshness; the status code reflects liveness only), and topic search. Computes dashboard stats directly on each request via `get_dashboard_stats()` — no cache; a TTL cache here used to lag every stat card for up to 60s after a mutation. |
| `web/routers/topics.py` | Topic CRUD, detail/articles pages, manual check + init triggers, and the knowledge revision timeline — `GET /topics/{topic_id}/knowledge-diff/{revision_id}` returns one revision's diff against its predecessor as an HTMX partial. |
| `web/routers/exports.py` | Data export endpoints: all-topics JSON (`/export/topics/json`) and per-topic JSON/CSV (`/topics/{id}/export/json`, `/topics/{id}/export/csv`). |
| `web/routers/settings.py` | Setup wizard, settings editor, and notification-test endpoint. Reads/writes config via `load_settings()` / `save_settings_to_yaml()`. |
| `web/routers/feed_health.py` | Global feed-health dashboard and feed-URL validation endpoint (rate-limited). |
| `web/routers/opml.py` | OPML import/export. Bulk topics-JSON export lives in `exports.py`, not here. |
| `web/routers/background.py` | Background-task helpers (`_run_init`, check-all) that run after the request connection closes, each opening its own DB connection. Coordinates via the shared `_checking_state`. |
| `web/routers/templates.py` | Shared `Jinja2Templates` instance and template filters (`timeago`, `sanitize_error`, `mask_url`, `confidence_badge`). Filters are module-level for unit testing. |
| `web/routers/_validation.py` | Shared topic-form validation (`validate_topic_form`) used by create and edit handlers. |
| `web/api.py` | JSON API v1 (`/api/v1`). Read-only endpoints (list/get topics, checks, knowledge) plus one CSRF-protected mutation to trigger a check. Reuses CRUD and Pydantic models. |
| `web/state.py` | Process-global web state: `CheckingState` (in-progress check tracking with stale-lock detection) and the in-memory feed-validation rate limiter. No cache — dashboard stats are queried at render time. |
| `web/csrf.py` | Signed double-submit cookie CSRF middleware + `verify_csrf` dependency. Sets an HMAC-signed token cookie on responses (an unsigned legacy cookie is adopted and upgraded in place), validates POST/PUT/DELETE via `Sec-Fetch-Site` first, then the `X-CSRF-Token` header (HTMX) or `csrf_token` form field. |
| `web/dependencies.py` | FastAPI dependency injection: `get_db_conn` (per-request connection with auto-commit/rollback), `get_settings` (from `app.state`). |
| `web/setup_middleware.py` | ASGI middleware that redirects all routes to `/setup` while `app.state.setup_required` is set (exempts `/setup`, `/health`, `/static`). |

### Infrastructure

| Module | Responsibility |
|--------|---------------|
| `main.py` | FastAPI app + lifespan. Runs migrations, starts/stops the scheduler, mounts the web routers, JSON API, and static files. Middleware stack (outermost first): `RequestIdMiddleware` (per-HTTP-request correlation id, distinct from a check's own correlation id — see `check_context.py` below) → `HostAllowlistMiddleware` (rejects an untrusted `Host` header before routing) → `SetupRedirectMiddleware` → `CSRFMiddleware`. |
| `config.py` | Pydantic `BaseSettings` with YAML source. Priority: env > YAML > defaults. `load_settings()` / `save_settings_to_yaml()`; `CLOUD_PROVIDERS` / `LOCAL_PROVIDER_DEFAULTS` provider lists. |
| `logging_config.py` | Plain text or JSON structured logging. Controlled by `TOPIC_WATCH_LOG_FORMAT` and `TOPIC_WATCH_LOG_LEVEL` env vars. |
| `check_context.py` | Correlation IDs via `contextvars.ContextVar`. `check_id_var` (per topic-check/init run) and `cycle_id_var` (per scheduler tick) are distinct from `main.py`'s `request_id_var` (per HTTP request) — a request that triggers a synchronous check carries both ids. `CheckIdFilter` injects whichever is set into all log records. |
| `url_validation.py` | SSRF protection. Blocks private/reserved IPs (localhost, 10.x, 172.16-31.x, 192.168.x, link-local, CGNAT 100.64.0.0/10, IPv6 ULA). |
| `feed_backoff.py` | `feed_backoff_until()` — stateless exponential backoff for persistently-failing feeds, computed from `feed_health` consecutive failures. Bounded by `feed_backoff_base_minutes` / `feed_backoff_cap_hours`. |
| `notifications.py` | Apprise wrapper. `format_notification()` formats a `NoveltyResult` into title/body. `send_single_notification()` delivers one per-target intent (sync Apprise call wrapped in `asyncio.to_thread()`). Re-exports `redact_url` from `log_redaction.py`. |
| `log_redaction.py` | Log-hygiene helper. `redact_url` strips userinfo, query strings, fragments, and long (likely-secret) path segments from notification/webhook URLs, keeping scheme + host + a short path prefix for diagnostics. |
| `webhooks.py` | JSON POST to configured webhook endpoints. `build_webhook_intents()` creates one durable per-target intent inside the check's C3 transaction; `deliver_webhook_intents()` claims, sends and applies the outcome for each, concurrently. `retry_pending_webhooks()` drains due intents alongside each check cycle's due topics. |
| `heartbeat.py` | Silence Heartbeat decision logic. `evaluate_heartbeat()` counts the leading run of failing `stage_error`s for a topic and returns a pure `HeartbeatAction` (alert, recovery, or nothing) with the formatted message. No DB writes, no sends — the checker owns the latch and the delivery. |
| `cli.py` | Argparse CLI: `list`, `check`, `check-all`, `init`, `doctor` (secret-safe diagnostic report for bug reports). |

### Frontend

- `templates/` - 19 Jinja2 templates. Pico CSS + HTMX base layout; partials for dynamic updates.
- `static/themes.css` - Color themes (Nord, Dracula, Solarized, High Contrast, Tokyo Night).
- `static/components.css` - Component styles (cards, badges, tables) layered on Pico.
- `static/theme.js` - Theme switcher with localStorage persistence.
- `static/notifications.js` - Browser push notification wrapper.
- `static/vendor/` - Vendored Pico CSS + HTMX. No build tooling.

## Key Design Decisions

**SQLite, not Postgres.** Single-user self-hosted tool. SQLite eliminates deployment complexity. WAL mode provides adequate concurrency for web server + background scheduler.

**LiteLLM for provider abstraction.** Users switch between OpenAI, Anthropic, Ollama, Gemini, or any supported provider by changing one config string. No provider-specific code in the app.

**Instructor for structured output.** Pydantic response models (`NoveltyResponse`, `KnowledgeStateUpdate`) with automatic validation retry against the live provider. `NoveltyResponse` — the strict wire contract — is converted field-by-field to `NoveltyResult`, the permissive shape actually stored. Eliminates JSON parsing fragility.

**Safe defaults on LLM failure.** `analyze_articles()` returns `has_new_info=False` on any error. Users miss an update rather than get a false alert. Knowledge operations raise because correctness is critical there.

**Knowledge state with token budget.** When a summary would exceed `knowledge_state_max_tokens`, the LLM is asked to compress it while preserving every fact; deterministic sentence-level truncation is only the fallback when that call fails or still overflows. Prevents unbounded context growth. Compression failure is swallowed, not raised — only the primary init/update generation raises.

**Knowledge revisions with capped retention.** Every knowledge write also appends a row to `knowledge_revisions` inside the SAME transaction as the state write (see C3 in Request Lifecycle below), not after it — a revision-append failure takes the state write down with it rather than leaving knowledge with no recorded history. Rows are never rewritten, only pruned oldest-first to `knowledge_revision_limit`, both on each write and once at startup for topics that have gone quiet. The checker never reads the table. Diffs are computed on read with `difflib`; no diff is ever stored.

**Apprise for notifications.** Supports 100+ services via URL format. No need for individual service integrations.

**No built-in authentication.** Deliberate choice for a single-user tool. Remote deployments use a reverse proxy with auth (Authelia, Caddy basicauth, etc.).

**Signed double-submit CSRF cookie.** Stateless CSRF protection compatible with HTMX: the cookie's value is HMAC-signed with a per-process secret, and a `Sec-Fetch-Site` check refuses cross-site submissions outright — clients that omit the header fall back to plain double-submit. Cookie is not httponly (HTMX reads it via JS). SameSite=Lax.

**Host allowlist against DNS rebinding.** `HostAllowlistMiddleware` rejects any `Host` header that is not `localhost`, an IP literal, or listed in `TOPIC_WATCH_ALLOWED_HOSTS`, so a hostile site cannot re-point its own domain at this machine and drive the console same-origin.

**Content hash dedup.** `content_hash` (`article_identity()` in `scraping/source.py`) digests the canonical URL, casefolded title, and the source's revision marker — not URL+title alone, so a correction or update published at the same URL is a new row rather than silently skipped (AUG-320). Cross-topic content reuse avoids re-fetching the same article content for overlapping topics.

**Scheduler ticks every minute.** Rather than scheduling one APScheduler job per topic, a single job ticks every minute and queries which topics are due. This avoids complex job lifecycle management when topics are added/removed/edited.

## Data Model

### Tables

| Table | Purpose |
|-------|---------|
| `topics` | Core entity. Name, description, `feed_urls` (JSON array), `feed_mode` (auto/manual/exa), `status`, `is_active`, `status_changed_at`, `check_interval_minutes`, `tags` (JSON array), per-topic `confidence_threshold` / `relevance_threshold` (m011, nullable overrides), `init_attempts` (m013), `novelty_instruction` (m022, nullable, ≤500 chars, injected into the novelty prompt), `importance_threshold` (m023, nullable 1-5; NULL = notify on any importance), `heartbeat_alerted_at` (m024, nullable Silence Heartbeat latch — stamped when a "sources failing" alert is sent, cleared on recovery), `generation` (m026, opaque per-incarnation id; fences a stale in-flight writer from a delete+recreate that recycled the rowid). |
| `articles` | Fetched articles linked to a topic. Deduped by `content_hash` (unique per topic) — a digest of canonical URL, title and the source's revision marker, not just URL+title, so a same-URL correction or update is treated as new instead of silently skipped (AUG-320). `source_provider` records the news provider (m009), `published_at` the feed entry's date (m018), `analysis_attempts` (m026, caps retries on a persistently-failing article at `MAX_ANALYSIS_ATTEMPTS`). `processed` flag tracks analysis completion. |
| `knowledge_states` | One per topic. Rolling LLM-generated summary. `token_count` tracks budget usage. `version` (m026) is a CAS counter: a write is rejected if the row moved since it was read, so two overlapping checks can never interleave a lost update. |
| `knowledge_revisions` | History of knowledge-state writes (m025), one row per init/update: `summary_text` (full copy), `token_count`, `source` (init/update), `change_note`, `created_at`, `model` / `basis_hash` (m029, nullable provenance — the LLM that wrote the summary and a fingerprint of the topic scope it was derived from, so the diff timeline can tell a real edit from a token-count unit change or a since-edited scope). Append-only, pruned oldest-first per topic to `knowledge_revision_limit` on each write and once at startup for quiet topics. |
| `check_results` | Audit log of every check cycle. Stores articles found/new, `has_new_info`, full LLM response JSON, `prompt_tokens` / `completion_tokens` (m012), `stage_error` recording which pipeline stage failed (m015), and `notify_disposition` (m026: `sent` / `pending` / `pending_knowledge_stale` / `suppressed_importance` / `below_confidence` / `below_relevance` / `no_new_info` / `analysis_failed` — why this check did or did not notify, distinct from a delivery failure). |
| `pending_notifications` | Durable per-target notification delivery intents (m026 lifecycle columns): `pending` → `sending` (claimed) → `sent` / `abandoned` / `revoked`. Created inside the check's durable transaction, before any send. Retried with backoff (`next_attempt_at`) up to `max_retries`, then `abandoned` and kept — sent/abandoned rows are the delivery ledger the dashboard reads, pruned by age on the daily maintenance tick. |
| `pending_webhooks` | The webhook half of the same thing (m010, plus m026's intent-lifecycle columns): `url`, `payload`, `retry_count`/`max_retries`, the same claim/retry/abandon lifecycle. |
| `check_intents` | Durable admission for an accepted manual check (m030): `request_id` (correlation only, not unique), `topic_id`, `baseline_check_id` (`MAX(check_results.id)` at admission — any newer row satisfies the intent), `status` (`pending` → `running` → `done` / `abandoned`), `attempts`/`max_attempts`, `next_attempt_at`, `claimed_at`/`claim_token`, `check_result_id`, `last_error`. One row per accepted topic is committed **before** the response, so a crash after it cannot lose the command. `attempts` counts claims, not completions, so a check that takes the process down is bounded too. A row a dead process left behind is re-armed and run by the scheduler's check cycle; terminal rows are pruned by age on the same daily tick as the delivery ledger. |
| `feed_health` | Per-feed-URL health. Consecutive failures, total fetches/failures, last success/error timestamps, and `etag` / `last_modified` for HTTP conditional requests (m019). |
| `schema_version` | Migration tracking. Single `version` column. |

### Topic Lifecycle

```
  ┌──────────────┐               ┌──────────────┐    success    ┌─────────┐
  │     NEW      │──────────────>│ RESEARCHING  │──────────────>│  READY  │
  │ (OPML queue) │  one per tick │ (init phase) │               │ (active)│
  └──────────────┘               └──────┬───────┘               └─────────┘
                                        │ init failure
                                        v
                                 ┌──────────────┐
                                 │    ERROR     │
                                 │ (user retry) │
                                 └──────────────┘
```

Topics created through the UI start in **RESEARCHING**: articles are fetched and an initial knowledge state is built via the LLM. OPML imports instead create topics in **NEW**; the every-minute scheduler tick promotes one NEW topic at a time through initialization (gradual processing to avoid hammering the LLM API). On success, the topic moves to **READY** and enters the normal check cycle. On failure — including an interrupted (Ctrl-C/cancelled) run — it moves to **ERROR** with a user-visible message. Users can retry from the dashboard.

**READY is a stable state.** A routine check's LLM or knowledge-update failure is recorded as that check's `stage_error` and does not move the topic out of READY — it keeps checking on schedule, and the next successful check clears the streak. Only two things move a topic to ERROR: initialization failure (above), and stuck-topic recovery for a RESEARCHING topic that outlives its timeout (see Error Handling below) — neither is a routine check outcome.

## Request Lifecycle

### Scheduled Check Cycle

1. APScheduler calls `_scheduled_check()` every 1 minute (with jitter), which runs `_run_check_cycle()` then initializes one NEW topic.
2. `_run_check_cycle()` snapshots the due topics on one short connection, then runs the notification/webhook retry drain and every due topic's check *concurrently* in one `asyncio.gather()` — the drain no longer runs ahead of due-topic work, so a retry backlog cannot delay a whole tick. `retry_pending_notifications()` / `retry_pending_webhooks()` each manage their own short-lived connections and commit per item.
3. `get_topics_due_for_check()` finds active READY topics whose last check exceeds their interval.
4. For each due topic, `check_topic()` runs with a unique correlation ID, in phases that never hold a database connection across a network or LLM call:
   - **P0 — snapshot.** Re-read the topic and its knowledge state on a short connection; a paused or deleted topic is caught here.
   - **P1 — fetch.** `fetch_new_articles_for_topic()`: fetch feeds/Exa results, dedup against the DB, extract content. Connection-free.
   - **P2 — analyze.** `analyze_articles()`: the LLM compares the batch (this cycle's fetch plus any article an earlier cycle stored but never finished analyzing) against the snapshotted knowledge state. Connection-free; never raises.
   - **P3 — knowledge plan.** If `has_new_info` and confidence/relevance clear their thresholds, generate the knowledge update. Still connection-free — nothing durable exists yet.
   - **C3 — the durable transaction.** One `BEGIN IMMEDIATE` commit applies the knowledge write, its revision, article disposition and the `CheckResult` together, fenced by the topic's generation and the knowledge version snapshotted at P0. Per-target notification and webhook delivery *intents* are created in this same commit, before any send is attempted — record always precedes notify, and a crash between "decided to notify" and "message sent" cannot lose the alert.
   - **P4 — send.** Only after C3 commits: each intent is claimed, sent, and its outcome applied. A finding below the topic's `importance_threshold` never reaches this step — the knowledge state already absorbed it in C3, so the same minor fact never re-flags as new.
   - **C4 — delivery outcome.** The `CheckResult` C3 committed is updated with the aggregate send outcome.
   - **P5 — Silence Heartbeat.** Runs last, over the row C3 just committed (see below).
5. Each topic is independent. Errors in one do not affect others.

**Silence Heartbeat (P5).** Each recorded check is classified by its `stage_error`:
`sources_failed`, `scrape_failed` and `sources_unavailable` all mean no source
produced usable results. `app/heartbeat.py` counts the leading run of those for a
topic; once it reaches `silence_heartbeat_checks`, the checker claims the
`topics.heartbeat_alerted_at` latch with a conditional UPDATE and sends one
alert. The latch is what makes it one alert per outage rather than one per check,
and the conditional UPDATE keeps it exactly-once even when a CLI `check-all` runs
alongside the server. The first check that sees a working source again clears the
latch and sends a recovery notice, only to the targets that actually received
the outage alert. Both messages go through the same per-target delivery-intent
path as a novelty notification and share the `pending_notifications` ledger;
webhooks are not fired for heartbeat events. Setting `silence_heartbeat_checks`
to 0 clears any outstanding latch on the next check, silently. The
dashboard/detail badge is derived from the newest check's `stage_error`, never
from the latch, so it always reflects the last check's real outcome.

### Manual Check (Web UI)

1. User clicks "Check Now" on the topic detail page.
2. POST to `/topics/{id}/check` with CSRF token.
3. One `check_intents` row per topic is committed before the response (AUG-286); a topic already being checked, or not READY, admits nothing.
4. The background task claims it and runs `check_topic()`; a row a dead process left behind is re-armed and run by the next scheduler check cycle — the page's poll may have ended by then, so the result shows on the next load. Check All Now is not admitted this way: it runs one scheduler cycle, and the next minute tick is its resume.

### Topic Creation

1. User submits topic form (name, description, feed URLs or auto mode).
2. Topic created in DB with `status=RESEARCHING` — an already-won claim, so `initialize_new_topic()` runs with `claimed=True`.
3. Background task: fetch articles (connection-free) → `prepare_initial_knowledge()` builds the LLM baseline (connection-free) → one durable transaction commits the knowledge state, its revision, article disposition and `status=READY` together.
4. On LLM/knowledge failure, or an interrupted (Ctrl-C/cancelled) run: `status=ERROR` with an error message. User can retry.

## Configuration

Settings are managed by Pydantic `BaseSettings` in `app/config.py`.

**Sources (highest priority first):**
1. Environment variables - prefix `TOPIC_WATCH_`, nested keys use `__` (e.g., `TOPIC_WATCH_LLM__API_KEY`)
2. YAML file - `data/config.yml`
3. Field defaults in `Settings` class

On first run, `config.example.yml` is auto-copied to `data/config.yml`.

**State root.** `resolve_state_root()` decides where `config.yml` and the default database
live, and both resolve through it so they cannot diverge. Highest priority first:

1. `TOPIC_WATCH_CONFIG_PATH` - the pinned file's directory.
2. Whichever candidate already holds a `config.yml` or a `topic_watch.db`, `data/` beside the
   package before the user-level directory. This is what makes the root sticky: an install
   that has written state keeps reading it, and a `data/` directory that appears later
   (`docker compose` creating the bind-mount source, a documented `mkdir -p data`) cannot
   take over from a populated user-level directory.
3. An existing `data/` beside the package - repo checkout, worktree, the container's
   `/app/data`.
4. `data/` beside the package when that directory is writable and is not `site-packages` /
   `dist-packages`. A source checkout lands here on a fresh install, created by the first
   write.
5. A user-level state directory (`$XDG_STATE_HOME` or `~/.local/state`, `%LOCALAPPDATA%` on
   Windows) - an installed wheel, or a package root that cannot be written to.

`TOPIC_WATCH_DB_PATH` stays authoritative for the database on top of that.

**Runtime access:**
- Web routes: `request.app.state.settings`
- Scheduler ticks: also `app.state.settings`, resolved live at the start of each tick (`_resolve_settings()`), so a Settings-page save takes effect on the next tick with no restart. Falls back to the settings bound at `start_scheduler()` only when no app is wired (unit tests calling the scheduler directly). Trigger-shaped fields (`scheduler_jitter_seconds`, `scheduler_misfire_grace_time`) are copied into the jobs at add time, so the minute tick re-applies them (`reconfigure_scheduler()`) when they change — live, within one tick, like every other setting.
- CLI: `load_settings()` — each invocation is a fresh process that reads env/YAML from disk
- Settings page: `POST /settings` saves YAML via `save_settings_to_yaml()` and replaces `app.state.settings` with the resolved object in the same request. Editing `data/config.yml` or the environment directly, outside that route, needs a restart to take effect.

**Writing YAML.** `save_settings_to_yaml()` patches the model's own dump onto the existing
document rather than rebuilding a hand-maintained key list: keys this version does not know
are preserved (they warn on load), a cleared optional value deletes its key, and any field
the environment currently supplies is skipped entirely so an env-derived value is never
materialized into the file. The write goes to a same-directory temp file and one atomic
rename, owner-readable only. A symlinked `config.yml` is resolved first, so the link survives
the rename and the change reaches the file it points at.

**Environment-owned controls.** `env_owned_field_paths()` is the single source of provenance.
The Settings page renders every control whose field the environment owns as disabled with a
read-only note, and the setup wizard does the same for the model and API key - a control the
save path strips must not present itself as editable.

### Configuration Key Reference

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `llm.model` | string | - | LiteLLM model string (e.g. `openai/gpt-5.4-nano`) |
| `llm.api_key` | string | - | API key for your LLM provider |
| `llm.base_url` | string | - | Base URL for a self-hosted (Ollama) or OpenAI-compatible gateway endpoint. Honored for any provider. |
| `exa.enabled` | bool | `false` | Enable the [Exa](https://exa.ai) AI search source for EXA-mode topics |
| `exa.api_key` | string | `""` | Exa API key. Also settable via `TOPIC_WATCH_EXA__API_KEY`, in which case the Settings page shows it read-only and never writes it to YAML |
| `exa.base_url` | string | - | Exa endpoint override (advanced/proxy). Defaults to `https://api.exa.ai` |
| `notifications.urls` | list | `[]` | [Apprise](https://github.com/caronc/apprise/wiki) notification URLs |
| `notifications.webhook_urls` | list | `[]` | Webhook endpoints for JSON POST (see [HTTP API](#http-api)) |
| `check_interval` | string | `"6h"` | Default check interval. Units: m, h, d, w, M. Combine: `1w 3d`, `2h 30m`. Min 10m, max 6M. |
| `max_articles_per_check` | int | `10` | Articles to process per check per topic (1-100) |
| `knowledge_state_max_tokens` | int | `2000` | Token budget for knowledge state (500-10,000) |
| `knowledge_revision_limit` | int | `50` | Knowledge revisions retained per topic for the diff timeline (2-200). Config-only; no Settings-page field |
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
| `silence_heartbeat_checks` | int | `3` | Consecutive checks with no usable source before a Silence Heartbeat alert (0-50, 0 disables) |
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

The check endpoint returns the recorded outcome: `status`, `check_result_id`, `has_new_info`, `articles_found`, `articles_new`, `stage_error`, `notification_sent`, `notification_error`, `notify_disposition` (AUG-203) — the pipeline is fail-safe, so an unreachable source or a failed delivery still returns `200` with `has_new_info=false`, and these fields are what let a caller tell that apart from a clean quiet run.

### Data Export & OPML

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/export/topics/json` | All topics as JSON |
| `GET` | `/export/opml` | All topics as OPML XML |
| `GET` | `/topics/{id}/export/json` | Single topic with articles, checks, knowledge state |
| `GET` | `/topics/{id}/export/csv` | Check history as CSV |

Knowledge revisions are not exported. The per-topic JSON carries the current
knowledge state; the revision history is a UI affordance for reading what
changed, not part of the data contract.

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
  "importance": 4,
  "timestamp": "2026-04-01T12:00:00+00:00"
}
```

`importance` is the model's 1-5 significance rating (1 = trivial, 5 = major). A
topic with an `importance_threshold` set only delivers findings that meet it;
below-threshold findings still update the knowledge state, they just do not send.

10-second timeout per endpoint, concurrent delivery, failures logged but non-blocking.

## Error Handling

**Fail safe on notifications.** LLM analysis failure returns `has_new_info=False`. Users miss an update rather than receive a false alert.

**Fail loud on knowledge — but only during initialization.** `initialize_new_topic()` raises on LLM/knowledge failure and the topic transitions to ERROR with a user-visible message. A routine check's knowledge update also raises internally, but `check_topic()` catches it, keeps the topic READY, and records the failure as that check's `stage_error` — a knowledge-update failure alone never disables a working topic.

**Independent topic checks.** One topic's failure doesn't affect other topics in the same check cycle.

**Durable notification/webhook delivery intents.** Every intended send becomes a durable per-target row inside the check's C3 transaction, before any network call — a crash between deciding to notify and the message leaving cannot lose it. The retry drain claims due rows (backed off via `next_attempt_at`) alongside each check cycle's due topics, up to `max_retries` (default 3) attempts, after which a row is marked `abandoned` and kept as the delivery ledger rather than deleted.

**Feed resilience.** Timeouts and 5xx errors get configurable retries. Feed health is tracked per-URL. Empty feeds are not errors.

**Stuck topic recovery.** Two distinct paths recover RESEARCHING topics to ERROR. At startup `recover_stuck_topics` clears *every* RESEARCHING topic immediately — after a restart the background task is dead, so any such topic is definitively stuck. During runtime the periodic scheduler job (every 5 min) calls `recover_stuck_researching`, which only recovers topics that have been RESEARCHING longer than the 15-minute timeout (via `status_changed_at`).

## Security Model

**No authentication.** Intentional for a single-user self-hosted tool. Remote deployments must use a reverse proxy with external auth (Authelia, Caddy, Nginx).

**CSRF.** Signed double-submit cookie on all POST/PUT/DELETE endpoints, plus a `Sec-Fetch-Site` check that refuses cross-site submissions outright. HTMX sends the token via `X-CSRF-Token` header. Regular forms use a hidden field. Timing-safe HMAC comparison.

**Host allowlist.** `HostAllowlistMiddleware` rejects any request whose `Host` header is not `localhost`/`*.localhost`/`*.local`, an IP literal, or listed in `TOPIC_WATCH_ALLOWED_HOSTS` — closes the DNS-rebinding path where a hostile site re-points its own domain at this machine to drive the console same-origin.

**SSRF protection.** `url_validation.py` blocks requests to private/reserved IP ranges (127.x, 10.x, 172.16-31.x, 192.168.x, 169.254.x, CGNAT 100.64.0.0/10, localhost, IPv6 ULA/link-local).

**XSS.** Jinja2 auto-escaping enabled. Error messages sanitized via `sanitize_error` template filter. Notification URLs masked in UI.

**SQL injection.** All queries use parameterized statements throughout `crud.py`.

**Rate limiting.** In-memory rate limiter on feed validation endpoint (10 requests per 60 seconds per IP).

**Docker hardening.** Non-root user (`appuser`), health check, `STOPSIGNAL SIGTERM`, 512 MB memory limit, log rotation.

**Sensitive data.** API keys and notification URLs stored in `data/config.yml` (gitignored). Notification URLs masked in the settings UI display.
