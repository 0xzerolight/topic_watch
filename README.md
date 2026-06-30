<h1 align="center">Topic Watch</h1>

<p align="center">
  <a href="https://github.com/0xzerolight/topic_watch/actions/workflows/ci.yml"><img src="https://github.com/0xzerolight/topic_watch/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://www.gnu.org/licenses/gpl-3.0"><img src="https://img.shields.io/badge/License-GPLv3-blue.svg" alt="License: GPL v3"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"></a>
  <a href="https://github.com/0xzerolight/topic_watch"><img src="https://img.shields.io/badge/coverage-94%25-brightgreen.svg" alt="Coverage"></a>
  <a href="https://github.com/0xzerolight/topic_watch/stargazers"><img src="https://img.shields.io/github/stars/0xzerolight/topic_watch?style=social" alt="GitHub stars"></a>
</p>

<p align="center">
Self-hosted news monitor that pings you only on genuinely new info.
</p>

<p align="center">
Please leave a ⭐ star if Topic Watch is useful — it helps others find it :).
</p>

<p align="center">
  <img src="assets/themes-showcase.gif" alt="Topic Watch — theme showcase" width="720">
</p>

An LLM tracks a per-topic knowledge state and stays silent until something actually changes. Not keyword matching, not summarization. Bring your own key, or run free against a local model.

<details>
<summary><strong>How It Works</strong></summary>

1. Define a topic with RSS feed URLs, or let it auto-generate a news-search feed (Bing News first, Google News as fallback).
2. On a schedule, articles are fetched and compared against a **knowledge state** — a rolling summary of what's already known.
3. An LLM decides if anything is actually new.
4. New info → notification with summary + sources. Nothing new → silence.

</details>

## Quick Start

Runs in Docker. Get it at [get.docker.com](https://get.docker.com) (or install [Docker Desktop](https://www.docker.com/products/docker-desktop/) on macOS/Windows), and make sure it's running.

**Linux / macOS:**

```bash
curl -fsSL https://raw.githubusercontent.com/0xzerolight/topic_watch/main/scripts/install.sh | bash
```

**Windows (PowerShell):**

```powershell
irm https://raw.githubusercontent.com/0xzerolight/topic_watch/main/scripts/install.ps1 | iex
```

Pulls the image, starts the container, creates a desktop/Start Menu shortcut, and opens the setup wizard at [http://localhost:8000](http://localhost:8000). Set your LLM API key there. Boot/login autostart is opt-in — the installer asks (or pass `TOPIC_WATCH_AUTOSTART=yes` for a non-interactive run).

> ⚠️ Piping a script to a shell runs whatever the URL returns. These scripts fetch from the mutable `main` branch by default. For a verifiable install, review the script first and pin a release with `TOPIC_WATCH_REF` — see [SECURITY.md](SECURITY.md#install-script-trust).

Override install location, port, ref, and autostart:

```bash
# Linux / macOS
TOPIC_WATCH_DIR=~/my-path TOPIC_WATCH_PORT=9000 TOPIC_WATCH_REF=v1.1.2 TOPIC_WATCH_AUTOSTART=yes \
  curl -fsSL https://raw.githubusercontent.com/0xzerolight/topic_watch/v1.1.2/scripts/install.sh | bash

# Windows (PowerShell)
$env:TOPIC_WATCH_DIR="C:\TopicWatch"; $env:TOPIC_WATCH_PORT="9000"; $env:TOPIC_WATCH_REF="v1.1.2"; $env:TOPIC_WATCH_AUTOSTART="yes"
irm https://raw.githubusercontent.com/0xzerolight/topic_watch/v1.1.2/scripts/install.ps1 | iex
```

<details>
<summary><strong>Manual setup (with or without Docker)</strong></summary>

#### With Docker

```bash
git clone https://github.com/0xzerolight/topic_watch.git
cd topic_watch
docker compose up -d
```

#### Without Docker

```bash
git clone https://github.com/0xzerolight/topic_watch.git
cd topic_watch
python -m venv .venv && source .venv/bin/activate
pip install .
mkdir -p data && cp config.example.yml data/config.yml
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

> **Run a single worker.** The scheduler, the in-memory dashboard stats cache, and the "already checking" guard all live in-process. Do not pass `--workers N` (or run multiple replicas) — extra workers each start their own scheduler and keep separate state. The default Docker image runs one worker.

</details>

<details>
<summary><strong>File permissions (PUID / PGID — native Linux only)</strong></summary>

Topic Watch stores its database and config in the bind-mounted `./data` directory. Docker keeps your host's file ownership on bind mounts, so the container has to write `./data` as *your* user. The image defaults to UID/GID `1000` (the first user on most Linux hosts), and the container entrypoint chowns `./data` to match `PUID`/`PGID` on startup.

If your host user is not `1000`, set `PUID`/`PGID` so the container writes as you. The one-line installer handles this automatically and is safe to re-run.

For manual installs, use `grep -q` to avoid duplicating lines on re-run:

```bash
# In the directory containing docker-compose.yml
for kv in "PUID=$(id -u)" "PGID=$(id -g)"; do
    key="${kv%%=*}"
    grep -q "^${key}=" .env 2>/dev/null \
        && sed -i "s|^${key}=.*|${kv}|" .env \
        || echo "${kv}" >> .env
done
docker compose up -d
```

macOS and Windows (Docker Desktop) handle ownership transparently, so `PUID`/`PGID` are only relevant on native Linux hosts.

</details>

<details>
<summary><strong>Run with Ollama (no API key)</strong></summary>

If you run [Ollama](https://ollama.ai) locally, Topic Watch can use it for free LLM-powered novelty detection:

```bash
# 1. Start Ollama and pull a model (8B+ recommended for novelty detection)
ollama pull llama3.3

# 2. Start Topic Watch with the Ollama config
cp docker-compose.override.example.yml docker-compose.override.yml
docker compose up -d
```

The override file sets `ollama/llama3.3` as the model and points to your local Ollama instance. No API key required.

On native Linux Docker Engine, `host.docker.internal` does not resolve on its own; the override file maps it via `extra_hosts: ["host.docker.internal:host-gateway"]` so the container can reach Ollama on the host. macOS and Windows Docker Desktop resolve `host.docker.internal` automatically.

Models with 8B+ parameters and 8K+ context windows work best. Tested with `llama3.3` (8B), `mistral` (7B), and `qwen2.5` (7B). Smaller models may miss subtle novelty signals.

</details>

## Features

- Novelty detection: per-topic knowledge state, not keyword matching or summarization — ignores the 10th article rehashing the same story
- Any LLM via [LiteLLM](https://docs.litellm.ai/docs/providers) — OpenAI, Anthropic, Gemini, Groq, and more. BYOK, or run free and local with Ollama
- Private and self-hosted on SQLite — no database server, no JavaScript build step. Outbound traffic only goes to RSS feeds, your LLM provider, and your notifier
- Auto feeds (Bing News, falling back to Google News) or manual RSS/Atom URLs
- Per-topic check intervals (10 min to 6 months, human-readable: `6h`, `1w 3d`, `2h 30m`)
- Topic tags
- 100+ notification services via [Apprise](https://github.com/caronc/apprise/wiki) (Discord, Slack, Telegram, email, ntfy, etc.)
- Custom JSON webhooks
- Notification retry queue
- Feed health dashboard
- Data export (JSON, CSV) and OPML import/export
- Bulk check/delete
- 5 color themes (Nord, Dracula, Solarized Dark, High Contrast, Tokyo Night)
- In-app settings page
- CLI: `list`, `check`, `check-all`, `init`

## Adding Topics

1. Dashboard → **Add Topic**.
2. Fill in **Name**, **Description** (what you care about in plain English), **Feed Source** (Automatic/Manual), **Feed URLs** (if Manual, one per line), **Check Interval**, **Tags**.
3. **Save**.

The topic enters a "Researching" phase where it fetches articles and builds an initial knowledge state. This takes under a minute. After that, it enters the normal check cycle.

**Finding RSS feeds:**

- Try appending `/rss`, `/feed`, or `/atom.xml` to a site URL.
- Reddit: `https://www.reddit.com/r/SUBREDDIT/search.rss?q=QUERY&sort=new`
- Most blogs use `/feed` or `/index.xml`.

## Setup Wizard

On first launch with no valid configuration, Topic Watch redirects to a setup wizard at `/setup`. Enter your LLM model string (LiteLLM `provider/model-name` format) and API key; a base URL field appears for self-hosted providers like Ollama. On submit, it runs a quick pre-flight check against your provider and tells you what failed instead of saving a broken config. Everything is editable later from the **Settings** page or in `data/config.yml`.

## LLM Providers

Uses [LiteLLM](https://docs.litellm.ai/docs/providers). Anything LiteLLM supports works.

| Provider | Model String | Notes |
|----------|-------------|-------|
| OpenAI | `openai/gpt-5.4-nano` | Cheapest OpenAI option |
| Anthropic | `anthropic/claude-haiku-4-5` | Fast, good quality |
| Ollama | `ollama/llama3.3` | Free, local. Set `llm.base_url` |
| Google Gemini | `gemini/gemini-2.5-flash` | |
| Groq | `groq/llama-3.3-70b-versatile` | Very fast inference |
| DeepSeek | `deepseek/deepseek-chat` | Very cheap |
| Azure OpenAI | `azure/your-deployment` | |
| Cohere | `cohere_chat/command-a-03-2025` | |
| Together AI | `together_ai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8` | |

Ollama config:

```yaml
llm:
  model: "ollama/llama3.3"
  api_key: "unused"
  base_url: "http://host.docker.internal:11434"  # or http://localhost:11434 outside Docker
```

## Notifications

100+ services via [Apprise](https://github.com/caronc/apprise/wiki) URL format:

| Service | URL Format |
|---------|-----------|
| Ntfy | `ntfy://your-topic` |
| Discord | `discord://webhook_id/webhook_token` |
| Telegram | `tgram://bot_token/chat_id` |
| Slack | `slack://token_a/token_b/token_c/channel` |
| Email (Gmail) | `mailto://user:app_password@gmail.com` |
| Pushover | `pover://user_key@api_token` |

Multiple URLs supported. Use the **Test Notification** button on the Settings page to verify.

```yaml
notifications:
  urls:
    - "ntfy://my-news-tracker"
    - "discord://webhook_id/webhook_token"
```

<details>
<summary><strong>Custom JSON webhooks</strong></summary>

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

</details>

## Configuration

Settings live in `data/config.yml`. First run auto-copies `config.example.yml` (or run `mkdir -p data && cp config.example.yml data/config.yml` yourself). Editable via the web UI Settings page or directly in the file.

**Priority (highest to lowest):** environment variables (`TOPIC_WATCH_` prefix) > `data/config.yml` > built-in defaults.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `llm.model` | string | - | LiteLLM model string (e.g. `openai/gpt-5.4-nano`) |
| `llm.api_key` | string | - | API key for your LLM provider |
| `llm.base_url` | string | - | Base URL for self-hosted providers (Ollama, etc.) |
| `notifications.urls` | list | `[]` | [Apprise](https://github.com/caronc/apprise/wiki) notification URLs |
| `notifications.webhook_urls` | list | `[]` | Webhook endpoints for JSON POST (see [Notifications](#notifications)) |
| `check_interval` | string | `"6h"` | Default check interval. Units: m, h, d, w, M. Combine: `1w 3d`, `2h 30m`. Min 10m, max 6M. |
| `max_articles_per_check` | int | `10` | Articles to process per check per topic (1-100) |
| `knowledge_state_max_tokens` | int | `2000` | Token budget for knowledge state (500-10,000) |
| `article_retention_days` | int | `90` | Days to keep articles before cleanup (1-3,650) |

<details>
<summary><strong>Advanced settings</strong></summary>

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `db_path` | string | `data/topic_watch.db` | SQLite database path (relative or absolute) |
| `feed_fetch_timeout` | float | `15.0` | RSS feed fetch timeout (seconds) |
| `article_fetch_timeout` | float | `20.0` | Article content fetch timeout (seconds) |
| `llm_analysis_timeout` | int | `60` | LLM novelty analysis timeout (seconds) |
| `llm_knowledge_timeout` | int | `120` | LLM knowledge generation timeout (seconds) |
| `apprise_timeout_seconds` | int | `30` | Timeout for a single Apprise notification send (seconds) |
| `web_page_size` | int | `20` | Items per page in the web UI (5-200) |
| `feed_max_retries` | int | `2` | RSS feed fetch retries (1-10) |
| `content_fetch_concurrency` | int | `3` | Concurrent article content fetches (1-20) |
| `scheduler_misfire_grace_time` | int | `300` | APScheduler misfire grace time (seconds, 30-3,600) |
| `scheduler_jitter_seconds` | int | `30` | Random jitter per scheduler tick (seconds, 0-120) |
| `llm_max_retries` | int | `2` | LLM API call retries (0-10) |
| `llm_temperature` | float | `0.2` | LLM sampling temperature (0.0-2.0, lower = more factual) |
| `min_confidence_threshold` | float | `0.7` | Minimum LLM confidence to send notifications (0.0-1.0) |
| `min_relevance_threshold` | float | `0.5` | Minimum relevance to topic description to send notifications (0.0-1.0) |
| `secure_cookies` | bool | `false` | Set the Secure flag on cookies (enable when TLS terminates at a reverse proxy) |

</details>

<details>
<summary><strong>Environment variables</strong></summary>

All settings can be overridden with the `TOPIC_WATCH_` prefix. Double underscores for nested keys:

```bash
TOPIC_WATCH_LLM__API_KEY=sk-abc123
TOPIC_WATCH_LLM__MODEL=openai/gpt-5.4-nano
TOPIC_WATCH_CHECK_INTERVAL=4h
TOPIC_WATCH_NOTIFICATIONS__WEBHOOK_URLS='["https://example.com/hook"]'
```

Environment-only settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `TOPIC_WATCH_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `TOPIC_WATCH_LOG_FORMAT` | `text` | `text` or `json` |

</details>

## CLI

```bash
python -m app.cli list                # List all topics
python -m app.cli check "Topic Name"  # Check single topic
python -m app.cli check-all           # Check all topics
python -m app.cli init "Topic Name"   # Re-initialize knowledge state
```

> **Run the CLI only when the server is stopped.** The in-flight check guards are process-local, so the CLI does not coordinate with a running server's scheduler. Pointing it at a live server's database can double-check a topic, double-spend the LLM, and send duplicate notifications. Stop the server first, or use a separate/offline database.

<details>
<summary><strong>JSON API</strong></summary>

A read-only JSON API lives under `/api/v1`, plus one endpoint to trigger a check. Interactive docs are at `/docs` (OpenAPI/Swagger).

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/topics` | List topics. Optional query params: `active` (bool), `tag` (string) |
| `GET` | `/api/v1/topics/{id}` | One topic plus its knowledge state |
| `GET` | `/api/v1/topics/{id}/checks` | Check history, paginated (`page`, `per_page`; `per_page` capped at 100) |
| `GET` | `/api/v1/topics/{id}/knowledge` | Current knowledge state |
| `POST` | `/api/v1/topics/{id}/check` | Trigger a check. Runs synchronously; requires `X-CSRF-Token`. Returns `409` unless the topic status is `ready` |

The check endpoint returns `{"status": "checked", "has_new_info": <bool>, "check_result_id": <int>}`.

</details>

<details>
<summary><strong>Data export & OPML</strong></summary>

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/export/topics/json` | All topics as JSON |
| `GET` | `/export/opml` | All topics as OPML XML |
| `GET` | `/topics/{id}/export/json` | Single topic with articles, checks, knowledge state |
| `GET` | `/topics/{id}/export/csv` | Check history as CSV |

Move feeds in and out of RSS readers (FreshRSS, Miniflux, Tiny Tiny RSS) via OPML:

- **Export:** `GET /export/opml` downloads all topics as an OPML file.
- **Import:** `POST /import/opml` accepts an OPML upload (`opml_file` form field, 1 MB max, UTF-8). Imported topics start as `new` and initialize gradually (~1/min). Same-named topics are skipped.

</details>

## Updating

**Docker (one-line install):**

```bash
cd ~/topic-watch  # or your install directory
docker compose pull
docker compose up -d
```

The database is automatically backed up before any schema migration.

**Docker (git clone):**

```bash
cd topic_watch
git pull
docker compose up -d --build
```

**Without Docker:**

```bash
cd topic_watch
git pull
pip install .
# Restart your uvicorn process
```

Check the [CHANGELOG](CHANGELOG.md) before upgrading for breaking changes.

## Security

**No built-in authentication** by design (single-user tool).

- **Localhost:** safe as-is.
- **Remote:** put it behind a reverse proxy with auth ([Authelia](https://www.authelia.com/), [Authentik](https://goauthentik.io/), Nginx basic auth, Caddy `basicauth`).

API keys are stored in `data/config.yml` (gitignored) or env vars. All data stays on your machine; outbound connections only go to RSS feeds, your LLM provider, and notification services. See [SECURITY.md](SECURITY.md) for vulnerability reporting.

<details>
<summary>Caddy example</summary>

```
topic-watch.example.com {
    basicauth {
        admin $2a$14$YOUR_HASHED_PASSWORD
    }
    reverse_proxy localhost:8000
}
```

Generate hash: `caddy hash-password`
</details>

<details>
<summary>Nginx example</summary>

```nginx
server {
    listen 443 ssl;
    server_name topic-watch.example.com;

    auth_basic "Topic Watch";
    auth_basic_user_file /etc/nginx/.htpasswd;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Create credentials: `htpasswd -c /etc/nginx/.htpasswd admin`
</details>

## Troubleshooting

| Issue | Fix |
|-------|-----|
| **Config file not found** | Run `mkdir -p data && cp config.example.yml data/config.yml`. |
| **LLM errors / checks failing** | Check your API key. Make sure the model string has the provider prefix (`openai/gpt-5.4-nano`, not `gpt-5.4-nano`). Check logs: `docker compose logs -f`. |
| **No notifications** | Check `notifications.urls` in config. Use the Test Notification button on Settings. Verify the [Apprise URL format](https://github.com/caronc/apprise/wiki). |
| **0 articles found** | Verify the RSS URL works in a browser. Check the Feed Health page. Some sites block bots. |
| **Topic stuck in "Researching"** | Auto-recovers after 15 minutes (set to Error). Retry from the topic page. Usually an LLM connectivity issue. |
| **Docker container exits** | `docker compose logs` for details. Check that `data/config.yml` exists and `data/` is writable. "Permission denied" on `data/` on native Linux usually means your UID isn't 1000 — set `PUID`/`PGID` (see [File permissions](#quick-start)). |
| **High memory** | Lower `max_articles_per_check` or `content_fetch_concurrency`. Increase check intervals. |

## FAQ

**Cost?** ~1,700 tokens per check. GPT-5.4 Nano: ~$0.0003/check. 5 topics, 4×/day = under $0.20/month. Ollama: free.

**Data privacy?** Everything runs locally. The only outbound traffic is to your LLM provider and notification services.

**No API key?** Use Ollama or any local LLM. Set `llm.base_url` and put any string for `llm.api_key`.

**No RSS feeds?** Pick "Automatic" when adding a topic. It builds a news-search RSS feed, using Bing News first and falling back to Google News if Bing is unhealthy.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

GNU General Public License v3.0. See [LICENSE](LICENSE).
