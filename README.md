# Topic Watch

[![CI](https://github.com/0xzerolight/topic_watch/actions/workflows/ci.yml/badge.svg)](https://github.com/0xzerolight/topic_watch/actions/workflows/ci.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Self-hosted news monitoring with LLM-powered novelty detection. Watches topics via RSS feeds, notifies you only when genuinely new information appears. BYOK (bring your own key).

## How It Works

1. Define topics with RSS feed URLs, or let it auto-generate a Google News feed
2. On a schedule, articles are fetched and compared against a **knowledge state** (a rolling summary of what's already known)
3. An LLM decides if anything is actually new
4. New info: notification with summary + sources. Nothing new: silence.

## Features

- Auto feeds (Google News) or manual RSS/Atom URLs
- Per-topic check intervals (10 min to 1 week)
- Topic tags
- 100+ notification services via [Apprise](https://github.com/caronc/apprise/wiki) (Discord, Slack, Telegram, email, ntfy, etc.)
- Custom JSON webhooks
- Notification retry queue
- Feed health dashboard
- Data export (JSON, CSV)
- Bulk check/delete
- 5 color themes (Nord, Dracula, Solarized Dark, High Contrast, Tokyo Night)
- In-app settings page
- CLI: `list`, `check`, `check-all`, `init`

## Quick Start

### One-line install (requires Docker)

**Linux / macOS:**

```bash
curl -fsSL https://raw.githubusercontent.com/0xzerolight/topic_watch/main/install.sh | bash
```

**Windows (PowerShell):**

```powershell
irm https://raw.githubusercontent.com/0xzerolight/topic_watch/main/install.ps1 | iex
```

Pulls the image, starts the container, creates a desktop shortcut + auto-start, and opens the setup wizard. Set your LLM API key in the wizard.

Override install location and port:

```bash
# Linux / macOS
TOPIC_WATCH_DIR=~/my-path TOPIC_WATCH_PORT=9000 curl -fsSL .../install.sh | bash

# Windows (PowerShell)
$env:TOPIC_WATCH_DIR="C:\TopicWatch"; $env:TOPIC_WATCH_PORT="9000"; irm .../install.ps1 | iex
```

<details>
<summary><strong>Manual setup</strong></summary>

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

</details>

Then visit [http://localhost:8000](http://localhost:8000) to configure.

## Configuration

Settings live in `data/config.yml`. First run auto-copies `config.example.yml`. Editable via the web UI Settings page or directly in the file.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `llm.model` | string | - | LiteLLM model string (e.g. `openai/gpt-4o-mini`) |
| `llm.api_key` | string | - | API key for your LLM provider |
| `llm.base_url` | string | - | Base URL for self-hosted providers (Ollama, etc.) |
| `notifications.urls` | list | `[]` | [Apprise](https://github.com/caronc/apprise/wiki) notification URLs |
| `notifications.webhook_urls` | list | `[]` | Webhook endpoints for JSON POST (see [Webhooks](#webhooks)) |
| `check_interval_hours` | int | `6` | Default hours between checks per topic (1-168) |
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
| `web_page_size` | int | `20` | Items per page in the web UI (5-200) |
| `feed_max_retries` | int | `2` | RSS feed fetch retries (1-10) |
| `content_fetch_concurrency` | int | `3` | Concurrent article content fetches (1-20) |
| `scheduler_misfire_grace_time` | int | `300` | APScheduler misfire grace time (seconds, 30-3,600) |
| `scheduler_jitter_seconds` | int | `30` | Random jitter per scheduler tick (seconds, 0-120) |
| `llm_max_retries` | int | `2` | LLM API call retries (0-10) |

</details>

### Environment Variables

All settings can be overridden with `TOPIC_WATCH_` prefix. Double underscores for nested keys:

```bash
TOPIC_WATCH_LLM__API_KEY=sk-abc123
TOPIC_WATCH_LLM__MODEL=openai/gpt-4o-mini
TOPIC_WATCH_CHECK_INTERVAL_HOURS=4
TOPIC_WATCH_NOTIFICATIONS__WEBHOOK_URLS='["https://example.com/hook"]'
```

Environment-only settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `TOPIC_WATCH_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `TOPIC_WATCH_LOG_FORMAT` | `text` | `text` or `json` |

## Adding Topics

1. Dashboard > **Add Topic**
2. Fill in: **Name**, **Description** (what you care about in plain English), **Feed Source** (Automatic/Manual), **Feed URLs** (if Manual, one per line), **Check Interval**, **Tags**
3. **Save**

The topic enters a "Researching" phase where it fetches articles and builds an initial knowledge state. This takes under a minute. After that, it enters the normal check cycle.

### Finding RSS Feeds

- Try appending `/rss`, `/feed`, or `/atom.xml` to a site URL
- Reddit: `https://www.reddit.com/r/SUBREDDIT/search.rss?q=QUERY&sort=new`
- Most blogs use `/feed` or `/index.xml`

## LLM Providers

Uses [LiteLLM](https://docs.litellm.ai/docs/providers). Anything LiteLLM supports works.

| Provider | Model String | Notes |
|----------|-------------|-------|
| OpenAI | `openai/gpt-4o-mini` | Good cost/quality balance |
| Anthropic | `anthropic/claude-3-5-haiku-latest` | |
| Ollama | `ollama/llama3` | Free, local. Set `llm.base_url` |
| Google Gemini | `gemini/gemini-2.0-flash` | |
| Azure OpenAI | `azure/your-deployment` | |
| Cohere | `cohere/command-r` | |
| Together AI | `together_ai/meta-llama/Llama-3-8b-chat-hf` | |

### Ollama

```yaml
llm:
  model: "ollama/llama3"
  api_key: "unused"
  base_url: "http://host.docker.internal:11434"  # or http://localhost:11434 outside Docker
```

## Notifications

100+ services supported via [Apprise](https://github.com/caronc/apprise/wiki) URL format:

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

### Webhooks

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
  "summary": "...",
  "key_facts": ["...", "..."],
  "source_urls": ["https://..."],
  "confidence": 0.92,
  "timestamp": "2026-04-01T12:00:00+00:00"
}
```

10-second timeout per endpoint, concurrent delivery, failures logged but non-blocking.

## Data Export

| Endpoint | Description |
|----------|-------------|
| `GET /export/topics/json` | All topics |
| `GET /topics/{id}/export/json` | Single topic with articles, checks, knowledge state |
| `GET /topics/{id}/export/csv` | Check history as CSV |

## CLI

```bash
python -m app.cli list                # List all topics
python -m app.cli check "Topic Name"  # Check single topic
python -m app.cli check-all           # Check all topics
python -m app.cli init "Topic Name"   # Re-initialize knowledge state
```

## Security

**No built-in authentication** by design (single-user tool).

- **Localhost:** safe as-is
- **Remote:** put it behind a reverse proxy with auth ([Authelia](https://www.authelia.com/), [Authentik](https://goauthentik.io/), Nginx basic auth, Caddy `basicauth`)

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

API keys stored in `data/config.yml` (gitignored) or env vars. All data stays on your machine; outbound connections only go to RSS feeds, your LLM provider, and notification services.

See [SECURITY.md](SECURITY.md) for vulnerability reporting.

## Troubleshooting

**Config file not found** - Run `mkdir -p data && cp config.example.yml data/config.yml`.

**LLM errors / checks failing** - Check your API key, make sure the model string has the provider prefix (`openai/gpt-4o-mini`, not `gpt-4o-mini`), check logs with `docker compose logs -f`.

**No notifications** - Check `notifications.urls` in config. Use the Test Notification button on the Settings page. Verify the [Apprise URL format](https://github.com/caronc/apprise/wiki).

**0 articles found** - Verify the RSS URL works in a browser. Check the Feed Health page. Some sites block bots.

**Topic stuck in "Researching"** - Auto-recovers after 15 minutes (set to Error). Retry from the topic page. Usually an LLM connectivity issue.

**Docker container exits** - `docker compose logs` for details. Check that `data/config.yml` exists and `data/` is writable.

**High memory** - Lower `max_articles_per_check` or `content_fetch_concurrency`. Increase check intervals.

## FAQ

**Cost?** ~1,700 tokens per check. GPT-4o-mini: ~$0.0003-0.001/check. 5 topics, 4x/day = ~$0.42/month. Ollama: free.

**Why not Google Alerts?** Google Alerts sends every mention. Topic Watch only notifies when something is *actually* new.

**Data privacy?** Everything runs locally. Only outbound traffic is to your LLM provider and notification services.

**No API key?** Use Ollama or any local LLM. Set `llm.base_url` and put any string for `llm.api_key`.

**No RSS feeds?** Pick "Automatic" when adding a topic. Uses Google News RSS.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
