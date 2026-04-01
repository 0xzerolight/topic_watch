# Topic Watch

[![CI](https://github.com/0xzerolight/topic_watch/actions/workflows/ci.yml/badge.svg)](https://github.com/0xzerolight/topic_watch/actions/workflows/ci.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

> Google Alerts but actually smart and self-hosted.

A self-hosted news watchdog that monitors user-defined topics and sends notifications **only when genuinely new information is found**. Uses an LLM to distinguish real updates from filler articles that rehash old news. You bring your own API key.

## How It Works

1. You define topics you care about and provide RSS feed URLs
2. On a schedule, Topic Watch fetches articles from those feeds
3. New articles are compared against a **knowledge state** — a rolling summary of everything already known about the topic
4. An LLM determines if the articles contain genuinely new information
5. If yes: you get a notification with a summary and source links. If no: silence.

The default state is **no notification**. You only hear from Topic Watch when something actually matters.

<!-- Screenshots: capture at 1200px width, save to docs/screenshots/, uncomment below.
![Dashboard](docs/screenshots/dashboard.png)
![Topic Detail](docs/screenshots/topic-detail.png)
-->

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

This pulls the Docker image, starts Topic Watch, sets up a desktop shortcut and auto-start, and opens the setup wizard in your browser. Configure your LLM API key in the wizard and you're ready to go.

Customize the install with environment variables:

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
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

</details>

Visit [http://localhost:8000](http://localhost:8000) — the setup wizard will guide you through configuring your LLM provider, then you can start adding topics.

## Configuration

All settings live in `data/config.yml`. On first run, the setup wizard at [http://localhost:8000/setup](http://localhost:8000/setup) will configure the essentials for you. You can also edit the file directly.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `llm.model` | string | — | LiteLLM model string (e.g. `openai/gpt-4o-mini`) |
| `llm.api_key` | string | — | API key for your LLM provider |
| `llm.base_url` | string | — | Base URL for self-hosted providers (e.g. Ollama) |
| `notifications.urls` | list | `[]` | Apprise notification URLs (see [Notification Services](#notification-services)) |
| `check_interval_hours` | int | `6` | Hours between automatic checks (1–168) |
| `max_articles_per_check` | int | `10` | Max articles to process per check per topic (1–100) |
| `knowledge_state_max_tokens` | int | `2000` | Token budget for the knowledge state summary (500–10000) |

### Environment Variable Overrides

All settings can be overridden with environment variables using the `TOPIC_WATCH_` prefix. Use double underscores for nested keys:

```bash
TOPIC_WATCH_LLM__API_KEY=sk-abc123
TOPIC_WATCH_LLM__MODEL=openai/gpt-4o-mini
TOPIC_WATCH_CHECK_INTERVAL_HOURS=4
```

This is useful for Docker deployments where you prefer not to mount a config file.

## Adding Topics

1. Open the dashboard at `http://localhost:8000`
2. Click **Add Topic**
3. Fill in:
   - **Name** — a short label (e.g. "GTA 6 Release")
   - **Description** — what you care about in plain English (e.g. "Release date, gameplay trailers, and official announcements for GTA 6")
   - **Feed URLs** — one RSS/Atom feed URL per line
4. Click **Save**

Topic Watch will immediately start an **initial research** phase: it fetches recent articles from your feeds and builds a knowledge state of everything currently known. This prevents false notifications for existing information. The topic status will show "Researching" until this completes (usually under a minute).

Once ready, the topic enters the normal check cycle and you'll only be notified when something genuinely new is published.

### Finding RSS Feeds

Most news sites have RSS feeds. Common patterns:

- Look for an RSS icon on the site, or try appending `/rss`, `/feed`, or `/atom.xml` to the URL
- Reddit: `https://www.reddit.com/r/SUBREDDIT/search.rss?q=QUERY&sort=new`
- Many blogs use `/feed` or `/index.xml`

## Supported LLM Providers

Topic Watch uses [LiteLLM](https://docs.litellm.ai/docs/providers) for provider abstraction. Any provider supported by LiteLLM works.

| Provider | Example Model String | Notes |
|----------|---------------------|-------|
| OpenAI | `openai/gpt-4o-mini` | Recommended for cost/quality balance |
| Anthropic | `anthropic/claude-3-haiku-20240307` | |
| Ollama | `ollama/llama3` | Free, runs locally. Set `llm.base_url` |
| Google Gemini | `gemini/gemini-pro` | |
| Azure OpenAI | `azure/your-deployment` | |
| Cohere | `cohere/command-r` | |
| Together AI | `together_ai/meta-llama/Llama-3-8b-chat-hf` | |

### Using Ollama (Free, Local)

```yaml
llm:
  model: "ollama/llama3"
  api_key: "unused"
  base_url: "http://host.docker.internal:11434"
```

Use `host.docker.internal` when Topic Watch runs in Docker and Ollama runs on the host machine. If both run outside Docker, use `http://localhost:11434`.

## Notification Services

Topic Watch uses [Apprise](https://github.com/caronc/apprise/wiki) for notifications. 100+ services are supported via URL configuration. Common examples:

| Service | URL Format |
|---------|-----------|
| Ntfy | `ntfy://your-topic` |
| Discord | `discord://webhook_id/webhook_token` |
| Telegram | `tgram://bot_token/chat_id` |
| Slack | `slack://token_a/token_b/token_c/channel` |
| Email (Gmail) | `mailto://user:app_password@gmail.com` |
| Pushover | `pover://user_key@api_token` |

See the [Apprise wiki](https://github.com/caronc/apprise/wiki) for the full list of supported services and URL formats.

Multiple notification URLs can be configured — notifications will be sent to all of them:

```yaml
notifications:
  urls:
    - "ntfy://my-news-tracker"
    - "discord://webhook_id/webhook_token"
```

## Running Without Docker

```bash
python -m venv .venv
source .venv/bin/activate
pip install .

mkdir -p data
cp config.example.yml data/config.yml
# Edit data/config.yml with your settings

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### CLI Commands

Topic Watch also includes a CLI for manual operations:

```bash
python -m app.cli list                # List all topics
python -m app.cli check "Topic Name"  # Check a single topic now
python -m app.cli check-all           # Check all topics now
python -m app.cli init "Topic Name"   # Re-initialize a topic's knowledge state
```

## Security

**Topic Watch has no built-in authentication.** The web UI is open by design for simplicity.

- **Local machine**: Safe to use as-is on `localhost`.
- **VPS / remote server**: You **must** place it behind a reverse proxy with authentication. Options include:
  - [Authelia](https://www.authelia.com/)
  - [Authentik](https://goauthentik.io/)
  - Nginx with HTTP basic auth
  - Caddy with `basicauth`

<details>
<summary>Example: Caddy reverse proxy with basic auth</summary>

```
topic-watch.example.com {
    basicauth {
        admin $2a$14$YOUR_HASHED_PASSWORD
    }
    reverse_proxy localhost:8000
}
```

Generate a password hash with: `caddy hash-password`
</details>

<details>
<summary>Example: Nginx with HTTP basic auth</summary>

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

Create credentials with: `htpasswd -c /etc/nginx/.htpasswd admin`
</details>

Your API keys are stored in `data/config.yml` (gitignored) or passed via environment variables. All data stays on your machine — the only outbound connections are to news sources and your LLM provider.

See [SECURITY.md](SECURITY.md) for reporting vulnerabilities.

## Troubleshooting

**App won't start / "Config file not found"**

Copy the example config: `cp config.example.yml data/config.yml`. The `data/` directory must exist and contain a valid `config.yml` before startup.

**"LLM API error" / checks fail**

- Verify your API key is correct and has credits
- Ensure the model string uses the provider prefix (e.g., `openai/gpt-4o-mini`, not just `gpt-4o-mini`)
- Check server logs: `docker compose logs -f` or the terminal output when running without Docker

**No notifications received**

- Verify notification URLs are set in `data/config.yml` under `notifications.urls`
- Use the **Test Notification** button on the Settings page to confirm delivery
- Double-check the Apprise URL format for your service in the [Apprise wiki](https://github.com/caronc/apprise/wiki)

**Feeds not fetching / "0 articles found"**

- Confirm the RSS URL is valid by opening it in a browser
- Check the Feed Health page in the dashboard for per-feed status
- Some sites block automated requests; try a different feed source for the same content

**Docker container exits immediately**

- Run `docker compose logs` to see the error details
- Ensure `data/config.yml` exists and is valid YAML
- Confirm the `data/` directory is writable

**High memory usage**

- Reduce `max_articles_per_check` in config (default: 10)
- Reduce `content_fetch_concurrency` if set
- Increase check intervals to reduce concurrent activity

## FAQ

**How much does it cost?**

Each check uses ~1,700 LLM tokens. With GPT-4o-mini, that's ~$0.0003–0.001 per check. For 5 topics checked 4 times per day: about **$0.42/month**. With Ollama (local): **$0**.

**Why not Google Alerts?**

Google Alerts sends every mention without filtering for significance. Topic Watch uses AI to determine whether news is *meaningfully new* versus rehashed content. The default state is silence — you only get pinged when something actually changes.

**Is my data private?**

Yes. Everything runs on your machine. No data is sent anywhere except to your configured LLM provider (for article analysis) and notification services (for alerts).

**Can I use it without an API key?**

Yes, if you run a local LLM via Ollama or another compatible server. Set `llm.base_url` to your local endpoint and use any string for `llm.api_key`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.
