<h1 align="center">Topic Watch</h1>

<p align="center">
  <a href="https://www.gnu.org/licenses/gpl-3.0"><img src="https://img.shields.io/badge/License-GPLv3-blue.svg" alt="License: GPL v3"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"></a>
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

An LLM tracks a per-topic knowledge state and stays silent until something actually changes. Bring your own key, or run free against a local model.

## How It Works

1. Define a topic with RSS feed URLs, or let it auto-generate a news-search feed (Bing News first, Google News as fallback).
2. On a schedule, articles are fetched and compared against a **knowledge state** — a rolling summary of what's already known.
3. An LLM decides if anything is actually new.
4. New info → notification with summary + sources. Nothing new → silence.

## Install

### 1. Install Docker

Topic Watch runs in Docker. Get it at [get.docker.com](https://get.docker.com), or install [Docker Desktop](https://www.docker.com/products/docker-desktop/) on macOS/Windows. Make sure it's running before you continue.

### 2. Install Topic Watch

**Linux / macOS:**

```bash
curl -fsSL https://raw.githubusercontent.com/0xzerolight/topic_watch/main/scripts/install.sh | bash
```

**Windows (PowerShell):**

```powershell
irm https://raw.githubusercontent.com/0xzerolight/topic_watch/main/scripts/install.ps1 | iex
```

This pulls the image, starts the container, and opens the setup wizard at [http://localhost:8000](http://localhost:8000) — set your LLM API key there.

<details>
<summary><strong>Manual install (without the script)</strong></summary>

**Docker, prebuilt image** — same image the script uses, you just supply the compose file:

```bash
mkdir -p topic-watch/data && cd topic-watch
curl -fsSL https://raw.githubusercontent.com/0xzerolight/topic_watch/main/docker-compose.prod.yml -o docker-compose.yml
printf 'PUID=%s\nPGID=%s\n' "$(id -u)" "$(id -g)" > .env   # only if your host UID isn't 1000
docker compose up -d
```

**Build from source** — no prebuilt image, builds from the Dockerfile:

```bash
git clone https://github.com/0xzerolight/topic_watch.git
cd topic_watch
docker compose up -d
```

**Without Docker** (Python 3.11+):

```bash
git clone https://github.com/0xzerolight/topic_watch.git
cd topic_watch
python -m venv .venv && source .venv/bin/activate
pip install .
mkdir -p data && cp config.example.yml data/config.yml
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

</details>

## Features

- Novelty detection: per-topic knowledge state, not keyword matching or summarization — ignores the 10th article rehashing the same story
- Any LLM via [LiteLLM](https://docs.litellm.ai/docs/providers) — OpenAI, Anthropic, Gemini, Groq, and more. BYOK, or run free and local with Ollama
- Cheap: ~$0.0003/check on GPT-5.4 Nano (under $0.20/month for 5 topics checked 4×/day), or free with Ollama
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

## Adding Topics

1. Dashboard → **Add Topic**.
2. Fill in **Name**, **Description** (what you care about in plain English), **Feed Source** (Automatic/Manual), **Feed URLs** (if Manual, one per line), **Check Interval**, **Tags**.
3. **Save**.

The topic enters a "Researching" phase where it fetches articles and builds an initial knowledge state (under a minute), then enters the normal check cycle.

**Finding RSS feeds:**

- Try appending `/rss`, `/feed`, or `/atom.xml` to a site URL.
- Reddit: `https://www.reddit.com/r/SUBREDDIT/search.rss?q=QUERY&sort=new`
- Most blogs use `/feed` or `/index.xml`.

## LLM Providers

Uses [LiteLLM](https://docs.litellm.ai/docs/providers). Anything LiteLLM supports works.

| Provider | Model String |
|----------|-------------|
| OpenAI | `openai/gpt-5.4-nano` |
| Anthropic | `anthropic/claude-haiku-4-5` |
| Ollama | `ollama/llama3.3` |
| Google Gemini | `gemini/gemini-2.5-flash` |
| Groq | `groq/llama-3.3-70b-versatile` |
| DeepSeek | `deepseek/deepseek-chat` |
| Azure OpenAI | `azure/your-deployment` |
| Cohere | `cohere_chat/command-a-03-2025` |
| Together AI | `together_ai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8` |

Ollama config:

```yaml
llm:
  model: "ollama/llama3.3"
  api_key: "unused"
  base_url: "http://host.docker.internal:11434"  # or http://localhost:11434 outside Docker
```

Running Ollama in Docker needs the override file: `cp docker-compose.override.example.yml docker-compose.override.yml`.

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

## Configuration

Settings live in `data/config.yml` (auto-copied from `config.example.yml` on first run) and can be edited there or on the **Settings** page. Override any key with the `TOPIC_WATCH_` env prefix (use `__` for nested keys, e.g. `TOPIC_WATCH_LLM__API_KEY`). Full key reference is in [`config.example.yml`](config.example.yml) and [ARCHITECTURE.md](ARCHITECTURE.md).

## Updating

```bash
cd ~/topic-watch  # or your install directory
docker compose pull
docker compose up -d
```

The database is automatically backed up before any schema migration.

## Security

**No built-in authentication** by design (single-user tool). Safe as-is on localhost. For remote access, put it behind a reverse proxy with auth ([Authelia](https://www.authelia.com/), [Authentik](https://goauthentik.io/), Caddy `basicauth`, Nginx basic auth). See [SECURITY.md](SECURITY.md).

## Troubleshooting

| Issue | Fix |
|-------|-----|
| **LLM errors / checks failing** | Check your API key. Make sure the model string has the provider prefix (`openai/gpt-5.4-nano`, not `gpt-5.4-nano`). Check logs: `docker compose logs -f`. |
| **No notifications** | Check `notifications.urls` in config. Use the Test Notification button on Settings. Verify the [Apprise URL format](https://github.com/caronc/apprise/wiki). |
| **0 articles found** | Verify the RSS URL works in a browser. Check the Feed Health page. Some sites block bots. |
| **Topic stuck in "Researching"** | Auto-recovers after 15 minutes (set to Error). Retry from the topic page. Usually an LLM connectivity issue. |
| **Docker container exits** | `docker compose logs` for details. Check that `data/` is writable. The installer sets `PUID`/`PGID` automatically; see [SECURITY.md](SECURITY.md). |
| **High memory** | Lower `max_articles_per_check` or `content_fetch_concurrency`. Increase check intervals. |

## Contributing

Contributions of any kind are welcome.

- New here? Start with [CONTRIBUTING.md](CONTRIBUTING.md).
- Architecture overview: [ARCHITECTURE.md](ARCHITECTURE.md).
- Security: [SECURITY.md](SECURITY.md).

Bug reports and feature requests → [Issues](https://github.com/0xzerolight/topic_watch/issues).
Questions and discussion → [Discussions](https://github.com/0xzerolight/topic_watch/discussions).

## License

GNU General Public License v3.0. See [LICENSE](LICENSE).
