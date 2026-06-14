# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Per-topic confidence and relevance thresholds (override global defaults; high-stakes topics can demand stricter novelty)
- Per-check LLM token cost shown in topic check history
- Multi-round topic initialization that retries thin first-pass knowledge across cycles instead of going READY too early
- Setup wizard pre-flight LLM credential validation (pings the model on submit; bad key/model is caught before completing setup)
- Persistent webhook retry queue that survives restarts and respects `max_retries`
- Knowledge compression that condenses over-budget knowledge via the LLM instead of truncating, preventing dropped facts from being re-detected as new
- Full settings UI that surfaces and persists all configurable fields (previously some were silently reset on save)
- Friendly empty-states, clearer error copy, and accessibility labels across the UI
- Docker PUID/PGID support for non-1000 host users (correct data-volume permissions)
- Dropped-duplicate count surfaced instead of silently discarding duplicate articles
- `apprise_timeout_seconds` to bound notification send time

### Changed

- Scheduler holds a database connection only per topic check (not across the whole tick); weekly VACUUM runs off the event loop
- Apprise sends are time-bounded so a hung notification can no longer freeze the scheduler
- Feed/LLM timeout config fields now reject zero or negative values
- Dependency updates

### Fixed

- Settings POST handler no longer silently resets `min_relevance_threshold`, `secure_cookies`, and other fields on save
- Stopped a re-analysis loop and corrected `status_changed_at` handling
- Dashboard now surfaces `?error=` flash messages
- RSS provider fallback: continue past single feed failures, fall back to a second provider when all are unhealthy, and don't mark a provider unhealthy on an empty-but-OK feed
- OPML import merges feeds for same-named outlines
- Config writer creates the parent directory before writing the YAML
- Install script writes/updates PUID/PGID in `.env` without truncating existing contents
- Hardened model parsing against malformed JSON and empty-string/corrupt datetimes
- Docker entrypoint guards `chown`, validates PUID/PGID, and warns when run as root

### Security

- Reject non-http(s) redirect schemes during URL fetches
- Re-validate redirect targets against SSRF on every hop
- Bump python-multipart and pytest to patch known CVEs

## [1.1.2] - 2026-04-04

### Fixed

- Fix failing test for empty dashboard message (test expected old wording)

## [1.1.1] - 2026-04-04

### Added

- Theme showcase GIF in README
- Contributor Covenant v3.0 Code of Conduct
- "Updating" section in README with upgrade instructions
- `--version` flag for the CLI
- GitHub Discussions enabled

### Fixed

- Generic 404/422 pages now render styled HTML instead of raw JSON for browser requests
- OpenAPI version synced with app version
- Version display now reads from pyproject.toml (single source of truth)
- Feed Health table column truncation on desktop
- Auto-mode topics now show all feed URLs on detail page
- Readable source names in articles table instead of raw feed URLs
- Page header alignment and footer positioning
- Button vertical alignment in action rows

## [1.1.0] - 2026-04-04

### Added

- OPML import/export for migrating feeds from RSS readers (FreshRSS, Miniflux, Tiny Tiny RSS)
- JSON API at `/api/v1/` for scripting and monitoring (topics, checks, knowledge state, trigger)
- Dashboard stats bar (total checks, new info found, last notification)
- Dark mode auto-detection via `prefers-color-scheme` media query
- Ollama quick start with `docker-compose.override.example.yml`
- `TopicStatus.NEW` for gradual OPML import initialization (~1 topic/min)
- Multi-provider RSS fallback (Bing News + Google News)
- Human-readable check intervals (`6h`, `1w 3d`, `2h 30m`) replacing integer hours. Range: 10 minutes to 6 months
- LLM confidence and relevance thresholds to reduce false notifications (`min_confidence_threshold`, `min_relevance_threshold`)
- Configurable LLM temperature (`llm_temperature`)
- Semantic status colors and UI design polish (table scroll, danger button)
- Docker image `latest` tag on GitHub releases (previously missing)
- Automatic cleanup of old untagged container images

### Changed

- Check interval config field renamed from `check_interval_hours` (integer) to `check_interval` (human-readable string). Old format auto-migrated.
- LLM prompts improved for more conservative novelty detection and better scope filtering
- Article truncation in prompts increased from 1000 to 1500 chars

### Fixed

- SSRF bypass via IPv6 and alternative IP encodings
- LLM novelty detection accuracy (reasoning field, relevance scoring, below-threshold article re-examination)
- Docker install script failing with `unauthorized` (GHCR package visibility + missing `latest` tag)

## [1.0.0] - 2026-03-20

### Added

- Topic monitoring with configurable RSS feeds
- LLM-powered novelty detection via LiteLLM + Instructor (structured Pydantic output)
- Knowledge state management with token budget and automatic compression
- Web dashboard with HTMX for topic management, search, and bulk operations
- Per-topic check intervals and feed health monitoring
- Apprise integration supporting 100+ notification services
- Webhook support with JSON payloads
- Notification retry queue for failed deliveries
- CLI for manual operations (`list`, `check`, `check-all`, `init`)
- Settings UI for in-app configuration
- Custom color themes (Nord, Dracula, Solarized Dark, High Contrast, Tokyo Night)
- CSRF protection on all mutation endpoints
- SSRF protection blocking private/internal network ranges on article fetches
- XSS protection with input sanitization on all user-facing outputs
- Export filename sanitization preventing header injection
- Rate limiting on API endpoints with automatic cleanup
- Docker multi-stage build with non-root user, HEALTHCHECK, and STOPSIGNAL
- Docker Compose resource limits (512M memory) and log rotation
- Auto-copy config on first run with clear setup instructions
- Configurable log level via `TOPIC_WATCH_LOG_LEVEL` environment variable
- Version display in web UI footer
- Ruff security lint rules (bandit) in CI
- CI testing on Python 3.11, 3.12, and 3.13
- Dependabot for automated dependency updates
- Reverse proxy examples (Caddy, Nginx) in README
- Comprehensive test suite (92% coverage)

[1.0.0]: https://github.com/0xzerolight/topic_watch/releases/tag/v1.0.0
