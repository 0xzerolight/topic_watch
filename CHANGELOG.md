# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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

### Changed

- Check interval config field renamed from `check_interval_hours` (integer) to `check_interval` (human-readable string). Old format auto-migrated.
- LLM prompts improved for more conservative novelty detection and better scope filtering
- Article truncation in prompts increased from 1000 to 1500 chars

### Fixed

- SSRF bypass via IPv6 and alternative IP encodings
- LLM novelty detection accuracy (reasoning field, relevance scoring, below-threshold article re-examination)

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
