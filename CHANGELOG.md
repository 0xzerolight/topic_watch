# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-03-20

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
- Docker and docker-compose packaging with health checks
- Comprehensive test suite (92% coverage)

[0.1.0]: https://github.com/0xzerolight/topic_watch/releases/tag/v0.1.0
