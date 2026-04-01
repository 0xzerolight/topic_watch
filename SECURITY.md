# Security Policy

## Reporting a Vulnerability

If you find a security vulnerability, please open a [GitHub Issue](https://github.com/0xzerolight/topic_watch/issues/new) with the label `security`. If you prefer to report privately, include your contact information and I'll follow up directly.

Please include: steps to reproduce, potential impact, and any suggested fix.

## Scope

**In scope:**

- Vulnerabilities in the application code (`app/`)
- Dependency vulnerabilities that affect Topic Watch
- Docker / docker-compose configuration issues
- CSRF or injection issues in the web UI

**Out of scope** (report to the relevant project instead):

- Your LLM provider (OpenAI, Anthropic, etc.)
- Apprise notification services ([Apprise project](https://github.com/caronc/apprise))
- Your reverse proxy or hosting configuration

## Why No Built-in Authentication?

Topic Watch is designed as a personal, self-hosted tool. Adding authentication would mean managing users, passwords, and sessions — complexity that doesn't make sense for a single-user application.

If you deploy on a remote server, place Topic Watch behind a reverse proxy with your preferred auth layer (Authelia, Caddy basicauth, Nginx basic auth, etc.). See the [README](README.md#security) for examples.

Your `data/config.yml` contains sensitive values (API keys, notification URLs). Ensure it is not world-readable.
