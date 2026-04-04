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

Topic Watch is designed as a personal, self-hosted tool. Adding authentication would mean managing users, passwords, and sessions - complexity that doesn't make sense for a single-user application.

If you deploy on a remote server, place Topic Watch behind a reverse proxy with your preferred auth layer (Authelia, Caddy basicauth, Nginx basic auth, etc.). See the [README](../README.md#security) for examples.

Your `data/config.yml` contains sensitive values (API keys, notification URLs). Ensure it is not world-readable.

## Deployment Security

When deploying Topic Watch on a public network:

- **TLS is required.** Terminate TLS at your reverse proxy (Caddy, Nginx, Traefik) before forwarding to Topic Watch. Without TLS, CSRF tokens and session cookies are transmitted in plaintext.
- **Enable secure cookies.** Set `TOPIC_WATCH_SECURE_COOKIES=true` (or `secure_cookies: true` in `data/config.yml`) so cookies are only sent over HTTPS connections.
- **Restrict network access.** Bind Topic Watch to `127.0.0.1` and proxy from your reverse proxy. Do not expose port 8000 directly to the internet.
- **Protect `data/config.yml`.** This file contains your LLM API key. Ensure it is not world-readable (`chmod 600 data/config.yml`).
- **Keep dependencies updated.** Dependabot is configured on the repository. For self-hosted installs, run `pip install --upgrade -r requirements.txt` periodically.
- **Use the Docker image.** It runs as a non-root user with resource limits.

## JSON API

The `/api/v1/` JSON API endpoints are unauthenticated, the same as the web UI. GET endpoints provide read access to all topic data including knowledge states. The single mutation endpoint (`POST /api/v1/topics/{id}/check`) is protected by CSRF.

If you expose Topic Watch to a network, apply the same reverse proxy authentication to API endpoints as you do to the web UI.
