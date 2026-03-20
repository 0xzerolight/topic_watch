# Security Policy

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Please use GitHub Security Advisories to report vulnerabilities:

- **GitHub Security Advisories:** [Report a vulnerability](https://github.com/0xzerolight/topic_watch/security/advisories/new)

Include as much detail as possible: steps to reproduce, impact, and any suggested fix.

### Response Timeline

- **Acknowledgement:** within 48 hours
- **Status update:** within 7 days
- **Fix for critical issues:** within 30 days

## Supported Versions

Only the **latest release** receives security fixes. If you are running an older version, please upgrade before reporting.

## Scope

### In scope

- Vulnerabilities in the application code (`app/`)
- Dependency vulnerabilities that affect Topic Watch
- Docker / docker-compose configuration issues
- CSRF or injection issues in the web UI

### Out of scope

- The LLM provider you configure (OpenAI, Anthropic, etc.) — report those to the provider directly
- Apprise notification services — report those to the [Apprise project](https://github.com/caronc/apprise)
- Your reverse proxy configuration or hosting environment
- Issues that require physical access to the server

## Security Considerations

**Topic Watch has no built-in authentication.** The web UI is intentionally unauthenticated so users can integrate any auth layer they choose.

If you deploy Topic Watch on a VPS or any publicly accessible machine, you **must** place it behind a reverse proxy (nginx, Caddy, Traefik, etc.) with authentication enabled. Running it exposed to the public internet without auth is a misconfiguration, not a vulnerability in Topic Watch itself.

Your `data/config.yml` contains sensitive values (API keys, notification URLs). Ensure it is not world-readable and is excluded from any backups that are stored insecurely.
