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

If you deploy on a remote server, place Topic Watch behind a reverse proxy with your preferred auth layer (Authelia, Caddy basicauth, Nginx basic auth, etc.). See [Reverse Proxy Auth Examples](#reverse-proxy-auth-examples) for Caddy and Nginx configs.

Your `data/config.yml` contains sensitive values (API keys, notification URLs). Ensure it is not world-readable.

## Deployment Security

When deploying Topic Watch on a public network:

- **TLS is required.** Terminate TLS at your reverse proxy (Caddy, Nginx, Traefik) before forwarding to Topic Watch. Without TLS, CSRF tokens and session cookies are transmitted in plaintext.
- **Enable secure cookies.** Set `TOPIC_WATCH_SECURE_COOKIES=true` (or `secure_cookies: true` in `data/config.yml`) so cookies are only sent over HTTPS connections.
- **Restrict network access.** Bind Topic Watch to `127.0.0.1` and proxy from your reverse proxy. Do not expose port 8000 directly to the internet.
- **Protect `data/config.yml`.** This file contains your LLM API key. Ensure it is not world-readable (`chmod 600 data/config.yml`).
- **Keep dependencies updated.** `requirements.txt` is a hash-pinned lockfile (exact `==` versions plus `--hash` entries), so `pip install --upgrade -r requirements.txt` cannot raise versions — it is a no-op for upgrades. Updates land through the configured Dependabot PRs; to bump versions locally, regenerate the lockfile with `make lock` and reinstall.
- **Use the Docker image.** It runs as a non-root user with resource limits.

### Reverse Proxy Auth Examples

#### Caddy

```
topic-watch.example.com {
    basicauth {
        admin $2a$14$YOUR_HASHED_PASSWORD
    }
    reverse_proxy localhost:8000
}
```

Generate hash: `caddy hash-password`

#### Nginx

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

## Install Script Trust

The one-line installers (`scripts/install.sh`, `scripts/install.ps1`) are meant to be run with `curl | bash` / `irm | iex`. That convenience carries the usual trade-off: your shell executes whatever the URL returns, and by default the scripts fetch the `docker-compose` file (which selects the container image) from the same source. Both are pulled from the **mutable `main` branch** with no commit pin, tag, signature, or checksum, so a repository/branch compromise or a man-in-the-middle proxy could run arbitrary code as the invoking user.

To reduce that trust before running either installer:

- **Read it first.** Download the script and review it, or run it from a local checkout, instead of piping straight to a shell.
- **Pin a ref.** Set `TOPIC_WATCH_REF` to a release tag or commit SHA and fetch the installer from that same ref. This pins both the installer and the `docker-compose` file it downloads:

  ```bash
  # Linux / macOS
  TOPIC_WATCH_REF=v1.1.2 curl -fsSL \
    https://raw.githubusercontent.com/0xzerolight/topic_watch/v1.1.2/scripts/install.sh | bash
  ```

  ```powershell
  # Windows (PowerShell)
  $env:TOPIC_WATCH_REF="v1.1.2"
  irm https://raw.githubusercontent.com/0xzerolight/topic_watch/v1.1.2/scripts/install.ps1 | iex
  ```

### Autostart persistence

The installers can set up boot/login autostart (a systemd user service + `loginctl enable-linger` on Linux; a Startup-folder shortcut on Windows). This is **opt-in**: the installer prompts when run interactively and skips autostart in a non-interactive piped run unless you pass `TOPIC_WATCH_AUTOSTART=yes`. The closing summary echoes the exact uninstall commands.

To remove autostart later:

```bash
# Linux
systemctl --user disable --now topic-watch
rm -f ~/.config/systemd/user/topic-watch.service
loginctl disable-linger "$USER"
```

```powershell
# Windows — delete the Startup-folder shortcut
Remove-Item "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Startup\Topic Watch.lnk"
```

## Known Limitations

- **SSRF / DNS rebinding (TOCTOU).** Feed and webhook URLs are validated against private/reserved addresses, including a DNS-resolution layer that now fails closed (an unresolvable host is blocked). Redirects are re-validated per hop. However, validation resolves DNS at check time while httpx re-resolves at connect time, leaving a narrow rebinding window between validation and fetch. Eliminating it would require a pinned-IP connect transport that risks breaking HTTPS feed fetching (SNI / cert verification), so it is an accepted limitation for this single-user self-hosted tool.

## JSON API

The `/api/v1/` JSON API endpoints are unauthenticated, the same as the web UI. GET endpoints provide read access to all topic data including knowledge states. The single mutation endpoint (`POST /api/v1/topics/{id}/check`) is protected by CSRF.

If you expose Topic Watch to a network, apply the same reverse proxy authentication to API endpoints as you do to the web UI.
