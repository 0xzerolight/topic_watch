#!/usr/bin/env bash
# Topic Watch installer
# Usage: curl -fsSL https://raw.githubusercontent.com/0xzerolight/topic_watch/main/scripts/install.sh | bash
#
# SUPPLY-CHAIN NOTE (OVH-146): curl|bash runs whatever this URL returns, and by
# default this script also fetches docker-compose.prod.yml (which selects the
# container image) from the same ref. Both are pulled from the mutable "main"
# branch with no commit pin, tag, signature, or checksum, so a repo/branch
# compromise or a MITM proxy means arbitrary code runs as you. To reduce trust:
#   1. Review this script before piping it to a shell, or download + run it.
#   2. Pin a specific commit or release tag instead of "main". TOPIC_WATCH_REF
#      must reach the `bash` process, not the `curl` process ahead of it in
#      the pipe — a "VAR=val curl ... | bash" prefix does not propagate
#      across the pipe, so set it on bash's side instead:
#        curl -fsSL \
#          https://raw.githubusercontent.com/0xzerolight/topic_watch/v1.1.2/scripts/install.sh \
#          | TOPIC_WATCH_REF=v1.1.2 bash
#      TOPIC_WATCH_REF also pins the docker-compose file this script downloads.
set -euo pipefail

REPO="0xzerolight/topic_watch"
IMAGE_REPO="ghcr.io/${REPO}"
# Pin to a commit SHA or release tag for a verifiable install (OVH-146).
# Defaults to "main" (mutable) — see the supply-chain note above.
BRANCH="${TOPIC_WATCH_REF:-main}"
INSTALL_DIR="${TOPIC_WATCH_DIR:-$HOME/topic-watch}"
PORT="${TOPIC_WATCH_PORT:-8000}"
# Host interface the container's port is published on. Loopback by default:
# Topic Watch has no authentication, and Docker's published ports bypass ufw /
# firewalld, so binding every interface would expose an unauthenticated app even
# on a host whose firewall denies incoming traffic. Asked interactively below.
BIND_ADDR="${TOPIC_WATCH_BIND_ADDR:-}"
# Autostart persistence is opt-in (OVH-147). Set TOPIC_WATCH_AUTOSTART=yes|no to
# answer non-interactively; default in a non-interactive (piped) run is "no".
AUTOSTART="${TOPIC_WATCH_AUTOSTART:-}"

# --- Colors (degrade gracefully) ---
if [ -t 1 ]; then
    BOLD='\033[1m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    RED='\033[0;31m'
    RESET='\033[0m'
else
    BOLD='' GREEN='' YELLOW='' RED='' RESET=''
fi

info()  { echo -e "${GREEN}[+]${RESET} $*"; }
warn()  { echo -e "${YELLOW}[!]${RESET} $*"; }
error() { echo -e "${RED}[x]${RESET} $*" >&2; }

# --- Interactive prompts ---
# The documented install path is `curl … | bash`, which makes stdin the script
# pipe. `[ -t 0 ]` is therefore false even in a real terminal, so gating prompts
# on it silently skips every question. /dev/tty is the controlling terminal
# regardless of how stdin is wired — probe that instead.
TTY_OK=0
if (exec 3</dev/tty) 2>/dev/null; then
    TTY_OK=1
fi

# Write to the terminal, bypassing any redirection of stdout.
say() { [ "$TTY_OK" = "1" ] && printf "%b" "$*" > /dev/tty; return 0; }

# Ask a question and echo the reply, or echo the default unchanged when there is
# no terminal. Never blocks: a non-interactive run answers itself.
prompt_tty() {
    local text="$1" default="$2" reply=""
    if [ "$TTY_OK" != "1" ]; then
        printf '%s' "$default"
        return 0
    fi
    printf "%b" "$text" > /dev/tty
    read -r reply < /dev/tty || reply=""
    printf '%s' "${reply:-$default}"
}

# True when something is already listening on the given TCP port. Uses bash's
# /dev/tcp rather than ss/lsof/netstat, none of which are present everywhere.
port_in_use() {
    (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null
}

# --- Prerequisite checks ---
check_docker() {
    if ! command -v docker &>/dev/null; then
        return 1
    fi
    if docker compose version &>/dev/null; then
        return 0
    fi
    return 1
}

if ! check_docker; then
    error "Docker with Compose plugin is required but not found."
    echo ""
    echo "Install Docker: https://docs.docker.com/engine/install/"
    exit 1
fi

info "Docker found: $(docker compose version 2>/dev/null | head -1)"

# --- Setup questions ---
# Asked up front so the install runs uninterrupted afterwards. Every question is
# skipped when its environment variable is already set, which keeps automated
# and re-run installs reproducible. With no terminal, each falls back to its
# default and nothing blocks.

# 1. Network exposure. Loopback unless the user opts into LAN access.
if [ -z "$BIND_ADDR" ]; then
    if [ "$TTY_OK" = "1" ]; then
        say "\n${BOLD}Who should be able to reach Topic Watch?${RESET}\n"
        say "  ${BOLD}1${RESET}) This computer only          (recommended)\n"
        say "  ${BOLD}2${RESET}) Any device on my network    (needs a reverse proxy to be safe)\n"
        case "$(prompt_tty "Choice [1]: " "1")" in
            2) BIND_ADDR="0.0.0.0" ;;
            *) BIND_ADDR="127.0.0.1" ;;
        esac
    else
        BIND_ADDR="127.0.0.1"
    fi
fi

if [ "$BIND_ADDR" = "0.0.0.0" ]; then
    warn "Topic Watch will be reachable from your whole network."
    warn "It has no login screen: anyone who can reach this machine can read your"
    warn "topics and spend your LLM API budget. Beyond a trusted home network, put"
    warn "it behind a reverse proxy with authentication — see SECURITY.md."
fi

# 2. Autostart at boot. Linux only — this script installs a systemd user
#    service, and there is no launchd equivalent here. Persistence stays opt-in
#    for unattended runs (OVH-147), but an interactive user is asked directly and
#    the recommended answer is yes: Topic Watch checks topics on a schedule, so
#    without autostart it stops monitoring after a reboot and says nothing.
want_autostart="no"
if [[ "${OSTYPE:-}" == linux* ]]; then
    case "${AUTOSTART}" in
        yes|y|YES|Y) want_autostart="yes" ;;
        no|n|NO|N)   want_autostart="no" ;;
        "")
            if [ "$TTY_OK" = "1" ]; then
                say "\n${BOLD}Start Topic Watch automatically at boot?${RESET}\n"
                say "  Recommended: it checks topics on a schedule, so without this it\n"
                say "  stops monitoring after a reboot until you start it by hand.\n"
                case "$(prompt_tty "Enable autostart? [Y/n]: " "y")" in
                    n|N|no|NO|No) want_autostart="no" ;;
                    *)            want_autostart="yes" ;;
                esac
            else
                warn "Skipping boot autostart (non-interactive). Set TOPIC_WATCH_AUTOSTART=yes to enable it."
            fi
            ;;
    esac
fi

# 3. Port, asked only when the default is already taken — most users have no
#    reason to think about it.
if [ -z "${TOPIC_WATCH_PORT:-}" ] && port_in_use "$PORT"; then
    if [ "$TTY_OK" = "1" ]; then
        say "\n"
        warn "Port ${PORT} is already in use on this machine."
        chosen="$(prompt_tty "Use a different port [8080]: " "8080")"
        case "$chosen" in
            ''|*[!0-9]*) warn "Not a port number — keeping ${PORT}; the install may fail." ;;
            *)           PORT="$chosen" ;;
        esac
    else
        warn "Port ${PORT} is already in use — the install will likely fail."
        warn "Set TOPIC_WATCH_PORT to choose another and re-run."
    fi
fi

# --- Create install directory ---
info "Installing to ${BOLD}${INSTALL_DIR}${RESET}"
mkdir -p "$INSTALL_DIR/data"

# --- Download production compose file ---
COMPOSE_URL="https://raw.githubusercontent.com/${REPO}/${BRANCH}/docker-compose.prod.yml"
info "Downloading docker-compose.yml..."
curl -fsSL "$COMPOSE_URL" -o "$INSTALL_DIR/docker-compose.yml"

# Also fetch the Ollama/local-LLM override example so the README's documented
# `cp docker-compose.override.example.yml docker-compose.override.yml` step
# works from a script install too, not only a source checkout. Optional (only
# needed for local LLM providers), so a failure here warns instead of aborting
# the install.
OVERRIDE_URL="https://raw.githubusercontent.com/${REPO}/${BRANCH}/docker-compose.override.example.yml"
if ! curl -fsSL "$OVERRIDE_URL" -o "$INSTALL_DIR/docker-compose.override.example.yml"; then
    warn "Could not download docker-compose.override.example.yml (only needed for Ollama/local LLM setups)."
fi

# --- Write PUID/PGID so bind-mounted ./data is writable by this host user ---
# Docker bind mounts keep host ownership. If this user's UID/GID is not the
# image default (1000), the container must chown ./data to match. The compose
# files read PUID/PGID from this .env; the entrypoint applies them at startup.
#
# Upsert: replace existing PUID=/PGID= lines in-place so a re-run never
# truncates user-added vars (e.g. a TZ or TOPIC_WATCH_PORT override). If the key
# is absent it is appended; if the file doesn't exist it is created.
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
ENV_FILE="$INSTALL_DIR/.env"

upsert_env() {
    local key="$1"
    local value="$2"
    local file="$3"
    # Owner-only on every write path so any secret a user adds to .env is never
    # even briefly group/world-readable, not just after the trailing chmod (OVH-063).
    if [ ! -f "$file" ]; then
        (umask 077; echo "${key}=${value}" > "$file")
    elif grep -q "^${key}=" "$file"; then
        # Replace the existing line via a temp file (portable, no sed -i portability issues)
        local tmp
        tmp="$(mktemp "${file}.XXXXXX")"
        grep -v "^${key}=" "$file" > "$tmp"
        echo "${key}=${value}" >> "$tmp"
        mv "$tmp" "$file"
    else
        echo "${key}=${value}" >> "$file"
    fi
    # Guarantee owner-only after every write path — including the append branch
    # above and any pre-existing world-readable .env — not just the trailing
    # chmod (OVH-063).
    chmod 600 "$file"
}

upsert_env "PUID" "${HOST_UID}" "${ENV_FILE}"
upsert_env "PGID" "${HOST_GID}" "${ENV_FILE}"
# The compose file reads both of these. Persisting them here is what makes the
# answers above survive a later `docker compose up -d` and any future re-run of
# this installer: .env is upserted, whereas docker-compose.yml is overwritten.
upsert_env "TOPIC_WATCH_PORT" "${PORT}" "${ENV_FILE}"
upsert_env "TOPIC_WATCH_BIND_ADDR" "${BIND_ADDR}" "${ENV_FILE}"

# Restrict the .env to the owner: it holds PUID/PGID and any secrets a user adds
# (proxy creds, etc.). Without this it is created world/group-readable by the
# default umask, leaking those to other users on a shared host (OVH-063).
chmod 600 "${ENV_FILE}"

if [ "$HOST_UID" != "1000" ] || [ "$HOST_GID" != "1000" ]; then
    info "Host UID/GID is ${HOST_UID}:${HOST_GID} (not 1000); wrote PUID/PGID to .env"
else
    info "Wrote PUID/PGID (${HOST_UID}:${HOST_GID}) to .env"
fi

# --- Pull and start ---
cd "$INSTALL_DIR"
info "Pulling Docker image..."
# Fail fast with an actionable hint: the most common failure here is the image
# not being publicly pullable. set -e would abort anyway, but with only Docker's
# raw "denied"/network error and no pointer to the fix.
if ! docker compose pull; then
    error "Could not pull the Docker image (${IMAGE_REPO})."
    echo ""
    echo "  Most likely the image is not publicly accessible, or ghcr.io is unreachable."
    echo "  - Check your network and that https://ghcr.io is reachable."
    echo "  - Maintainers: confirm the GHCR package visibility is set to Public."
    echo "  - Pin a known release instead of latest: TOPIC_WATCH_REF=<tag> re-run this installer."
    exit 1
fi

# Pin the exact digest just pulled into .env (TW-AUD-032): docker-compose.yml
# reads TOPIC_WATCH_IMAGE for its image reference, so once this is set, a
# later restart (reboot, systemd, `docker compose up`) reruns this specific,
# already-verified image instead of silently re-resolving the movable
# "latest" tag to whatever the registry has by then. Skipped, not fatal, if
# the digest can't be resolved — `up -d` then falls back to `:latest`.
IMAGE_DIGEST="$(docker inspect --format '{{index .RepoDigests 0}}' "${IMAGE_REPO}:latest" 2>/dev/null || true)"
if [ -n "$IMAGE_DIGEST" ]; then
    upsert_env "TOPIC_WATCH_IMAGE" "$IMAGE_DIGEST" "${ENV_FILE}"
fi

info "Starting Topic Watch..."
docker compose up -d

# --- Wait for health check ---
info "Waiting for Topic Watch to start..."
HEALTHY=0
for i in $(seq 1 30); do
    if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        HEALTHY=1
        break
    fi
    sleep 1
done

# AUG-059: a failed health check must not be reported as a successful
# install — stop here, before desktop integration, autostart, or the
# "running!" message, so a broken install never looks like a working one.
if [ "$HEALTHY" != "1" ]; then
    error "Health check did not pass after starting Topic Watch."
    echo ""
    echo "  Diagnose with: docker compose -f ${INSTALL_DIR}/docker-compose.yml logs"
    exit 1
fi

# --- Desktop integration (Linux only) ---
if [[ "${OSTYPE:-}" == linux* ]]; then
    # Desktop entry
    DESKTOP_DIR="$HOME/.local/share/applications"
    mkdir -p "$DESKTOP_DIR"
    cat > "$DESKTOP_DIR/topic-watch.desktop" << DESKTOP_EOF
[Desktop Entry]
Type=Application
Name=Topic Watch
Comment=Self-hosted news monitoring with AI-powered novelty detection
Exec=xdg-open http://localhost:${PORT}
Icon=applications-internet
Terminal=false
Categories=Network;Monitor;
StartupNotify=false
DESKTOP_EOF
    info "Desktop entry installed (find 'Topic Watch' in your app launcher)"

    # --- Autostart at boot (opt-in, OVH-147) ---
    # A systemd user service + enable-linger starts the container at boot even
    # when you are not logged in. That is real persistence, so it is never
    # installed silently: want_autostart was decided by the question above, or by
    # TOPIC_WATCH_AUTOSTART, and defaults to "no" without a terminal.
    if [ "$want_autostart" = "yes" ]; then
        # Systemd user service. systemd requires an absolute ExecStart, and
        # docker is not at /usr/bin everywhere (Docker Desktop and several
        # distros put it under /usr/local/bin), so resolve it rather than
        # hardcoding a path that leaves the unit failing silently at boot.
        DOCKER_BIN="$(command -v docker)"
        SYSTEMD_DIR="$HOME/.config/systemd/user"
        mkdir -p "$SYSTEMD_DIR"
        cat > "$SYSTEMD_DIR/topic-watch.service" << SERVICE_EOF
[Unit]
Description=Topic Watch - Self-hosted news monitoring
After=network-online.target docker.service
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=${INSTALL_DIR}
ExecStart=${DOCKER_BIN} compose up
ExecStop=${DOCKER_BIN} compose down
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
SERVICE_EOF

        systemctl --user daemon-reload
        systemctl --user enable topic-watch 2>/dev/null || true
        info "Systemd service installed and enabled"

        # Enable lingering so service starts at boot (may require password)
        if command -v loginctl &>/dev/null; then
            loginctl enable-linger "$USER" 2>/dev/null || \
                warn "Could not enable lingering. Run: sudo loginctl enable-linger $USER"
        fi
        info "To remove autostart later: systemctl --user disable --now topic-watch &&"
        info "  rm -f \"$HOME/.config/systemd/user/topic-watch.service\" && loginctl disable-linger \"$USER\""
    else
        info "Boot autostart not installed. Enable later by re-running with TOPIC_WATCH_AUTOSTART=yes."
    fi
fi

# --- Open browser ---
echo ""
info "${BOLD}Topic Watch is running!${RESET}"
echo ""
echo "  Open http://localhost:${PORT} to complete setup."
echo "  Data stored in: ${INSTALL_DIR}/data/"
if [ "$BIND_ADDR" = "127.0.0.1" ]; then
    echo "  Reachable from: this computer only"
    echo "    To allow other devices, set TOPIC_WATCH_BIND_ADDR=0.0.0.0 in"
    echo "    ${INSTALL_DIR}/.env and run: docker compose up -d"
else
    echo "  Reachable from: any device on your network (no login required)"
fi
echo ""
echo "  Manage with:"
echo "    cd ${INSTALL_DIR} && docker compose logs    # View logs"
echo "    cd ${INSTALL_DIR} && docker compose restart  # Restart"
echo "    cd ${INSTALL_DIR} && docker compose down     # Stop"
echo ""
echo "  Uninstall:"
echo "    cd ${INSTALL_DIR} && docker compose down      # Stop the container"
echo "    systemctl --user disable --now topic-watch    # Remove boot autostart (if enabled)"
echo "    rm -f ~/.config/systemd/user/topic-watch.service ~/.local/share/applications/topic-watch.desktop"
echo "    loginctl disable-linger \"\$USER\"               # Stop running at boot when logged out"
echo "    rm -rf ${INSTALL_DIR}                          # Remove install dir + data (irreversible)"
echo ""

# Try to open browser
if command -v xdg-open &>/dev/null; then
    xdg-open "http://localhost:${PORT}" 2>/dev/null &
elif command -v open &>/dev/null; then
    open "http://localhost:${PORT}" 2>/dev/null &
fi
