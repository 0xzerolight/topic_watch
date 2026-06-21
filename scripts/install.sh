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
#   2. Pin a specific commit or release tag instead of "main":
#        TOPIC_WATCH_REF=v1.1.2 curl -fsSL \
#          https://raw.githubusercontent.com/0xzerolight/topic_watch/v1.1.2/scripts/install.sh | bash
#      TOPIC_WATCH_REF also pins the docker-compose file this script downloads.
set -euo pipefail

REPO="0xzerolight/topic_watch"
# Pin to a commit SHA or release tag for a verifiable install (OVH-146).
# Defaults to "main" (mutable) — see the supply-chain note above.
BRANCH="${TOPIC_WATCH_REF:-main}"
INSTALL_DIR="${TOPIC_WATCH_DIR:-$HOME/topic-watch}"
PORT="${TOPIC_WATCH_PORT:-8000}"
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

# --- Create install directory ---
info "Installing to ${BOLD}${INSTALL_DIR}${RESET}"
mkdir -p "$INSTALL_DIR/data"

# --- Download production compose file ---
COMPOSE_URL="https://raw.githubusercontent.com/${REPO}/${BRANCH}/docker-compose.prod.yml"
info "Downloading docker-compose.yml..."
curl -fsSL "$COMPOSE_URL" -o "$INSTALL_DIR/docker-compose.yml"

# --- Write PUID/PGID so bind-mounted ./data is writable by this host user ---
# Docker bind mounts keep host ownership. If this user's UID/GID is not the
# image default (1000), the container must chown ./data to match. The compose
# files read PUID/PGID from this .env; the entrypoint applies them at startup.
#
# Upsert: replace existing PUID=/PGID= lines in-place so a re-run never
# truncates user-added vars (e.g. TOPIC_WATCH_LLM__API_KEY). If the key is
# absent it is appended; if the file doesn't exist it is created.
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
ENV_FILE="$INSTALL_DIR/.env"

upsert_env() {
    local key="$1"
    local value="$2"
    local file="$3"
    # Owner-only on every write path so the .env (LLM API key) is never even
    # briefly group/world-readable, not just after the trailing chmod (OVH-063).
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
}

upsert_env "PUID" "${HOST_UID}" "${ENV_FILE}"
upsert_env "PGID" "${HOST_GID}" "${ENV_FILE}"

# Restrict the .env to the owner: it holds the LLM API key (and any user-added
# secrets). Without this it is created world/group-readable by the default umask,
# leaking the key to other users on a shared host (OVH-063).
chmod 600 "${ENV_FILE}"

if [ "$HOST_UID" != "1000" ] || [ "$HOST_GID" != "1000" ]; then
    info "Host UID/GID is ${HOST_UID}:${HOST_GID} (not 1000); wrote PUID/PGID to .env"
else
    info "Wrote PUID/PGID (${HOST_UID}:${HOST_GID}) to .env"
fi

# --- Pull and start ---
cd "$INSTALL_DIR"
info "Pulling Docker image..."
docker compose pull

info "Starting Topic Watch..."
docker compose up -d

# --- Wait for health check ---
info "Waiting for Topic Watch to start..."
for i in $(seq 1 30); do
    if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

if ! curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    warn "Health check not responding yet. Check: docker compose -f ${INSTALL_DIR}/docker-compose.yml logs"
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
    # when you are not logged in. That is real persistence, so ask first instead
    # of installing it silently. Non-interactive runs default to "no".
    want_autostart="no"
    case "${AUTOSTART}" in
        yes|y|YES|Y) want_autostart="yes" ;;
        no|n|NO|N)   want_autostart="no" ;;
        "")
            if [ -t 0 ]; then
                printf "%b" "${YELLOW}[?]${RESET} Start Topic Watch automatically at boot (systemd user service + linger)? [y/N] "
                read -r reply </dev/tty || reply=""
                case "$reply" in y|Y|yes|YES) want_autostart="yes" ;; esac
            else
                warn "Skipping boot autostart (non-interactive). Set TOPIC_WATCH_AUTOSTART=yes to enable it."
            fi
            ;;
    esac

    if [ "$want_autostart" = "yes" ]; then
        # Systemd user service
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
ExecStart=/usr/bin/docker compose up
ExecStop=/usr/bin/docker compose down
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
