#!/usr/bin/env bash
# Topic Watch installer
# Usage: curl -fsSL https://raw.githubusercontent.com/0xzerolight/topic_watch/main/scripts/install.sh | bash
set -euo pipefail

REPO="0xzerolight/topic_watch"
BRANCH="main"
INSTALL_DIR="${TOPIC_WATCH_DIR:-$HOME/topic-watch}"
PORT="${TOPIC_WATCH_PORT:-8000}"

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

# Try to open browser
if command -v xdg-open &>/dev/null; then
    xdg-open "http://localhost:${PORT}" 2>/dev/null &
elif command -v open &>/dev/null; then
    open "http://localhost:${PORT}" 2>/dev/null &
fi
