#!/usr/bin/env bash
# Topic Watch updater
# Usage: curl -fsSL https://raw.githubusercontent.com/0xzerolight/topic_watch/main/scripts/update.sh | bash
set -euo pipefail

INSTALL_DIR="${TOPIC_WATCH_DIR:-$HOME/topic-watch}"
ENV_FILE="$INSTALL_DIR/.env"

# --- Colors (degrade gracefully) ---
if [ -t 1 ]; then
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    RED='\033[0;31m'
    RESET='\033[0m'
else
    GREEN='' YELLOW='' RED='' RESET=''
fi

info()  { echo -e "${GREEN}[+]${RESET} $*"; }
warn()  { echo -e "${YELLOW}[!]${RESET} $*"; }
error() { echo -e "${RED}[x]${RESET} $*" >&2; }

# Read a KEY's value out of .env, or print nothing if absent/missing.
read_env() {
    local key="$1"
    local file="$2"
    [ -f "$file" ] || return 0
    grep -m1 "^${key}=" "$file" 2>/dev/null | cut -d= -f2- || true
}

# --- Validate install ---
if [ ! -f "$INSTALL_DIR/docker-compose.yml" ]; then
    error "Topic Watch not found at $INSTALL_DIR"
    echo "  Install first: curl -fsSL https://raw.githubusercontent.com/0xzerolight/topic_watch/main/scripts/install.sh | bash"
    exit 1
fi

cd "$INSTALL_DIR"

# AUG-060: derive the effective port from the persisted .env, the same
# source docker-compose.yml itself reads, not just the process environment —
# an install on a custom port otherwise gets probed on 8000 here regardless
# of what it actually published.
PORT="${TOPIC_WATCH_PORT:-$(read_env TOPIC_WATCH_PORT "$ENV_FILE")}"
PORT="${PORT:-8000}"

# --- Show current version ---
CURRENT=$(docker compose exec -T topic-watch python -c "from app import __version__; print(__version__)" 2>/dev/null || echo "unknown")
info "Current version: ${CURRENT}"

# --- Pull new image ---
info "Pulling latest image..."
docker compose pull

# --- Restart (migrations run automatically on startup, with auto-backup) ---
info "Restarting Topic Watch..."
docker compose up -d

# --- Wait for health ---
info "Waiting for health check..."
HEALTHY=0
for i in $(seq 1 30); do
    if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        HEALTHY=1
        break
    fi
    sleep 1
done

# AUG-059: a failed post-update health check must not exit successfully.
if [ "$HEALTHY" = "1" ]; then
    NEW=$(docker compose exec -T topic-watch python -c "from app import __version__; print(__version__)" 2>/dev/null || echo "unknown")
    info "Updated: ${CURRENT} → ${NEW}"
    info "Database backups: ${INSTALL_DIR}/data/backups/"
else
    error "Health check failed after update!"
    echo ""
    echo "  Check logs:    docker compose logs"
    echo "  Roll back:     docker compose down"
    echo "                 cp data/backups/<latest>.db data/topic_watch.db"
    echo "                 docker compose up -d"
    exit 1
fi
