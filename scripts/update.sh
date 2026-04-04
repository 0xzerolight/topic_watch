#!/usr/bin/env bash
# Topic Watch updater
# Usage: curl -fsSL https://raw.githubusercontent.com/0xzerolight/topic_watch/main/scripts/update.sh | bash
set -euo pipefail

INSTALL_DIR="${TOPIC_WATCH_DIR:-$HOME/topic-watch}"

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

# --- Validate install ---
if [ ! -f "$INSTALL_DIR/docker-compose.yml" ]; then
    error "Topic Watch not found at $INSTALL_DIR"
    echo "  Install first: curl -fsSL https://raw.githubusercontent.com/0xzerolight/topic_watch/main/scripts/install.sh | bash"
    exit 1
fi

cd "$INSTALL_DIR"

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
for i in $(seq 1 30); do
    if curl -sf "http://localhost:${TOPIC_WATCH_PORT:-8000}/health" >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

if curl -sf "http://localhost:${TOPIC_WATCH_PORT:-8000}/health" >/dev/null 2>&1; then
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
fi
