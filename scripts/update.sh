#!/usr/bin/env bash
# Topic Watch updater
# Usage: curl -fsSL https://raw.githubusercontent.com/0xzerolight/topic_watch/main/scripts/update.sh | bash
set -euo pipefail

REPO="0xzerolight/topic_watch"
IMAGE_REPO="ghcr.io/${REPO}"
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

# Upsert a KEY=value line into .env, owner-only on every write path (mirrors
# scripts/install.sh's helper — no shared shell module between the two).
upsert_env() {
    local key="$1"
    local value="$2"
    local file="$3"
    if [ ! -f "$file" ]; then
        (umask 077; echo "${key}=${value}" > "$file")
    elif grep -q "^${key}=" "$file"; then
        local tmp
        tmp="$(mktemp "${file}.XXXXXX")"
        grep -v "^${key}=" "$file" > "$tmp"
        echo "${key}=${value}" >> "$tmp"
        mv "$tmp" "$file"
    else
        echo "${key}=${value}" >> "$file"
    fi
    chmod 600 "$file"
}

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

# TW-AUD-032: capture the digest currently pinned in .env before touching
# anything, so a failed update can restart on this exact known-good image
# rather than whatever "latest" now resolves to.
PREV_IMAGE="$(read_env TOPIC_WATCH_IMAGE "$ENV_FILE")"

# --- Pull new image ---
# Pull the floating tag explicitly (shell env overrides .env for Compose
# interpolation): otherwise, once TOPIC_WATCH_IMAGE is pinned to a digest,
# `docker compose pull` would just re-fetch that same pinned digest forever
# and updates would never advance.
info "Pulling latest image..."
TOPIC_WATCH_IMAGE="${IMAGE_REPO}:latest" docker compose pull

# Re-pin .env to the digest just pulled, so the *next* restart (reboot,
# systemd) reruns this specific image instead of a movable "latest" tag.
NEW_DIGEST="$(docker inspect --format '{{index .RepoDigests 0}}' "${IMAGE_REPO}:latest" 2>/dev/null || true)"
if [ -n "$NEW_DIGEST" ]; then
    upsert_env "TOPIC_WATCH_IMAGE" "$NEW_DIGEST" "$ENV_FILE"
    info "Restarting Topic Watch..."
    docker compose up -d
else
    warn "Could not resolve a pinned digest for ${IMAGE_REPO}:latest; running unpinned this cycle."
    info "Restarting Topic Watch..."
    TOPIC_WATCH_IMAGE="${IMAGE_REPO}:latest" docker compose up -d
fi

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

# AUG-059: a failed post-update health check must not exit successfully —
# and, since the recorded digest was just advanced above, it must not leave
# the deployment pointed at the new (possibly broken) image either.
if [ "$HEALTHY" = "1" ]; then
    NEW=$(docker compose exec -T topic-watch python -c "from app import __version__; print(__version__)" 2>/dev/null || echo "unknown")
    info "Updated: ${CURRENT} → ${NEW}"
    info "Database backups: ${INSTALL_DIR}/data/backups/"
else
    error "Health check failed after update!"
    if [ -n "$PREV_IMAGE" ]; then
        warn "Rolling back to the previously running image: ${PREV_IMAGE}"
        upsert_env "TOPIC_WATCH_IMAGE" "$PREV_IMAGE" "$ENV_FILE"
        docker compose up -d
    else
        warn "No previously recorded image digest to roll back to."
    fi
    echo ""
    echo "  Check logs:    docker compose logs"
    echo "  Restore data if needed:"
    echo "                 docker compose down"
    echo "                 cp data/backups/<latest>.db data/topic_watch.db"
    echo "                 docker compose up -d"
    exit 1
fi
