#!/bin/sh
# Topic Watch container entrypoint.
#
# Bind-mounted volumes (./data:/app/data) keep their HOST ownership, so a host
# user whose UID/GID is not 1000 cannot write /app/data and initialization
# fails with "permission denied". To stay portable, this entrypoint:
#
#   1. When started as root, aligns the runtime user with the host-provided
#      PUID/PGID (default 1000:1000), chowns /app/data to match, then drops
#      privileges with gosu so the app never runs as root.
#   2. When already started as a non-root user (e.g. compose `user:` override),
#      runs the command as-is without attempting any privileged operation.
#
# Idempotent: re-running adjusts the existing appuser/appgroup in place.
set -eu

PUID="${PUID:-1000}"
PGID="${PGID:-1000}"
DATA_DIR="/app/data"

# If we are not root, we cannot chown or change identity; just exec the app.
if [ "$(id -u)" != "0" ]; then
    exec "$@"
fi

# Align the appgroup GID with PGID (idempotent).
current_gid="$(getent group appgroup | cut -d: -f3 || true)"
if [ "$current_gid" != "$PGID" ]; then
    groupmod -o -g "$PGID" appgroup
fi

# Align the appuser UID with PUID (idempotent).
current_uid="$(id -u appuser 2>/dev/null || true)"
if [ "$current_uid" != "$PUID" ]; then
    usermod -o -u "$PUID" appuser
fi

# Ensure the data volume is writable by the runtime user.
mkdir -p "$DATA_DIR"
chown -R "$PUID:$PGID" "$DATA_DIR"

# Drop privileges and run the app as the (now host-aligned) appuser.
exec gosu "$PUID:$PGID" "$@"
