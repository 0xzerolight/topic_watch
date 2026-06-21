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

# Validate that PUID and PGID are numeric.
case "$PUID" in
    *[!0-9]*) echo "ERROR: PUID must be a numeric user ID (got: '$PUID')" >&2; exit 1;;
esac
case "$PGID" in
    *[!0-9]*) echo "ERROR: PGID must be a numeric group ID (got: '$PGID')" >&2; exit 1;;
esac

# Warn loudly when asked to run as root (privilege drop would be a no-op).
if [ "$PUID" = "0" ]; then
    echo "WARNING: PUID=0 — the app will run as root, which defeats privilege drop." >&2
fi

# If we are not root, we cannot chown or change identity. Verify the data
# volume is writable by the current (non-root) user and fail with an actionable
# message if not, instead of crashing deep inside init_db with permission denied.
# This is a diagnostic, not a privileged op (which couldn't run as non-root).
if [ "$(id -u)" != "0" ]; then
    mkdir -p "$DATA_DIR" 2>/dev/null || true
    if ! { touch "$DATA_DIR/.wtest" 2>/dev/null && rm -f "$DATA_DIR/.wtest"; }; then
        cur_uid="$(id -u)"
        cur_gid="$(id -g)"
        dir_owner_uid="$(stat -c %u "$DATA_DIR" 2>/dev/null || echo '?')"
        dir_owner_gid="$(stat -c %g "$DATA_DIR" 2>/dev/null || echo '?')"
        echo "ERROR: $DATA_DIR is not writable by uid=$cur_uid gid=$cur_gid" >&2
        echo "       (the volume is owned by uid=$dir_owner_uid gid=$dir_owner_gid)." >&2
        echo "       The container was started as a non-root user (e.g. a compose 'user:' override)," >&2
        echo "       so it cannot fix volume ownership itself. Remediate on the host with:" >&2
        echo "         chown -R $cur_uid:$cur_gid ./data" >&2
        echo "       or drop the 'user:' override and set PUID/PGID to match the host owner." >&2
        exit 1
    fi
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
# Skip the recursive chown when ownership already matches — it is expensive on
# large bind-mounted volumes and unnecessary when the UID/GID haven't changed.
# Check BOTH owner UID and group GID so a GID-only change is not missed.
dir_uid="$(stat -c %u "$DATA_DIR")"
dir_gid="$(stat -c %g "$DATA_DIR")"
if [ "$dir_uid" != "$PUID" ] || [ "$dir_gid" != "$PGID" ]; then
    chown -R "$PUID:$PGID" "$DATA_DIR"
fi

# Drop privileges and run the app as the (now host-aligned) appuser.
exec gosu "$PUID:$PGID" "$@"
