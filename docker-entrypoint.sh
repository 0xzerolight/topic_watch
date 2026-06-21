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

# Resolve the directory that will hold the SQLite database. db_path is a
# first-class configurable (TOPIC_WATCH_DB_PATH); it defaults to a relative
# path under /app/data, but an absolute or relocated value can point outside
# the bind-mounted volume. Such a directory is NOT chowned/probed by the
# /app/data logic below, so the dropped appuser would crash in init_db with
# "permission denied" (OVH-121). Surface it as its own path to handle.
DEFAULT_DB_PATH="data/topic_watch.db"
db_path="${TOPIC_WATCH_DB_PATH:-$DEFAULT_DB_PATH}"
case "$db_path" in
    /*) db_dir="$(dirname "$db_path")" ;;        # absolute → use as-is
    *)  db_dir="/app/$(dirname "$db_path")" ;;   # relative → resolved from /app
esac
# When the DB dir already lives inside the data volume, the /app/data handling
# covers it — collapse to DATA_DIR so the common case touches a single path.
case "$db_dir/" in
    "$DATA_DIR"/* | "$DATA_DIR/") db_dir="$DATA_DIR" ;;
esac

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

# Probe one directory for writability by the current user, failing with an
# actionable message instead of crashing deep inside init_db. Used on the
# non-root path, where we cannot chown to fix ownership ourselves.
probe_writable() {
    probe_dir="$1"
    mkdir -p "$probe_dir" 2>/dev/null || true
    if { touch "$probe_dir/.wtest" 2>/dev/null && rm -f "$probe_dir/.wtest"; }; then
        return 0
    fi
    cur_uid="$(id -u)"
    cur_gid="$(id -g)"
    dir_owner_uid="$(stat -c %u "$probe_dir" 2>/dev/null || echo '?')"
    dir_owner_gid="$(stat -c %g "$probe_dir" 2>/dev/null || echo '?')"
    echo "ERROR: $probe_dir is not writable by uid=$cur_uid gid=$cur_gid" >&2
    echo "       (the directory is owned by uid=$dir_owner_uid gid=$dir_owner_gid)." >&2
    echo "       The container was started as a non-root user (e.g. a compose 'user:' override)," >&2
    echo "       so it cannot fix volume ownership itself. Remediate on the host with:" >&2
    echo "         chown -R $cur_uid:$cur_gid <host path mounted at $probe_dir>" >&2
    echo "       or drop the 'user:' override and set PUID/PGID to match the host owner." >&2
    return 1
}

# If we are not root, we cannot chown or change identity. Verify the data
# volume — and the DB directory when it lives outside the volume — are writable
# by the current (non-root) user. This is a diagnostic, not a privileged op
# (which couldn't run as non-root anyway).
if [ "$(id -u)" != "0" ]; then
    probe_writable "$DATA_DIR" || exit 1
    if [ "$db_dir" != "$DATA_DIR" ]; then
        probe_writable "$db_dir" || exit 1
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

# Ensure a directory exists and is owned by the runtime user. Skip the
# recursive chown when ownership already matches — it is expensive on large
# bind-mounted volumes and unnecessary when the UID/GID haven't changed. Check
# BOTH owner UID and group GID so a GID-only change is not missed.
chown_runtime() {
    target_dir="$1"
    mkdir -p "$target_dir"
    dir_uid="$(stat -c %u "$target_dir")"
    dir_gid="$(stat -c %g "$target_dir")"
    if [ "$dir_uid" != "$PUID" ] || [ "$dir_gid" != "$PGID" ]; then
        chown -R "$PUID:$PGID" "$target_dir"
    fi
}

# Ensure the data volume is writable by the runtime user.
chown_runtime "$DATA_DIR"
# When the DB lives outside the data volume (absolute/relocated db_path), its
# parent is not covered above and would stay root-owned — chown it too (OVH-121).
if [ "$db_dir" != "$DATA_DIR" ]; then
    chown_runtime "$db_dir"
fi

# Drop privileges and run the app as the (now host-aligned) appuser.
exec gosu "$PUID:$PGID" "$@"
