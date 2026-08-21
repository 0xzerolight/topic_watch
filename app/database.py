"""SQLite database setup and connection management.

Configures WAL mode for concurrent access from FastAPI and APScheduler.
Provides connection factory and schema initialization.
"""

import logging
import os
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from app import config as config_module

logger = logging.getLogger(__name__)

# The database holds notification and webhook URLs (secret-bearing) alongside the
# user's whole monitored corpus, so it is owner-only — as are its WAL/SHM sidecars
# and every backup (AUG-149). Modes, not settings: there is no reason a
# single-user self-hosted install would want another account reading them.
_DB_FILE_MODE = 0o600
_BACKUP_DIR_MODE = 0o700
_MAX_BACKUPS = 5

# Sidecars SQLite creates beside the main database file in WAL mode.
_DB_SIDECAR_SUFFIXES = ("-wal", "-shm")


class SchemaLedgerError(RuntimeError):
    """The recorded schema_version ledger does not describe a database this binary can run.

    Raised before any migration runs, so an unusable ledger stops startup instead
    of being reduced to ``MAX(version)`` and declared current (TW-AUD-011).
    """


class BackupVerificationError(RuntimeError):
    """A freshly written backup failed its integrity check and was discarded."""


# Both roots come from the ONE state-root helper in app.config, so the database
# can never land somewhere the config file does not (TW-AUD-029).
PROJECT_ROOT = config_module.PROJECT_ROOT
DATA_DIR = config_module.STATE_ROOT
DEFAULT_DB_PATH = config_module.DEFAULT_DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS topics (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL,
    feed_urls TEXT NOT NULL DEFAULT '[]',
    feed_mode TEXT NOT NULL DEFAULT 'auto',
    created_at TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'researching',
    error_message TEXT,
    check_interval_hours INTEGER
);

CREATE TABLE IF NOT EXISTS articles (
    id INTEGER PRIMARY KEY,
    topic_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    raw_content TEXT,
    source_feed TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    processed INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (topic_id) REFERENCES topics(id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_articles_content_hash
    ON articles(topic_id, content_hash);

CREATE INDEX IF NOT EXISTS idx_articles_topic_processed
    ON articles(topic_id, processed);

CREATE TABLE IF NOT EXISTS knowledge_states (
    id INTEGER PRIMARY KEY,
    topic_id INTEGER NOT NULL UNIQUE,
    summary_text TEXT NOT NULL,
    token_count INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (topic_id) REFERENCES topics(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS check_results (
    id INTEGER PRIMARY KEY,
    topic_id INTEGER NOT NULL,
    checked_at TEXT NOT NULL,
    articles_found INTEGER NOT NULL DEFAULT 0,
    articles_new INTEGER NOT NULL DEFAULT 0,
    has_new_info INTEGER NOT NULL DEFAULT 0,
    llm_response TEXT,
    notification_sent INTEGER NOT NULL DEFAULT 0,
    notification_error TEXT,
    FOREIGN KEY (topic_id) REFERENCES topics(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_check_results_topic_time
    ON check_results(topic_id, checked_at DESC);
"""


def _secure_db_file(path: Path) -> None:
    """Tighten a database file and its WAL/SHM sidecars to owner-only (AUG-149).

    Best effort: a filesystem that cannot represent the mode (a bind-mounted
    volume, a FAT device) must not stop the app from starting — the data is still
    only as exposed as it was before.
    """
    for candidate in (path, *(path.with_name(path.name + suffix) for suffix in _DB_SIDECAR_SUFFIXES)):
        try:
            if candidate.exists():
                candidate.chmod(_DB_FILE_MODE)
        except OSError as exc:  # pragma: no cover - platform/filesystem dependent
            logger.debug("Could not set owner-only mode on %s: %s", candidate.name, exc)


def _create_owner_only(path: Path) -> None:
    """Pre-create a database file with mode 0600 so it is never briefly world-readable."""
    try:
        os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, _DB_FILE_MODE))
    except FileExistsError:
        pass
    except OSError as exc:  # pragma: no cover - platform/filesystem dependent
        logger.debug("Could not pre-create %s with owner-only mode: %s", path.name, exc)


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Create a new database connection with WAL mode and pragmas.

    Args:
        db_path: Path to the database file. Defaults to data/topic_watch.db.

    Returns:
        Configured sqlite3.Connection with Row factory.
    """
    path = db_path or DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    if is_new:
        _create_owner_only(path)

    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    if is_new:
        # The WAL pragma above created the sidecars; they carry the same data.
        _secure_db_file(path)
    return conn


@contextmanager
def get_db(db_path: Path | None = None) -> Generator[sqlite3.Connection, None, None]:
    """Context manager for database connections with auto-commit/rollback.

    Usage:
        with get_db() as conn:
            conn.execute("INSERT INTO topics ...")
    """
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def short_conn(
    conn: sqlite3.Connection | None = None,
    db_path: Path | None = None,
) -> Generator[sqlite3.Connection, None, None]:
    """Yield a connection for a short DB interaction without auto-commit.

    Used by the retry queues, which must apply results per item (committing
    explicitly after each) rather than once at the end.

    * If ``conn`` is given it is reused and NOT closed (the caller owns its
      lifecycle); on error it is rolled back.
    * Otherwise a fresh connection is opened and closed on exit.

    Unlike ``get_db`` this does not commit on success — callers commit
    explicitly so a failure mid-batch leaves prior commits intact.
    """
    if conn is not None:
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise
        return

    own = get_connection(db_path)
    try:
        yield own
    except Exception:
        own.rollback()
        raise
    finally:
        own.close()


def backup_database(src_conn: sqlite3.Connection, dest_path: Path) -> None:
    """Copy a live database to ``dest_path`` with the SQLite online backup API.

    Copying the main database file (``shutil.copy2``) is wrong under WAL: commits
    live in the ``-wal`` sidecar until a checkpoint, so a file copy silently omits
    recent durable data (TW-AUD-012). The online backup runs from the active
    connection, so it captures exactly what that connection can read.

    The result is verified with ``PRAGMA integrity_check`` before it is offered as
    a backup, and written owner-only (AUG-149). A backup that fails verification is
    deleted and ``BackupVerificationError`` raised — an unusable backup must not sit
    on disk looking like a usable one.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        dest_path.parent.chmod(_BACKUP_DIR_MODE)
    except OSError as exc:  # pragma: no cover - platform/filesystem dependent
        logger.debug("Could not set owner-only mode on %s: %s", dest_path.parent, exc)

    dest_path.unlink(missing_ok=True)
    _create_owner_only(dest_path)

    dest_conn = sqlite3.connect(str(dest_path))
    try:
        src_conn.backup(dest_conn)
        integrity = dest_conn.execute("PRAGMA integrity_check").fetchone()
    finally:
        dest_conn.close()
    _secure_db_file(dest_path)

    if integrity is None or integrity[0] != "ok":
        detail = "no result" if integrity is None else str(integrity[0])
        dest_path.unlink(missing_ok=True)
        for suffix in _DB_SIDECAR_SUFFIXES:
            dest_path.with_name(dest_path.name + suffix).unlink(missing_ok=True)
        raise BackupVerificationError(f"Backup {dest_path.name} failed integrity_check ({detail}); it was discarded")


def _backup_db(conn: sqlite3.Connection, db_path: Path) -> Path | None:
    """Create a timestamped backup of the database before running migrations.

    Keeps at most ``_MAX_BACKUPS`` backups, removing the oldest when exceeded.
    Returns the backup path, or None if the DB file doesn't exist yet.
    """
    if not db_path.exists():
        return None

    backup_dir = db_path.parent / "backups"
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup_path = backup_dir / f"topic_watch.{timestamp}.db"
    backup_database(conn, backup_path)
    logger.info("Database backup created: %s", backup_path.name)

    backups = sorted(backup_dir.glob("topic_watch.*.db"))
    for old_backup in backups[:-_MAX_BACKUPS]:
        old_backup.unlink()

    return backup_path


def _validate_ledger(applied: list[object], known: list[int]) -> None:
    """Reject a schema_version ledger that is not an exact prefix of the registry.

    ``MAX(version)`` alone declares a database current whenever the highest row is
    present, so a ledger with a hole (a partially restored dump, a hand-repaired
    database) passes while the schema those migrations build is missing, and a
    ledger written by a newer binary passes while this one cannot read the schema
    (TW-AUD-011). The applied versions must therefore be exactly the first N
    registered versions, in order.
    """
    versions: list[int] = []
    for value in applied:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise SchemaLedgerError(f"schema_version contains an invalid version {value!r}")
        versions.append(value)
    versions.sort()

    expected = known[: len(versions)]
    if versions == expected:
        return

    unknown = sorted(set(versions) - set(known))
    if unknown:
        raise SchemaLedgerError(
            f"schema_version records migration(s) {unknown} that this version of Topic Watch does not know. "
            "The database was written by a newer release; upgrade Topic Watch or restore a matching backup."
        )
    missing = sorted(set(expected) - set(versions))
    raise SchemaLedgerError(
        f"schema_version is missing migration(s) {missing} below the highest applied version {versions[-1]}. "
        "The database is incompletely migrated; restore a backup rather than running against it."
    )


def run_migrations(conn: sqlite3.Connection, db_path: Path | None = None) -> None:
    """Apply any pending database migrations.

    Validates the recorded ledger, backs the database up, then applies each
    pending migration and its ``schema_version`` row as one transaction.
    """
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY)")
    applied = [row[0] for row in conn.execute("SELECT version FROM schema_version ORDER BY version").fetchall()]

    from app.migrations import MIGRATIONS

    # Sort by version, not list position (OVH-109): the registry order is
    # hand-maintained, so an append-only migration inserted out of position would
    # otherwise apply/record out of numeric order — after which a lower version is
    # silently skipped (current = MAX(version)). Sorting makes numeric order the
    # contract, independent of list order.
    registry = sorted(MIGRATIONS, key=lambda m: m[0])
    _validate_ledger(applied, [m[0] for m in registry])

    current = max(applied) if applied else 0
    pending = [m for m in registry if m[0] > current]
    if not pending:
        return

    path = db_path or DEFAULT_DB_PATH
    backup_path = _backup_db(conn, path)
    logger.info("Running %d pending migration(s) from version %d", len(pending), current)

    for version, description, up_func in pending:
        try:
            # One transaction per migration, taking the write lock up front: SQLite
            # runs DDL in autocommit unless a transaction is already open, so the
            # body's ALTER used to persist even when the ledger INSERT never ran —
            # leaving an unrecorded schema change that the next start re-applied and
            # failed on (TW-AUD-010).
            conn.execute("BEGIN IMMEDIATE")
            up_func(conn)
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
            # Commit per-migration so progress is durable: a crash between two
            # migrations leaves the DB at a clean recorded version and the next
            # run resumes from there, rather than re-running already-applied
            # (possibly non-idempotent) migrations against a changed schema.
            conn.commit()
        except Exception:
            conn.rollback()
            # No restore happens here, and none is needed: each migration commits
            # on success and the connection is rolled back on the way out, so the
            # DB is left at the last successfully applied version. Saying
            # "restored" told operators a recovery had run that never did.
            if backup_path is not None:
                logger.exception(
                    "Migration %d (%s) failed; database left at version %d. Pre-migration backup: %s",
                    version,
                    description,
                    get_schema_version(conn),
                    backup_path,
                )
            else:
                logger.exception(
                    "Migration %d (%s) failed; database left at version %d. "
                    "No pre-migration backup was taken (no database file at that path).",
                    version,
                    description,
                    get_schema_version(conn),
                )
            raise
        logger.info("Applied migration %d: %s", version, description)


def get_schema_version(conn: sqlite3.Connection) -> int:
    """Return the current schema version (0 if no migrations recorded).

    Read-only, unlike ``run_migrations``: this never creates the
    ``schema_version`` table. The ``sqlite_master`` existence guard is
    load-bearing — a bare ``SELECT MAX(version) FROM schema_version`` raises
    ``OperationalError`` on a read-only (``mode=ro``) connection when the table
    is absent, so the ``doctor`` CLI command relies on this guard to degrade
    cleanly instead of raising.
    """
    exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_version'").fetchone()
    if exists is None:
        return 0
    row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    return (row[0] if row else 0) or 0


def init_db(db_path: Path | None = None) -> None:
    """Create all tables if they don't exist, then run migrations.

    Safe to call multiple times (uses CREATE TABLE IF NOT EXISTS).
    """
    path = db_path or DEFAULT_DB_PATH
    with get_db(path) as conn:
        conn.executescript(_SCHEMA)
        run_migrations(conn, db_path=path)
    # Startup is also where an install that predates the owner-only modes gets
    # tightened — the file was created under the ambient umask back then (AUG-149).
    _secure_db_file(path)
    logger.info("Database initialized at %s", path)
