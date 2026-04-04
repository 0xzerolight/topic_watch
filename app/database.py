"""SQLite database setup and connection management.

Configures WAL mode for concurrent access from FastAPI and APScheduler.
Provides connection factory and schema initialization.
"""

import logging
import shutil
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DB_PATH = DATA_DIR / "topic_watch.db"

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


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Create a new database connection with WAL mode and pragmas.

    Args:
        db_path: Path to the database file. Defaults to data/topic_watch.db.

    Returns:
        Configured sqlite3.Connection with Row factory.
    """
    path = db_path or DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
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


def _backup_db(db_path: Path) -> Path | None:
    """Create a timestamped backup of the database before running migrations.

    Keeps at most 5 backups, removing the oldest when exceeded.
    Returns the backup path, or None if the DB file doesn't exist yet.
    """
    if not db_path.exists():
        return None

    backup_dir = db_path.parent / "backups"
    backup_dir.mkdir(exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup_path = backup_dir / f"topic_watch.{timestamp}.db"
    shutil.copy2(db_path, backup_path)
    logger.info("Database backup created: %s", backup_path.name)

    backups = sorted(backup_dir.glob("topic_watch.*.db"))
    for old_backup in backups[:-5]:
        old_backup.unlink()

    return backup_path


def run_migrations(conn: sqlite3.Connection, db_path: Path | None = None) -> None:
    """Apply any pending database migrations.

    Creates a backup before applying new migrations. Tracks applied
    migrations in a schema_version table.
    """
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY)")
    row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    current = row[0] or 0

    from app.migrations import MIGRATIONS

    pending = [(v, d, f) for v, d, f in MIGRATIONS if v > current]
    if not pending:
        return

    path = db_path or DEFAULT_DB_PATH
    _backup_db(path)
    logger.info("Running %d pending migration(s) from version %d", len(pending), current)

    for version, description, up_func in pending:
        up_func(conn)
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
        logger.info("Applied migration %d: %s", version, description)
    conn.commit()


def init_db(db_path: Path | None = None) -> None:
    """Create all tables if they don't exist, then run migrations.

    Safe to call multiple times (uses CREATE TABLE IF NOT EXISTS).
    """
    with get_db(db_path) as conn:
        conn.executescript(_SCHEMA)
        run_migrations(conn, db_path=db_path)
    logger.info("Database initialized at %s", db_path or DEFAULT_DB_PATH)
