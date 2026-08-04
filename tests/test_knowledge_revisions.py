"""Tests for knowledge revisions and the diff timeline.

Covers the m025 schema + backfill, revision CRUD and pruning, the knowledge.py
write path, the pure diff module, and the web timeline + diff fragment.
Model-level defensive loading lives in tests/test_models_from_row.py.
"""

import sqlite3
from datetime import UTC, datetime

from app.database import get_connection, init_db
from app.models import Topic, TopicStatus


def _revision_settings(**overrides):
    """Offline-safe Settings for revision tests (mirrors smoke's _settings())."""
    from app.config import LLMSettings, Settings

    defaults = {
        "llm": LLMSettings(model="openai/gpt-4o-mini", api_key="test-key-12345678"),
        "knowledge_state_max_tokens": 2000,
    }
    defaults.update(overrides)
    return Settings(**defaults)


class TestKnowledgeRevisionSchema:
    def test_expected_columns(self, db_conn: sqlite3.Connection) -> None:
        columns = {r[1] for r in db_conn.execute("PRAGMA table_info(knowledge_revisions)").fetchall()}
        assert columns == {
            "id",
            "topic_id",
            "summary_text",
            "token_count",
            "source",
            "change_note",
            "created_at",
        }

    def test_index_exists(self, db_conn: sqlite3.Connection) -> None:
        row = db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_knowledge_revisions_topic'",
        ).fetchone()
        assert row is not None

    def test_registered_in_migrations_list(self) -> None:
        from app.migrations import MIGRATIONS

        entry = next((m for m in MIGRATIONS if m[0] == 25), None)
        assert entry is not None, "m025 not found in MIGRATIONS"
        assert "knowledge_revisions" in entry[1]

    def test_cascade_delete_with_topic(self, db_conn: sqlite3.Connection) -> None:
        from app.crud import create_topic

        topic = create_topic(db_conn, Topic(name="Cascade", description="d", status=TopicStatus.READY))
        db_conn.execute(
            "INSERT INTO knowledge_revisions (topic_id, summary_text, token_count, source, created_at)"
            " VALUES (?, 'x', 1, 'init', ?)",
            (topic.id, datetime.now(UTC).isoformat()),
        )
        db_conn.commit()
        db_conn.execute("DELETE FROM topics WHERE id = ?", (topic.id,))
        db_conn.commit()
        row = db_conn.execute(
            "SELECT COUNT(*) FROM knowledge_revisions WHERE topic_id = ?",
            (topic.id,),
        ).fetchone()
        assert row[0] == 0


def _legacy_db(tmp_path, name: str):
    """A migrated DB with one topic, ready for a hand-invoked m025 backfill.

    ``init_db`` runs m025 before this helper inserts anything, so
    ``knowledge_revisions`` is already empty and the backfill saw no
    ``knowledge_states``. Tests then seed a state and call ``up()`` directly.
    Caller must close the returned connection.
    """
    db_path = tmp_path / name
    init_db(db_path)
    conn = get_connection(db_path)
    conn.execute(
        "INSERT INTO topics (name, description, feed_urls, feed_mode, created_at, status)"
        " VALUES ('T', 'd', '[]', 'manual', ?, 'ready')",
        (datetime.now(UTC).isoformat(),),
    )
    topic_id = conn.execute("SELECT id FROM topics WHERE name='T'").fetchone()[0]
    conn.commit()
    return conn, topic_id


class TestM025Backfill:
    def test_backfills_existing_knowledge_state(self, tmp_path) -> None:
        from app.migrations.m025_knowledge_revisions import up as m025_up

        conn, topic_id = _legacy_db(tmp_path, "backfill.db")
        try:
            conn.execute(
                "INSERT OR REPLACE INTO knowledge_states (topic_id, summary_text, token_count, updated_at)"
                " VALUES (?, 'Legacy summary.', 42, ?)",
                (topic_id, "2026-01-02T03:04:05+00:00"),
            )
            conn.commit()

            m025_up(conn)
            conn.commit()

            rows = conn.execute(
                "SELECT summary_text, token_count, source, created_at FROM knowledge_revisions WHERE topic_id = ?",
                (topic_id,),
            ).fetchall()
            assert len(rows) == 1
            assert rows[0][0] == "Legacy summary."
            assert rows[0][1] == 42
            assert rows[0][2] == "init"
            assert rows[0][3] == "2026-01-02T03:04:05+00:00"
        finally:
            conn.close()

    def test_rerun_does_not_duplicate(self, tmp_path) -> None:
        from app.migrations.m025_knowledge_revisions import up as m025_up

        conn, topic_id = _legacy_db(tmp_path, "idem.db")
        try:
            conn.execute(
                "INSERT OR REPLACE INTO knowledge_states (topic_id, summary_text, token_count, updated_at)"
                " VALUES (?, 'Legacy.', 5, ?)",
                (topic_id, datetime.now(UTC).isoformat()),
            )
            conn.commit()
            m025_up(conn)
            m025_up(conn)
            conn.commit()
            assert conn.execute("SELECT COUNT(*) FROM knowledge_revisions").fetchone()[0] == 1
        finally:
            conn.close()

    def test_orphaned_knowledge_state_does_not_break_the_migration(self, tmp_path) -> None:
        """A knowledge_states row whose topic is gone must not abort startup.

        PRAGMA foreign_keys=ON (app/database.py:98) makes an unguarded
        INSERT..SELECT raise IntegrityError on such a row; run_migrations
        re-raises, leaving the app unbootable.
        """
        from app.migrations.m025_knowledge_revisions import up as m025_up

        conn, topic_id = _legacy_db(tmp_path, "orphan.db")
        try:
            conn.execute(
                "INSERT OR REPLACE INTO knowledge_states (topic_id, summary_text, token_count, updated_at)"
                " VALUES (?, 'Live.', 5, ?)",
                (topic_id, datetime.now(UTC).isoformat()),
            )
            conn.commit()
            # Plant an orphan behind the FK's back.
            conn.execute("PRAGMA foreign_keys=OFF")
            conn.execute(
                "INSERT INTO knowledge_states (topic_id, summary_text, token_count, updated_at)"
                " VALUES (99999, 'Orphan.', 1, ?)",
                (datetime.now(UTC).isoformat(),),
            )
            conn.commit()
            conn.execute("PRAGMA foreign_keys=ON")

            m025_up(conn)
            conn.commit()

            rows = conn.execute("SELECT summary_text FROM knowledge_revisions").fetchall()
            assert [r[0] for r in rows] == ["Live."]
        finally:
            conn.close()
