"""Tests for knowledge revisions and the diff timeline.

Covers the m025 schema + backfill, revision CRUD and pruning, the knowledge.py
write path, the pure diff module, and the web timeline + diff fragment.
Model-level defensive loading lives in tests/test_models_from_row.py.
"""

import sqlite3
from datetime import UTC, datetime

from app.database import get_connection, init_db
from app.models import KnowledgeRevision, KnowledgeRevisionSource, Topic, TopicStatus


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


def _seed_topic(conn: sqlite3.Connection, name: str = "CRUD Topic") -> Topic:
    from app.crud import create_topic

    topic = create_topic(conn, Topic(name=name, description="d", status=TopicStatus.READY))
    conn.commit()
    return topic


def _seed_revision(conn: sqlite3.Connection, topic_id: int, text: str, **overrides) -> KnowledgeRevision:
    from app.crud import create_knowledge_revision

    created = create_knowledge_revision(conn, KnowledgeRevision(topic_id=topic_id, summary_text=text, **overrides))
    conn.commit()
    return created


class TestKnowledgeRevisionCRUD:
    def test_create_assigns_id(self, db_conn: sqlite3.Connection) -> None:
        topic = _seed_topic(db_conn)
        created = _seed_revision(db_conn, topic.id, "First.", token_count=3)
        assert created.id is not None
        assert created.token_count == 3

    def test_headers_are_newest_first(self, db_conn: sqlite3.Connection) -> None:
        from app.crud import list_knowledge_revision_headers

        topic = _seed_topic(db_conn)
        first = _seed_revision(db_conn, topic.id, "One.")
        second = _seed_revision(db_conn, topic.id, "Two.")
        third = _seed_revision(db_conn, topic.id, "Three.")
        headers = list_knowledge_revision_headers(db_conn, topic.id, limit=10)
        assert [h.id for h in headers] == [third.id, second.id, first.id]

    def test_headers_omit_summary_text(self, db_conn: sqlite3.Connection) -> None:
        """The listing must not ship every revision's full summary."""
        from app.crud import list_knowledge_revision_headers

        topic = _seed_topic(db_conn)
        _seed_revision(db_conn, topic.id, "A very long summary body.")
        headers = list_knowledge_revision_headers(db_conn, topic.id, limit=10)
        assert headers[0].summary_text == ""

    def test_headers_carry_metadata(self, db_conn: sqlite3.Connection) -> None:
        from app.crud import list_knowledge_revision_headers

        topic = _seed_topic(db_conn)
        _seed_revision(db_conn, topic.id, "x", token_count=17, source=KnowledgeRevisionSource.INIT)
        header = list_knowledge_revision_headers(db_conn, topic.id, limit=10)[0]
        assert header.token_count == 17
        assert header.source == KnowledgeRevisionSource.INIT

    def test_headers_respect_limit(self, db_conn: sqlite3.Connection) -> None:
        from app.crud import list_knowledge_revision_headers

        topic = _seed_topic(db_conn)
        for i in range(5):
            _seed_revision(db_conn, topic.id, f"Rev {i}.", token_count=i)
        headers = list_knowledge_revision_headers(db_conn, topic.id, limit=2)
        assert [h.token_count for h in headers] == [4, 3]

    def test_headers_scoped_to_topic(self, db_conn: sqlite3.Connection) -> None:
        from app.crud import list_knowledge_revision_headers

        a = _seed_topic(db_conn, "A")
        b = _seed_topic(db_conn, "B")
        a_rev = _seed_revision(db_conn, a.id, "A rev.")
        _seed_revision(db_conn, b.id, "B rev.")
        assert [h.id for h in list_knowledge_revision_headers(db_conn, a.id, limit=10)] == [a_rev.id]

    def test_get_by_id_returns_full_summary(self, db_conn: sqlite3.Connection) -> None:
        from app.crud import get_knowledge_revision

        topic = _seed_topic(db_conn)
        created = _seed_revision(db_conn, topic.id, "Only.")
        loaded = get_knowledge_revision(db_conn, created.id)
        assert loaded is not None
        assert loaded.summary_text == "Only."
        assert get_knowledge_revision(db_conn, created.id + 999) is None

    def test_get_previous(self, db_conn: sqlite3.Connection) -> None:
        from app.crud import get_previous_knowledge_revision

        topic = _seed_topic(db_conn)
        first = _seed_revision(db_conn, topic.id, "First.")
        second = _seed_revision(db_conn, topic.id, "Second.")
        previous = get_previous_knowledge_revision(db_conn, topic.id, second.id)
        assert previous is not None
        assert previous.id == first.id
        assert previous.summary_text == "First."
        assert get_previous_knowledge_revision(db_conn, topic.id, first.id) is None

    def test_get_previous_ignores_other_topics(self, db_conn: sqlite3.Connection) -> None:
        from app.crud import get_previous_knowledge_revision

        a = _seed_topic(db_conn, "A")
        b = _seed_topic(db_conn, "B")
        _seed_revision(db_conn, a.id, "A older.")
        b_rev = _seed_revision(db_conn, b.id, "B only.")
        assert get_previous_knowledge_revision(db_conn, b.id, b_rev.id) is None

    def test_prune_keeps_newest(self, db_conn: sqlite3.Connection) -> None:
        from app.crud import list_knowledge_revision_headers, prune_knowledge_revisions

        topic = _seed_topic(db_conn)
        for i in range(5):
            _seed_revision(db_conn, topic.id, f"Rev {i}.", token_count=i)
        deleted = prune_knowledge_revisions(db_conn, topic.id, keep=2)
        db_conn.commit()
        assert deleted == 3
        headers = list_knowledge_revision_headers(db_conn, topic.id, limit=10)
        assert [h.token_count for h in headers] == [4, 3]

    def test_prune_noop_under_cap(self, db_conn: sqlite3.Connection) -> None:
        from app.crud import prune_knowledge_revisions

        topic = _seed_topic(db_conn)
        _seed_revision(db_conn, topic.id, "One.")
        assert prune_knowledge_revisions(db_conn, topic.id, keep=10) == 0

    def test_prune_scoped_to_topic(self, db_conn: sqlite3.Connection) -> None:
        from app.crud import list_knowledge_revision_headers, prune_knowledge_revisions

        a = _seed_topic(db_conn, "A")
        b = _seed_topic(db_conn, "B")
        for i in range(3):
            _seed_revision(db_conn, a.id, f"A {i}.")
            _seed_revision(db_conn, b.id, f"B {i}.")
        prune_knowledge_revisions(db_conn, a.id, keep=1)
        db_conn.commit()
        assert len(list_knowledge_revision_headers(db_conn, a.id, limit=10)) == 1
        assert len(list_knowledge_revision_headers(db_conn, b.id, limit=10)) == 3
