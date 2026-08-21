"""Tests for knowledge revisions and the diff timeline.

Covers the m025 schema + backfill, revision CRUD and pruning, the knowledge.py
write path, the pure diff module, and the web timeline + diff fragment.
Model-level defensive loading lives in tests/test_models_from_row.py.
"""

import sqlite3
from datetime import UTC, datetime

import pytest

from app.database import get_connection, init_db
from app.models import KnowledgeRevision, KnowledgeRevisionSource, KnowledgeState, Topic, TopicStatus


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
            "model",
            "basis_hash",
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

    def test_real_v24_database_upgrades_to_head(self, tmp_path, monkeypatch) -> None:
        """AUG-157: the registered v24 -> head route, not a hand-invoked ``up()``.

        The other tests here call ``m025_up`` directly on a database that is
        already at HEAD, so the one thing this stateful backfill exists for — an
        existing install crossing version 24 — was never exercised: not the
        registry ordering, not the ledger advancing, and not the actual pre-m025
        schema that migrations m001-m024 produce.
        """
        import app.migrations as migrations_mod
        from app.database import get_schema_version

        real = list(migrations_mod.MIGRATIONS)
        db_path = tmp_path / "v24.db"

        monkeypatch.setattr(migrations_mod, "MIGRATIONS", [m for m in real if m[0] <= 24])
        init_db(db_path)

        conn = get_connection(db_path)
        try:
            assert get_schema_version(conn) == 24
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            assert "knowledge_revisions" not in tables, "precondition: v24 predates the history table"
            conn.execute(
                "INSERT INTO topics (name, description, feed_urls, feed_mode, created_at, status)"
                " VALUES ('Upgrader', 'd', '[]', 'manual', ?, 'ready')",
                (datetime.now(UTC).isoformat(),),
            )
            topic_id = conn.execute("SELECT id FROM topics WHERE name='Upgrader'").fetchone()[0]
            conn.execute(
                "INSERT INTO knowledge_states (topic_id, summary_text, token_count, updated_at)"
                " VALUES (?, 'Known before the upgrade.', 77, ?)",
                (topic_id, "2026-01-02T03:04:05+00:00"),
            )
            conn.commit()
        finally:
            conn.close()

        monkeypatch.setattr(migrations_mod, "MIGRATIONS", real)
        init_db(db_path)

        conn = get_connection(db_path)
        try:
            assert get_schema_version(conn) == max(v for v, _, _ in real)
            rows = conn.execute(
                "SELECT summary_text, token_count, source, change_note, created_at"
                " FROM knowledge_revisions WHERE topic_id = ?",
                (topic_id,),
            ).fetchall()
            assert len(rows) == 1, "the upgrade must backfill one baseline revision"
            assert tuple(rows[0]) == ("Known before the upgrade.", 77, "init", None, "2026-01-02T03:04:05+00:00")
            upgraded_columns = {r[1] for r in conn.execute("PRAGMA table_info(knowledge_revisions)").fetchall()}
        finally:
            conn.close()

        # The upgraded schema must match what a fresh install gets.
        fresh_path = tmp_path / "fresh.db"
        init_db(fresh_path)
        fresh = get_connection(fresh_path)
        try:
            fresh_columns = {r[1] for r in fresh.execute("PRAGMA table_info(knowledge_revisions)").fetchall()}
        finally:
            fresh.close()
        assert upgraded_columns == fresh_columns

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


class TestM028TokenCountRepair:
    """TW-AUD-014: m021 rewrote summary_text and left token_count behind."""

    def _state(self, conn, topic_id: int, summary: str, token_count: int) -> None:
        conn.execute(
            "INSERT OR REPLACE INTO knowledge_states (topic_id, summary_text, token_count, updated_at)"
            " VALUES (?, ?, ?, ?)",
            (topic_id, summary, token_count, datetime.now(UTC).isoformat()),
        )
        conn.commit()

    def test_empty_summary_reports_zero_tokens(self, tmp_path) -> None:
        from app.migrations.m028_repair_knowledge_token_counts import up as m028_up

        conn, topic_id = _legacy_db(tmp_path, "repair_empty.db")
        try:
            # What m021 leaves behind when the whole summary was a quality note.
            self._state(conn, topic_id, "\n", 480)
            m028_up(conn)
            conn.commit()
            assert conn.execute("SELECT token_count FROM knowledge_states").fetchone()[0] == 0
        finally:
            conn.close()

    def test_count_larger_than_the_text_is_impossible(self, tmp_path) -> None:
        from app.migrations.m028_repair_knowledge_token_counts import up as m028_up

        conn, topic_id = _legacy_db(tmp_path, "repair_impossible.db")
        try:
            summary = "Trial began."  # 12 characters, so 12 tokens is the ceiling
            self._state(conn, topic_id, summary, 400)
            m028_up(conn)
            conn.commit()
            repaired = conn.execute("SELECT token_count FROM knowledge_states").fetchone()[0]
            assert 0 < repaired <= len(summary)
        finally:
            conn.close()

    def test_plausible_counts_are_left_alone(self, tmp_path) -> None:
        """A model-accurate count must not be replaced by a character estimate."""
        from app.migrations.m028_repair_knowledge_token_counts import up as m028_up

        conn, topic_id = _legacy_db(tmp_path, "repair_noop.db")
        try:
            self._state(conn, topic_id, "A substantial summary of the case so far.", 9)
            m028_up(conn)
            conn.commit()
            assert conn.execute("SELECT token_count FROM knowledge_states").fetchone()[0] == 9
        finally:
            conn.close()

    def test_rerun_is_a_noop(self, tmp_path) -> None:
        from app.migrations.m028_repair_knowledge_token_counts import up as m028_up

        conn, topic_id = _legacy_db(tmp_path, "repair_rerun.db")
        try:
            self._state(conn, topic_id, "", 99)
            m028_up(conn)
            conn.commit()
            first = conn.execute("SELECT token_count FROM knowledge_states").fetchone()[0]
            m028_up(conn)
            conn.commit()
            assert conn.execute("SELECT token_count FROM knowledge_states").fetchone()[0] == first
        finally:
            conn.close()


class TestM029Provenance:
    def test_registered_in_migrations_list(self) -> None:
        from app.migrations import MIGRATIONS

        versions = [m[0] for m in MIGRATIONS]
        assert versions == sorted(versions)
        assert versions[-2:] == [28, 29]

    def test_columns_exist_and_default_null(self, db_conn: sqlite3.Connection) -> None:
        topic = _seed_topic(db_conn, "Provenance Schema Topic")
        db_conn.execute(
            "INSERT INTO knowledge_revisions (topic_id, summary_text, token_count, source, created_at)"
            " VALUES (?, 'x', 1, 'init', ?)",
            (topic.id, datetime.now(UTC).isoformat()),
        )
        db_conn.commit()
        row = db_conn.execute("SELECT model, basis_hash FROM knowledge_revisions").fetchone()
        assert row["model"] is None
        assert row["basis_hash"] is None

    def test_rerun_does_not_fail_on_existing_columns(self, db_conn: sqlite3.Connection) -> None:
        from app.migrations.m029_revision_provenance import up as m029_up

        m029_up(db_conn)  # already applied by init_db
        m029_up(db_conn)


def _seed_topic(conn: sqlite3.Connection, name: str = "CRUD Topic") -> Topic:
    from app.crud import create_topic

    topic = create_topic(conn, Topic(name=name, description="d", status=TopicStatus.READY))
    conn.commit()
    return topic


TEST_MODEL = "openai/gpt-4o-mini"


def _seed_revision(conn: sqlite3.Connection, topic_id: int, text: str, **overrides) -> KnowledgeRevision:
    """Seed one revision. Defaults to a known model so token deltas are
    comparable (AUG-255); pass ``model=None`` for a pre-provenance row."""
    from app.crud import create_knowledge_revision

    overrides.setdefault("model", TEST_MODEL)
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

    def test_prune_all_reclaims_dormant_history(self, db_conn: sqlite3.Connection) -> None:
        """AUG-034: write-time pruning only touches the topic being written, so a
        lowered limit never reached a topic that has stopped updating."""
        from app.crud import list_knowledge_revision_headers, prune_all_knowledge_revisions

        quiet = _seed_topic(db_conn, "Dormant Topic")
        busy = _seed_topic(db_conn, "Other Topic")
        for i in range(5):
            _seed_revision(db_conn, quiet.id, f"Quiet {i}.")
        for i in range(3):
            _seed_revision(db_conn, busy.id, f"Busy {i}.")

        assert prune_all_knowledge_revisions(db_conn, keep=2) == 4
        db_conn.commit()

        assert [h.summary_text for h in list_knowledge_revision_headers(db_conn, quiet.id, limit=10)] == ["", ""]
        assert len(list_knowledge_revision_headers(db_conn, quiet.id, limit=10)) == 2
        assert len(list_knowledge_revision_headers(db_conn, busy.id, limit=10)) == 2

    def test_prune_all_keeps_the_newest_of_each_topic(self, db_conn: sqlite3.Connection) -> None:
        from app.crud import get_knowledge_revision, list_knowledge_revision_headers, prune_all_knowledge_revisions

        topic = _seed_topic(db_conn, "Order Topic")
        for i in range(4):
            _seed_revision(db_conn, topic.id, f"Rev {i}.")

        prune_all_knowledge_revisions(db_conn, keep=2)
        db_conn.commit()

        headers = list_knowledge_revision_headers(db_conn, topic.id, limit=10)
        assert [get_knowledge_revision(db_conn, h.id).summary_text for h in headers] == ["Rev 3.", "Rev 2."]

    def test_prune_all_is_a_noop_under_the_cap(self, db_conn: sqlite3.Connection) -> None:
        from app.crud import prune_all_knowledge_revisions

        topic = _seed_topic(db_conn, "Under Cap Topic")
        _seed_revision(db_conn, topic.id, "Only.")
        assert prune_all_knowledge_revisions(db_conn, keep=5) == 0

    def test_headers_carry_the_change_note(self, db_conn: sqlite3.Connection) -> None:
        """AUG-124: the header query used to replace it with NULL."""
        from app.crud import list_knowledge_revision_headers

        topic = _seed_topic(db_conn, "Note Header Topic")
        _seed_revision(db_conn, topic.id, "Body.", change_note="Verdict announced.")

        headers = list_knowledge_revision_headers(db_conn, topic.id, limit=10)
        assert headers[0].change_note == "Verdict announced."
        assert headers[0].summary_text == ""

    def test_headers_carry_provenance(self, db_conn: sqlite3.Connection) -> None:
        from app.crud import list_knowledge_revision_headers

        topic = _seed_topic(db_conn, "Provenance Header Topic")
        _seed_revision(db_conn, topic.id, "Body.", model="openai/gpt-4o-mini", basis_hash="abc123")

        headers = list_knowledge_revision_headers(db_conn, topic.id, limit=10)
        assert headers[0].model == "openai/gpt-4o-mini"
        assert headers[0].basis_hash == "abc123"

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


def _one_article(topic_id: int):
    from app.models import Article

    return [
        Article(
            topic_id=topic_id,
            title="A",
            url="https://example.com/a",
            content_hash="h1",
            source_feed="https://example.com/feed.xml",
        )
    ]


class TestRevisionWritePath:
    async def test_initialize_records_init_revision(self, db_conn: sqlite3.Connection) -> None:
        from app.crud import get_knowledge_revision, list_knowledge_revision_headers
        from tests.helpers import init_knowledge as initialize_knowledge
        from tests.helpers import make_knowledge_update, stub_llm_boundary

        topic = _seed_topic(db_conn, "Init Topic")
        with stub_llm_boundary(knowledge_init=make_knowledge_update("Baseline summary.")):
            await initialize_knowledge(topic, _one_article(topic.id), db_conn, _revision_settings())

        headers = list_knowledge_revision_headers(db_conn, topic.id, limit=10)
        assert len(headers) == 1
        assert headers[0].source == KnowledgeRevisionSource.INIT
        assert headers[0].token_count > 0
        full = get_knowledge_revision(db_conn, headers[0].id)
        assert full.summary_text == "Baseline summary."
        assert full.change_note is None

    async def test_initialize_with_insufficient_data_still_records(self, db_conn: sqlite3.Connection) -> None:
        """Thin/off-topic articles still persist a baseline state, so they get a revision."""
        from app.crud import list_knowledge_revision_headers
        from tests.helpers import init_knowledge as initialize_knowledge
        from tests.helpers import make_knowledge_update, stub_llm_boundary

        topic = _seed_topic(db_conn, "Thin Topic")
        thin = make_knowledge_update("Not enough information yet.", sufficient_data=False)
        with stub_llm_boundary(knowledge_init=thin):
            result = await initialize_knowledge(topic, _one_article(topic.id), db_conn, _revision_settings())

        assert result.sufficient_data is False
        assert len(list_knowledge_revision_headers(db_conn, topic.id, limit=10)) == 1

    async def test_update_records_update_revision_with_change_note(self, db_conn: sqlite3.Connection) -> None:
        from app.crud import create_knowledge_state, get_knowledge_revision, list_knowledge_revision_headers
        from tests.helpers import make_knowledge_update, make_novelty_result, stub_llm_boundary, update_knowledge

        topic = _seed_topic(db_conn, "Update Topic")
        create_knowledge_state(db_conn, KnowledgeState(topic_id=topic.id, summary_text="Old.", token_count=3))
        db_conn.commit()

        novelty = make_novelty_result(summary="Court ruled on Tuesday.")
        with stub_llm_boundary(knowledge_init=make_knowledge_update("New summary.")):
            await update_knowledge(topic, novelty, db_conn, _revision_settings())

        headers = list_knowledge_revision_headers(db_conn, topic.id, limit=10)
        assert len(headers) == 1
        assert headers[0].source == KnowledgeRevisionSource.UPDATE
        full = get_knowledge_revision(db_conn, headers[0].id)
        assert full.summary_text == "New summary."
        assert full.change_note == "Court ruled on Tuesday."

    async def test_insufficient_update_records_no_revision(self, db_conn: sqlite3.Connection) -> None:
        from app.crud import create_knowledge_state, list_knowledge_revision_headers
        from tests.helpers import make_knowledge_update, make_novelty_result, stub_llm_boundary, update_knowledge

        topic = _seed_topic(db_conn, "Insufficient Topic")
        create_knowledge_state(db_conn, KnowledgeState(topic_id=topic.id, summary_text="Old.", token_count=2))
        db_conn.commit()

        thin = make_knowledge_update("Ignored.", sufficient_data=False)
        with stub_llm_boundary(knowledge_init=thin):
            result = await update_knowledge(topic, make_novelty_result(), db_conn, _revision_settings())

        assert result.sufficient_data is False
        assert list_knowledge_revision_headers(db_conn, topic.id, limit=10) == []

    async def test_prunes_to_configured_limit(self, db_conn: sqlite3.Connection) -> None:
        from app.crud import create_knowledge_state, get_knowledge_revision, list_knowledge_revision_headers
        from tests.helpers import make_knowledge_update, make_novelty_result, stub_llm_boundary, update_knowledge

        settings = _revision_settings(knowledge_revision_limit=2)
        topic = _seed_topic(db_conn, "Prune Topic")
        create_knowledge_state(db_conn, KnowledgeState(topic_id=topic.id, summary_text="Start.", token_count=2))
        db_conn.commit()

        for i in range(4):
            # A fresh boundary per iteration resets the stub's call counter, so
            # knowledge_init= is what each single update call receives.
            with stub_llm_boundary(knowledge_init=make_knowledge_update(f"Summary {i}.")):
                await update_knowledge(topic, make_novelty_result(), db_conn, settings)

        headers = list_knowledge_revision_headers(db_conn, topic.id, limit=10)
        assert len(headers) == 2
        assert [get_knowledge_revision(db_conn, h.id).summary_text for h in headers] == [
            "Summary 3.",
            "Summary 2.",
        ]

    async def test_revision_records_its_provenance(self, db_conn: sqlite3.Connection) -> None:
        """TW-AUD-017/AUG-255: which model wrote it, and for which topic scope."""
        from app.analysis.knowledge import topic_basis_hash
        from app.crud import get_knowledge_revision, list_knowledge_revision_headers
        from tests.helpers import init_knowledge as initialize_knowledge
        from tests.helpers import make_knowledge_update, stub_llm_boundary

        topic = _seed_topic(db_conn, "Provenance Topic")
        settings = _revision_settings()
        with stub_llm_boundary(knowledge_init=make_knowledge_update("Baseline summary.")):
            await initialize_knowledge(topic, _one_article(topic.id), db_conn, settings)

        headers = list_knowledge_revision_headers(db_conn, topic.id, limit=10)
        full = get_knowledge_revision(db_conn, headers[0].id)
        assert full.model == settings.llm.model
        assert full.basis_hash == topic_basis_hash(topic)

    async def test_basis_hash_tracks_semantic_topic_fields_only(self, db_conn: sqlite3.Connection) -> None:
        from app.analysis.knowledge import topic_basis_hash

        topic = _seed_topic(db_conn, "Basis Topic")
        baseline = topic_basis_hash(topic)

        topic.confidence_threshold = 0.9
        topic.check_interval_minutes = 120
        assert topic_basis_hash(topic) == baseline, "filtering and scheduling are not scope"

        topic.feed_urls = ["https://example.com/a.xml", "https://example.com/b.xml"]
        changed = topic_basis_hash(topic)
        assert changed != baseline

        topic.feed_urls = list(reversed(topic.feed_urls))
        assert topic_basis_hash(topic) == changed, "reordering feeds is not scope"

        topic.novelty_instruction = "Only rulings."
        assert topic_basis_hash(topic) != changed

    async def test_stale_generation_appends_no_revision(self, db_conn: sqlite3.Connection) -> None:
        """AUG-164: topics.id is a recyclable rowid, so a revision written after a
        delete+recreate would land on the replacement topic's timeline. The
        generation fence inside the init transaction is what stops it."""
        import pytest

        from app.checker import CheckTransitionAborted, _commit_init_transition, _snapshot_topic
        from app.crud import list_knowledge_revision_headers
        from tests.helpers import make_knowledge_update, stub_llm_boundary

        topic = _seed_topic(db_conn, "Recycled Topic")
        snapshot = _snapshot_topic(db_conn, topic.id)
        with stub_llm_boundary(knowledge_init=make_knowledge_update("Baseline.")):
            from app.analysis.knowledge import prepare_initial_knowledge

            plan = await prepare_initial_knowledge(topic, _one_article(topic.id), _revision_settings())

        # The rowid now belongs to a different topic than the one this plan was built for.
        db_conn.execute("UPDATE topics SET generation = 'recycled' WHERE id = ?", (topic.id,))
        db_conn.commit()

        with pytest.raises(CheckTransitionAborted):
            _commit_init_transition(db_conn, snapshot, plan, [], _revision_settings())
        db_conn.rollback()

        assert list_knowledge_revision_headers(db_conn, topic.id, limit=10) == []

    async def test_revision_failure_rolls_the_whole_write_back(self, db_conn: sqlite3.Connection) -> None:
        """The revision append is part of the durable transition, not an afterthought.

        It used to run after the state write had already committed and swallow its
        own failures, which meant a full disk could leave knowledge advanced while
        the check that produced it was never recorded. Inside the one transaction
        the failure takes the state write with it, so the next cycle re-runs from a
        consistent starting point.

        AUG-053: failing inside ``create_knowledge_revision`` itself (as this test
        used to) never lets its own INSERT run, so it stays green even without any
        rollback discipline at all. Pruning runs after that INSERT commits within
        the transaction, so failing there is what actually proves a later step can
        unwind an already-applied revision write.
        """
        from unittest.mock import patch

        from app.analysis.knowledge import prepare_knowledge_update
        from app.crud import create_knowledge_state, get_knowledge_state, list_knowledge_revision_headers
        from app.models import KnowledgeRevisionSource
        from tests.helpers import apply_plan, make_knowledge_update, make_novelty_result, stub_llm_boundary

        topic = _seed_topic(db_conn, "Resilient Topic")
        create_knowledge_state(db_conn, KnowledgeState(topic_id=topic.id, summary_text="Old.", token_count=2))
        db_conn.commit()

        with stub_llm_boundary(knowledge_init=make_knowledge_update("Never lands.")):
            plan = await prepare_knowledge_update(topic, make_novelty_result(), "Old.", _revision_settings())

        with (
            patch(
                "app.analysis.knowledge.prune_knowledge_revisions",
                side_effect=sqlite3.OperationalError("database or disk is full"),
            ),
            pytest.raises(sqlite3.OperationalError),
        ):
            apply_plan(db_conn, topic, plan, KnowledgeRevisionSource.UPDATE, _revision_settings())

        # The revision INSERT really ran before pruning failed — visible on this
        # same connection while the transaction is still open.
        assert (
            db_conn.execute("SELECT COUNT(*) FROM knowledge_revisions WHERE topic_id = ?", (topic.id,)).fetchone()[0]
            == 1
        )

        db_conn.rollback()

        assert not db_conn.in_transaction
        assert get_knowledge_state(db_conn, topic.id).summary_text == "Old."
        assert list_knowledge_revision_headers(db_conn, topic.id, limit=10) == []

        # No stuck write transaction blocks a fresh writer on the same connection.
        db_conn.execute("UPDATE knowledge_states SET summary_text = 'proof' WHERE topic_id = ?", (topic.id,))
        db_conn.commit()
        assert get_knowledge_state(db_conn, topic.id).summary_text == "proof"


class TestSplitSegments:
    def test_splits_sentences_within_a_line(self) -> None:
        from app.analysis.knowledge_diff import split_segments

        assert [s.text for s in split_segments("One fact. Two facts! Three?")] == [
            "One fact.",
            "Two facts!",
            "Three?",
        ]

    def test_splits_on_line_boundaries(self) -> None:
        from app.analysis.knowledge_diff import split_segments

        text = "**Current Status**\n- Trial began.\n- Verdict pending."
        assert [s.text for s in split_segments(text)] == ["**Current Status**", "- Trial began.", "- Verdict pending."]

    def test_keeps_indentation_in_the_display_text(self) -> None:
        """AUG-254: indentation is what makes a bullet a child of the one above."""
        from app.analysis.knowledge_diff import split_segments

        assert [s.text for s in split_segments("- Parent.\n  - Child.")] == ["- Parent.", "  - Child."]

    def test_trailing_whitespace_is_not_display_text(self) -> None:
        from app.analysis.knowledge_diff import split_segments

        assert [s.text for s in split_segments("A fact.  \nB fact.")] == ["A fact.", "B fact."]

    def test_empty_text_yields_no_segments(self) -> None:
        from app.analysis.knowledge_diff import split_segments

        assert split_segments("") == []
        assert split_segments("   \n  ") == []

    def test_nesting_change_produces_a_different_key(self) -> None:
        """AUG-254: "- Parent\\n  - Child" and "- Parent\\n- Child" render
        differently, so they must not compare equal."""
        from app.analysis.knowledge_diff import split_segments

        nested = split_segments("- Parent.\n  - Child.")
        flat = split_segments("- Parent.\n- Child.")
        assert nested[1].text.strip() == flat[1].text.strip()
        assert nested[1].key != flat[1].key

    def test_hard_break_change_produces_a_different_key(self) -> None:
        """A trailing two-space hard break renders as <br>; losing it is a change."""
        from app.analysis.knowledge_diff import split_segments

        assert split_segments("A fact.  \nB.")[0].key != split_segments("A fact.\nB.")[0].key

    def test_paragraph_boundary_change_produces_a_different_key(self) -> None:
        """Blank lines produce no segment but are not discarded (AUG-254)."""
        from app.analysis.knowledge_diff import split_segments

        joined = split_segments("A fact.\nB fact.")
        split_apart = split_segments("A fact.\n\nB fact.")
        assert [s.text for s in joined] == [s.text for s in split_apart]
        assert joined[1].key != split_apart[1].key

    def test_canonically_equivalent_text_shares_a_key(self) -> None:
        """AUG-170: NFC and NFD spellings of the same word are not a rewrite."""
        from app.analysis.knowledge_diff import split_segments

        assert split_segments("Café opened.")[0].key == split_segments("Café opened.")[0].key

    def test_cjk_sentences_split_without_spaces(self) -> None:
        """AUG-170: 。！？ end a sentence with no whitespace after them."""
        from app.analysis.knowledge_diff import split_segments

        assert [s.text for s in split_segments("裁判が始まった。判決は保留中である。")] == [
            "裁判が始まった。",
            "判決は保留中である。",
        ]

    def test_ascii_terminator_without_whitespace_does_not_split(self) -> None:
        """A version number is not two sentences."""
        from app.analysis.knowledge_diff import split_segments

        assert [s.text for s in split_segments("Released v1.2 today.")] == ["Released v1.2 today."]


class TestDiffSegments:
    def test_identical_text_is_all_equal(self) -> None:
        from app.analysis.knowledge_diff import MODE_DIFF, diff_segments

        result = diff_segments("A fact. B fact.", "A fact. B fact.")
        assert result.mode == MODE_DIFF
        assert [s.kind for s in result.segments] == ["equal", "equal"]

    def test_pure_addition(self) -> None:
        from app.analysis.knowledge_diff import diff_segments

        segments = diff_segments("A fact.", "A fact. B fact.").segments
        assert [(s.kind, s.text) for s in segments] == [("equal", "A fact."), ("insert", "B fact.")]

    def test_pure_deletion(self) -> None:
        from app.analysis.knowledge_diff import diff_segments

        segments = diff_segments("A fact. B fact.", "A fact.").segments
        assert [(s.kind, s.text) for s in segments] == [("equal", "A fact."), ("delete", "B fact.")]

    def test_replacement_emits_delete_then_insert(self) -> None:
        from app.analysis.knowledge_diff import diff_segments

        segments = diff_segments("Verdict pending.", "Verdict returned.").segments
        assert [(s.kind, s.text) for s in segments] == [
            ("delete", "Verdict pending."),
            ("insert", "Verdict returned."),
        ]

    def test_nested_to_flat_bullet_is_a_change(self) -> None:
        """AUG-254: stripping the indentation used to report this as no change."""
        from app.analysis.knowledge_diff import diff_segments

        result = diff_segments("- Parent.\n  - Child.", "- Parent.\n- Child.")
        assert {s.kind for s in result.segments} == {"equal", "delete", "insert"}

    def test_empty_old_is_a_snapshot_not_a_pile_of_insertions(self) -> None:
        """AUG-222: nothing was compared, so nothing was added."""
        from app.analysis.knowledge_diff import MODE_SNAPSHOT, diff_segments

        result = diff_segments("", "A fact. B fact.")
        assert result.mode == MODE_SNAPSHOT
        assert [s.kind for s in result.segments] == ["equal", "equal"]

    def test_empty_both_yields_nothing(self) -> None:
        from app.analysis.knowledge_diff import MODE_SNAPSHOT, diff_segments

        result = diff_segments("", "")
        assert result.mode == MODE_SNAPSHOT
        assert result.segments == []

    def test_preserves_document_order(self) -> None:
        from app.analysis.knowledge_diff import diff_segments

        old = "Alpha.\nBravo.\nCharlie."
        new = "Alpha.\nDelta.\nCharlie."
        assert [(s.kind, s.text) for s in diff_segments(old, new).segments] == [
            ("equal", "Alpha."),
            ("delete", "Bravo."),
            ("insert", "Delta."),
            ("equal", "Charlie."),
        ]

    def test_oversized_input_falls_back_to_a_labelled_snapshot(self) -> None:
        """Beyond the cap, skip the quadratic matcher rather than burn CPU.

        AUG-222: the fallback used to return every segment of the new revision as
        an insertion with no way for the caller to tell — a probe on two oversized
        but *identical* texts reported 1501 additions.
        """
        from app.analysis.knowledge_diff import MAX_DIFF_SEGMENTS, MODE_OVERSIZE, diff_segments

        old = "\n".join(f"- Old fact {i}." for i in range(MAX_DIFF_SEGMENTS + 1))
        new = "\n".join(f"- New fact {i}." for i in range(MAX_DIFF_SEGMENTS + 1))
        result = diff_segments(old, new)
        assert result.mode == MODE_OVERSIZE
        assert {s.kind for s in result.segments} == {"equal"}
        assert result.segments[0].text == "- New fact 0."

    def test_oversized_old_against_empty_new_is_not_a_diff(self) -> None:
        """AUG-222: this rendered as "No textual change" — the emptiest possible lie."""
        from app.analysis.knowledge_diff import MAX_DIFF_SEGMENTS, MODE_OVERSIZE, diff_segments

        old = "\n".join(f"- Old fact {i}." for i in range(MAX_DIFF_SEGMENTS + 1))
        result = diff_segments(old, "")
        assert result.mode == MODE_OVERSIZE
        assert result.segments == []

    def test_exact_cap_still_uses_the_real_matcher(self) -> None:
        """At exactly MAX_DIFF_SEGMENTS the matcher still runs; only strictly
        more than the cap degrades to the snapshot fallback."""
        from app.analysis.knowledge_diff import MAX_DIFF_SEGMENTS, MODE_DIFF, diff_segments

        text = "\n".join(f"- Fact {i}." for i in range(MAX_DIFF_SEGMENTS))
        result = diff_segments(text, text)  # identical input either way
        assert result.mode == MODE_DIFF
        assert {s.kind for s in result.segments} == {"equal"}

    def test_repeated_segments_use_exact_matching_not_autojunk(self) -> None:
        """AUG-052: below the 1500 cap but past difflib's 200-item autojunk
        threshold, a segment that recurs in >1% of the sequence — a common
        shape for LLM-written knowledge summaries with repeated bullet
        wording — is exactly the case ``autojunk=False`` exists for. With the
        default heuristic left on, this same 250-segment input with one
        inserted line renders as 125 wholesale deletions plus 126 insertions
        instead of the one real insertion, making the audit timeline
        materially misleading.
        """
        from app.analysis.knowledge_diff import diff_segments

        old_lines = ["- Fact confirmed."] * 250
        new_lines = old_lines[:125] + ["- New fact inserted here."] + old_lines[125:]
        old = "\n".join(old_lines)
        new = "\n".join(new_lines)

        segments = diff_segments(old, new).segments

        assert [s.kind for s in segments if s.kind == "delete"] == []
        assert [s.text for s in segments if s.kind == "insert"] == ["- New fact inserted here."]
        # old/new reconstruct exactly from the segment stream, in document order.
        assert [s.text for s in segments if s.kind in ("equal", "delete")] == old_lines
        assert [s.text for s in segments if s.kind in ("equal", "insert")] == new_lines


CSRF_TEST_TOKEN = "test-csrf-token-for-tests"


@pytest.fixture
async def client(db_conn: sqlite3.Connection):
    """httpx client bound to the test DB (mirrors tests/test_web.py::client)."""
    from unittest.mock import patch

    import httpx

    from app.main import app
    from app.web.dependencies import get_db_conn, get_settings

    settings = _revision_settings(notifications={"urls": ["json://localhost"]})

    def override_db():
        yield db_conn

    def override_settings():
        return settings

    app.dependency_overrides[get_db_conn] = override_db
    app.dependency_overrides[get_settings] = override_settings
    with patch("app.web.routers.settings.load_settings", return_value=settings):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            cookies={"csrf_token": CSRF_TEST_TOKEN},
            headers={"X-CSRF-Token": CSRF_TEST_TOKEN},
        ) as ac:
            yield ac
    app.dependency_overrides.clear()


class TestKnowledgeHistoryUI:
    async def test_timeline_lists_revisions(self, client, db_conn: sqlite3.Connection) -> None:
        topic = _seed_topic(db_conn, "Timeline Topic")
        _seed_revision(db_conn, topic.id, "First.", source=KnowledgeRevisionSource.INIT)
        _seed_revision(db_conn, topic.id, "Second.", source=KnowledgeRevisionSource.UPDATE)

        response = await client.get(f"/topics/{topic.id}")
        assert response.status_code == 200
        assert "Knowledge History" in response.text
        assert "showing the newest 2" in response.text

    async def test_timeline_emits_the_lazy_load_wiring(self, client, db_conn: sqlite3.Connection) -> None:
        """Without these attributes the diff never loads, yet every other test still passes."""
        topic = _seed_topic(db_conn, "Wiring Topic")
        revision = _seed_revision(db_conn, topic.id, "Only.")

        response = await client.get(f"/topics/{topic.id}")
        assert f'hx-get="/topics/{topic.id}/knowledge-diff/{revision.id}"' in response.text
        assert 'hx-trigger="toggle once from:closest details"' in response.text

    async def test_timeline_ships_a_durable_retry(self, client, db_conn: sqlite3.Connection) -> None:
        """AUG-233: ``toggle once`` is consumed the instant the request goes out,
        win or lose (vendored HTMX 2.0.4 does not reset it on failure), so a
        failed first load could never be retried by closing/reopening the
        disclosure. A retry button independent of that trigger must ship."""
        topic = _seed_topic(db_conn, "Retry Wiring Topic")
        revision = _seed_revision(db_conn, topic.id, "Only.")

        response = await client.get(f"/topics/{topic.id}")
        assert "hx-on:htmx:response-error" in response.text
        assert "hx-on:htmx:send-error" in response.text
        assert f'hx-get="/topics/{topic.id}/knowledge-diff/{revision.id}"' in response.text
        assert 'hx-target="closest .rev-body"' in response.text
        assert "Retry" in response.text

    async def test_timeline_does_not_ship_summaries(self, client, db_conn: sqlite3.Connection) -> None:
        topic = _seed_topic(db_conn, "Light Topic")
        _seed_revision(db_conn, topic.id, "SECRET-SUMMARY-BODY")

        response = await client.get(f"/topics/{topic.id}")
        assert "SECRET-SUMMARY-BODY" not in response.text

    async def test_timeline_honors_the_revision_limit(self, client, db_conn: sqlite3.Connection) -> None:
        """Pins the fetch to knowledge_revision_limit, not web_page_size."""
        from app.main import app
        from app.web.dependencies import get_settings

        topic = _seed_topic(db_conn, "Capped Topic")
        for i in range(3):
            _seed_revision(db_conn, topic.id, f"Rev {i}.")
        capped = _revision_settings(
            knowledge_revision_limit=2,
            notifications={"urls": ["json://localhost"]},
        )
        app.dependency_overrides[get_settings] = lambda: capped

        response = await client.get(f"/topics/{topic.id}")
        assert "showing the newest 2" in response.text

    async def test_empty_state_when_no_revisions(self, client, db_conn: sqlite3.Connection) -> None:
        topic = _seed_topic(db_conn, "No History Topic")
        response = await client.get(f"/topics/{topic.id}")
        assert response.status_code == 200
        assert "No knowledge revisions recorded yet" in response.text

    async def test_diff_fragment_marks_added_and_removed(self, client, db_conn: sqlite3.Connection) -> None:
        topic = _seed_topic(db_conn, "Diff Topic")
        _seed_revision(db_conn, topic.id, "Alpha fact.\nBravo fact.", token_count=5)
        newer = _seed_revision(db_conn, topic.id, "Alpha fact.\nCharlie fact.", token_count=9)

        response = await client.get(f"/topics/{topic.id}/knowledge-diff/{newer.id}")
        assert response.status_code == 200
        assert "Charlie fact." in response.text
        assert "Bravo fact." in response.text
        assert "diff-ins" in response.text
        assert "diff-del" in response.text
        assert "+4" in response.text  # token delta renders its sign

    async def test_diff_route_offloads_the_diff_to_a_worker_thread(self, client, db_conn: sqlite3.Connection) -> None:
        """AUG-054: replacing the ``asyncio.to_thread`` offload with a direct
        ``diff_segments(...)`` call would block the event loop (documented at
        up to ~0.32s on repetitive input at the cap) while still rendering
        identical output — so only pinning the offload itself catches it.
        """
        from unittest.mock import AsyncMock, patch

        from app.analysis.knowledge_diff import diff_segments

        topic = _seed_topic(db_conn, "Diff Offload Topic")
        _seed_revision(db_conn, topic.id, "Alpha fact.\nBravo fact.", token_count=5)
        newer = _seed_revision(db_conn, topic.id, "Alpha fact.\nCharlie fact.", token_count=9)

        with patch(
            "app.web.routers.topics.asyncio.to_thread",
            new=AsyncMock(side_effect=lambda fn, *args: fn(*args)),
        ) as mock_to_thread:
            response = await client.get(f"/topics/{topic.id}/knowledge-diff/{newer.id}")

        assert response.status_code == 200
        mock_to_thread.assert_awaited_once_with(diff_segments, "Alpha fact.\nBravo fact.", "Alpha fact.\nCharlie fact.")

    async def test_diff_fragment_shows_change_note(self, client, db_conn: sqlite3.Connection) -> None:
        topic = _seed_topic(db_conn, "Note Topic")
        _seed_revision(db_conn, topic.id, "Old.")
        newer = _seed_revision(db_conn, topic.id, "New.", change_note="Verdict announced.")

        response = await client.get(f"/topics/{topic.id}/knowledge-diff/{newer.id}")
        assert "Verdict announced." in response.text

    async def test_oldest_revision_renders_as_snapshot(self, client, db_conn: sqlite3.Connection) -> None:
        topic = _seed_topic(db_conn, "Oldest Topic")
        oldest = _seed_revision(db_conn, topic.id, "Baseline fact.")

        response = await client.get(f"/topics/{topic.id}/knowledge-diff/{oldest.id}")
        assert response.status_code == 200
        assert "No earlier revision is retained" in response.text
        assert "Baseline fact." in response.text

    async def test_reinitialize_renders_as_snapshot(self, client, db_conn: sqlite3.Connection) -> None:
        """An init revision starts a new lineage; diffing it against the old one is nonsense."""
        topic = _seed_topic(db_conn, "Reinit Topic")
        _seed_revision(db_conn, topic.id, "Old lineage fact.", source=KnowledgeRevisionSource.UPDATE)
        reinit = _seed_revision(db_conn, topic.id, "Fresh baseline.", source=KnowledgeRevisionSource.INIT)

        response = await client.get(f"/topics/{topic.id}/knowledge-diff/{reinit.id}")
        assert response.status_code == 200
        assert "Old lineage fact." not in response.text
        assert "Fresh baseline." in response.text

    async def test_unchanged_revision_says_so(self, client, db_conn: sqlite3.Connection) -> None:
        topic = _seed_topic(db_conn, "Unchanged Topic")
        _seed_revision(db_conn, topic.id, "Identical body.")
        newer = _seed_revision(db_conn, topic.id, "Identical body.")

        response = await client.get(f"/topics/{topic.id}/knowledge-diff/{newer.id}")
        assert "No textual change" in response.text

    async def test_diff_escapes_html_in_summary(self, client, db_conn: sqlite3.Connection) -> None:
        """Summaries are LLM output derived from articles — never trust them as markup."""
        topic = _seed_topic(db_conn, "XSS Topic")
        _seed_revision(db_conn, topic.id, "Safe.")
        newer = _seed_revision(db_conn, topic.id, "Safe.\n<script>alert(1)</script>")

        response = await client.get(f"/topics/{topic.id}/knowledge-diff/{newer.id}")
        assert "<script>alert(1)</script>" not in response.text
        assert "&lt;script&gt;" in response.text

    async def test_revision_from_another_topic_is_404(self, client, db_conn: sqlite3.Connection) -> None:
        a = _seed_topic(db_conn, "Topic A")
        b = _seed_topic(db_conn, "Topic B")
        b_rev = _seed_revision(db_conn, b.id, "B content.")

        response = await client.get(f"/topics/{a.id}/knowledge-diff/{b_rev.id}")
        assert response.status_code == 404

    async def test_missing_revision_is_404(self, client, db_conn: sqlite3.Connection) -> None:
        topic = _seed_topic(db_conn, "Missing Rev Topic")
        response = await client.get(f"/topics/{topic.id}/knowledge-diff/999999")
        assert response.status_code == 404

    async def test_oldest_revision_reports_no_counts(self, client, db_conn: sqlite3.Connection) -> None:
        """AUG-222: nothing was compared, so "N added" would be invented."""
        topic = _seed_topic(db_conn, "No Counts Topic")
        oldest = _seed_revision(db_conn, topic.id, "Alpha.\nBravo.", token_count=7)

        response = await client.get(f"/topics/{topic.id}/knowledge-diff/{oldest.id}")
        assert "added" not in response.text
        assert "removed" not in response.text
        assert "tokens" not in response.text

    async def test_reinitialize_is_labelled_as_a_new_baseline(self, client, db_conn: sqlite3.Connection) -> None:
        """AUG-222: retained older history is not "no earlier revision"."""
        topic = _seed_topic(db_conn, "Relabel Topic")
        _seed_revision(db_conn, topic.id, "Old lineage.", source=KnowledgeRevisionSource.UPDATE)
        reinit = _seed_revision(db_conn, topic.id, "Fresh baseline.", source=KnowledgeRevisionSource.INIT)

        response = await client.get(f"/topics/{topic.id}/knowledge-diff/{reinit.id}")
        assert "Research was re-run" in response.text
        assert "No earlier revision is retained" not in response.text

    async def test_first_init_is_still_the_oldest_not_a_reinitialize(self, client, db_conn: sqlite3.Connection) -> None:
        """An init with nothing behind it is the first research, not a re-run."""
        topic = _seed_topic(db_conn, "First Init Topic")
        first = _seed_revision(db_conn, topic.id, "Baseline.", source=KnowledgeRevisionSource.INIT)

        response = await client.get(f"/topics/{topic.id}/knowledge-diff/{first.id}")
        assert "No earlier revision is retained" in response.text
        assert "Research was re-run" not in response.text

    async def test_oversized_revision_is_labelled_a_snapshot(self, client, db_conn: sqlite3.Connection) -> None:
        """AUG-222: the bounded-cost fallback used to report every segment as added."""
        from app.analysis.knowledge_diff import MAX_DIFF_SEGMENTS

        topic = _seed_topic(db_conn, "Oversize Topic")
        body = "\n".join(f"- Fact {i}." for i in range(MAX_DIFF_SEGMENTS + 1))
        _seed_revision(db_conn, topic.id, body, token_count=100)
        newer = _seed_revision(db_conn, topic.id, body, token_count=100)

        response = await client.get(f"/topics/{topic.id}/knowledge-diff/{newer.id}")
        assert response.status_code == 200
        assert "too large to compare" in response.text
        assert "added" not in response.text

    async def test_unknown_source_is_a_lineage_boundary(self, client, db_conn: sqlite3.Connection) -> None:
        """AUG-155: a source this version does not know is not an ordinary update."""
        topic = _seed_topic(db_conn, "Unknown Source Topic")
        _seed_revision(db_conn, topic.id, "Old lineage.")
        db_conn.execute(
            "INSERT INTO knowledge_revisions (topic_id, summary_text, token_count, source, created_at)"
            " VALUES (?, 'From the future.', 4, 'compaction', ?)",
            (topic.id, datetime.now(UTC).isoformat()),
        )
        db_conn.commit()
        future_id = db_conn.execute("SELECT MAX(id) FROM knowledge_revisions").fetchone()[0]

        response = await client.get(f"/topics/{topic.id}/knowledge-diff/{future_id}")
        assert response.status_code == 200
        assert "does not recognise" in response.text
        assert "Old lineage." not in response.text

    async def test_timeline_labels_an_unknown_source_unknown(self, client, db_conn: sqlite3.Connection) -> None:
        topic = _seed_topic(db_conn, "Unknown Badge Topic")
        db_conn.execute(
            "INSERT INTO knowledge_revisions (topic_id, summary_text, token_count, source, created_at)"
            " VALUES (?, 'x', 1, 'compaction', ?)",
            (topic.id, datetime.now(UTC).isoformat()),
        )
        db_conn.commit()

        response = await client.get(f"/topics/{topic.id}")
        assert ">Unknown</span>" in response.text
        assert ">Update</span>" not in response.text

    async def test_token_delta_hidden_across_different_models(self, client, db_conn: sqlite3.Connection) -> None:
        """AUG-255: counts from two tokenizers are not subtractable."""
        topic = _seed_topic(db_conn, "Mixed Model Topic")
        _seed_revision(db_conn, topic.id, "Alpha.", token_count=5, model="openai/gpt-4o-mini")
        newer = _seed_revision(db_conn, topic.id, "Bravo.", token_count=900, model="ollama/llama3")

        response = await client.get(f"/topics/{topic.id}/knowledge-diff/{newer.id}")
        assert "added" in response.text  # the text diff itself is unaffected
        assert "tokens" not in response.text

    async def test_token_delta_hidden_for_pre_provenance_rows(self, client, db_conn: sqlite3.Connection) -> None:
        topic = _seed_topic(db_conn, "Legacy Token Topic")
        _seed_revision(db_conn, topic.id, "Alpha.", token_count=5, model=None)
        newer = _seed_revision(db_conn, topic.id, "Bravo.", token_count=9, model=None)

        response = await client.get(f"/topics/{topic.id}/knowledge-diff/{newer.id}")
        assert "tokens" not in response.text

    async def test_timeline_hides_pre_provenance_token_counts(self, client, db_conn: sqlite3.Connection) -> None:
        topic = _seed_topic(db_conn, "Legacy Timeline Topic")
        _seed_revision(db_conn, topic.id, "Alpha.", token_count=4242, model=None)

        response = await client.get(f"/topics/{topic.id}")
        assert "4242" not in response.text

    async def test_timeline_shows_the_change_headline(self, client, db_conn: sqlite3.Connection) -> None:
        """AUG-124: the note is already stored; expanding each row to find it is the bug."""
        topic = _seed_topic(db_conn, "Headline Topic")
        _seed_revision(db_conn, topic.id, "SECRET-SUMMARY-BODY", change_note="Court ruled on Tuesday.")

        response = await client.get(f"/topics/{topic.id}")
        assert "Court ruled on Tuesday." in response.text
        # Still no summary bodies — the headline is metadata, not the diff.
        assert "SECRET-SUMMARY-BODY" not in response.text

    async def test_timeline_escapes_the_change_headline(self, client, db_conn: sqlite3.Connection) -> None:
        topic = _seed_topic(db_conn, "Headline XSS Topic")
        _seed_revision(db_conn, topic.id, "Body.", change_note="<script>alert(1)</script>")

        response = await client.get(f"/topics/{topic.id}")
        assert "<script>alert(1)</script>" not in response.text
        assert "&lt;script&gt;" in response.text

    async def test_oversized_revision_id_is_not_a_500(self, client, db_conn: sqlite3.Connection) -> None:
        """AUG-257: SQLite binds signed 64-bit; anything larger used to raise
        OverflowError from the query itself, before the 404 check."""
        topic = _seed_topic(db_conn, "Big Id Topic")

        response = await client.get(f"/topics/{topic.id}/knowledge-diff/9223372036854775808")
        assert response.status_code == 422

    async def test_zero_and_negative_revision_ids_are_rejected(self, client, db_conn: sqlite3.Connection) -> None:
        topic = _seed_topic(db_conn, "Small Id Topic")

        assert (await client.get(f"/topics/{topic.id}/knowledge-diff/0")).status_code == 422
        assert (await client.get(f"/topics/{topic.id}/knowledge-diff/-1")).status_code == 422

    async def test_pruned_revision_ends_the_htmx_interaction(self, client, db_conn: sqlite3.Connection) -> None:
        """AUG-318: a revision pruned between page load and expand is normal
        retention, not a server fault — 404 fires the global error toast and
        offers a retry that can never succeed."""
        topic = _seed_topic(db_conn, "Pruned Topic")
        doomed = _seed_revision(db_conn, topic.id, "About to be pruned.")
        db_conn.execute("DELETE FROM knowledge_revisions WHERE id = ?", (doomed.id,))
        db_conn.commit()

        response = await client.get(
            f"/topics/{topic.id}/knowledge-diff/{doomed.id}",
            headers={"HX-Request": "true"},
        )
        assert response.status_code == 200
        assert "has since been pruned" in response.text
        assert "Retry" not in response.text

    async def test_pruned_revision_still_404s_on_direct_navigation(self, client, db_conn: sqlite3.Connection) -> None:
        topic = _seed_topic(db_conn, "Pruned Direct Topic")
        doomed = _seed_revision(db_conn, topic.id, "Gone.")
        db_conn.execute("DELETE FROM knowledge_revisions WHERE id = ?", (doomed.id,))
        db_conn.commit()

        response = await client.get(f"/topics/{topic.id}/knowledge-diff/{doomed.id}")
        assert response.status_code == 404

    async def test_another_topics_revision_is_404_even_over_htmx(self, client, db_conn: sqlite3.Connection) -> None:
        """The pruning fragment must not become a cross-topic read oracle."""
        a = _seed_topic(db_conn, "HX Topic A")
        b = _seed_topic(db_conn, "HX Topic B")
        b_rev = _seed_revision(db_conn, b.id, "B CONTENT.")

        response = await client.get(
            f"/topics/{a.id}/knowledge-diff/{b_rev.id}",
            headers={"HX-Request": "true"},
        )
        assert "B CONTENT." not in response.text

    async def test_edited_topic_scope_is_flagged_on_the_knowledge_panel(
        self, client, db_conn: sqlite3.Connection
    ) -> None:
        """TW-AUD-017: knowledge answers the question the topic asked when it was
        written; an edit moves the question and leaves the baseline behind."""
        from app.analysis.knowledge import topic_basis_hash
        from app.crud import create_knowledge_state, update_topic

        topic = _seed_topic(db_conn, "Scope Topic")
        create_knowledge_state(db_conn, KnowledgeState(topic_id=topic.id, summary_text="Baseline.", token_count=2))
        _seed_revision(db_conn, topic.id, "Baseline.", basis_hash=topic_basis_hash(topic))
        db_conn.commit()

        fresh = await client.get(f"/topics/{topic.id}")
        assert "built before the topic" not in fresh.text

        topic.description = "A materially different question."
        update_topic(db_conn, topic)
        db_conn.commit()

        stale = await client.get(f"/topics/{topic.id}")
        assert "built before the topic" in stale.text

    async def test_pre_provenance_knowledge_is_not_flagged_stale(self, client, db_conn: sqlite3.Connection) -> None:
        """A NULL basis means unknown, which is not a mismatch."""
        from app.crud import create_knowledge_state

        topic = _seed_topic(db_conn, "Legacy Scope Topic")
        create_knowledge_state(db_conn, KnowledgeState(topic_id=topic.id, summary_text="Baseline.", token_count=2))
        _seed_revision(db_conn, topic.id, "Baseline.", basis_hash=None)
        db_conn.commit()

        response = await client.get(f"/topics/{topic.id}")
        assert "built before the topic" not in response.text


class TestInitialPlanCarriesTheAnalyzedSubset:
    """A baseline built from a fitted prompt reports what it was built from."""

    async def test_reported_ids_reach_the_plan(self, db_conn: sqlite3.Connection) -> None:
        from unittest.mock import AsyncMock, patch

        from app.analysis.knowledge import prepare_initial_knowledge
        from app.analysis.llm import KnowledgeStateUpdate
        from app.models import Article

        topic = _seed_topic(db_conn, "Fitted Init")
        articles = [
            Article(
                id=i,
                topic_id=topic.id,
                title=f"A{i}",
                url=f"https://example.com/{i}",
                content_hash=f"h{i}",
                source_feed="https://example.com/feed.xml",
            )
            for i in (1, 2)
        ]
        result = KnowledgeStateUpdate(
            updated_summary="Baseline.",
            token_count=3,
            confidence=0.9,
            sufficient_data=True,
            analyzed_article_ids=[1],
        )

        with patch(
            "app.analysis.knowledge.generate_initial_knowledge",
            new_callable=AsyncMock,
            return_value=result,
        ):
            plan = await prepare_initial_knowledge(topic, articles, _revision_settings())

        assert plan.analyzed_article_ids == [1]

    async def test_silent_llm_leaves_the_plan_unconstrained(self, db_conn: sqlite3.Connection) -> None:
        """An LLM layer that reports no subset still means "the whole input was used"."""
        from unittest.mock import AsyncMock, patch

        from app.analysis.knowledge import prepare_initial_knowledge
        from app.analysis.llm import KnowledgeStateUpdate

        topic = _seed_topic(db_conn, "Silent Init")
        result = KnowledgeStateUpdate(updated_summary="Baseline.", token_count=3, confidence=0.9, sufficient_data=True)
        with patch(
            "app.analysis.knowledge.generate_initial_knowledge",
            new_callable=AsyncMock,
            return_value=result,
        ):
            plan = await prepare_initial_knowledge(topic, [], _revision_settings())

        assert plan.analyzed_article_ids is None
