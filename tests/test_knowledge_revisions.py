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

    async def test_revision_failure_rolls_the_whole_write_back(self, db_conn: sqlite3.Connection) -> None:
        """The revision append is part of the durable transition, not an afterthought.

        It used to run after the state write had already committed and swallow its
        own failures, which meant a full disk could leave knowledge advanced while
        the check that produced it was never recorded. Inside the one transaction
        the failure takes the state write with it, so the next cycle re-runs from a
        consistent starting point.
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
                "app.analysis.knowledge.create_knowledge_revision",
                side_effect=sqlite3.OperationalError("database or disk is full"),
            ),
            pytest.raises(sqlite3.OperationalError),
        ):
            apply_plan(db_conn, topic, plan, KnowledgeRevisionSource.UPDATE, _revision_settings())
        db_conn.rollback()

        assert get_knowledge_state(db_conn, topic.id).summary_text == "Old."
        assert list_knowledge_revision_headers(db_conn, topic.id, limit=10) == []


class TestSplitSegments:
    def test_splits_sentences_within_a_line(self) -> None:
        from app.analysis.knowledge_diff import split_segments

        assert split_segments("One fact. Two facts! Three?") == ["One fact.", "Two facts!", "Three?"]

    def test_splits_on_line_boundaries(self) -> None:
        from app.analysis.knowledge_diff import split_segments

        text = "**Current Status**\n- Trial began.\n- Verdict pending."
        assert split_segments(text) == ["**Current Status**", "- Trial began.", "- Verdict pending."]

    def test_drops_blank_lines_and_strips(self) -> None:
        from app.analysis.knowledge_diff import split_segments

        assert split_segments("  A fact.  \n\n\n   \n  B fact.  ") == ["A fact.", "B fact."]

    def test_empty_text_yields_no_segments(self) -> None:
        from app.analysis.knowledge_diff import split_segments

        assert split_segments("") == []
        assert split_segments("   \n  ") == []


class TestDiffSegments:
    def test_identical_text_is_all_equal(self) -> None:
        from app.analysis.knowledge_diff import diff_segments

        assert [s.kind for s in diff_segments("A fact. B fact.", "A fact. B fact.")] == ["equal", "equal"]

    def test_pure_addition(self) -> None:
        from app.analysis.knowledge_diff import diff_segments

        segments = diff_segments("A fact.", "A fact. B fact.")
        assert [(s.kind, s.text) for s in segments] == [("equal", "A fact."), ("insert", "B fact.")]

    def test_pure_deletion(self) -> None:
        from app.analysis.knowledge_diff import diff_segments

        segments = diff_segments("A fact. B fact.", "A fact.")
        assert [(s.kind, s.text) for s in segments] == [("equal", "A fact."), ("delete", "B fact.")]

    def test_replacement_emits_delete_then_insert(self) -> None:
        from app.analysis.knowledge_diff import diff_segments

        segments = diff_segments("Verdict pending.", "Verdict returned.")
        assert [(s.kind, s.text) for s in segments] == [
            ("delete", "Verdict pending."),
            ("insert", "Verdict returned."),
        ]

    def test_empty_old_is_all_insert(self) -> None:
        """The oldest retained revision has no predecessor — render it as a snapshot."""
        from app.analysis.knowledge_diff import diff_segments

        assert [s.kind for s in diff_segments("", "A fact. B fact.")] == ["insert", "insert"]

    def test_empty_both_yields_nothing(self) -> None:
        from app.analysis.knowledge_diff import diff_segments

        assert diff_segments("", "") == []

    def test_preserves_document_order(self) -> None:
        from app.analysis.knowledge_diff import diff_segments

        old = "Alpha.\nBravo.\nCharlie."
        new = "Alpha.\nDelta.\nCharlie."
        assert [(s.kind, s.text) for s in diff_segments(old, new)] == [
            ("equal", "Alpha."),
            ("delete", "Bravo."),
            ("insert", "Delta."),
            ("equal", "Charlie."),
        ]

    def test_oversized_input_falls_back_to_snapshot(self) -> None:
        """Beyond the cap, skip the quadratic matcher rather than burn CPU."""
        from app.analysis.knowledge_diff import MAX_DIFF_SEGMENTS, diff_segments

        old = "\n".join(f"- Old fact {i}." for i in range(MAX_DIFF_SEGMENTS + 1))
        new = "\n".join(f"- New fact {i}." for i in range(MAX_DIFF_SEGMENTS + 1))
        segments = diff_segments(old, new)
        assert {s.kind for s in segments} == {"insert"}
        assert segments[0].text == "- New fact 0."


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


class TestInitialPlanCarriesTheAnalyzedSubset:
    """A baseline built from a fitted prompt reports what it was built from."""

    async def test_reported_ids_reach_the_plan(self, db_conn: sqlite3.Connection) -> None:
        from unittest.mock import AsyncMock, patch

        from app.analysis.knowledge import prepare_initial_knowledge
        from app.analysis.llm import KnowledgeStateUpdate
        from app.models import Article

        class _PartialInit(KnowledgeStateUpdate):
            analyzed_article_ids: list[int] = []

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
        result = _PartialInit(
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
        from app.analysis.knowledge import prepare_initial_knowledge
        from tests.helpers import make_knowledge_update, stub_llm_boundary

        topic = _seed_topic(db_conn, "Silent Init")
        with stub_llm_boundary(knowledge_init=make_knowledge_update("Baseline.")):
            plan = await prepare_initial_knowledge(topic, [], _revision_settings())

        assert plan.analyzed_article_ids is None
