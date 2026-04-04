"""Tests for the JSON API endpoints."""

import sqlite3

from fastapi.testclient import TestClient

from app.config import LLMSettings, Settings
from app.crud import create_check_result, create_knowledge_state, create_topic
from app.main import app
from app.models import CheckResult, KnowledgeState, Topic, TopicStatus


def _make_settings(**overrides) -> Settings:
    defaults = {
        "llm": LLMSettings(model="openai/gpt-4o-mini", api_key="test-key"),
        "check_interval_hours": 6,
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _seed_topic(conn: sqlite3.Connection, name: str = "Test Topic", status: str = "ready") -> Topic:
    topic = Topic(name=name, description=f"About {name}", status=TopicStatus(status))
    return create_topic(conn, topic)


def _seed_check(conn: sqlite3.Connection, topic_id: int, has_new_info: bool = False) -> CheckResult:
    result = CheckResult(topic_id=topic_id, has_new_info=has_new_info, articles_found=5, articles_new=2)
    return create_check_result(conn, result)


def _seed_knowledge(conn: sqlite3.Connection, topic_id: int) -> KnowledgeState:
    state = KnowledgeState(topic_id=topic_id, summary_text="Test knowledge", token_count=50)
    return create_knowledge_state(conn, state)


class TestAPIListTopics:
    def test_list_all_topics(self, db_conn: sqlite3.Connection):
        _seed_topic(db_conn, "Topic A")
        _seed_topic(db_conn, "Topic B")
        db_conn.commit()

        app.state.db_path = None
        with TestClient(app) as client:
            # Override dependency to use test db
            from app.web.dependencies import get_db_conn

            app.dependency_overrides[get_db_conn] = lambda: db_conn
            try:
                resp = client.get("/api/v1/topics")
                assert resp.status_code == 200
                data = resp.json()
                assert len(data) == 2
            finally:
                app.dependency_overrides.pop(get_db_conn, None)

    def test_filter_active_only(self, db_conn: sqlite3.Connection):
        _seed_topic(db_conn, "Active")
        inactive = _seed_topic(db_conn, "Inactive")
        inactive.is_active = False
        from app.crud import update_topic

        update_topic(db_conn, inactive)
        db_conn.commit()

        from app.web.dependencies import get_db_conn

        app.dependency_overrides[get_db_conn] = lambda: db_conn
        try:
            with TestClient(app) as client:
                resp = client.get("/api/v1/topics?active=true")
                assert resp.status_code == 200
                data = resp.json()
                assert len(data) == 1
                assert data[0]["name"] == "Active"
        finally:
            app.dependency_overrides.pop(get_db_conn, None)


class TestAPIGetTopic:
    def test_get_existing_topic(self, db_conn: sqlite3.Connection):
        topic = _seed_topic(db_conn)
        _seed_knowledge(db_conn, topic.id)
        db_conn.commit()

        from app.web.dependencies import get_db_conn

        app.dependency_overrides[get_db_conn] = lambda: db_conn
        try:
            with TestClient(app) as client:
                resp = client.get(f"/api/v1/topics/{topic.id}")
                assert resp.status_code == 200
                data = resp.json()
                assert data["topic"]["name"] == "Test Topic"
                assert data["knowledge"]["summary_text"] == "Test knowledge"
        finally:
            app.dependency_overrides.pop(get_db_conn, None)

    def test_get_nonexistent_topic(self, db_conn: sqlite3.Connection):
        from app.web.dependencies import get_db_conn

        app.dependency_overrides[get_db_conn] = lambda: db_conn
        try:
            with TestClient(app) as client:
                resp = client.get("/api/v1/topics/9999")
                assert resp.status_code == 404
        finally:
            app.dependency_overrides.pop(get_db_conn, None)


class TestAPIChecks:
    def test_list_checks_with_pagination(self, db_conn: sqlite3.Connection):
        topic = _seed_topic(db_conn)
        for _ in range(5):
            _seed_check(db_conn, topic.id)
        db_conn.commit()

        from app.web.dependencies import get_db_conn

        app.dependency_overrides[get_db_conn] = lambda: db_conn
        try:
            with TestClient(app) as client:
                resp = client.get(f"/api/v1/topics/{topic.id}/checks?per_page=2&page=1")
                assert resp.status_code == 200
                data = resp.json()
                assert len(data["checks"]) == 2
                assert data["total"] == 5
                assert data["pages"] == 3
        finally:
            app.dependency_overrides.pop(get_db_conn, None)

    def test_checks_nonexistent_topic(self, db_conn: sqlite3.Connection):
        from app.web.dependencies import get_db_conn

        app.dependency_overrides[get_db_conn] = lambda: db_conn
        try:
            with TestClient(app) as client:
                resp = client.get("/api/v1/topics/9999/checks")
                assert resp.status_code == 404
        finally:
            app.dependency_overrides.pop(get_db_conn, None)


class TestAPIKnowledge:
    def test_get_knowledge(self, db_conn: sqlite3.Connection):
        topic = _seed_topic(db_conn)
        _seed_knowledge(db_conn, topic.id)
        db_conn.commit()

        from app.web.dependencies import get_db_conn

        app.dependency_overrides[get_db_conn] = lambda: db_conn
        try:
            with TestClient(app) as client:
                resp = client.get(f"/api/v1/topics/{topic.id}/knowledge")
                assert resp.status_code == 200
                data = resp.json()
                assert data["summary_text"] == "Test knowledge"
        finally:
            app.dependency_overrides.pop(get_db_conn, None)

    def test_knowledge_not_found(self, db_conn: sqlite3.Connection):
        topic = _seed_topic(db_conn)
        db_conn.commit()

        from app.web.dependencies import get_db_conn

        app.dependency_overrides[get_db_conn] = lambda: db_conn
        try:
            with TestClient(app) as client:
                resp = client.get(f"/api/v1/topics/{topic.id}/knowledge")
                assert resp.status_code == 404
        finally:
            app.dependency_overrides.pop(get_db_conn, None)


class TestAPITriggerCheck:
    def test_trigger_requires_csrf(self, db_conn: sqlite3.Connection):
        topic = _seed_topic(db_conn)
        db_conn.commit()

        from app.web.dependencies import get_db_conn

        app.dependency_overrides[get_db_conn] = lambda: db_conn
        try:
            with TestClient(app) as client:
                resp = client.post(f"/api/v1/topics/{topic.id}/check")
                # CSRF should block without token
                assert resp.status_code in (403, 422)
        finally:
            app.dependency_overrides.pop(get_db_conn, None)

    def test_trigger_nonexistent_topic(self, db_conn: sqlite3.Connection):
        from app.web.dependencies import get_db_conn

        app.dependency_overrides[get_db_conn] = lambda: db_conn
        try:
            with TestClient(app) as client:
                # Get CSRF token first
                home = client.get("/")
                csrf_token = home.cookies.get("csrf_token", "")
                resp = client.post(
                    "/api/v1/topics/9999/check",
                    headers={"X-CSRF-Token": csrf_token},
                )
                assert resp.status_code == 404
        finally:
            app.dependency_overrides.pop(get_db_conn, None)

    def test_trigger_non_ready_topic(self, db_conn: sqlite3.Connection):
        topic = _seed_topic(db_conn, status="new")
        db_conn.commit()

        from app.web.dependencies import get_db_conn

        app.dependency_overrides[get_db_conn] = lambda: db_conn
        try:
            with TestClient(app) as client:
                home = client.get("/")
                csrf_token = home.cookies.get("csrf_token", "")
                resp = client.post(
                    f"/api/v1/topics/{topic.id}/check",
                    headers={"X-CSRF-Token": csrf_token},
                )
                assert resp.status_code == 409
        finally:
            app.dependency_overrides.pop(get_db_conn, None)
