"""Tests for the web UI: routes, templates, and HTMX interactions."""

import sqlite3
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.config import LLMSettings, NotificationSettings, Settings
from app.crud import (
    create_check_result,
    create_knowledge_state,
    create_topic,
)
from app.main import app
from app.models import (
    CheckResult,
    FeedMode,
    KnowledgeState,
    Topic,
    TopicStatus,
)
from app.web.dependencies import get_db_conn, get_settings


def _make_settings(**overrides) -> Settings:
    defaults = {
        "llm": LLMSettings(model="openai/gpt-4o-mini", api_key="test-key-12345678"),
        "notifications": NotificationSettings(urls=["json://localhost"]),
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _make_topic(conn: sqlite3.Connection, **overrides) -> Topic:
    defaults = {
        "name": "Test Topic",
        "description": "A test topic",
        "feed_urls": ["https://example.com/feed.xml"],
        "feed_mode": FeedMode.MANUAL,
        "status": TopicStatus.READY,
    }
    defaults.update(overrides)
    topic = create_topic(conn, Topic(**defaults))
    conn.commit()
    return topic


CSRF_TEST_TOKEN = "test-csrf-token-for-tests"


@pytest.fixture
async def client(
    db_conn: sqlite3.Connection,
) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Create a test client with database dependency overridden."""
    settings = _make_settings()

    def override_db():
        yield db_conn

    def override_settings():
        return settings

    app.dependency_overrides[get_db_conn] = override_db
    app.dependency_overrides[get_settings] = override_settings

    # GET /settings calls load_settings() directly instead of using Depends
    with patch("app.web.routers.settings.load_settings", return_value=settings):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            cookies={"csrf_token": CSRF_TEST_TOKEN},
            headers={"X-CSRF-Token": CSRF_TEST_TOKEN},
        ) as ac:
            yield ac

    app.dependency_overrides.clear()


# --- Dashboard ---


class TestDashboard:
    """Tests for GET / (dashboard)."""

    async def test_dashboard_empty(self, client: httpx.AsyncClient) -> None:
        """Empty database shows 'no topics' message."""
        response = await client.get("/")
        assert response.status_code == 200
        assert "adding your first topic" in response.text

    async def test_dashboard_shows_error_banner(self, client: httpx.AsyncClient) -> None:
        """The ?error= query param (e.g. from a failed OPML import redirect) is surfaced."""
        response = await client.get("/?error=No+file+selected")
        assert response.status_code == 200
        assert "No file selected" in response.text
        assert "error-banner" in response.text

    async def test_dashboard_shows_topics(self, client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
        """Dashboard lists topics with names."""
        _make_topic(db_conn, name="Topic A")
        _make_topic(db_conn, name="Topic B", status=TopicStatus.RESEARCHING)

        response = await client.get("/")
        assert response.status_code == 200
        assert "Topic A" in response.text
        assert "Topic B" in response.text

    async def test_dashboard_shows_last_check(self, client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
        """Dashboard shows check info when a check has been performed."""
        topic = _make_topic(db_conn)
        create_check_result(
            db_conn,
            CheckResult(topic_id=topic.id, articles_found=3),
        )
        db_conn.commit()

        response = await client.get("/")
        assert response.status_code == 200
        assert "Never" not in response.text

    async def test_dashboard_shows_check_now_for_ready(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """Ready topics have a Check Now button."""
        _make_topic(db_conn, status=TopicStatus.READY)

        response = await client.get("/")
        assert "Check Now" in response.text

    async def test_dashboard_no_check_button_for_researching(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """Researching topics do not have a Check Now button."""
        _make_topic(db_conn, status=TopicStatus.RESEARCHING)

        response = await client.get("/")
        assert "Check Now" not in response.text


# --- Add Topic ---


class TestAddTopic:
    """Tests for GET /topics/new and POST /topics."""

    async def test_add_form_renders(self, client: httpx.AsyncClient) -> None:
        """The add topic form page loads successfully."""
        response = await client.get("/topics/new")
        assert response.status_code == 200
        assert "Add Topic" in response.text
        assert "<form" in response.text

    async def test_create_topic_redirects_to_detail(self, client: httpx.AsyncClient) -> None:
        """POST /topics creates a topic and redirects to its detail page."""
        with patch(
            "app.web.routers.background._run_init",
            new_callable=AsyncMock,
        ):
            response = await client.post(
                "/topics",
                data={
                    "name": "New Topic",
                    "description": "Testing creation",
                    "feed_mode": "manual",
                    "feed_urls": "https://example.com/feed1.xml\nhttps://example.com/feed2.xml",
                },
                follow_redirects=False,
            )

        assert response.status_code == 303
        assert "/topics/" in response.headers["location"]

    async def test_create_topic_parses_feed_urls(self, client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
        """Feed URLs textarea is parsed into a list (one URL per line)."""
        with patch(
            "app.web.routers.background._run_init",
            new_callable=AsyncMock,
        ):
            await client.post(
                "/topics",
                data={
                    "name": "Feed Parse Test",
                    "description": "Test",
                    "feed_mode": "manual",
                    "feed_urls": "https://a.com/feed\n\nhttps://b.com/feed\n  ",
                },
                follow_redirects=False,
            )

        from app.crud import get_topic_by_name

        topic = get_topic_by_name(db_conn, "Feed Parse Test")
        assert topic is not None
        assert topic.feed_urls == ["https://a.com/feed", "https://b.com/feed"]
        assert topic.feed_mode == FeedMode.MANUAL

    async def test_create_topic_auto_mode_default(self, client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
        """Topic created with auto mode has empty feed_urls."""
        with patch(
            "app.web.routers.background._run_init",
            new_callable=AsyncMock,
        ):
            await client.post(
                "/topics",
                data={
                    "name": "Auto Topic",
                    "description": "Test auto mode",
                    "feed_mode": "auto",
                    "feed_urls": "",
                },
                follow_redirects=False,
            )

        from app.crud import get_topic_by_name

        topic = get_topic_by_name(db_conn, "Auto Topic")
        assert topic is not None
        assert topic.feed_mode == FeedMode.AUTO
        assert topic.feed_urls == []

    async def test_create_topic_auto_mode_ignores_feed_urls(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """Auto mode ignores any feed_urls provided in the form."""
        with patch(
            "app.web.routers.background._run_init",
            new_callable=AsyncMock,
        ):
            response = await client.post(
                "/topics",
                data={
                    "name": "Auto Ignore URLs",
                    "description": "Test",
                    "feed_mode": "auto",
                    "feed_urls": "not-a-valid-url",
                },
                follow_redirects=False,
            )

        # Should succeed (not 422) because auto mode skips URL validation
        assert response.status_code == 303

    async def test_create_topic_empty_feed_urls(self, client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
        """Empty feed_urls textarea with auto mode results in empty list."""
        with patch(
            "app.web.routers.background._run_init",
            new_callable=AsyncMock,
        ):
            await client.post(
                "/topics",
                data={
                    "name": "No Feeds",
                    "description": "Test",
                    "feed_urls": "",
                },
                follow_redirects=False,
            )

        from app.crud import get_topic_by_name

        topic = get_topic_by_name(db_conn, "No Feeds")
        assert topic is not None
        assert topic.feed_urls == []

    async def test_create_topic_status_is_researching(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """Newly created topic starts in RESEARCHING status."""
        with patch(
            "app.web.routers.background._run_init",
            new_callable=AsyncMock,
        ):
            await client.post(
                "/topics",
                data={"name": "Status Test", "description": "Test", "feed_urls": ""},
                follow_redirects=False,
            )

        from app.crud import get_topic_by_name

        topic = get_topic_by_name(db_conn, "Status Test")
        assert topic.status == TopicStatus.RESEARCHING

    async def test_create_topic_kicks_off_init(self, client: httpx.AsyncClient) -> None:
        """POST /topics schedules the init background task."""
        with patch(
            "app.web.routers.background._run_init",
            new_callable=AsyncMock,
        ) as mock_init:
            await client.post(
                "/topics",
                data={"name": "Init Test", "description": "Test", "feed_urls": ""},
                follow_redirects=False,
            )

        mock_init.assert_called_once()

    async def test_create_topic_rejects_invalid_urls(self, client: httpx.AsyncClient) -> None:
        """Invalid feed URLs are rejected with 422 and error messages."""
        response = await client.post(
            "/topics",
            data={
                "name": "Bad URL Topic",
                "description": "Test",
                "feed_mode": "manual",
                "feed_urls": "not-a-url\nhttps://valid.com/feed.xml",
            },
            follow_redirects=False,
        )
        assert response.status_code == 422
        assert "Invalid feed URL" in response.text
        assert "not-a-url" in response.text
        # Form values should be preserved
        assert "Bad URL Topic" in response.text

    async def test_create_topic_accepts_valid_urls(self, client: httpx.AsyncClient) -> None:
        """Valid http/https URLs pass validation."""
        with patch(
            "app.web.routers.background._run_init",
            new_callable=AsyncMock,
        ):
            response = await client.post(
                "/topics",
                data={
                    "name": "Good URL Topic",
                    "description": "Test",
                    "feed_mode": "manual",
                    "feed_urls": "https://example.com/feed.xml\nhttp://other.com/rss",
                },
                follow_redirects=False,
            )
        assert response.status_code == 303

    async def test_create_topic_persists_thresholds(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """Per-topic confidence/relevance thresholds are persisted on create."""
        with patch("app.web.routers.background._run_init", new_callable=AsyncMock):
            await client.post(
                "/topics",
                data={
                    "name": "Thresholded",
                    "description": "Test",
                    "feed_urls": "",
                    "confidence_threshold": "0.9",
                    "relevance_threshold": "0.5",
                },
                follow_redirects=False,
            )

        from app.crud import get_topic_by_name

        topic = get_topic_by_name(db_conn, "Thresholded")
        assert topic is not None
        assert topic.confidence_threshold == 0.9
        assert topic.relevance_threshold == 0.5

    async def test_create_topic_blank_thresholds_inherit(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """Blank threshold inputs store NULL (inherit global)."""
        with patch("app.web.routers.background._run_init", new_callable=AsyncMock):
            await client.post(
                "/topics",
                data={"name": "NoThresh", "description": "Test", "feed_urls": ""},
                follow_redirects=False,
            )

        from app.crud import get_topic_by_name

        topic = get_topic_by_name(db_conn, "NoThresh")
        assert topic is not None
        assert topic.confidence_threshold is None
        assert topic.relevance_threshold is None

    async def test_create_topic_rejects_out_of_range_threshold(self, client: httpx.AsyncClient) -> None:
        """Out-of-range threshold values return 422."""
        response = await client.post(
            "/topics",
            data={
                "name": "BadThresh",
                "description": "Test",
                "feed_urls": "",
                "confidence_threshold": "1.5",
            },
            follow_redirects=False,
        )
        assert response.status_code == 422
        assert "between 0.0 and 1.0" in response.text


# --- Topic Detail ---


class TestTopicDetail:
    """Tests for GET /topics/{id}."""

    async def test_detail_page_renders(self, client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
        """Detail page shows topic name and description."""
        topic = _make_topic(db_conn)
        response = await client.get(f"/topics/{topic.id}")
        assert response.status_code == 200
        assert topic.name in response.text

    async def test_detail_shows_auto_feed_url(self, client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
        """Detail page for auto mode shows the generated Google News URL."""
        topic = _make_topic(db_conn, name="Auto Detail", feed_mode=FeedMode.AUTO, feed_urls=[])
        response = await client.get(f"/topics/{topic.id}")
        assert response.status_code == 200
        assert "Automatic" in response.text
        assert "news.google.com" in response.text

    async def test_detail_shows_manual_feed_urls(self, client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
        """Detail page for manual mode shows the configured feed URLs."""
        topic = _make_topic(db_conn, name="Manual Detail")
        response = await client.get(f"/topics/{topic.id}")
        assert response.status_code == 200
        assert "Manual" in response.text
        assert "example.com/feed.xml" in response.text

    async def test_detail_shows_knowledge_state(self, client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
        """Detail page shows the knowledge state summary."""
        topic = _make_topic(db_conn)
        create_knowledge_state(
            db_conn,
            KnowledgeState(
                topic_id=topic.id,
                summary_text="This is the knowledge summary.",
                token_count=50,
            ),
        )
        db_conn.commit()

        response = await client.get(f"/topics/{topic.id}")
        assert "This is the knowledge summary." in response.text

    async def test_detail_shows_check_history(self, client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
        """Detail page shows recent check results."""
        topic = _make_topic(db_conn)
        create_check_result(
            db_conn,
            CheckResult(topic_id=topic.id, articles_found=42, has_new_info=True),
        )
        db_conn.commit()

        response = await client.get(f"/topics/{topic.id}")
        assert response.status_code == 200
        assert "42" in response.text

    async def test_detail_404_for_nonexistent(self, client: httpx.AsyncClient) -> None:
        """Requesting a nonexistent topic renders the HTML error page with a 404."""
        response = await client.get("/topics/9999", headers={"accept": "text/html"})
        assert response.status_code == 404
        assert "text/html" in response.headers["content-type"]
        assert "404" in response.text
        assert "Back to Dashboard" in response.text

    async def test_detail_researching_shows_polling(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """RESEARCHING status shows HTMX polling attribute."""
        topic = _make_topic(db_conn, status=TopicStatus.RESEARCHING)
        response = await client.get(f"/topics/{topic.id}")
        assert "hx-get" in response.text
        assert "every 3s" in response.text

    async def test_detail_error_shows_retry_button(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """ERROR status shows error message and retry button."""
        topic = _make_topic(
            db_conn,
            status=TopicStatus.ERROR,
            error_message="LLM failed",
        )
        response = await client.get(f"/topics/{topic.id}")
        assert "LLM failed" in response.text
        assert "Retry Research" in response.text


# --- Topic Status (HTMX partial) ---


class TestTopicStatus:
    """Tests for GET /topics/{id}/status (HTMX partial)."""

    async def test_status_researching_includes_polling(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """RESEARCHING status fragment includes hx-trigger for polling."""
        topic = _make_topic(db_conn, status=TopicStatus.RESEARCHING)
        response = await client.get(f"/topics/{topic.id}/status")
        assert response.status_code == 200
        assert "hx-trigger" in response.text
        assert "every 3s" in response.text

    async def test_status_ready_shows_knowledge(self, client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
        """READY status fragment shows knowledge state without polling."""
        topic = _make_topic(db_conn, status=TopicStatus.READY)
        create_knowledge_state(
            db_conn,
            KnowledgeState(topic_id=topic.id, summary_text="Summary here.", token_count=20),
        )
        db_conn.commit()

        response = await client.get(f"/topics/{topic.id}/status")
        assert "Summary here." in response.text
        assert "hx-trigger" not in response.text

    async def test_status_error_shows_retry(self, client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
        """ERROR status fragment shows error and retry button."""
        topic = _make_topic(db_conn, status=TopicStatus.ERROR, error_message="Init failed")
        response = await client.get(f"/topics/{topic.id}/status")
        assert "Init failed" in response.text
        assert "Retry" in response.text
        assert "hx-trigger" not in response.text

    async def test_status_404(self, client: httpx.AsyncClient) -> None:
        """Nonexistent topic returns 404."""
        response = await client.get("/topics/9999/status")
        assert response.status_code == 404


# --- Re-init ---


class TestReinitTopic:
    """Tests for POST /topics/{id}/init."""

    async def test_reinit_resets_status(self, client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
        """Re-init sets status to RESEARCHING and clears error."""
        topic = _make_topic(
            db_conn,
            status=TopicStatus.ERROR,
            error_message="Previous failure",
        )

        with patch("app.web.routers.background._run_init", new_callable=AsyncMock):
            response = await client.post(f"/topics/{topic.id}/init", follow_redirects=False)

        assert response.status_code == 303

        from app.crud import get_topic

        updated = get_topic(db_conn, topic.id)
        assert updated.status == TopicStatus.RESEARCHING
        assert updated.error_message is None

    async def test_reinit_schedules_background_task(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """Re-init schedules the init background task."""
        topic = _make_topic(db_conn, status=TopicStatus.ERROR)

        with patch("app.web.routers.background._run_init", new_callable=AsyncMock) as mock_init:
            await client.post(f"/topics/{topic.id}/init", follow_redirects=False)

        mock_init.assert_called_once()

    async def test_reinit_404(self, client: httpx.AsyncClient) -> None:
        """Re-init for nonexistent topic returns 404."""
        response = await client.post("/topics/9999/init", follow_redirects=False)
        assert response.status_code == 404


# --- Check Now ---


class TestCheckNow:
    """Tests for POST /topics/{id}/check."""

    async def test_check_defers_to_background_not_inline(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """OVH-013: /check enqueues the background task; the pipeline does not run inline."""
        topic = _make_topic(db_conn)

        with (
            patch("app.checker.check_topic", new_callable=AsyncMock) as mock_inline,
            patch("app.web.routers.background._run_single_check", new_callable=AsyncMock) as mock_bg,
        ):
            response = await client.post(
                f"/topics/{topic.id}/check",
                headers={"HX-Request": "true"},
            )

        assert response.status_code == 200
        # Pipeline must be deferred to the background task, never run inline.
        mock_inline.assert_not_called()
        mock_bg.assert_called_once()
        assert mock_bg.call_args[0][0] == topic.id

    async def test_check_htmx_returns_row_partial(self, client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
        """OVH-005: HTMX /check returns the topic-row partial."""
        topic = _make_topic(db_conn)

        with patch("app.web.routers.background._run_single_check", new_callable=AsyncMock):
            response = await client.post(
                f"/topics/{topic.id}/check",
                headers={"HX-Request": "true"},
                follow_redirects=False,
            )

        assert response.status_code == 200
        assert f'id="topic-{topic.id}"' in response.text

    async def test_check_non_htmx_redirects(self, client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
        """OVH-005: a plain-form /check redirects to the topic detail page (no orphan <tr>)."""
        topic = _make_topic(db_conn)

        with patch("app.web.routers.background._run_single_check", new_callable=AsyncMock):
            response = await client.post(
                f"/topics/{topic.id}/check",
                follow_redirects=False,
            )

        assert response.status_code == 303
        assert response.headers["location"] == f"/topics/{topic.id}"
        # Must not return a bare table row to a full-page navigation.
        assert "<tr" not in response.text

    async def test_check_already_checking_htmx_returns_partial(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """OVH-005: the already-checking early return honors HX-Request (partial)."""
        topic = _make_topic(db_conn)
        from app.web.state import _checking_state

        await _checking_state.start_check(topic.id)
        try:
            with patch("app.web.routers.background._run_single_check", new_callable=AsyncMock) as mock_bg:
                response = await client.post(
                    f"/topics/{topic.id}/check",
                    headers={"HX-Request": "true"},
                    follow_redirects=False,
                )
            assert response.status_code == 200
            assert f'id="topic-{topic.id}"' in response.text
            # Already checking — do not enqueue a second pipeline run.
            mock_bg.assert_not_called()
        finally:
            await _checking_state.finish_check(topic.id)

    async def test_check_already_checking_non_htmx_redirects(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """OVH-005: the already-checking early return redirects for a plain form."""
        topic = _make_topic(db_conn)
        from app.web.state import _checking_state

        await _checking_state.start_check(topic.id)
        try:
            response = await client.post(
                f"/topics/{topic.id}/check",
                follow_redirects=False,
            )
            assert response.status_code == 303
            assert response.headers["location"] == f"/topics/{topic.id}"
        finally:
            await _checking_state.finish_check(topic.id)

    async def test_check_counts_articles_with_count_query(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """OVH-138: the article count comes from COUNT(*), not by hydrating every row."""
        topic = _make_topic(db_conn)

        with (
            patch("app.web.routers.background._run_single_check", new_callable=AsyncMock),
            patch("app.web.routers.topics.count_articles_for_topic", return_value=7) as mock_count,
            patch("app.web.routers.topics.list_articles_for_topic") as mock_list,
        ):
            response = await client.post(
                f"/topics/{topic.id}/check",
                headers={"HX-Request": "true"},
                follow_redirects=False,
            )

        assert response.status_code == 200
        mock_count.assert_called_once()
        mock_list.assert_not_called()
        assert ">7</td>" in response.text

    async def test_check_404(self, client: httpx.AsyncClient) -> None:
        """Check for nonexistent topic returns 404."""
        response = await client.post("/topics/9999/check")
        assert response.status_code == 404


# --- Delete ---


class TestDeleteTopic:
    """Tests for POST /topics/{id}/delete."""

    async def test_delete_redirects_to_dashboard(self, client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
        """Delete topic redirects to dashboard."""
        topic = _make_topic(db_conn)
        response = await client.post(f"/topics/{topic.id}/delete", follow_redirects=False)
        assert response.status_code == 303
        assert response.headers["location"] == "/"

    async def test_delete_removes_topic(self, client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
        """Delete actually removes the topic from the database."""
        topic = _make_topic(db_conn)
        await client.post(f"/topics/{topic.id}/delete", follow_redirects=False)

        from app.crud import get_topic

        assert get_topic(db_conn, topic.id) is None


# --- Settings ---


class TestSettings:
    """Tests for GET /settings."""

    async def test_settings_renders(self, client: httpx.AsyncClient) -> None:
        """Settings page loads and shows configuration."""
        response = await client.get("/settings")
        assert response.status_code == 200
        assert "openai/gpt-4o-mini" in response.text

    async def test_settings_masks_api_key(self, client: httpx.AsyncClient) -> None:
        """Settings page masks the API key."""
        response = await client.get("/settings")
        assert response.status_code == 200
        # Full key should NOT be visible
        assert "test-key-12345678" not in response.text
        # The masked format should be shown (first 4 chars...last 4 chars)
        assert "test...5678" in response.text


# --- CSRF Protection ---


class TestCSRFProtection:
    """Tests for CSRF token validation on POST routes."""

    async def test_post_without_csrf_returns_403(self, db_conn: sqlite3.Connection) -> None:
        """POST without CSRF token is rejected with 403."""
        settings = _make_settings()

        def override_db():
            yield db_conn

        def override_settings():
            return settings

        app.dependency_overrides[get_db_conn] = override_db
        app.dependency_overrides[get_settings] = override_settings

        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
            ) as ac:
                response = await ac.post(
                    "/topics",
                    data={
                        "name": "No CSRF",
                        "description": "Should fail",
                        "feed_urls": "",
                    },
                    follow_redirects=False,
                )
            assert response.status_code == 403
        finally:
            app.dependency_overrides.clear()

    async def test_post_with_mismatched_csrf_returns_403(self, db_conn: sqlite3.Connection) -> None:
        """POST with a CSRF token that doesn't match the cookie is rejected."""
        settings = _make_settings()

        def override_db():
            yield db_conn

        def override_settings():
            return settings

        app.dependency_overrides[get_db_conn] = override_db
        app.dependency_overrides[get_settings] = override_settings

        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
                cookies={"csrf_token": "real-token"},
                headers={"X-CSRF-Token": "wrong-token"},
            ) as ac:
                response = await ac.post(
                    "/topics",
                    data={
                        "name": "Bad CSRF",
                        "description": "Should fail",
                        "feed_urls": "",
                    },
                    follow_redirects=False,
                )
            assert response.status_code == 403
        finally:
            app.dependency_overrides.clear()

    async def test_csrf_cookie_set_on_first_get(self, db_conn: sqlite3.Connection) -> None:
        """First GET request sets a CSRF cookie."""
        settings = _make_settings()

        def override_db():
            yield db_conn

        def override_settings():
            return settings

        app.dependency_overrides[get_db_conn] = override_db
        app.dependency_overrides[get_settings] = override_settings

        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
            ) as ac:
                response = await ac.get("/")
            assert "csrf_token" in response.cookies
        finally:
            app.dependency_overrides.clear()

    async def test_csrf_form_field_validation(self, db_conn: sqlite3.Connection) -> None:
        """POST with CSRF token only in form field (no header) succeeds."""
        settings = _make_settings()

        def override_db():
            yield db_conn

        def override_settings():
            return settings

        app.dependency_overrides[get_db_conn] = override_db
        app.dependency_overrides[get_settings] = override_settings

        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
                cookies={"csrf_token": CSRF_TEST_TOKEN},
            ) as ac:
                with patch("app.web.routers.background._run_init", new_callable=AsyncMock):
                    response = await ac.post(
                        "/topics",
                        data={
                            "name": "Form CSRF Test",
                            "description": "Test",
                            "feed_urls": "",
                            "csrf_token": CSRF_TEST_TOKEN,
                        },
                        follow_redirects=False,
                    )
            assert response.status_code == 303
        finally:
            app.dependency_overrides.clear()


# --- Health Check ---


class TestHealthCheck:
    """Tests for GET /health."""

    async def test_health_returns_ok(self, client: httpx.AsyncClient) -> None:
        """Health endpoint returns status ok."""
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "topics" in data

    async def test_health_counts_topics(self, client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
        """Health endpoint reports correct topic count."""
        _make_topic(db_conn, name="T1")
        _make_topic(db_conn, name="T2")
        response = await client.get("/health")
        assert response.json()["topics"] == 2


# --- Timeago Filter ---


class TestTimeagoFilter:
    """Tests for the timeago Jinja2 filter."""

    def test_just_now(self) -> None:
        from app.web.routers.templates import _timeago

        now = datetime.now(UTC)
        assert _timeago(now) == "just now"

    def test_minutes_ago(self) -> None:
        from datetime import timedelta

        from app.web.routers.templates import _timeago

        dt = datetime.now(UTC) - timedelta(minutes=5)
        assert _timeago(dt) == "5m ago"

    def test_hours_ago(self) -> None:
        from datetime import timedelta

        from app.web.routers.templates import _timeago

        dt = datetime.now(UTC) - timedelta(hours=3)
        assert _timeago(dt) == "3h ago"

    def test_days_ago(self) -> None:
        from datetime import timedelta

        from app.web.routers.templates import _timeago

        dt = datetime.now(UTC) - timedelta(days=5)
        assert _timeago(dt) == "5d ago"

    def test_over_30_days_shows_date(self) -> None:
        from datetime import timedelta

        from app.web.routers.templates import _timeago

        dt = datetime.now(UTC) - timedelta(days=45)
        result = _timeago(dt)
        assert "-" in result
        assert "ago" not in result

    def test_naive_datetime(self) -> None:
        from app.web.routers.templates import _timeago

        dt = datetime.now(UTC).replace(tzinfo=None)
        result = _timeago(dt)
        assert isinstance(result, str)


# --- SSRF URL Validation ---


class TestSSRFProtection:
    """Tests for SSRF protection in feed URL validation."""

    async def test_rejects_localhost(self, client: httpx.AsyncClient) -> None:
        """Feed URL pointing to localhost is rejected."""
        response = await client.post(
            "/topics",
            data={
                "name": "SSRF Test",
                "description": "Test",
                "feed_mode": "manual",
                "feed_urls": "http://localhost/feed.xml",
            },
            follow_redirects=False,
        )
        assert response.status_code == 422
        assert "private" in response.text.lower()

    async def test_rejects_127(self, client: httpx.AsyncClient) -> None:
        """Feed URL pointing to 127.0.0.1 is rejected."""
        response = await client.post(
            "/topics",
            data={
                "name": "SSRF Test 2",
                "description": "Test",
                "feed_mode": "manual",
                "feed_urls": "http://127.0.0.1/feed.xml",
            },
            follow_redirects=False,
        )
        assert response.status_code == 422

    async def test_rejects_private_10(self, client: httpx.AsyncClient) -> None:
        """Feed URL pointing to 10.x.x.x is rejected."""
        response = await client.post(
            "/topics",
            data={
                "name": "SSRF Test 3",
                "description": "Test",
                "feed_mode": "manual",
                "feed_urls": "http://10.0.0.1/feed.xml",
            },
            follow_redirects=False,
        )
        assert response.status_code == 422

    async def test_rejects_private_192(self, client: httpx.AsyncClient) -> None:
        """Feed URL pointing to 192.168.x.x is rejected."""
        response = await client.post(
            "/topics",
            data={
                "name": "SSRF Test 4",
                "description": "Test",
                "feed_mode": "manual",
                "feed_urls": "http://192.168.1.1/feed.xml",
            },
            follow_redirects=False,
        )
        assert response.status_code == 422


# --- Topic Editing ---


class TestTopicEdit:
    """Tests for GET/POST /topics/{id}/edit."""

    async def test_edit_form_renders(self, client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
        """Edit form shows current topic values."""
        topic = _make_topic(db_conn, name="Editable Topic")
        response = await client.get(f"/topics/{topic.id}/edit")
        assert response.status_code == 200
        assert "Editable Topic" in response.text
        assert "<form" in response.text

    async def test_edit_form_404(self, client: httpx.AsyncClient) -> None:
        """Edit form for nonexistent topic returns 404."""
        response = await client.get("/topics/9999/edit")
        assert response.status_code == 404

    async def test_edit_updates_topic(self, client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
        """POST to edit updates the topic's fields."""
        topic = _make_topic(db_conn, name="Old Name")
        response = await client.post(
            f"/topics/{topic.id}/edit",
            data={
                "name": "New Name",
                "description": "New description",
                "feed_mode": "manual",
                "feed_urls": "https://new.example.com/feed.xml",
            },
            follow_redirects=False,
        )
        assert response.status_code == 303

        from app.crud import get_topic

        updated = get_topic(db_conn, topic.id)
        assert updated.name == "New Name"
        assert updated.description == "New description"
        assert updated.feed_urls == ["https://new.example.com/feed.xml"]
        assert updated.feed_mode == FeedMode.MANUAL

    async def test_edit_switch_to_auto_mode(self, client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
        """Editing a topic can switch feed_mode from manual to auto."""
        topic = _make_topic(db_conn, name="Switch Mode")
        response = await client.post(
            f"/topics/{topic.id}/edit",
            data={
                "name": "Switch Mode",
                "description": topic.description,
                "feed_mode": "auto",
                "feed_urls": "",
            },
            follow_redirects=False,
        )
        assert response.status_code == 303

        from app.crud import get_topic

        updated = get_topic(db_conn, topic.id)
        assert updated.feed_mode == FeedMode.AUTO
        assert updated.feed_urls == []

    async def test_edit_validates_urls(self, client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
        """Edit rejects invalid feed URLs in manual mode."""
        topic = _make_topic(db_conn)
        response = await client.post(
            f"/topics/{topic.id}/edit",
            data={
                "name": "Test",
                "description": "Test",
                "feed_mode": "manual",
                "feed_urls": "not-a-url",
            },
            follow_redirects=False,
        )
        assert response.status_code == 422

    async def test_edit_404(self, client: httpx.AsyncClient) -> None:
        """Edit for nonexistent topic returns 404."""
        response = await client.post(
            "/topics/9999/edit",
            data={"name": "X", "description": "X", "feed_urls": ""},
            follow_redirects=False,
        )
        assert response.status_code == 404


# --- Check All ---


class TestCheckAll:
    """Tests for POST /check-all."""

    async def test_check_all_redirects(self, client: httpx.AsyncClient) -> None:
        """Check all returns redirect to dashboard."""
        with patch("app.web.routers.background._run_check_all", new_callable=AsyncMock):
            response = await client.post("/check-all", follow_redirects=False)
        assert response.status_code == 303
        assert response.headers["location"] == "/"
