"""Tests for the web UI: routes, templates, and HTMX interactions."""

import json
import logging
import re
import sqlite3
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.analysis.llm import NoveltyResult
from app.config import ExaSettings, LLMSettings, NotificationSettings, Settings
from app.crud import (
    create_article,
    create_check_result,
    create_knowledge_state,
    create_topic,
    get_topic_by_name,
    list_article_headers_for_topic,
)
from app.main import REQUEST_ID_PATTERN, app
from app.models import (
    Article,
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


# Scope assertions to the dashboard's "Active" stat card. The bare ``num`` markup is
# shared by the Checks/New-info cards, so match the card by its "Active" kicker and read
# both its active count and the "of N topics" total together (dashboard.html:31-33).
_ACTIVE_CARD_RE = re.compile(
    r'card-kicker">Active</span>\s*'
    r'<span class="stat-value"><span class="num">(\d+)</span></span>\s*'
    r'<span class="stat-sub">of (\d+) topics'
)


def _active_card_counts(html: str) -> tuple[int, int]:
    """Return (active_topics, total_topics) rendered in the dashboard Active card."""
    match = _ACTIVE_CARD_RE.search(html)
    assert match is not None, "Active stat card not found in dashboard HTML"
    return int(match.group(1)), int(match.group(2))


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


# --- Request correlation id (OVH-043) ---


class TestRequestId:
    """OVH-043: every request carries an X-Request-ID echoed back and visible to logs."""

    async def test_response_echoes_generated_request_id(self, client: httpx.AsyncClient) -> None:
        """A request with no X-Request-ID gets one generated and echoed in the response."""
        response = await client.get("/")
        rid = response.headers.get("X-Request-ID")
        assert rid, "Response should carry an X-Request-ID header"
        assert rid != "-"
        assert len(rid) >= 8

    async def test_response_echoes_inbound_request_id(self, client: httpx.AsyncClient) -> None:
        """A client-supplied X-Request-ID is preserved and echoed back."""
        response = await client.get("/", headers={"X-Request-ID": "client-supplied-123"})
        assert response.headers.get("X-Request-ID") == "client-supplied-123"

    async def test_oversized_inbound_id_is_replaced(self, client: httpx.AsyncClient) -> None:
        """AUG-331: an unbounded client id is neither trusted nor echoed."""
        oversized = "a" * 4096
        response = await client.get("/", headers={"X-Request-ID": oversized})
        echoed = response.headers["X-Request-ID"]
        assert echoed != oversized
        assert len(echoed) <= 128

    @pytest.mark.parametrize("hostile", ["with space", "tab\there", "semi;colon", 'quote"here', ""])
    async def test_ids_outside_the_grammar_are_replaced(self, client: httpx.AsyncClient, hostile: str) -> None:
        response = await client.get("/", headers={"X-Request-ID": hostile})
        assert response.headers["X-Request-ID"] != hostile

    @pytest.mark.parametrize("hostile", [b"\x9bcsi-here", b"\xc2\xadsoft-hyphen", b"\x7fdel"])
    async def test_non_ascii_bytes_are_replaced(self, client: httpx.AsyncClient, hostile: bytes) -> None:
        """Bytes the plain formatter would emit raw never become the correlation id."""
        response = await client.get("/", headers={"X-Request-ID": hostile})
        echoed = response.headers["X-Request-ID"]
        assert echoed != hostile.decode("latin-1")
        assert REQUEST_ID_PATTERN.match(echoed)

    async def test_max_length_id_is_still_accepted(self, client: httpx.AsyncClient) -> None:
        at_limit = "b" * 128
        response = await client.get("/", headers={"X-Request-ID": at_limit})
        assert response.headers["X-Request-ID"] == at_limit

    @pytest.mark.parametrize("proxy_id", ["9c1b2f4e8a", "trace-1-5759e988-bd862e3f", "req.42_ok"])
    async def test_ordinary_proxy_ids_are_preserved(self, client: httpx.AsyncClient, proxy_id: str) -> None:
        response = await client.get("/", headers={"X-Request-ID": proxy_id})
        assert response.headers["X-Request-ID"] == proxy_id

    async def test_request_id_set_in_context_during_request(self) -> None:
        """While a request is in flight, request_id_var (and thus logs) carries the inbound id."""
        from app.check_context import CheckIdFilter, request_id_var
        from app.main import RequestIdMiddleware

        seen: dict[str, str] = {}

        async def inner_app(scope, receive, send):
            # Resolve through the filter exactly as a log record would mid-request.
            record = logging.makeLogRecord({})
            CheckIdFilter().filter(record)
            seen["ctx"] = request_id_var.get()
            seen["filter"] = record.check_id
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        middleware = RequestIdMiddleware(inner_app)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=middleware), base_url="http://test") as ac:
            response = await ac.get("/", headers={"X-Request-ID": "trace-me-42"})

        assert seen["ctx"] == "trace-me-42"
        assert seen["filter"] == "trace-me-42", "CheckIdFilter must surface the request id"
        assert response.headers.get("X-Request-ID") == "trace-me-42"

    async def test_request_id_cleared_after_request(self) -> None:
        """The contextvar is reset after the request so ids do not leak across requests."""
        from app.check_context import request_id_var
        from app.main import RequestIdMiddleware

        async def inner_app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        middleware = RequestIdMiddleware(inner_app)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=middleware), base_url="http://test") as ac:
            await ac.get("/", headers={"X-Request-ID": "leaky-1"})

        assert request_id_var.get() is None


# --- Dashboard ---


class TestDashboard:
    """Tests for GET / (dashboard)."""

    async def test_dashboard_empty(self, client: httpx.AsyncClient) -> None:
        """Empty database shows 'no topics' message."""
        response = await client.get("/")
        assert response.status_code == 200
        assert "Add your first topic" in response.text

    async def test_failing_sources_badge(self, client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
        """A topic whose newest check saw no usable source is badged on the dashboard."""
        topic = _make_topic(db_conn, name="HBWeb")
        create_check_result(
            db_conn,
            CheckResult(topic_id=topic.id, stage_error="sources_failed: all feed source(s) failed (see logs)"),
        )
        db_conn.commit()

        page = await client.get("/")
        assert "Sources failing" in page.text

    async def test_no_badge_for_a_healthy_quiet_topic(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        topic = _make_topic(db_conn, name="HBOk")
        create_check_result(db_conn, CheckResult(topic_id=topic.id))
        db_conn.commit()

        page = await client.get("/")
        assert "Sources failing" not in page.text

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

    async def test_dashboard_notification_tag_is_per_topic_row(self, client: httpx.AsyncClient) -> None:
        """AUG-219: the fresh-check browser notification is tagged with the row id.

        A fixed shared tag let one topic's alert silently replace another's; the tag
        must now be scoped to the row so distinct topics never collide.
        """
        response = await client.get("/")
        assert "tag: detailTarget.id" in response.text
        assert 'tag: "topic-watch"' not in response.text

    async def test_dashboard_shows_msg_banner(self, client: httpx.AsyncClient) -> None:
        """AUG-197: a successful/partial OPML import summary (redirected to
        /?msg=...) must actually render — the dashboard used to accept and
        render only ?error=, discarding every non-error import outcome."""
        response = await client.get("/?msg=Imported+3+topic(s).")
        assert response.status_code == 200
        assert "Imported 3 topic(s)." in response.text

    async def test_search_trigger_reacts_to_paste(self, client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
        """AUG-226: the search box listens for `input` (fires on a context-menu
        paste), not just `keyup`/`search`, which paste does not trigger."""
        _make_topic(db_conn)

        response = await client.get("/")
        assert "input changed delay:300ms, search" in response.text

    async def test_dashboard_opml_file_input_has_accessible_name(self, client: httpx.AsyncClient) -> None:
        """AUG-238: the OPML file picker has a programmatically associated label."""
        response = await client.get("/")
        assert '<label for="opml_file" class="sr-only">OPML File</label>' in response.text
        assert 'id="opml_file"' in response.text
        assert 'aria-describedby="opml-file-hint"' in response.text
        assert 'id="opml-file-hint"' in response.text


class TestOpmlImportErrorRedirect:
    """AUG-206: a rejected feed URL must not travel in the redirect Location."""

    _OPML_WITH_SECRET_FEED = (
        '<?xml version="1.0"?><opml version="2.0"><body>'
        '<outline text="Leaky" type="rss" xmlUrl="https://user:s3cr3t@feeds.example.com/x?token=abc123"/>'
        "</body></opml>"
    )

    async def test_a_valid_upload_is_actually_read(self, client: httpx.AsyncClient) -> None:
        """The uploaded file reaches parse_opml instead of being read as absent.

        request.form() yields starlette.datastructures.UploadFile; the route used
        to isinstance-check against fastapi.UploadFile, a subclass, so every real
        upload fell through to "No file selected".
        """
        from urllib.parse import unquote

        from app.opml import OPMLResult

        parsed = OPMLResult()
        parsed.topics.append({"name": "Imported", "feed_urls": ["https://feeds.example.com/ok"], "tags": []})

        with patch("app.opml.parse_opml", return_value=parsed) as mock_parse:
            response = await client.post(
                "/import/opml",
                files={"opml_file": ("feeds.opml", self._OPML_WITH_SECRET_FEED, "text/xml")},
                follow_redirects=False,
            )

        assert mock_parse.call_count == 1
        assert "Imported 1 topic(s)" in unquote(response.headers["location"])

    async def test_rejected_url_never_reaches_the_location_header(self, client: httpx.AsyncClient) -> None:
        from app.opml import OPMLResult

        rejected = OPMLResult()
        rejected.skipped_invalid = 1
        rejected.warnings.append("Invalid feed URL: https://user:s3cr3t@feeds.example.com/x?token=abc123 (blocked)")

        with patch("app.opml.parse_opml", return_value=rejected):
            response = await client.post(
                "/import/opml",
                files={"opml_file": ("feeds.opml", self._OPML_WITH_SECRET_FEED, "text/xml")},
                follow_redirects=False,
            )

        assert response.status_code == 303
        location = response.headers["location"]
        assert "s3cr3t" not in location
        assert "token=abc123" not in location
        assert "feeds.example.com" not in location

    async def test_imported_topics_inherit_the_global_cadence(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """TW-AUD-025: OPML has no interval, so the topic must not get an override."""
        from app.crud import get_topic_by_name
        from app.opml import OPMLResult

        parsed = OPMLResult()
        parsed.topics.append({"name": "Inheriting", "feed_urls": ["https://feeds.example.com/ok"], "tags": []})

        with patch("app.opml.parse_opml", return_value=parsed):
            await client.post(
                "/import/opml",
                files={"opml_file": ("feeds.opml", self._OPML_WITH_SECRET_FEED, "text/xml")},
                follow_redirects=False,
            )

        topic = get_topic_by_name(db_conn, "Inheriting")
        assert topic is not None
        assert topic.check_interval_minutes is None

    async def test_redirect_still_says_how_many_were_rejected(self, client: httpx.AsyncClient) -> None:
        from urllib.parse import unquote

        from app.opml import OPMLResult

        rejected = OPMLResult()
        rejected.skipped_invalid = 2
        rejected.warnings.append("Invalid feed URL: https://feeds.example.com/x (blocked)")

        with patch("app.opml.parse_opml", return_value=rejected):
            response = await client.post(
                "/import/opml",
                files={"opml_file": ("feeds.opml", self._OPML_WITH_SECRET_FEED, "text/xml")},
                follow_redirects=False,
            )

        assert "2 feed URL(s) rejected" in unquote(response.headers["location"])


class TestOpmlImportRechecksAfterValidation:
    """AUG-287: admission is decided against live state, after the DNS work."""

    _OPML = (
        '<?xml version="1.0"?><opml version="2.0"><body>'
        '<outline text="Dup" type="rss" xmlUrl="https://feeds.example.com/dup"/>'
        "</body></opml>"
    )

    async def test_name_taken_during_validation_is_skipped_not_a_500(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """A concurrent import that wins the name loses this batch, not the request.

        The snapshot of existing names is taken before the DNS-validating parse, so
        an overlapping import of the same file passed admission twice; the loser hit
        the UNIQUE constraint uncaught and rolled its whole batch back.
        """
        from urllib.parse import unquote

        from app.opml import OPMLResult

        parsed = OPMLResult()
        parsed.topics.append({"name": "Dup", "feed_urls": ["https://feeds.example.com/dup"], "tags": []})
        parsed.topics.append({"name": "Fresh", "feed_urls": ["https://feeds.example.com/fresh"], "tags": []})

        def _parse_and_race(*args, **kwargs):
            # The competing import commits while this one is inside parse_opml.
            create_topic(db_conn, Topic(name="Dup", description="d", status=TopicStatus.NEW))
            db_conn.commit()
            return parsed

        with patch("app.opml.parse_opml", new=_parse_and_race):
            response = await client.post(
                "/import/opml",
                files={"opml_file": ("feeds.opml", self._OPML, "text/xml")},
                follow_redirects=False,
            )

        assert response.status_code == 303
        location = unquote(response.headers["location"])
        assert "Imported 1 topic(s)" in location
        assert "skipped 1 duplicate(s)" in location
        # The topic that was still unique survived the batch.
        assert get_topic_by_name(db_conn, "Fresh") is not None

    async def test_feed_url_taken_during_validation_is_skipped(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """A URL that became a duplicate during validation does not import twice."""
        from app.opml import OPMLResult

        parsed = OPMLResult()
        parsed.topics.append({"name": "Second Name", "feed_urls": ["https://feeds.example.com/dup"], "tags": []})

        def _parse_and_race(*args, **kwargs):
            create_topic(
                db_conn,
                Topic(
                    name="First Name",
                    description="d",
                    status=TopicStatus.NEW,
                    feed_urls=["https://feeds.example.com/dup"],
                ),
            )
            db_conn.commit()
            return parsed

        with patch("app.opml.parse_opml", new=_parse_and_race):
            response = await client.post(
                "/import/opml",
                files={"opml_file": ("feeds.opml", self._OPML, "text/xml")},
                follow_redirects=False,
            )

        assert response.status_code == 303
        assert get_topic_by_name(db_conn, "Second Name") is None


class TestDashboardStatsFreshness:
    """The Active/Total stat cards must reflect mutations on the very next load.

    Regression guard for the stale-dashboard-stats bug: the stats were served from a
    60s TTL cache that no mutation invalidated, so the count lagged while the topic
    list (queried fresh) already updated. Each test issues a second ``GET /`` after a
    mutation in the SAME request session; if a stale, uninvalidated cache is ever
    reintroduced, the second read returns the old counts and these fail.
    """

    async def test_active_count_drops_after_delete(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """Deleting a topic decrements both active and total on the next dashboard load."""
        topics = [_make_topic(db_conn, name=f"Topic {i}") for i in range(3)]

        before = await client.get("/")
        assert _active_card_counts(before.text) == (3, 3)

        deleted = await client.post(f"/topics/{topics[0].id}/delete", follow_redirects=False)
        assert deleted.status_code == 303  # assert on the redirect, not its body

        after = await client.get("/")  # a fresh load, not the redirect target
        assert _active_card_counts(after.text) == (2, 2)

    async def test_active_count_drops_after_toggle_inactive(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """Toggling a topic inactive decrements active but leaves total unchanged."""
        topics = [_make_topic(db_conn, name=f"Topic {i}") for i in range(2)]

        before = await client.get("/")
        assert _active_card_counts(before.text) == (2, 2)

        # Plain POST (no HX-Request header) -> 303 redirect; an HX request would return
        # the _topic_row.html partial instead of the dashboard.
        toggled = await client.post(
            f"/topics/{topics[0].id}/toggle-active", data={"active": "false"}, follow_redirects=False
        )
        assert toggled.status_code == 303

        after = await client.get("/")
        assert _active_card_counts(after.text) == (1, 2)


class TestNewInfoSeenBadge:
    """The 'Ready · new info' badge (``badge--signal``) clears once the topic is
    acknowledged (TW-AUD-024: an explicit POST fired after the detail page
    renders, not a GET-time side effect)."""

    async def test_get_alone_does_not_clear_the_badge(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """TW-AUD-024: GET is query-only now — a prefetch/retry/render failure must
        not silently clear the indicator before the user has actually acknowledged it."""
        topic = _make_topic(db_conn, name="Query Only Topic", status=TopicStatus.READY)
        create_check_result(
            db_conn,
            CheckResult(topic_id=topic.id, articles_found=2, has_new_info=True),
        )
        db_conn.commit()

        detail = await client.get(f"/topics/{topic.id}")
        assert detail.status_code == 200

        after = await client.get("/")
        assert "badge--signal" in after.text

    async def test_detail_page_carries_the_ack_trigger(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """The rendered page fires the acknowledgement itself, keyed to the
        displayed check, once content has actually loaded (``hx-trigger="load"``)."""
        topic = _make_topic(db_conn, name="Ack Wiring Topic", status=TopicStatus.READY)
        check = create_check_result(
            db_conn,
            CheckResult(topic_id=topic.id, articles_found=2, has_new_info=True),
        )
        db_conn.commit()

        detail = await client.get(f"/topics/{topic.id}")
        assert f'hx-post="/topics/{topic.id}/checks/{check.id}/seen"' in detail.text
        assert 'hx-trigger="load"' in detail.text

    async def test_badge_clears_after_ack_and_returns_on_new_check(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """badge present -> open + ack -> badge gone -> a newer new-info check re-badges."""
        topic = _make_topic(db_conn, name="Seen Topic", status=TopicStatus.READY)
        now = datetime.now(UTC)
        check = create_check_result(
            db_conn,
            CheckResult(topic_id=topic.id, articles_found=2, has_new_info=True, checked_at=now),
        )
        db_conn.commit()

        before = await client.get("/")
        assert before.status_code == 200
        assert "badge--signal" in before.text

        detail = await client.get(f"/topics/{topic.id}")
        assert detail.status_code == 200
        ack = await client.post(f"/topics/{topic.id}/checks/{check.id}/seen")
        assert ack.status_code == 204

        after = await client.get("/")
        assert "badge--signal" not in after.text

        # A strictly-later check that finds new info re-badges (per-row seen_at).
        create_check_result(
            db_conn,
            CheckResult(
                topic_id=topic.id,
                articles_found=1,
                has_new_info=True,
                checked_at=now + timedelta(hours=1),
            ),
        )
        db_conn.commit()
        again = await client.get("/")
        assert "badge--signal" in again.text

    async def test_ack_for_a_stale_check_id_is_a_noop(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """An ack keyed to a check that is no longer the latest must not clear the
        badge for a newer check it never actually displayed."""
        topic = _make_topic(db_conn, name="Stale Ack Topic", status=TopicStatus.READY)
        now = datetime.now(UTC)
        older = create_check_result(
            db_conn,
            CheckResult(topic_id=topic.id, articles_found=1, has_new_info=True, checked_at=now),
        )
        db_conn.commit()
        create_check_result(
            db_conn,
            CheckResult(topic_id=topic.id, articles_found=1, has_new_info=True, checked_at=now + timedelta(hours=1)),
        )
        db_conn.commit()

        ack = await client.post(f"/topics/{topic.id}/checks/{older.id}/seen")
        assert ack.status_code == 204

        after = await client.get("/")
        assert "badge--signal" in after.text

    async def test_filtered_search_drops_badge_after_ack(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """The filtered search partial (search_dashboard_data) honors the same gate."""
        topic = _make_topic(db_conn, name="Filter Topic", status=TopicStatus.READY)
        check = create_check_result(
            db_conn,
            CheckResult(topic_id=topic.id, articles_found=1, has_new_info=True),
        )
        db_conn.commit()

        before = await client.get("/topics/search?q=Filter")
        assert "badge--signal" in before.text

        await client.get(f"/topics/{topic.id}")
        await client.post(f"/topics/{topic.id}/checks/{check.id}/seen")

        after = await client.get("/topics/search?q=Filter")
        assert "badge--signal" not in after.text

    async def test_open_preserves_history_and_notify(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """Acknowledging must NOT mutate has_new_info: the detail history cell still
        reads 'Yes' and the Notify button remains."""
        topic = _make_topic(db_conn, name="History Topic", status=TopicStatus.READY)
        check = create_check_result(
            db_conn,
            CheckResult(topic_id=topic.id, articles_found=1, has_new_info=True),
        )
        db_conn.commit()

        await client.get(f"/topics/{topic.id}")
        await client.post(f"/topics/{topic.id}/checks/{check.id}/seen")
        detail = await client.get(f"/topics/{topic.id}")
        assert detail.status_code == 200
        assert "Notify" in detail.text
        assert ">Yes<" in detail.text


# --- Check detail (novelty findings disclosure, AUG-110) ---


class TestCheckDetail:
    """GET /topics/{topic_id}/checks/{check_id}/detail.

    A lazy HTMX panel exposing the novelty findings a check already stored —
    summary, key facts, source links, and the three scores. No new LLM call;
    this is a read path over ``llm_response``. ``reasoning`` (the model's raw
    chain-of-thought) must never render.
    """

    async def test_renders_stored_findings(self, client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
        topic = _make_topic(db_conn, name="Findings Topic")
        novelty = NoveltyResult(
            has_new_info=True,
            summary="A new development happened.",
            key_facts=["Fact one", "Fact two"],
            source_urls=["https://example.com/a"],
            confidence=0.87,
            relevance=0.91,
            importance=4,
            reasoning="secret chain of thought",
        )
        check = create_check_result(
            db_conn,
            CheckResult(topic_id=topic.id, has_new_info=True, llm_response=novelty.model_dump_json()),
        )
        db_conn.commit()

        response = await client.get(f"/topics/{topic.id}/checks/{check.id}/detail")
        assert response.status_code == 200
        assert "A new development happened." in response.text
        assert "Fact one" in response.text
        assert "Fact two" in response.text
        assert "https://example.com/a" in response.text
        assert "0.87" in response.text
        assert "4/5" in response.text
        assert "secret chain of thought" not in response.text

    async def test_wrong_topic_returns_404(self, client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
        topic_a = _make_topic(db_conn, name="Owner A")
        topic_b = _make_topic(db_conn, name="Owner B")
        check = create_check_result(
            db_conn,
            CheckResult(
                topic_id=topic_b.id,
                has_new_info=True,
                llm_response=NoveltyResult(has_new_info=True, confidence=0.5).model_dump_json(),
            ),
        )
        db_conn.commit()

        response = await client.get(f"/topics/{topic_a.id}/checks/{check.id}/detail")
        assert response.status_code == 404

    async def test_missing_check_returns_404(self, client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
        topic = _make_topic(db_conn)
        response = await client.get(f"/topics/{topic.id}/checks/999999/detail")
        assert response.status_code == 404

    async def test_check_without_llm_response_renders_no_findings(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        topic = _make_topic(db_conn)
        check = create_check_result(
            db_conn,
            CheckResult(topic_id=topic.id, has_new_info=False, llm_response=None),
        )
        db_conn.commit()

        response = await client.get(f"/topics/{topic.id}/checks/{check.id}/detail")
        assert response.status_code == 200
        assert "No findings" in response.text

    async def test_detail_page_wires_lazy_disclosure_for_new_info_row(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        topic = _make_topic(db_conn, name="Disclosure Topic")
        check = create_check_result(
            db_conn,
            CheckResult(
                topic_id=topic.id,
                has_new_info=True,
                llm_response=NoveltyResult(has_new_info=True, summary="x", confidence=0.5).model_dump_json(),
            ),
        )
        db_conn.commit()

        page = await client.get(f"/topics/{topic.id}")
        assert f'hx-get="/topics/{topic.id}/checks/{check.id}/detail"' in page.text
        assert "toggle once from:closest details" in page.text


# --- Add Topic ---


class TestAddTopic:
    """Tests for GET /topics/new and POST /topics."""

    async def test_add_form_renders(self, client: httpx.AsyncClient) -> None:
        """The add topic form page loads successfully."""
        response = await client.get("/topics/new")
        assert response.status_code == 200
        assert "Add Topic" in response.text
        assert "<form" in response.text

    async def test_add_form_feed_mode_fieldset_has_accessible_name(self, client: httpx.AsyncClient) -> None:
        """AUG-237: the feed-mode radio group is labeled via the visible kicker."""
        response = await client.get("/topics/new")
        assert '<span class="card-kicker" id="feed-source-kicker">Feed Source</span>' in response.text
        assert '<fieldset aria-labelledby="feed-source-kicker">' in response.text

    async def test_add_form_feed_validation_results_is_live_region(self, client: httpx.AsyncClient) -> None:
        """AUG-235: the feed-URL validation result target announces to screen readers."""
        response = await client.get("/topics/new")
        assert '<div id="feed-validation-results" role="status" aria-live="polite" aria-atomic="true"></div>' in (
            response.text
        )

    async def test_automatic_radio_does_not_claim_google_news(self, client: httpx.AsyncClient) -> None:
        """AUG-122: routing is Bing-first with Google fallback, not Google News."""
        response = await client.get("/topics/new")
        assert "Automatic (Google News)" not in response.text

    async def test_validate_urls_button_has_pending_state(self, client: httpx.AsyncClient) -> None:
        """AUG-113: the button disables itself and shows a spinner during validation,
        guarding against a duplicate-submit re-triggering the serial fetch."""
        response = await client.get("/topics/new")
        assert 'hx-disabled-elt="this"' in response.text
        assert "Validate URLs" in response.text
        assert 'class="htmx-indicator spinner"' in response.text

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

    async def test_manual_mode_without_feed_urls_is_rejected(self, client: httpx.AsyncClient) -> None:
        """AUG-097: an empty MANUAL list can never fetch, so refuse it up front."""
        response = await client.post(
            "/topics",
            data={
                "name": "No Sources",
                "description": "Test",
                "feed_mode": "manual",
                "feed_urls": "   \n  \n",
            },
            follow_redirects=False,
        )
        assert response.status_code == 422
        assert "at least one feed URL" in response.text

    async def test_blank_topic_name_is_rejected(self, client: httpx.AsyncClient) -> None:
        """AUG-150: whitespace-only names pass ``required`` and UNIQUE — not here."""
        response = await client.post(
            "/topics",
            data={"name": "   ", "description": "Test", "feed_mode": "auto"},
            follow_redirects=False,
        )
        assert response.status_code == 422
        assert "Topic name is required" in response.text

    async def test_topic_name_is_trimmed(self, client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
        """AUG-150: surrounding whitespace never becomes part of the identity."""
        from app.crud import get_topic_by_name

        with patch("app.web.routers.background._run_init", new_callable=AsyncMock):
            await client.post(
                "/topics",
                data={"name": "  Padded  ", "description": "Test", "feed_mode": "auto"},
                follow_redirects=False,
            )

        assert get_topic_by_name(db_conn, "Padded") is not None

    async def test_tags_are_one_per_line_and_canonicalized(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """AUG-338/AUG-339: a comma is tag text, and equivalent tags collapse."""
        from app.crud import get_topic_by_name

        with patch("app.web.routers.background._run_init", new_callable=AsyncMock):
            await client.post(
                "/topics",
                data={
                    "name": "Tagged",
                    "description": "Test",
                    "feed_mode": "auto",
                    "tags": "Policy, Europe\n  Tech   News \nTech News\n\n",
                },
                follow_redirects=False,
            )

        topic = get_topic_by_name(db_conn, "Tagged")
        assert topic is not None
        assert topic.tags == ["Policy, Europe", "Tech News"]

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

    async def test_create_topic_duplicate_name_returns_422(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """Re-adding a topic with an existing name returns an inline 422, not a 500."""
        _make_topic(db_conn, name="Dupe Topic")

        with patch(
            "app.web.routers.background._run_init",
            new_callable=AsyncMock,
        ):
            response = await client.post(
                "/topics",
                data={
                    "name": "Dupe Topic",
                    "description": "Trying to re-add",
                    "feed_mode": "manual",
                    "feed_urls": "https://example.com/feed.xml",
                },
                follow_redirects=False,
            )

        assert response.status_code == 422
        assert "A topic with that name already exists" in response.text
        # Submitted form input is preserved (not discarded by a 500).
        assert "Dupe Topic" in response.text
        assert "Trying to re-add" in response.text
        assert "https://example.com/feed.xml" in response.text

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

    async def test_create_topic_persists_novelty_instruction(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """The per-topic novelty instruction is persisted on create; blank stores NULL."""
        with patch("app.web.routers.background._run_init", new_callable=AsyncMock):
            await client.post(
                "/topics",
                data={
                    "name": "Instructed",
                    "description": "Test",
                    "feed_urls": "",
                    "novelty_instruction": "  Only official announcements.  ",
                },
                follow_redirects=False,
            )
            await client.post(
                "/topics",
                data={"name": "Uninstructed", "description": "Test", "feed_urls": "", "novelty_instruction": "   "},
                follow_redirects=False,
            )

        from app.crud import get_topic_by_name

        instructed = get_topic_by_name(db_conn, "Instructed")
        assert instructed is not None
        assert instructed.novelty_instruction == "Only official announcements."
        uninstructed = get_topic_by_name(db_conn, "Uninstructed")
        assert uninstructed is not None
        assert uninstructed.novelty_instruction is None

    async def test_create_topic_persists_importance_threshold(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """The per-topic importance threshold is persisted on create; blank stores NULL."""
        with patch("app.web.routers.background._run_init", new_callable=AsyncMock):
            await client.post(
                "/topics",
                data={
                    "name": "ImportantOnly",
                    "description": "Test",
                    "feed_urls": "",
                    "importance_threshold": "4",
                },
                follow_redirects=False,
            )
            await client.post(
                "/topics",
                data={"name": "AnyImportance", "description": "Test", "feed_urls": "", "importance_threshold": ""},
                follow_redirects=False,
            )

        from app.crud import get_topic_by_name

        important = get_topic_by_name(db_conn, "ImportantOnly")
        assert important is not None
        assert important.importance_threshold == 4
        any_importance = get_topic_by_name(db_conn, "AnyImportance")
        assert any_importance is not None
        assert any_importance.importance_threshold is None

    async def test_create_topic_rejects_out_of_range_importance(self, client: httpx.AsyncClient) -> None:
        """Out-of-range or non-integer importance threshold returns 422."""
        for bad_value in ("6", "0", "abc"):
            response = await client.post(
                "/topics",
                data={
                    "name": f"BadImportance{bad_value}",
                    "description": "Test",
                    "feed_urls": "",
                    "importance_threshold": bad_value,
                },
                follow_redirects=False,
            )
            assert response.status_code == 422
            assert "between 1 and 5" in response.text

    async def test_create_topic_rejects_over_length_novelty_instruction(self, client: httpx.AsyncClient) -> None:
        """A novelty instruction over the cap returns 422."""
        from app.models import NOVELTY_INSTRUCTION_MAX_CHARS

        response = await client.post(
            "/topics",
            data={
                "name": "TooLong",
                "description": "Test",
                "feed_urls": "",
                "novelty_instruction": "x" * (NOVELTY_INSTRUCTION_MAX_CHARS + 1),
            },
            follow_redirects=False,
        )
        assert response.status_code == 422
        assert f"at most {NOVELTY_INSTRUCTION_MAX_CHARS} characters" in response.text


# --- Topic Detail ---


class TestTopicDetail:
    """Tests for GET /topics/{id}."""

    async def test_detail_page_renders(self, client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
        """Detail page shows topic name and description."""
        topic = _make_topic(db_conn)
        response = await client.get(f"/topics/{topic.id}")
        assert response.status_code == 200
        assert topic.name in response.text

    async def test_detail_status_announce_is_sibling_live_region(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """AUG-236: a hidden live-region sibling of #status-area announces completion.

        Must stay a SIBLING, never a child — #status-area is a leaf that the 3s
        researching poll replaces wholesale via innerHTML.
        """
        topic = _make_topic(db_conn)
        response = await client.get(f"/topics/{topic.id}")
        assert (
            '<div id="status-announce" class="sr-only" role="status" aria-live="polite" '
            'aria-atomic="true"></div>' in response.text
        )
        # Sibling, not nested: the announce div closes before #status-area opens.
        assert response.text.index('id="status-announce"') < response.text.index('id="status-area"')
        assert 'target.id !== "status-area"' in response.text
        assert "data-status-terminal" in response.text

    async def test_above_final_page_clamps_to_the_last_page_with_history(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """TW-AUD-023: an out-of-range page showed the false 'no history' state."""
        topic = _make_topic(db_conn, name="Paged")
        create_check_result(db_conn, CheckResult(topic_id=topic.id, articles_found=1))
        db_conn.commit()

        response = await client.get(f"/topics/{topic.id}?page=999")
        assert response.status_code == 200
        assert "No checks performed yet" not in response.text

    async def test_absurd_page_does_not_overflow_the_offset_binding(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """TW-AUD-023: a huge page used to bind an OFFSET SQLite cannot take."""
        topic = _make_topic(db_conn, name="Overflowing")
        response = await client.get(f"/topics/{topic.id}?page={2**63}")
        assert response.status_code == 200

    async def test_nonpositive_page_clamps_to_one(self, client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
        topic = _make_topic(db_conn, name="Negative Page")
        create_check_result(db_conn, CheckResult(topic_id=topic.id, articles_found=1))
        db_conn.commit()

        response = await client.get(f"/topics/{topic.id}?page=-5")
        assert response.status_code == 200
        assert "No checks performed yet" not in response.text

    async def test_failing_sources_callout(self, client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
        """The detail page warns when the most recent check found no usable source."""
        topic = _make_topic(db_conn, name="HBDetail")
        create_check_result(db_conn, CheckResult(topic_id=topic.id, stage_error="sources_failed: x"))
        db_conn.commit()

        page = await client.get(f"/topics/{topic.id}")
        assert "Sources failing" in page.text

    async def test_no_callout_for_a_healthy_topic(self, client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
        topic = _make_topic(db_conn, name="HBDetailOk")
        create_check_result(db_conn, CheckResult(topic_id=topic.id))
        db_conn.commit()

        page = await client.get(f"/topics/{topic.id}")
        assert "Sources failing" not in page.text

    async def test_failing_sources_callout_survives_older_history_pages(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """AUG-224: the callout used to read checks[0] from the paginated history
        and only render on page==1, so it vanished while viewing an older page even
        though the topic's actual latest check was still failing."""
        settings = _make_settings(web_page_size=5)
        app.dependency_overrides[get_settings] = lambda: settings
        try:
            topic = _make_topic(db_conn, name="HBPaged")
            now = datetime.now(UTC)
            # 6 older, healthy checks fill page 1 (page_size=5); the latest, failing
            # check lands alone on page 2 once ordered checked_at DESC.
            for i in range(6):
                create_check_result(db_conn, CheckResult(topic_id=topic.id, checked_at=now + timedelta(minutes=i)))
            create_check_result(
                db_conn,
                CheckResult(
                    topic_id=topic.id,
                    stage_error="sources_failed: x",
                    checked_at=now + timedelta(hours=1),
                ),
            )
            db_conn.commit()

            page2 = await client.get(f"/topics/{topic.id}?page=2")
            assert page2.status_code == 200
            assert "Sources failing" in page2.text
        finally:
            app.dependency_overrides[get_settings] = lambda: _make_settings()

    async def test_failing_sources_callout_is_not_line_clamped(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """AUG-106: the callout gets its own bordered component, not the two-line
        clamped .row-error class used for table-cell text."""
        topic = _make_topic(db_conn, name="HBClamp")
        create_check_result(db_conn, CheckResult(topic_id=topic.id, stage_error="sources_failed: x"))
        db_conn.commit()

        page = await client.get(f"/topics/{topic.id}")
        assert 'class="source-warning"' in page.text

    async def test_failing_sources_callout_appears_in_feed_source_fragment(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """AUG-224: living in the Feed Source fragment means it refreshes on that
        section's own 30s poll, independent of a full page reload."""
        topic = _make_topic(db_conn, name="HBFragment")
        create_check_result(db_conn, CheckResult(topic_id=topic.id, stage_error="sources_failed: x"))
        db_conn.commit()

        fragment = await client.get(f"/topics/{topic.id}/feed-source")
        assert fragment.status_code == 200
        assert "Sources failing" in fragment.text

    async def test_detail_shows_importance_threshold(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """Detail page shows the per-topic importance threshold, or 'off' when NULL."""
        topic = _make_topic(db_conn, name="Important Detail", importance_threshold=4)
        response = await client.get(f"/topics/{topic.id}")
        assert response.status_code == 200
        assert "Importance threshold" in response.text
        assert "4/5" in response.text

        plain = _make_topic(db_conn, name="Plain Detail")
        response = await client.get(f"/topics/{plain.id}")
        assert response.status_code == 200
        assert "notify on all" in response.text

    async def test_detail_shows_novelty_instruction(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """Detail page shows the per-topic novelty instruction, or 'none' when NULL."""
        topic = _make_topic(db_conn, name="Instructed Detail", novelty_instruction="Official announcements only.")
        response = await client.get(f"/topics/{topic.id}")
        assert response.status_code == 200
        assert "Novelty instruction" in response.text
        assert "Official announcements only." in response.text

        plain = _make_topic(db_conn, name="Uninstructed Detail")
        response = await client.get(f"/topics/{plain.id}")
        assert response.status_code == 200
        assert "default criteria" in response.text

    async def test_detail_routine_actions_in_masthead_above_history(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """AUG-115: Check Now/Edit/Enable-Disable live in the masthead, above Check History."""
        topic = _make_topic(db_conn, name="Routine Actions")
        for _ in range(3):
            create_check_result(db_conn, CheckResult(topic_id=topic.id))
        db_conn.commit()

        response = await client.get(f"/topics/{topic.id}")
        assert response.status_code == 200
        text = response.text

        check_now_pos = text.index(f'action="/topics/{topic.id}/check"')
        edit_pos = text.index(f'href="/topics/{topic.id}/edit"')
        toggle_pos = text.index(f'action="/topics/{topic.id}/toggle-active"')
        history_pos = text.index("<h2>Check History</h2>")
        maintenance_pos = text.index("<h2>Maintenance</h2>")

        assert check_now_pos < history_pos
        assert edit_pos < history_pos
        assert toggle_pos < history_pos
        assert check_now_pos < maintenance_pos

    async def test_detail_maintenance_block_keeps_secondary_actions(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """AUG-115: reinitialize/export/delete stay grouped as Maintenance, below history."""
        topic = _make_topic(db_conn, name="Maintenance Actions")
        response = await client.get(f"/topics/{topic.id}")
        text = response.text

        assert "<h2>Maintenance</h2>" in text
        assert "<h2>Actions</h2>" not in text
        maintenance_pos = text.index("<h2>Maintenance</h2>")
        reinit_pos = text.index("Re-initialize")
        export_pos = text.index("Export JSON")
        delete_pos = text.index("Delete Topic")
        assert maintenance_pos < reinit_pos < export_pos
        assert maintenance_pos < delete_pos

    async def test_detail_novelty_policy_is_one_labeled_block(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """AUG-121: instruction + confidence/relevance/importance gates render as one card."""
        topic = _make_topic(
            db_conn,
            name="Policy Card",
            novelty_instruction="Official announcements only.",
            confidence_threshold=0.75,
            relevance_threshold=0.6,
            importance_threshold=3,
        )
        response = await client.get(f"/topics/{topic.id}")
        text = response.text

        kicker_pos = text.index('card-kicker">Novelty Policy')
        card_close_pos = text.index("</section>", kicker_pos)
        instruction_pos = text.index("Official announcements only.")
        confidence_pos = text.index("Confidence threshold")
        relevance_pos = text.index("Relevance threshold")
        importance_pos = text.index("Importance threshold")

        # All four live inside the same <section class="card">...</section> block.
        assert kicker_pos < instruction_pos < card_close_pos
        assert kicker_pos < confidence_pos < card_close_pos
        assert kicker_pos < relevance_pos < card_close_pos
        assert kicker_pos < importance_pos < card_close_pos
        # Not duplicated into the top-level at-a-glance meta-grid anymore.
        assert text.count("Confidence threshold") == 1
        top_meta_grid_close = text.index("</dl>")
        assert top_meta_grid_close < confidence_pos

    async def test_detail_check_history_shows_importance(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """Each check's importance score is rendered; pre-m023 blobs render as '-'."""
        topic = _make_topic(db_conn, name="Importance History")
        create_check_result(
            db_conn,
            CheckResult(
                topic_id=topic.id,
                has_new_info=True,
                llm_response=json.dumps({"has_new_info": True, "confidence": 0.9, "importance": 5}),
            ),
        )
        db_conn.commit()

        response = await client.get(f"/topics/{topic.id}")
        assert response.status_code == 200
        assert "5/5" in response.text

        legacy = _make_topic(db_conn, name="Legacy History")
        create_check_result(
            db_conn,
            CheckResult(
                topic_id=legacy.id,
                has_new_info=True,
                llm_response=json.dumps({"has_new_info": True, "confidence": 0.9}),
            ),
        )
        db_conn.commit()

        response = await client.get(f"/topics/{legacy.id}")
        assert response.status_code == 200
        assert "Importance" in response.text

    async def test_detail_labels_importance_suppressed_check(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """A check muted by the importance gate reads 'Suppressed', not a bare '-'.

        Without this the row is byte-identical to a notification that silently
        failed to send.
        """
        topic = _make_topic(db_conn, name="Suppressed Topic", importance_threshold=4)
        create_check_result(
            db_conn,
            CheckResult(
                topic_id=topic.id,
                has_new_info=True,
                notification_sent=False,
                llm_response=json.dumps({"has_new_info": True, "confidence": 0.9, "importance": 2}),
            ),
        )
        db_conn.commit()

        response = await client.get(f"/topics/{topic.id}")
        assert response.status_code == 200
        assert "Suppressed" in response.text

    async def test_detail_does_not_label_above_threshold_check_suppressed(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """A check that cleared the importance gate is never labelled 'Suppressed'."""
        topic = _make_topic(db_conn, name="Delivered Topic", importance_threshold=2)
        create_check_result(
            db_conn,
            CheckResult(
                topic_id=topic.id,
                has_new_info=True,
                notification_sent=True,
                llm_response=json.dumps({"has_new_info": True, "confidence": 0.9, "importance": 4}),
            ),
        )
        db_conn.commit()

        response = await client.get(f"/topics/{topic.id}")
        assert response.status_code == 200
        assert "Suppressed" not in response.text

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


class TestArticleHeadersForTopic:
    """AUG-038: the topic-detail article list stops hydrating raw_content, which
    the template never renders."""

    def test_raw_content_is_not_hydrated(self, db_conn: sqlite3.Connection) -> None:
        topic = create_topic(db_conn, Topic(name="Headers Only", description="d"))
        db_conn.commit()
        create_article(
            db_conn,
            Article(
                topic_id=topic.id,
                title="Full Article",
                url="https://example.com/a",
                content_hash="hash1",
                raw_content="x" * 5000,
                source_feed="https://example.com/feed.xml",
            ),
        )
        db_conn.commit()

        headers = list_article_headers_for_topic(db_conn, topic.id)
        assert len(headers) == 1
        assert headers[0].title == "Full Article"
        assert headers[0].raw_content is None

    async def test_detail_page_never_leaks_raw_content(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        topic = _make_topic(db_conn, name="No Leak Topic")
        create_article(
            db_conn,
            Article(
                topic_id=topic.id,
                title="Some Article",
                url="https://example.com/a",
                content_hash="hash2",
                raw_content="SECRET-BODY-TEXT-MARKER",
                source_feed="https://example.com/feed.xml",
            ),
        )
        db_conn.commit()

        response = await client.get(f"/topics/{topic.id}")
        assert response.status_code == 200
        assert "SECRET-BODY-TEXT-MARKER" not in response.text


class TestEmptyArticlesGuidance:
    """AUG-240: the zero-article empty state names the action actually available
    for the topic's current state, instead of always pointing at "Check Now"
    (which renders only for an active READY topic)."""

    async def test_ready_active_points_at_check_now(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        topic = _make_topic(db_conn, name="Empty Ready", status=TopicStatus.READY)
        response = await client.get(f"/topics/{topic.id}")
        assert "Check Now" in response.text

    async def test_disabled_points_at_enable(self, client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
        topic = _make_topic(db_conn, name="Empty Disabled", status=TopicStatus.READY, is_active=False)
        response = await client.get(f"/topics/{topic.id}")
        assert "Enable it above" in response.text
        assert "Check Now" not in response.text

    async def test_new_points_at_initialize(self, client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
        topic = _make_topic(db_conn, name="Empty New", status=TopicStatus.NEW)
        response = await client.get(f"/topics/{topic.id}")
        assert "Initialize Now" in response.text
        assert 'Check Now" above' not in response.text

    async def test_researching_points_at_waiting(self, client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
        topic = _make_topic(db_conn, name="Empty Researching", status=TopicStatus.RESEARCHING)
        response = await client.get(f"/topics/{topic.id}")
        assert "Research is in progress" in response.text

    async def test_error_points_at_retry(self, client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
        topic = _make_topic(db_conn, name="Empty Error", status=TopicStatus.ERROR, error_message="boom")
        response = await client.get(f"/topics/{topic.id}")
        assert 'Retry Research" below' in response.text


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

    async def test_status_ready_marked_terminal_for_announce(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """AUG-236: the ready fragment carries the marker the afterSwap handler announces."""
        topic = _make_topic(db_conn, status=TopicStatus.READY)
        create_knowledge_state(
            db_conn,
            KnowledgeState(topic_id=topic.id, summary_text="Summary here.", token_count=20),
        )
        db_conn.commit()

        response = await client.get(f"/topics/{topic.id}/status")
        assert 'data-status-terminal="ready"' in response.text

    async def test_status_token_meter_has_accessible_name(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """AUG-239: the token-budget <progress> element is linked to its visible label."""
        topic = _make_topic(db_conn, status=TopicStatus.READY)
        create_knowledge_state(
            db_conn,
            KnowledgeState(topic_id=topic.id, summary_text="Summary here.", token_count=20),
        )
        db_conn.commit()

        response = await client.get(f"/topics/{topic.id}/status")
        assert '<span id="token-budget-label">Token budget</span>' in response.text
        assert 'aria-labelledby="token-budget-label"' in response.text

    async def test_status_renders_markdown_summary(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """The knowledge summary's markdown renders to HTML, not literal asterisks.

        Guards against the ``| markdown`` filter being dropped from the template or
        the result being double-escaped — neither of which a unit test would catch.
        """
        topic = _make_topic(db_conn, status=TopicStatus.READY)
        create_knowledge_state(
            db_conn,
            KnowledgeState(
                topic_id=topic.id,
                summary_text="**Current Status:** ongoing\n- fact a",
                token_count=20,
            ),
        )
        db_conn.commit()

        response = await client.get(f"/topics/{topic.id}/status")
        assert "<strong>Current Status:</strong>" in response.text
        assert "<li>fact a</li>" in response.text
        assert "**Current Status:**" not in response.text

    async def test_status_error_shows_retry(self, client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
        """ERROR status fragment shows error and retry button."""
        topic = _make_topic(db_conn, status=TopicStatus.ERROR, error_message="Init failed")
        response = await client.get(f"/topics/{topic.id}/status")
        assert "Init failed" in response.text
        assert "Retry" in response.text
        assert "hx-trigger" not in response.text
        # AUG-236: marker the detail page's afterSwap handler announces as failure.
        assert 'data-status-terminal="error"' in response.text

    async def test_status_deleted_returns_terminal_fragment(self, client: httpx.AsyncClient) -> None:
        """Deleted/nonexistent topic returns a 200 terminal fragment that stops polling (OVH-048)."""
        response = await client.get("/topics/9999/status")
        # 200 so HTMX swaps the fragment instead of leaving an eternal spinner.
        assert response.status_code == 200
        # No polling trigger remains, so the every-3s poll stops.
        assert "hx-trigger" not in response.text
        # Surfaces the failure to the user.
        assert "no longer exists" in response.text.lower()
        # AUG-236: marker the detail page's afterSwap handler announces as removal.
        assert 'data-status-terminal="removed"' in response.text

    async def test_status_since_mismatch_sets_hx_refresh(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """Status moved on since the page rendered -> HX-Refresh triggers a one-shot reload."""
        topic = _make_topic(db_conn, status=TopicStatus.READY)
        response = await client.get(f"/topics/{topic.id}/status?since=researching")
        assert response.status_code == 200
        assert response.headers["HX-Refresh"] == "true"

    async def test_status_since_match_no_refresh(self, client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
        """Unchanged status keeps polling: fragment as usual, no refresh header."""
        topic = _make_topic(db_conn, status=TopicStatus.RESEARCHING)
        response = await client.get(f"/topics/{topic.id}/status?since=researching")
        assert response.status_code == 200
        assert "HX-Refresh" not in response.headers
        assert "every 3s" in response.text
        assert "since=researching" in response.text

    async def test_status_without_since_never_refreshes(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """Plain GET without ?since behaves exactly as before (regression guard)."""
        topic = _make_topic(db_conn, status=TopicStatus.READY)
        response = await client.get(f"/topics/{topic.id}/status")
        assert response.status_code == 200
        assert "HX-Refresh" not in response.headers

    async def test_status_deleted_with_since_no_refresh(self, client: httpx.AsyncClient) -> None:
        """Deleted topic never triggers a reload — that would land on a 404 page."""
        response = await client.get("/topics/9999/status?since=researching")
        assert response.status_code == 200
        assert "HX-Refresh" not in response.headers
        assert "hx-trigger" not in response.text

    async def test_status_new_fragment_polls(self, client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
        """NEW topics poll too (slow cadence) so NEW -> RESEARCHING appears without a reload."""
        topic = _make_topic(db_conn, status=TopicStatus.NEW)
        response = await client.get(f"/topics/{topic.id}/status")
        assert response.status_code == 200
        assert "hx-trigger" in response.text
        assert "every 30s" in response.text
        assert "since=new" in response.text


class TestTopicRow:
    """Tests for GET /topics/{id}/row (dashboard row status poll)."""

    async def test_row_returns_row(self, client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
        """Plain poll (no since) renders the row without the just-checked marker (OVH-119)."""
        topic = _make_topic(db_conn, status=TopicStatus.RESEARCHING)
        response = await client.get(f"/topics/{topic.id}/row", headers={"HX-Request": "true"})
        assert response.status_code == 200
        assert f'id="topic-{topic.id}"' in response.text
        assert "data-just-checked" not in response.text

    async def test_row_since_match_returns_204(self, client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
        """Unchanged status -> 204 so htmx skips the swap and checkbox state survives."""
        topic = _make_topic(db_conn, status=TopicStatus.RESEARCHING)
        response = await client.get(f"/topics/{topic.id}/row?since=researching", headers={"HX-Request": "true"})
        assert response.status_code == 204
        assert response.text == ""

    async def test_row_since_mismatch_returns_terminated_row(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """Transition re-renders the row once; the ready row carries no poll attrs."""
        topic = _make_topic(db_conn, status=TopicStatus.READY)
        response = await client.get(f"/topics/{topic.id}/row?since=researching", headers={"HX-Request": "true"})
        assert response.status_code == 200
        assert "Ready" in response.text
        assert "hx-trigger" not in response.text

    async def test_row_missing_topic_returns_empty_200(self, client: httpx.AsyncClient) -> None:
        """Deleted mid-poll: 200 empty body removes the row via outerHTML swap (OVH-048)."""
        response = await client.get("/topics/9999/row", headers={"HX-Request": "true"})
        assert response.status_code == 200
        assert response.text == ""

    async def test_row_poll_attrs_by_status(self, client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
        """Only new/researching rows poll; ready/error rows emit no poll attrs."""
        cases = {
            TopicStatus.RESEARCHING: "every 3s",
            TopicStatus.NEW: "every 30s",
        }
        for status, interval in cases.items():
            topic = _make_topic(db_conn, name=f"Poll {status.value}", status=status)
            response = await client.get(f"/topics/{topic.id}/row", headers={"HX-Request": "true"})
            assert f"/topics/{topic.id}/row?since={status.value}" in response.text
            assert interval in response.text
        for status in (TopicStatus.READY, TopicStatus.ERROR):
            topic = _make_topic(db_conn, name=f"NoPoll {status.value}", status=status)
            response = await client.get(f"/topics/{topic.id}/row", headers={"HX-Request": "true"})
            assert "/row?since=" not in response.text
            assert "hx-trigger" not in response.text

    async def test_row_non_htmx_redirects_to_detail(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """A direct browser GET of the fragment redirects to the detail page (helper fallback)."""
        topic = _make_topic(db_conn, status=TopicStatus.READY)
        response = await client.get(f"/topics/{topic.id}/row")
        assert response.status_code == 303
        assert response.headers["location"] == f"/topics/{topic.id}"


# --- Generic exception handler (OVH-046) ---


class TestGenericExceptionHandler:
    """Tests for @app.exception_handler(Exception) — unhandled errors."""

    @pytest.fixture(autouse=True)
    def _boom_route(self):
        """Register a throwaway route that raises an unhandled exception."""

        async def boom():
            raise RuntimeError("kaboom secret detail")

        app.add_api_route("/_test/boom", boom, methods=["GET"])
        yield
        app.router.routes = [r for r in app.router.routes if getattr(r, "path", None) != "/_test/boom"]

    @pytest.fixture
    async def error_client(self) -> AsyncGenerator[httpx.AsyncClient, None]:
        """Client that returns the handler's 500 response rather than re-raising.

        Starlette's ServerErrorMiddleware sends the handler response AND re-raises
        so the ASGI server can log/close; raise_app_exceptions=False mirrors what a
        real server (uvicorn) shows the client — the rendered 500.
        """
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac

    async def test_browser_gets_branded_html(self, error_client: httpx.AsyncClient) -> None:
        """A generic unhandled 500 renders the branded error.html for browser requests."""
        response = await error_client.get("/_test/boom", headers={"accept": "text/html"})
        assert response.status_code == 500
        assert "text/html" in response.headers["content-type"]
        # Branded error page chrome (base.html footer / heading).
        assert "Topic Watch" in response.text
        assert "500" in response.text
        # No traceback / internal detail leaked.
        assert "kaboom" not in response.text
        assert "RuntimeError" not in response.text
        assert "Traceback" not in response.text

    async def test_json_client_gets_envelope(self, error_client: httpx.AsyncClient) -> None:
        """A generic unhandled 500 returns the JSON envelope for API clients."""
        response = await error_client.get("/_test/boom", headers={"accept": "application/json"})
        assert response.status_code == 500
        assert response.json() == {"detail": "Internal server error"}
        # No traceback / internal detail leaked.
        assert "kaboom" not in response.text
        assert "RuntimeError" not in response.text


# --- Global HTMX error surfacing (OVH-011) ---


class TestHtmxErrorSurfacing:
    """Tests asserting the global HTMX error listener is wired into the client JS."""

    def test_response_and_send_error_listeners_present(self) -> None:
        """notifications.js registers global htmx:responseError and htmx:sendError listeners."""
        from pathlib import Path

        js = (Path(__file__).resolve().parent.parent / "app" / "static" / "notifications.js").read_text()
        assert "htmx:responseError" in js
        assert "htmx:sendError" in js
        # Surfaces a toast and offers a retry affordance.
        assert "toast" in js.lower()
        assert "retry" in js.lower()

    def test_retry_replays_only_safe_verbs(self) -> None:
        """AUG-218: a failed POST is never re-issued — its outcome is unknown."""
        from pathlib import Path

        js = (Path(__file__).resolve().parent.parent / "app" / "static" / "notifications.js").read_text()
        # The retry path decides on the verb before touching htmx.ajax.
        ajax_idx = js.find("window.htmx.ajax(")
        assert ajax_idx != -1
        guard = js[:ajax_idx]
        assert "isSafeVerb" in guard
        # Unsafe failures offer a reload to verify state instead.
        assert "Reload" in js
        assert "may or may not" in js


# --- Browser-notification robustness (OVH-117 / OVH-118 / OVH-119) ---


class TestBrowserNotificationRobustness:
    """Source-presence assertions for client-side notification guards."""

    def _notifications_js(self) -> str:
        from pathlib import Path

        return (Path(__file__).resolve().parent.parent / "app" / "static" / "notifications.js").read_text()

    def _settings_html(self) -> str:
        from pathlib import Path

        return (Path(__file__).resolve().parent.parent / "app" / "templates" / "settings.html").read_text()

    def _dashboard_html(self) -> str:
        from pathlib import Path

        return (Path(__file__).resolve().parent.parent / "app" / "templates" / "dashboard.html").read_text()

    def test_notification_construction_is_guarded(self) -> None:
        """OVH-117: new Notification() is wrapped in try/catch so an Illegal-constructor
        throw (Android Chrome) can't abort the afterSwap handler / leave inconsistent UI."""
        js = self._notifications_js()
        # The constructor *call* (not the comment) must sit inside a try block that
        # has a matching catch immediately after it.
        ctor_idx = js.find("n = new Notification(")
        assert ctor_idx != -1, "expected an assigned `new Notification(...)` call"
        # The nearest preceding `try {` opens the guard block...
        try_idx = js.rfind("try {", 0, ctor_idx)
        assert try_idx != -1, "Notification construction is not inside a try block"
        # ...and a catch closes it after the constructor.
        catch_idx = js.find("} catch", ctor_idx)
        assert catch_idx != -1, "Notification construction has no matching catch"

    def test_request_permission_chain_has_rejection_handler(self) -> None:
        """OVH-118: the requestPermission() chain has a rejection handler that
        unchecks the toggle and persists setEnabled(false) instead of leaving it
        checked-but-disabled."""
        html = self._settings_html()
        assert ".catch(" in html
        # The rejection handler mirrors the denial path: uncheck + persist false.
        catch_idx = html.find(".catch(")
        catch_body = html[catch_idx:]
        assert "checked = false" in catch_body
        assert "setEnabled(false)" in catch_body

    def test_dashboard_notification_gated_on_just_checked(self) -> None:
        """OVH-119: the dashboard afterSwap handler only fires on a fresh check
        (data-just-checked), not on any topic-row re-render (e.g. toggle-active)."""
        html = self._dashboard_html()
        assert "data-just-checked" in html
        # The fire condition requires the just-checked marker alongside new-info.
        assert 'justChecked === "true"' in html

    def test_show_reports_construction_failure(self) -> None:
        """AUG-128: show() returns a boolean capability result instead of failing
        silently, so callers can tell permission-granted from actually-displayed."""
        js = self._notifications_js()
        ctor_idx = js.find("n = new Notification(")
        catch_idx = js.find("} catch", ctor_idx)
        catch_body = js[catch_idx : catch_idx + 80]
        assert "return false" in catch_body
        # The success path returns true after scheduling the auto-close.
        assert "return true;" in js[catch_idx:]

    def test_notification_tag_is_not_hardcoded_shared(self) -> None:
        """AUG-219: no fixed "topic-watch" tag — same tag replaces same-origin
        notifications, so a shared literal collapsed alerts across topics."""
        js = self._notifications_js()
        assert 'tag: "topic-watch"' not in js
        assert "options.tag" in js


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

        with patch("app.web.routers.background._run_single_check", new_callable=AsyncMock) as mock_bg:
            response = await client.post(
                f"/topics/{topic.id}/check",
                headers={"HX-Request": "true"},
            )

        assert response.status_code == 200
        # Pipeline must be deferred to the background task, never run inline.
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

    async def test_check_htmx_row_defers_just_checked_until_fresh_result(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """AUG-217: the queuing response has not actually re-checked anything yet — the
        background task hasn't run — so it must not claim data-just-checked. Marking a
        stale pre-check row as fresh could re-fire a notification for old unseen info
        while the real completion produced no swap at all. Instead it renders a checking
        row that polls toward completion."""
        topic = _make_topic(db_conn)
        create_check_result(db_conn, CheckResult(topic_id=topic.id, has_new_info=True))

        with patch("app.web.routers.background._run_single_check", new_callable=AsyncMock):
            response = await client.post(
                f"/topics/{topic.id}/check",
                headers={"HX-Request": "true"},
                follow_redirects=False,
            )

        assert response.status_code == 200
        assert 'data-just-checked="true"' not in response.text
        assert f"/topics/{topic.id}/row?since_check_id=" in response.text

    async def test_check_completion_poll_marks_just_checked_on_new_result(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """AUG-217: the checking row's completion poll only marks data-just-checked once
        a newer check_results row actually exists, and keeps polling (204, no swap)
        while the background task is still running with no new row yet."""
        topic = _make_topic(db_conn)
        baseline = create_check_result(db_conn, CheckResult(topic_id=topic.id))
        from app.web.state import _checking_state

        owner = await _checking_state.start_check(topic.id)
        try:
            response = await client.get(
                f"/topics/{topic.id}/row?since_check_id={baseline.id}", headers={"HX-Request": "true"}
            )
            assert response.status_code == 204

            create_check_result(db_conn, CheckResult(topic_id=topic.id, has_new_info=True))
        finally:
            await _checking_state.finish_check(topic.id, owner)

        response = await client.get(
            f"/topics/{topic.id}/row?since_check_id={baseline.id}", headers={"HX-Request": "true"}
        )
        assert response.status_code == 200
        assert 'data-just-checked="true"' in response.text

    async def test_check_completion_poll_stops_without_spinning_forever(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """AUG-217: if the background task is no longer running and produced no newer
        check (e.g. it crashed before check_topic's transaction committed), the poll
        must not spin forever — render the row as-is, with no marker and no further
        poll trigger."""
        topic = _make_topic(db_conn)
        baseline = create_check_result(db_conn, CheckResult(topic_id=topic.id))

        response = await client.get(
            f"/topics/{topic.id}/row?since_check_id={baseline.id}", headers={"HX-Request": "true"}
        )
        assert response.status_code == 200
        assert 'data-just-checked="true"' not in response.text
        assert "hx-trigger" not in response.text

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

        owner = await _checking_state.start_check(topic.id)
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
            await _checking_state.finish_check(topic.id, owner)

    async def test_check_already_checking_non_htmx_redirects(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """OVH-005: the already-checking early return redirects for a plain form."""
        topic = _make_topic(db_conn)
        from app.web.state import _checking_state

        owner = await _checking_state.start_check(topic.id)
        try:
            response = await client.post(
                f"/topics/{topic.id}/check",
                follow_redirects=False,
            )
            assert response.status_code == 303
            assert response.headers["location"] == f"/topics/{topic.id}"
        finally:
            await _checking_state.finish_check(topic.id, owner)

    async def test_check_counts_articles_with_count_query(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """OVH-138: the article count comes from COUNT(*), not by hydrating every row."""
        topic = _make_topic(db_conn)

        with (
            patch("app.web.routers.background._run_single_check", new_callable=AsyncMock),
            patch("app.web.routers.topics.count_articles_for_topic", return_value=7) as mock_count,
            patch("app.web.routers.topics.list_article_headers_for_topic") as mock_list,
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

    async def test_settings_page_renders_silence_heartbeat_field(self, client: httpx.AsyncClient) -> None:
        page = await client.get("/settings")
        assert 'name="silence_heartbeat_checks"' in page.text

    async def test_settings_masks_api_key(self, client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """Settings page masks the API key (editable, non-env-sourced path)."""
        # Clear the env override so the editable (masked) API-key field renders (OVH-003).
        monkeypatch.delenv("TOPIC_WATCH_LLM__API_KEY", raising=False)
        response = await client.get("/settings")
        assert response.status_code == 200
        # Full key should NOT be visible
        assert "test-key-12345678" not in response.text
        # The masked format should be shown (first 4 chars...last 4 chars)
        assert "test...5678" in response.text

    async def test_settings_env_key_readonly(self, client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """When the key is env-sourced, the field is read-only and the full key is hidden."""
        monkeypatch.setenv("TOPIC_WATCH_LLM__API_KEY", "test-key-12345678")
        response = await client.get("/settings")
        assert response.status_code == 200
        assert "set via environment" in response.text.lower()
        assert "test-key-12345678" not in response.text

    async def test_settings_notification_results_are_live_regions(self, client: httpx.AsyncClient) -> None:
        """AUG-235: the test-notification result and browser-notif status are live regions."""
        page = await client.get("/settings")
        assert (
            '<div id="notification-test-result" style="margin-top: 1rem;" role="status" '
            'aria-live="polite" aria-atomic="true"></div>' in page.text
        )
        assert (
            '<span id="browser-notif-status" style="font-size: 0.875rem;" role="status" '
            'aria-live="polite" aria-atomic="true"></span>' in page.text
        )

    async def test_settings_browser_notif_reflects_display_failure(self, client: httpx.AsyncClient) -> None:
        """AUG-128: enabling notifications checks show()'s return value before claiming Active."""
        page = await client.get("/settings")
        assert "var displayed = TopicWatchNotifications.show(" in page.text
        assert "Notifications can't display on this device" in page.text
        assert "updateBrowserNotifUI(true)" in page.text

    async def test_settings_browser_notifications_use_distinct_tags(self, client: httpx.AsyncClient) -> None:
        """AUG-219: settings-page test notifications no longer share the "topic-watch" tag."""
        page = await client.get("/settings")
        assert '{ tag: "topic-watch-enable-test" }' in page.text
        assert '{ tag: "topic-watch-test-notification" }' in page.text


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

    async def test_edit_form_feed_mode_fieldset_has_accessible_name(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """AUG-237: the feed-mode radio group is labeled via the visible kicker."""
        topic = _make_topic(db_conn)
        response = await client.get(f"/topics/{topic.id}/edit")
        assert '<span class="card-kicker" id="feed-source-kicker">Feed Source</span>' in response.text
        assert '<fieldset aria-labelledby="feed-source-kicker">' in response.text

    async def test_edit_form_feed_validation_results_is_live_region(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """AUG-235: the feed-URL validation result target announces to screen readers."""
        topic = _make_topic(db_conn)
        response = await client.get(f"/topics/{topic.id}/edit")
        assert '<div id="feed-validation-results" role="status" aria-live="polite" aria-atomic="true"></div>' in (
            response.text
        )

    async def test_automatic_radio_does_not_claim_google_news(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """AUG-122: routing is Bing-first with Google fallback, not Google News."""
        topic = _make_topic(db_conn)
        response = await client.get(f"/topics/{topic.id}/edit")
        assert "Automatic (Google News)" not in response.text

    async def test_validate_urls_button_has_pending_state(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """AUG-113: the button disables itself and shows a spinner during validation."""
        topic = _make_topic(db_conn)
        response = await client.get(f"/topics/{topic.id}/edit")
        assert 'hx-disabled-elt="this"' in response.text
        assert 'class="htmx-indicator spinner"' in response.text

    async def test_rename_onto_an_existing_name_is_a_validation_error(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """AUG-147: a duplicate rename used to reach the global 500 handler."""
        _make_topic(db_conn, name="Taken")
        topic = _make_topic(db_conn, name="Renamable")

        response = await client.post(
            f"/topics/{topic.id}/edit",
            data={"name": "Taken", "description": "d", "feed_mode": "auto"},
            follow_redirects=False,
        )
        assert response.status_code == 422
        assert "already exists" in response.text

        from app.crud import get_topic

        assert get_topic(db_conn, topic.id).name == "Renamable"

    async def test_saving_an_unchanged_name_is_not_a_conflict(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """AUG-147: a topic's own name must not collide with itself."""
        topic = _make_topic(db_conn, name="Same Name")
        response = await client.post(
            f"/topics/{topic.id}/edit",
            data={"name": "Same Name", "description": "changed", "feed_mode": "auto"},
            follow_redirects=False,
        )
        assert response.status_code == 303

    async def test_unchanged_save_keeps_a_comma_bearing_tag_whole(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """AUG-339: an OPML folder named "Policy, Europe" is one tag, not two."""
        from app.crud import get_topic

        topic = _make_topic(db_conn, name="Imported", tags=["Policy, Europe"])

        form = await client.get(f"/topics/{topic.id}/edit")
        assert "Policy, Europe" in form.text

        response = await client.post(
            f"/topics/{topic.id}/edit",
            data={
                "name": "Imported",
                "description": topic.description,
                "feed_mode": "auto",
                "tags": "Policy, Europe",
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert get_topic(db_conn, topic.id).tags == ["Policy, Europe"]

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

    async def test_edit_does_not_overwrite_a_concurrent_lifecycle_transition(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """AUG-022: an edit writes configuration only, never lifecycle columns.

        The handler snapshots the topic, awaits DNS validation of the submitted
        feeds, then writes. An initialization finishing inside that await used to
        be undone by the stale snapshot: the topic went back to RESEARCHING (or was
        marked READY while still initializing) on the strength of an unrelated
        rename.
        """
        from app.crud import get_topic, update_topic_init_status

        topic = _make_topic(db_conn, name="Racing", status=TopicStatus.RESEARCHING)

        async def _validate_then_finish_init(feed_mode, feed_urls, check_interval):
            # Stand in for the initializer committing READY during the DNS await.
            update_topic_init_status(
                db_conn,
                topic.id,
                status=TopicStatus.READY,
                status_changed_at=datetime.now(UTC),
                error_message=None,
                init_attempts=0,
            )
            db_conn.commit()
            return FeedMode.AUTO, [], None, []

        with patch("app.web.routers.topics.validate_topic_form", new=_validate_then_finish_init):
            response = await client.post(
                f"/topics/{topic.id}/edit",
                data={"name": "Renamed", "description": "d", "feed_mode": "auto"},
                follow_redirects=False,
            )

        assert response.status_code == 303
        updated = get_topic(db_conn, topic.id)
        assert updated.name == "Renamed"
        assert updated.status == TopicStatus.READY
        assert updated.error_message is None

    async def test_edit_persists_novelty_instruction(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """POST to edit persists the novelty instruction (guards the silent-drop trap).

        create_topic and update_topic bind named params from to_insert_dict(), and
        sqlite3 silently ignores surplus keys — so a missing handler assignment or a
        missing SET column would drop the field on edit with no error while create
        still works. This asserts the full edit round-trip, then clearing back to NULL.
        """
        topic = _make_topic(db_conn, name="Instructable")
        response = await client.post(
            f"/topics/{topic.id}/edit",
            data={
                "name": "Instructable",
                "description": topic.description,
                "feed_mode": "manual",
                "feed_urls": "https://example.com/feed.xml",
                "novelty_instruction": "Ignore rumors; official sources only.",
            },
            follow_redirects=False,
        )
        assert response.status_code == 303

        from app.crud import get_topic

        updated = get_topic(db_conn, topic.id)
        assert updated is not None
        assert updated.novelty_instruction == "Ignore rumors; official sources only."

        response = await client.post(
            f"/topics/{topic.id}/edit",
            data={
                "name": "Instructable",
                "description": topic.description,
                "feed_mode": "manual",
                "feed_urls": "https://example.com/feed.xml",
                "novelty_instruction": "",
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        cleared = get_topic(db_conn, topic.id)
        assert cleared is not None
        assert cleared.novelty_instruction is None

    async def test_edit_form_repopulates_novelty_instruction(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """GET edit form shows the stored novelty instruction in the textarea."""
        topic = _make_topic(db_conn, name="Prefilled", novelty_instruction="Official sources only.")
        response = await client.get(f"/topics/{topic.id}/edit")
        assert response.status_code == 200
        assert "Official sources only." in response.text

    async def test_edit_persists_importance_threshold(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """POST to edit persists the importance threshold, then clears it back to NULL."""
        topic = _make_topic(db_conn, name="ImportanceEdit")
        base = {
            "name": "ImportanceEdit",
            "description": topic.description,
            "feed_mode": "manual",
            "feed_urls": "https://example.com/feed.xml",
        }
        response = await client.post(
            f"/topics/{topic.id}/edit", data={**base, "importance_threshold": "3"}, follow_redirects=False
        )
        assert response.status_code == 303

        from app.crud import get_topic

        updated = get_topic(db_conn, topic.id)
        assert updated is not None
        assert updated.importance_threshold == 3

        response = await client.post(
            f"/topics/{topic.id}/edit", data={**base, "importance_threshold": ""}, follow_redirects=False
        )
        assert response.status_code == 303
        cleared = get_topic(db_conn, topic.id)
        assert cleared is not None
        assert cleared.importance_threshold is None

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


@asynccontextmanager
async def _exa_client(db_conn: sqlite3.Connection, *, enabled: bool):
    """A test client whose settings have Exa enabled/disabled (with a key when enabled)."""
    settings = _make_settings(exa=ExaSettings(enabled=enabled, api_key="exa-key" if enabled else ""))

    def override_db():
        yield db_conn

    def override_settings():
        return settings

    app.dependency_overrides[get_db_conn] = override_db
    app.dependency_overrides[get_settings] = override_settings
    try:
        with patch("app.web.routers.settings.load_settings", return_value=settings):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
                cookies={"csrf_token": CSRF_TEST_TOKEN},
                headers={"X-CSRF-Token": CSRF_TEST_TOKEN},
            ) as ac:
                yield ac
    finally:
        app.dependency_overrides.clear()


class TestExaFeedModeWeb:
    """Web layer for the EXA feed mode: create/edit guards, strict validation, detail render."""

    async def test_create_exa_topic_when_enabled(self, db_conn: sqlite3.Connection) -> None:
        async with _exa_client(db_conn, enabled=True) as client:
            with patch("app.web.routers.background._run_init", new_callable=AsyncMock):
                response = await client.post(
                    "/topics",
                    data={"name": "Exa Topic", "description": "d", "feed_mode": "exa", "feed_urls": "ignored"},
                    follow_redirects=False,
                )
        assert response.status_code == 303
        from app.crud import get_topic_by_name

        topic = get_topic_by_name(db_conn, "Exa Topic")
        assert topic is not None
        assert topic.feed_mode == FeedMode.EXA
        assert topic.feed_urls == []  # EXA carries no manual feed URLs

    async def test_create_exa_topic_rejected_when_disabled(self, db_conn: sqlite3.Connection) -> None:
        async with _exa_client(db_conn, enabled=False) as client:
            response = await client.post(
                "/topics",
                data={"name": "Exa Off", "description": "d", "feed_mode": "exa"},
                follow_redirects=False,
            )
        assert response.status_code == 422
        assert "Exa search is not enabled" in response.text
        from app.crud import get_topic_by_name

        assert get_topic_by_name(db_conn, "Exa Off") is None

    async def test_unknown_feed_mode_rejected(self, db_conn: sqlite3.Connection) -> None:
        async with _exa_client(db_conn, enabled=True) as client:
            response = await client.post(
                "/topics",
                data={"name": "Bad Mode", "description": "d", "feed_mode": "bogus"},
                follow_redirects=False,
            )
        assert response.status_code == 422
        assert "Invalid feed mode" in response.text

    async def test_edit_conversion_into_exa_rejected_when_disabled(self, db_conn: sqlite3.Connection) -> None:
        """Converting a working AUTO topic into EXA while Exa is off is blocked."""
        topic = _make_topic(db_conn, name="Auto2Exa", feed_mode=FeedMode.AUTO, feed_urls=[])
        async with _exa_client(db_conn, enabled=False) as client:
            response = await client.post(
                f"/topics/{topic.id}/edit",
                data={"name": "Auto2Exa", "description": "d", "feed_mode": "exa"},
                follow_redirects=False,
            )
        assert response.status_code == 422
        assert "Exa search is not enabled" in response.text
        from app.crud import get_topic

        assert get_topic(db_conn, topic.id).feed_mode == FeedMode.AUTO  # unchanged

    async def test_edit_already_exa_topic_allowed_when_disabled(self, db_conn: sqlite3.Connection) -> None:
        """An already-EXA topic can still be edited while Exa is disabled (degrades gracefully)."""
        topic = _make_topic(db_conn, name="StaysExa", feed_mode=FeedMode.EXA, feed_urls=[])
        async with _exa_client(db_conn, enabled=False) as client:
            response = await client.post(
                f"/topics/{topic.id}/edit",
                data={"name": "StaysExa", "description": "updated desc", "feed_mode": "exa"},
                follow_redirects=False,
            )
        assert response.status_code == 303
        from app.crud import get_topic

        updated = get_topic(db_conn, topic.id)
        assert updated.feed_mode == FeedMode.EXA
        assert updated.description == "updated desc"

    async def test_detail_page_labels_exa_mode(self, db_conn: sqlite3.Connection) -> None:
        topic = _make_topic(db_conn, name="ExaDetail", feed_mode=FeedMode.EXA, feed_urls=[])
        async with _exa_client(db_conn, enabled=True) as client:
            response = await client.get(f"/topics/{topic.id}")
        assert response.status_code == 200
        assert "Exa AI search" in response.text
        assert "No feed URLs configured" not in response.text

    async def test_feed_source_fragment_for_exa(self, db_conn: sqlite3.Connection) -> None:
        topic = _make_topic(db_conn, name="ExaFrag", feed_mode=FeedMode.EXA, feed_urls=[])
        async with _exa_client(db_conn, enabled=True) as client:
            response = await client.get(f"/topics/{topic.id}/feed-source")
        assert response.status_code == 200
        assert "Exa AI semantic search" in response.text
        assert "No feed URLs configured" not in response.text
