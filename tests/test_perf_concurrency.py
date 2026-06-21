"""Performance tasks: DNS offload + bounded concurrency (OVH-052..056).

Covers:
  * OVH-052: dashboard reads confidence via SQL json_extract, never shipping the
    full llm_response blob per topic.
  * OVH-053: OPML validation pass resolves the deduped URL set with bounded
    concurrency while still validating every URL through validate_feed_url.
  * OVH-054: topic create/edit handlers offload blocking getaddrinfo to a thread.
  * OVH-055: per-topic checks are bounded by topic_check_concurrency.
  * OVH-056: Google News resolution parallelizes under a small Semaphore and
    still aborts on the first 429.
"""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.config import Settings
from app.crud import create_check_result, create_topic, get_dashboard_data
from app.models import CheckResult, Topic, TopicStatus


def _make_topic(conn: sqlite3.Connection, name: str = "Perf Topic") -> Topic:
    topic = create_topic(
        conn,
        Topic(
            name=name,
            description="desc",
            status=TopicStatus.READY,
            status_changed_at=datetime.now(UTC),
        ),
    )
    # Commit so the connection holds no write lock while check_all_topics opens
    # its own short-lived connections (otherwise the fixture conn deadlocks them).
    conn.commit()
    return topic


# --- OVH-052: dashboard confidence via json_extract ---------------------------


class TestDashboardConfidenceExtract:
    def test_dashboard_select_uses_json_extract(self) -> None:
        """The dashboard SELECT extracts confidence in SQL, not the blob."""
        from app.crud import _DASHBOARD_SELECT

        assert "json_extract(cr.llm_response, '$.confidence')" in _DASHBOARD_SELECT
        # The full blob column must NOT be shipped on the dashboard path.
        assert "cr.llm_response AS cr_llm_response" not in _DASHBOARD_SELECT

    def test_dashboard_confidence_populated_without_blob(self, db_conn: sqlite3.Connection) -> None:
        """get_dashboard_data exposes the confidence scalar but not the blob."""
        topic = _make_topic(db_conn)
        blob = json.dumps(
            {
                "has_new_info": True,
                "confidence": 0.77,
                "reasoning": "x" * 5000,  # large payload we must NOT ship
                "summary": "y" * 5000,
                "key_facts": ["z" * 1000],
            }
        )
        create_check_result(db_conn, CheckResult(topic_id=topic.id, llm_response=blob, has_new_info=True))
        db_conn.commit()

        data = get_dashboard_data(db_conn)
        assert len(data) == 1
        last_check = data[0]["last_check"]
        assert last_check is not None
        # Confidence extracted by SQL...
        assert last_check.confidence == pytest.approx(0.77)
        # ...and the multi-KB blob never loaded into the dashboard CheckResult.
        assert last_check.llm_response is None

    def test_dashboard_confidence_none_when_no_check(self, db_conn: sqlite3.Connection) -> None:
        _make_topic(db_conn)
        db_conn.commit()
        data = get_dashboard_data(db_conn)
        assert data[0]["last_check"] is None

    def test_confidence_value_filter_renders_badge(self) -> None:
        from app.web.routers.templates import _confidence_value

        assert "#2ecc40" in _confidence_value(0.9)
        assert "#ffdc00" in _confidence_value(0.6)
        assert "#ff4136" in _confidence_value(0.2)
        assert _confidence_value(None) == "-"


# --- OVH-053: OPML concurrent bounded validation ------------------------------


class TestOPMLConcurrentValidation:
    def _opml(self, n: int) -> str:
        outlines = "\n".join(f'<outline text="Feed {i}" xmlUrl="https://example{i}.com/feed" />' for i in range(n))
        return f'<?xml version="1.0"?><opml version="2.0"><body>{outlines}</body></opml>'

    def test_validation_uses_bounded_thread_pool(self) -> None:
        """The validation pass resolves URLs concurrently via a bounded pool."""
        import app.opml as opml_mod

        # The OPML module must declare a bounded validation concurrency cap.
        assert 10 <= opml_mod._VALIDATION_CONCURRENCY <= 20

    def test_all_urls_validated_concurrently(self) -> None:
        """Every deduped URL still flows through validate_feed_url (SSRF)."""
        import threading
        import time

        from app import opml as opml_mod

        seen: list[str] = []
        max_in_flight = 0
        in_flight = 0
        lock = threading.Lock()

        def fake_validate(url: str) -> None:
            nonlocal in_flight, max_in_flight
            with lock:
                in_flight += 1
                max_in_flight = max(max_in_flight, in_flight)
                seen.append(url)
            time.sleep(0.02)  # hold the slot so concurrency is observable
            with lock:
                in_flight -= 1
            return None

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(opml_mod, "validate_feed_url", fake_validate)
            result = opml_mod.parse_opml(self._opml(12), set())

        # Every URL validated exactly once.
        assert len(seen) == 12
        assert len(set(seen)) == 12
        assert len(result.topics) == 12
        # Concurrency actually happened (more than one validate in flight at once).
        assert max_in_flight > 1

    def test_concurrency_is_bounded(self) -> None:
        """No more than _VALIDATION_CONCURRENCY validations run at once."""
        import threading
        import time

        from app import opml as opml_mod

        max_in_flight = 0
        in_flight = 0
        lock = threading.Lock()

        def fake_validate(url: str) -> None:
            nonlocal in_flight, max_in_flight
            with lock:
                in_flight += 1
                max_in_flight = max(max_in_flight, in_flight)
            time.sleep(0.02)
            with lock:
                in_flight -= 1
            return None

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(opml_mod, "validate_feed_url", fake_validate)
            opml_mod.parse_opml(self._opml(60), set())

        assert max_in_flight <= opml_mod._VALIDATION_CONCURRENCY

    def test_invalid_urls_still_skipped(self) -> None:
        from app import opml as opml_mod

        def fake_validate(url: str) -> str | None:
            return "bad" if "bad" in url else None

        opml = (
            '<?xml version="1.0"?><opml version="2.0"><body>'
            '<outline text="Good" xmlUrl="https://good.example.com/feed" />'
            '<outline text="Bad" xmlUrl="https://bad.example.com/feed" />'
            "</body></opml>"
        )
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(opml_mod, "validate_feed_url", fake_validate)
            result = opml_mod.parse_opml(opml, set())

        assert result.skipped_invalid == 1
        assert len(result.topics) == 1
        assert result.topics[0]["name"] == "Good"


# --- OVH-054: topic create/edit DNS offloaded to a thread ---------------------


class TestTopicFormDNSOffload:
    async def test_validate_topic_form_is_async(self) -> None:
        """validate_topic_form must be awaitable so DNS can be threaded."""
        import inspect

        from app.web.routers._validation import validate_topic_form

        assert inspect.iscoroutinefunction(validate_topic_form)

    async def test_dns_validation_runs_off_event_loop(self) -> None:
        """The blocking validate_feed_urls runs in a worker thread, not the loop."""
        import threading

        from app.web.routers import _validation

        main_thread = threading.get_ident()
        ran_on: list[int] = []

        def fake_validate(urls: list[str]) -> list[str]:
            ran_on.append(threading.get_ident())
            return []

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_validation, "validate_feed_urls", fake_validate)
            mode, urls, interval, errors = await _validation.validate_topic_form(
                "manual", "https://example.com/feed", ""
            )

        assert errors == []
        assert urls == ["https://example.com/feed"]
        # validate_feed_urls executed, and NOT on the event-loop thread (OVH-054).
        assert ran_on and ran_on[0] != main_thread

    async def test_auto_mode_skips_dns(self) -> None:
        """AUTO mode has no manual feeds, so no DNS work is scheduled."""
        from app.web.routers import _validation

        called = False

        def fake_validate(urls: list[str]) -> list[str]:
            nonlocal called
            called = True
            return []

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_validation, "validate_feed_urls", fake_validate)
            mode, urls, interval, errors = await _validation.validate_topic_form("auto", "", "")

        assert urls == []
        assert called is False


# --- OVH-055: bounded per-topic check concurrency -----------------------------


class TestTopicCheckConcurrency:
    def test_config_field_exists_with_bounds(self) -> None:
        from pydantic import ValidationError

        s = Settings(
            llm={"model": "openai/gpt-4o-mini", "api_key": "k"},
            notifications={"urls": ["json://localhost"]},
        )
        assert s.topic_check_concurrency == 3  # default 3-5 range
        # Bounds enforced.
        with pytest.raises(ValidationError):
            Settings(
                llm={"model": "openai/gpt-4o-mini", "api_key": "k"},
                notifications={"urls": ["json://localhost"]},
                topic_check_concurrency=0,
            )
        with pytest.raises(ValidationError):
            Settings(
                llm={"model": "openai/gpt-4o-mini", "api_key": "k"},
                notifications={"urls": ["json://localhost"]},
                topic_check_concurrency=999,
            )

    async def test_per_topic_checks_cap_at_n(self, db_conn: sqlite3.Connection, tmp_path: Path) -> None:
        """No more than topic_check_concurrency per-topic checks run at once."""
        import asyncio
        from unittest.mock import patch

        from app.checker import check_all_topics

        for i in range(8):
            _make_topic(db_conn, name=f"Topic {i}")

        settings = Settings(
            llm={"model": "openai/gpt-4o-mini", "api_key": "k"},
            notifications={"urls": ["json://localhost"]},
            topic_check_concurrency=3,
        )

        in_flight = 0
        max_in_flight = 0

        async def fake_check_topic(topic, conn, settings):
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.02)
            in_flight -= 1
            return CheckResult(topic_id=topic.id)

        with patch("app.checker.check_topic", side_effect=fake_check_topic):
            results = await check_all_topics(settings, db_path=tmp_path / "test.db")

        assert len(results) == 8
        assert max_in_flight > 1  # actually parallel
        assert max_in_flight <= 3  # but bounded

    async def test_concurrency_one_is_sequential(self, db_conn: sqlite3.Connection, tmp_path: Path) -> None:
        import asyncio
        from unittest.mock import patch

        from app.checker import check_all_topics

        for i in range(4):
            _make_topic(db_conn, name=f"Seq {i}")

        settings = Settings(
            llm={"model": "openai/gpt-4o-mini", "api_key": "k"},
            notifications={"urls": ["json://localhost"]},
            topic_check_concurrency=1,
        )

        in_flight = 0
        max_in_flight = 0

        async def fake_check_topic(topic, conn, settings):
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return CheckResult(topic_id=topic.id)

        with patch("app.checker.check_topic", side_effect=fake_check_topic):
            results = await check_all_topics(settings, db_path=tmp_path / "test.db")

        assert len(results) == 4
        assert max_in_flight == 1


# --- OVH-056: Google News resolver bounded concurrency + 429 abort ------------


class TestGoogleNewsResolverConcurrency:
    def test_resolver_declares_bounded_concurrency(self) -> None:
        from app.scraping import google_news as gn

        assert 2 <= gn._RESOLVE_CONCURRENCY <= 3

    async def test_resolution_runs_in_parallel(self) -> None:
        """Resolutions overlap (more than one in flight) under the Semaphore."""
        import asyncio
        from unittest.mock import patch

        from app.scraping.google_news import resolve_google_news_urls

        in_flight = 0
        max_in_flight = 0

        async def fake_resolve(url, client):
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.02)
            in_flight -= 1
            return f"https://real.example.com/{url[-12:]}"

        urls = [f"https://news.google.com/rss/articles/CBMiArticle{i}?oc=5" for i in range(6)]
        with patch("app.scraping.google_news._resolve_or_raise", side_effect=fake_resolve):
            resolved = await resolve_google_news_urls(urls, request_delay=0)

        assert len(resolved) == 6
        assert max_in_flight > 1  # genuinely parallel, not serialized

    async def test_concurrency_bounded(self) -> None:
        """Never more than _RESOLVE_CONCURRENCY resolutions at once."""
        import asyncio
        from unittest.mock import patch

        from app.scraping import google_news as gn
        from app.scraping.google_news import resolve_google_news_urls

        in_flight = 0
        max_in_flight = 0

        async def fake_resolve(url, client):
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.02)
            in_flight -= 1
            return f"https://real.example.com/{url[-12:]}"

        urls = [f"https://news.google.com/rss/articles/CBMiArticle{i}?oc=5" for i in range(15)]
        with patch("app.scraping.google_news._resolve_or_raise", side_effect=fake_resolve):
            await resolve_google_news_urls(urls, request_delay=0)

        assert max_in_flight <= gn._RESOLVE_CONCURRENCY

    async def test_429_aborts_remaining(self) -> None:
        """First 429 short-circuits not-yet-started resolutions."""
        from unittest.mock import patch

        from app.scraping.google_news import _RESOLVE_CONCURRENCY, _RateLimitedError, resolve_google_news_urls

        attempts = 0

        async def fake_resolve(url, client):
            nonlocal attempts
            attempts += 1
            raise _RateLimitedError

        urls = [f"https://news.google.com/rss/articles/CBMiArticle{i}?oc=5" for i in range(20)]
        with patch("app.scraping.google_news._resolve_or_raise", side_effect=fake_resolve):
            resolved = await resolve_google_news_urls(urls, request_delay=0)

        assert resolved == {}
        assert attempts < 20  # aborted, did not attempt all
        assert attempts <= _RESOLVE_CONCURRENCY
