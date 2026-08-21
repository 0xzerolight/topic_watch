"""Tests for the Exa AI search source (app/scraping/exa.py) and EXA-mode dispatch."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import httpx

from app.config import ExaSettings
from app.crud import create_topic, get_feed_health, list_articles_for_topic
from app.models import FeedMode, Topic
from app.scraping import fetch_new_articles_for_topic
from app.scraping.exa import _map_exa_result, fetch_exa_entries
from app.scraping.rss import compute_article_hash, fetch_feeds_for_topic
from app.scraping.source import FeedHealthOutcome, FetchStatus

_EXA_TOPIC = Topic(name="AI safety", description="news about AI safety", feed_mode=FeedMode.EXA, feed_urls=[])
_ENABLED = ExaSettings(enabled=True, api_key="test-exa-key")
_EXA_ENDPOINT = "https://api.exa.ai/search"  # default effective endpoint (base_url + /search)


class _Recorder:
    """Records the typed feed-health outcomes a fetch reports."""

    def __init__(self) -> None:
        self.calls: list[FeedHealthOutcome] = []

    def __call__(self, outcome: FeedHealthOutcome) -> None:
        self.calls.append(outcome)


def _exa_response(results: list[object]) -> httpx.MockTransport:
    """A MockTransport returning a canned Exa /search JSON payload."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": results})

    return httpx.MockTransport(handler)


class TestMapExaResult:
    def test_maps_core_fields(self) -> None:
        entry = _map_exa_result(
            {"url": "https://x.com/a", "title": "One", "publishedDate": "2024-01-02T03:04:05Z", "text": "full body"}
        )
        assert entry is not None
        assert entry.url == "https://x.com/a"
        assert entry.title == "One"
        assert entry.source_feed == "exa"
        assert entry.summary == ""  # summary means "RSS summary"; Exa text rides on content
        assert entry.content == "full body"
        assert entry.published is not None and entry.published.tzinfo is not None

    def test_date_only_is_made_tz_aware(self) -> None:
        entry = _map_exa_result({"url": "https://x.com/a", "title": "T", "publishedDate": "2023-11-15"})
        assert entry is not None
        assert entry.published is not None
        assert entry.published.tzinfo is not None

    def test_non_string_date_survives_as_none(self) -> None:
        """A non-string publishedDate degrades to None; the result is still kept."""
        entry = _map_exa_result({"url": "https://x.com/a", "title": "T", "publishedDate": 1699999999, "text": "b"})
        assert entry is not None
        assert entry.published is None

    def test_missing_url_or_title_dropped(self) -> None:
        assert _map_exa_result({"url": "https://x.com/a"}) is None
        assert _map_exa_result({"title": "no url"}) is None
        assert _map_exa_result({"url": "", "title": "blank"}) is None

    def test_non_http_url_dropped(self) -> None:
        assert _map_exa_result({"url": "javascript:alert(1)", "title": "x"}) is None
        assert _map_exa_result({"url": "data:text/html,x", "title": "x"}) is None

    def test_empty_text_yields_none_content(self) -> None:
        entry = _map_exa_result({"url": "https://x.com/a", "title": "T", "text": ""})
        assert entry is not None
        assert entry.content is None

    def test_whitespace_only_text_yields_none_content(self) -> None:
        """AUG-308: blank text is truthy, so it would short-circuit the publisher fetch."""
        entry = _map_exa_result({"url": "https://x.com/a", "title": "T", "text": "   \n\t "})
        assert entry is not None
        assert entry.content is None

    def test_non_string_text_keeps_the_row(self) -> None:
        """AUG-308: a structured ``text`` value must not discard a usable URL/title."""
        entry = _map_exa_result({"url": "https://x.com/a", "title": "T", "text": {"value": "x"}})
        assert entry is not None
        assert entry.url == "https://x.com/a"
        assert entry.content is None

    def test_text_keeps_its_own_whitespace(self) -> None:
        """Only the outer padding is trimmed; the body itself is untouched."""
        entry = _map_exa_result({"url": "https://x.com/a", "title": "T", "text": "  one\n\ntwo  "})
        assert entry is not None
        assert entry.content == "one\n\ntwo"


class TestFetchExaEntries:
    async def test_request_contract(self) -> None:
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["path"] = request.url.path
            captured["api_key"] = request.headers.get("x-api-key")
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"results": []})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await fetch_exa_entries(_EXA_TOPIC, _ENABLED, max_results=7, timeout=5.0, client=client)

        assert captured["path"].endswith("/search")
        assert captured["api_key"] == "test-exa-key"
        body = captured["body"]
        assert body["query"] == "AI safety news about AI safety"
        assert body["numResults"] == 7
        assert body["type"] == "auto"
        assert body["category"] == "news"
        assert body["contents"]["text"]["maxCharacters"] == 5000

    async def test_maps_results_to_entries(self) -> None:
        transport = _exa_response(
            [
                {"url": "https://x.com/1", "title": "One", "publishedDate": "2024-01-01T00:00:00Z", "text": "one"},
                {"url": "https://x.com/2", "title": "Two", "text": "two"},
            ]
        )
        async with httpx.AsyncClient(transport=transport) as client:
            resp = await fetch_exa_entries(_EXA_TOPIC, _ENABLED, max_results=5, timeout=5.0, client=client)
        assert resp.provider_name == "exa"
        assert resp.needs_url_resolution is False
        assert resp.feeds_total == 1 and resp.feeds_failed == 0
        assert [e.url for e in resp.entries] == ["https://x.com/1", "https://x.com/2"]
        assert resp.entries[0].content == "one"

    async def test_malformed_result_isolated(self) -> None:
        """One unusable result does not zero out the valid ones."""
        transport = _exa_response(
            [
                {"url": "https://x.com/good", "title": "Good", "text": "g"},
                {"title": "no url"},  # dropped
                "not-a-dict",  # would raise in mapping -> skipped
            ]
        )
        async with httpx.AsyncClient(transport=transport) as client:
            resp = await fetch_exa_entries(_EXA_TOPIC, _ENABLED, max_results=5, timeout=5.0, client=client)
        assert [e.url for e in resp.entries] == ["https://x.com/good"]
        assert resp.feeds_failed == 0

    async def test_healthy_empty(self) -> None:
        async with httpx.AsyncClient(transport=_exa_response([])) as client:
            resp = await fetch_exa_entries(_EXA_TOPIC, _ENABLED, max_results=5, timeout=5.0, client=client)
        assert resp.entries == []
        assert resp.feeds_total == 1 and resp.feeds_failed == 0

    async def test_a_batch_with_no_usable_result_is_a_failure(self) -> None:
        """AUG-307: rows that all fail to map are a protocol failure, not quiet news.

        Recorded as success it reset the source's failure state and told the
        silence heartbeat this was a healthy empty check, while monitoring
        received nothing at all.
        """
        recorder = _Recorder()
        transport = _exa_response([{"title": "no url"}, "not-a-dict"])
        async with httpx.AsyncClient(transport=transport) as client:
            resp = await fetch_exa_entries(
                _EXA_TOPIC, _ENABLED, max_results=5, timeout=5.0, client=client, health_callback=recorder
            )
        assert resp.entries == []
        assert resp.feeds_total == 1 and resp.feeds_failed == 1
        assert [c.status for c in recorder.calls] == [FetchStatus.FAILED]

    async def test_a_malformed_envelope_is_one_failed_fetch(self) -> None:
        """AUG-174: schema drift is recorded, never raised out of a never-raises call.

        ``results: null`` used to raise during iteration and take the whole check
        down; a non-object envelope or a non-list ``results`` was accepted and
        cleared the source's failure state with no articles to show for it.
        """
        for payload in ({"results": None}, ["not", "an", "object"], {"results": "rows"}, {"results": {"a": 1}}):
            recorder = _Recorder()

            def handler(request: httpx.Request, body: object = payload) -> httpx.Response:
                return httpx.Response(200, json=body)

            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                resp = await fetch_exa_entries(
                    _EXA_TOPIC, _ENABLED, max_results=5, timeout=5.0, client=client, health_callback=recorder
                )
            assert resp.entries == [], payload
            assert resp.feeds_failed == 1, payload
            assert [c.status for c in recorder.calls] == [FetchStatus.FAILED], payload

    async def test_date_mix_flows_through_select_candidates(self) -> None:
        """Mixed publishedDate shapes all normalize so recency sort never raises (load-bearing)."""
        from app.scraping import _prepare_entries, _select_candidates

        transport = _exa_response(
            [
                {"url": "https://x.com/1", "title": "date-only", "publishedDate": "2023-11-15"},
                {"url": "https://x.com/2", "title": "full-z", "publishedDate": "2024-06-01T12:00:00Z"},
                {"url": "https://x.com/3", "title": "null-date", "publishedDate": None},
                {"url": "https://x.com/4", "title": "int-date", "publishedDate": 1699999999},
            ]
        )
        async with httpx.AsyncClient(transport=transport) as client:
            resp = await fetch_exa_entries(_EXA_TOPIC, _ENABLED, max_results=10, timeout=5.0, client=client)
        assert len(resp.entries) == 4
        prepared = _prepare_entries(resp.entries)
        new_entries = [(e, compute_article_hash(e.url, e.title)) for e in prepared]
        reuse_batch, fetch_batch = _select_candidates(new_entries, [], 10)  # must not raise
        urls = [e.url for e, _ in fetch_batch]
        # The full-Z 2024 entry is the newest dated one, so it leads; the two
        # undated entries take the date of the entry they follow (AUG-184), which
        # keeps them beside it and ahead of the 2023 one.
        assert urls[0] == "https://x.com/2"
        assert urls.index("https://x.com/2") < urls.index("https://x.com/1")
        assert urls.index("https://x.com/3") < urls.index("https://x.com/1")

    async def test_not_enabled_makes_no_request(self) -> None:
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url))
            return httpx.Response(200, json={"results": []})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            resp = await fetch_exa_entries(
                _EXA_TOPIC, ExaSettings(enabled=False, api_key="k"), max_results=5, timeout=5.0, client=client
            )
        assert resp.feeds_total == 0 and resp.feeds_failed == 0
        assert calls == []

    async def test_no_key_makes_no_request(self) -> None:
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url))
            return httpx.Response(200, json={"results": []})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            resp = await fetch_exa_entries(
                _EXA_TOPIC, ExaSettings(enabled=True, api_key=""), max_results=5, timeout=5.0, client=client
            )
        assert resp.feeds_total == 0 and resp.feeds_failed == 0
        assert calls == []

    async def test_spent_deadline_makes_no_request(self) -> None:
        """TW-AUD-018: Exa draws on the topic's source budget like any other fetch."""
        from unittest.mock import MagicMock

        from app.scraping.source import DEADLINE_ERROR, Deadline

        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url))
            return httpx.Response(200, json={"results": []})

        callback = MagicMock()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            resp = await fetch_exa_entries(
                _EXA_TOPIC,
                _ENABLED,
                max_results=5,
                timeout=5.0,
                client=client,
                health_callback=callback,
                deadline=Deadline.after(-1.0),
            )

        assert resp.feeds_total == 1 and resp.feeds_failed == 1
        assert resp.entries == []
        assert calls == []
        assert DEADLINE_ERROR in callback.call_args[0][0].error_msg

    async def test_http_4xx_fails_safe(self) -> None:
        transport = httpx.MockTransport(lambda r: httpx.Response(401, json={"error": "bad key"}))
        async with httpx.AsyncClient(transport=transport) as client:
            resp = await fetch_exa_entries(_EXA_TOPIC, _ENABLED, max_results=5, timeout=5.0, client=client)
        assert resp.feeds_total == 1 and resp.feeds_failed == 1
        assert resp.entries == []

    async def test_http_5xx_fails_safe(self) -> None:
        transport = httpx.MockTransport(lambda r: httpx.Response(503, text="down"))
        async with httpx.AsyncClient(transport=transport) as client:
            resp = await fetch_exa_entries(_EXA_TOPIC, _ENABLED, max_results=5, timeout=5.0, client=client)
        assert resp.feeds_failed == 1

    async def test_timeout_fails_safe(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.TimeoutException("timed out", request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            resp = await fetch_exa_entries(_EXA_TOPIC, _ENABLED, max_results=5, timeout=5.0, client=client)
        assert resp.feeds_failed == 1

    async def test_invalid_json_fails_safe(self) -> None:
        transport = httpx.MockTransport(lambda r: httpx.Response(200, text="not json {{{"))
        async with httpx.AsyncClient(transport=transport) as client:
            resp = await fetch_exa_entries(_EXA_TOPIC, _ENABLED, max_results=5, timeout=5.0, client=client)
        assert resp.feeds_failed == 1

    async def test_private_endpoint_blocked(self) -> None:
        """A base_url on a private host is blocked before any request (SSRF)."""
        # A real RFC-1918 literal, so the gate runs its own classification rather
        # than a patched verdict.
        settings = ExaSettings(enabled=True, api_key="k", base_url="https://192.168.1.50")
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url))
            return httpx.Response(200, json={"results": []})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            resp = await fetch_exa_entries(_EXA_TOPIC, settings, max_results=5, timeout=5.0, client=client)
        assert resp.feeds_total == 1 and resp.feeds_failed == 1
        assert calls == []

    async def test_non_http_base_url_blocked(self) -> None:
        """A non-http(s) base_url is blocked by the scheme allowlist before any request."""
        settings = ExaSettings(enabled=True, api_key="k", base_url="ftp://host")
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url))
            return httpx.Response(200, json={"results": []})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            resp = await fetch_exa_entries(_EXA_TOPIC, settings, max_results=5, timeout=5.0, client=client)
        assert resp.feeds_total == 1 and resp.feeds_failed == 1
        assert calls == []


class TestExaHealthCallback:
    """Feed-health recording: every attempted fetch records one typed outcome."""

    async def test_success_records_healthy(self) -> None:
        rec = _Recorder()
        async with httpx.AsyncClient(transport=_exa_response([])) as client:
            await fetch_exa_entries(
                _EXA_TOPIC, _ENABLED, max_results=5, timeout=5.0, client=client, health_callback=rec
            )
        assert rec.calls == [FeedHealthOutcome(_EXA_ENDPOINT, FetchStatus.EMPTY)]

    async def test_http_error_records_failure(self) -> None:
        rec = _Recorder()
        transport = httpx.MockTransport(lambda r: httpx.Response(401, json={"error": "bad key"}))
        async with httpx.AsyncClient(transport=transport) as client:
            await fetch_exa_entries(
                _EXA_TOPIC, _ENABLED, max_results=5, timeout=5.0, client=client, health_callback=rec
            )
        assert len(rec.calls) == 1
        outcome = rec.calls[0]
        assert outcome.feed_url == _EXA_ENDPOINT
        assert outcome.status is FetchStatus.FAILED
        assert outcome.error_msg  # non-empty reason
        assert outcome.etag is None and outcome.last_modified is None

    async def test_timeout_records_failure(self) -> None:
        rec = _Recorder()

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.TimeoutException("timed out", request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await fetch_exa_entries(
                _EXA_TOPIC, _ENABLED, max_results=5, timeout=5.0, client=client, health_callback=rec
            )
        assert len(rec.calls) == 1
        assert rec.calls[0].feed_url == _EXA_ENDPOINT
        assert rec.calls[0].status is FetchStatus.FAILED
        assert rec.calls[0].error_msg

    async def test_generic_error_records_failure(self) -> None:
        """Invalid JSON hits the generic except path and records a failure."""
        rec = _Recorder()
        transport = httpx.MockTransport(lambda r: httpx.Response(200, text="not json {{{"))
        async with httpx.AsyncClient(transport=transport) as client:
            await fetch_exa_entries(
                _EXA_TOPIC, _ENABLED, max_results=5, timeout=5.0, client=client, health_callback=rec
            )
        assert len(rec.calls) == 1
        assert rec.calls[0].status is FetchStatus.FAILED
        assert rec.calls[0].error_msg

    async def test_non_http_endpoint_records_failure(self) -> None:
        rec = _Recorder()
        settings = ExaSettings(enabled=True, api_key="k", base_url="ftp://host")
        async with httpx.AsyncClient(transport=_exa_response([])) as client:
            await fetch_exa_entries(
                _EXA_TOPIC, settings, max_results=5, timeout=5.0, client=client, health_callback=rec
            )
        assert len(rec.calls) == 1
        assert rec.calls[0].feed_url == "ftp://host/search"
        assert rec.calls[0].status is FetchStatus.FAILED
        assert rec.calls[0].error_msg

    async def test_private_endpoint_records_failure(self) -> None:
        rec = _Recorder()
        settings = ExaSettings(enabled=True, api_key="k", base_url="https://192.168.1.50")
        async with httpx.AsyncClient(transport=_exa_response([])) as client:
            await fetch_exa_entries(
                _EXA_TOPIC, settings, max_results=5, timeout=5.0, client=client, health_callback=rec
            )
        assert len(rec.calls) == 1
        assert rec.calls[0].feed_url == "https://192.168.1.50/search"
        assert rec.calls[0].status is FetchStatus.FAILED
        assert rec.calls[0].error_msg

    async def test_malformed_endpoint_records_failure(self) -> None:
        """The endpoint-gate's unexpected-exception path records a failure."""
        rec = _Recorder()
        with patch("app.url_validation._classify_url", side_effect=RuntimeError):
            async with httpx.AsyncClient(transport=_exa_response([])) as client:
                await fetch_exa_entries(
                    _EXA_TOPIC, _ENABLED, max_results=5, timeout=5.0, client=client, health_callback=rec
                )
        assert len(rec.calls) == 1
        assert rec.calls[0].feed_url == _EXA_ENDPOINT
        assert rec.calls[0].status is FetchStatus.FAILED
        assert rec.calls[0].error_msg

    async def test_disabled_records_nothing(self) -> None:
        rec = _Recorder()
        async with httpx.AsyncClient(transport=_exa_response([])) as client:
            await fetch_exa_entries(
                _EXA_TOPIC,
                ExaSettings(enabled=False, api_key="k"),
                max_results=5,
                timeout=5.0,
                client=client,
                health_callback=rec,
            )
        assert rec.calls == []

    async def test_no_key_records_nothing(self) -> None:
        rec = _Recorder()
        async with httpx.AsyncClient(transport=_exa_response([])) as client:
            await fetch_exa_entries(
                _EXA_TOPIC,
                ExaSettings(enabled=True, api_key=""),
                max_results=5,
                timeout=5.0,
                client=client,
                health_callback=rec,
            )
        assert rec.calls == []

    async def test_none_callback_all_paths_safe(self) -> None:
        """Default health_callback=None: both success and failure paths still work."""
        async with httpx.AsyncClient(transport=_exa_response([])) as client:
            ok = await fetch_exa_entries(_EXA_TOPIC, _ENABLED, max_results=5, timeout=5.0, client=client)
        assert ok.feeds_failed == 0
        transport = httpx.MockTransport(lambda r: httpx.Response(500, text="down"))
        async with httpx.AsyncClient(transport=transport) as client:
            bad = await fetch_exa_entries(_EXA_TOPIC, _ENABLED, max_results=5, timeout=5.0, client=client)
        assert bad.feeds_failed == 1


class TestExaDispatch:
    async def test_exa_mode_routes_to_exa(self) -> None:
        transport = _exa_response([{"url": "https://x.com/1", "title": "One", "text": "one"}])
        original_init = httpx.AsyncClient.__init__

        def patched_init(self_client, **kwargs):
            kwargs["transport"] = transport
            original_init(self_client, **kwargs)

        with patch.object(httpx.AsyncClient, "__init__", patched_init):
            resp = await fetch_feeds_for_topic(_EXA_TOPIC, exa_settings=_ENABLED, max_results=5)
        assert resp.provider_name == "exa"
        assert [e.url for e in resp.entries] == ["https://x.com/1"]

    async def test_exa_mode_without_settings_returns_empty(self) -> None:
        resp = await fetch_feeds_for_topic(_EXA_TOPIC, exa_settings=None)
        assert resp.provider_name == "exa"
        assert resp.feeds_total == 0 and resp.feeds_failed == 0
        assert resp.entries == []


class TestExaPipelineStore:
    async def test_stores_exa_articles_with_provider_and_prefetched_content(
        self, db_conn: sqlite3.Connection, db_path: Path
    ) -> None:
        """End to end: Exa text lands as raw_content, source_provider='exa', published_at tz-aware."""
        topic = create_topic(db_conn, Topic(name="AI", description="ai news", feed_mode=FeedMode.EXA, feed_urls=[]))
        db_conn.commit()
        assert topic.id is not None

        transport = _exa_response(
            [{"url": "https://x.com/1", "title": "One", "publishedDate": "2024-01-01T00:00:00Z", "text": "exa body"}]
        )
        original_init = httpx.AsyncClient.__init__

        def patched_init(self_client, **kwargs):
            kwargs["transport"] = transport
            original_init(self_client, **kwargs)

        with patch.object(httpx.AsyncClient, "__init__", patched_init):
            result = await fetch_new_articles_for_topic(topic, db_path=db_path, max_articles=5, exa_settings=_ENABLED)

        assert len(result.articles) == 1
        stored = list_articles_for_topic(db_conn, topic.id)
        assert len(stored) == 1
        assert stored[0].source_provider == "exa"
        assert stored[0].raw_content == "exa body"  # prefetched, not a second fetch
        assert isinstance(stored[0].published_at, datetime)
        assert stored[0].published_at.tzinfo is not None

    async def test_check_records_exa_feed_health(self, db_conn: sqlite3.Connection, db_path: Path) -> None:
        """An EXA-mode check writes a feed_health row keyed on the Exa endpoint."""
        topic = create_topic(db_conn, Topic(name="AI", description="ai news", feed_mode=FeedMode.EXA, feed_urls=[]))
        db_conn.commit()

        transport = _exa_response([{"url": "https://x.com/1", "title": "One", "text": "b"}])
        original_init = httpx.AsyncClient.__init__

        def patched_init(self_client, **kwargs):
            kwargs["transport"] = transport
            original_init(self_client, **kwargs)

        with patch.object(httpx.AsyncClient, "__init__", patched_init):
            await fetch_new_articles_for_topic(topic, db_path=db_path, max_articles=5, exa_settings=_ENABLED)

        health = get_feed_health(db_conn, _EXA_ENDPOINT)
        assert health is not None
        assert health.consecutive_failures == 0
        assert health.total_fetches == 1
        assert health.last_success_at is not None


class TestExaEndpointTransport:
    """A public Exa override must not carry the paid API key in cleartext (AUG-304)."""

    async def test_public_http_base_url_blocked(self) -> None:
        settings = ExaSettings(enabled=True, api_key="k", base_url="http://proxy.example.com")
        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(200, json={"results": []})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            resp = await fetch_exa_entries(_EXA_TOPIC, settings, max_results=5, timeout=5.0, client=client)

        assert resp.feeds_total == 1 and resp.feeds_failed == 1
        assert calls == []

    async def test_https_base_url_still_allowed(self) -> None:
        settings = ExaSettings(enabled=True, api_key="k", base_url="https://proxy.example.com")
        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(200, json={"results": []})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            resp = await fetch_exa_entries(_EXA_TOPIC, settings, max_results=5, timeout=5.0, client=client)

        assert resp.feeds_failed == 0
        assert len(calls) == 1
