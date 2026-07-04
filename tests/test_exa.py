"""Tests for the Exa AI search source (app/scraping/exa.py) and EXA-mode dispatch."""

import json
import sqlite3
from datetime import datetime
from unittest.mock import patch

import httpx

from app.config import ExaSettings
from app.crud import create_topic, list_articles_for_topic
from app.models import FeedMode, Topic
from app.scraping import fetch_new_articles_for_topic
from app.scraping.exa import _map_exa_result, fetch_exa_entries
from app.scraping.rss import compute_article_hash, fetch_feeds_for_topic

_EXA_TOPIC = Topic(name="AI safety", description="news about AI safety", feed_mode=FeedMode.EXA, feed_urls=[])
_ENABLED = ExaSettings(enabled=True, api_key="test-exa-key")


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

    async def test_date_mix_flows_through_select_candidates(self) -> None:
        """Mixed publishedDate shapes all normalize so recency sort never raises (load-bearing)."""
        from app.scraping import _select_candidates

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
        new_entries = [(e, compute_article_hash(e.url, e.title)) for e in resp.entries]
        reuse_batch, fetch_batch = _select_candidates(new_entries, [], 10)  # must not raise
        # Newest-first: the full-Z 2024 entry sorts ahead of the date-only 2023 one.
        assert fetch_batch[0][0].url == "https://x.com/2"

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
        """A base_url resolving to a private host is blocked before any request (SSRF)."""
        settings = ExaSettings(enabled=True, api_key="k", base_url="http://internal.local")
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url))
            return httpx.Response(200, json={"results": []})

        with patch("app.scraping.exa.is_private_url", return_value=True):
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
    async def test_stores_exa_articles_with_provider_and_prefetched_content(self, db_conn: sqlite3.Connection) -> None:
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
            result = await fetch_new_articles_for_topic(topic, db_conn, max_articles=5, exa_settings=_ENABLED)

        assert len(result.articles) == 1
        stored = list_articles_for_topic(db_conn, topic.id)
        assert len(stored) == 1
        assert stored[0].source_provider == "exa"
        assert stored[0].raw_content == "exa body"  # prefetched, not a second fetch
        assert isinstance(stored[0].published_at, datetime)
        assert stored[0].published_at.tzinfo is not None
