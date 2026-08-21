"""Tests for per-feed health tracking."""

import sqlite3
from unittest.mock import MagicMock, patch

import httpx

from app.crud import (
    get_feed_health,
    list_all_feed_health,
    upsert_feed_health_failure,
    upsert_feed_health_success,
)
from app.models import FeedHealth
from app.scraping.source import FeedHealthOutcome, FetchStatus


class TestFeedHealthFromRow:
    """Tests for FeedHealth.from_row datetime handling."""

    def test_empty_string_datetimes_become_none(self) -> None:
        """Empty-string datetime cells must coerce to None, not raise."""
        row = {
            "id": 1,
            "feed_url": "https://example.com/feed.xml",
            "last_success_at": "",
            "last_error_at": "",
            "last_error_message": None,
            "consecutive_failures": 0,
            "total_fetches": 0,
            "total_failures": 0,
        }
        health = FeedHealth.from_row(row)
        assert health.last_success_at is None
        assert health.last_error_at is None

    def test_whitespace_only_datetimes_become_none(self) -> None:
        """Whitespace-only datetime cells must coerce to None, not raise."""
        row = {
            "id": 1,
            "feed_url": "https://example.com/feed.xml",
            "last_success_at": "   ",
            "last_error_at": "\t",
            "last_error_message": None,
            "consecutive_failures": 0,
            "total_fetches": 0,
            "total_failures": 0,
        }
        health = FeedHealth.from_row(row)
        assert health.last_success_at is None
        assert health.last_error_at is None

    def test_valid_isoformat_datetimes_parse(self) -> None:
        """Valid isoformat strings must parse into datetime instances."""
        row = {
            "id": 1,
            "feed_url": "https://example.com/feed.xml",
            "last_success_at": "2026-06-13T12:00:00+00:00",
            "last_error_at": "2026-06-12T09:30:00+00:00",
            "last_error_message": "boom",
            "consecutive_failures": 2,
            "total_fetches": 5,
            "total_failures": 2,
        }
        health = FeedHealth.from_row(row)
        assert health.last_success_at is not None
        assert health.last_success_at.year == 2026
        assert health.last_error_at is not None
        assert health.last_error_at.month == 6

    def test_malformed_datetimes_become_none(self) -> None:
        """Unparseable non-empty strings fall back to None."""
        row = {
            "id": 1,
            "feed_url": "https://example.com/feed.xml",
            "last_success_at": "not-a-date",
            "last_error_at": "also-bad",
            "last_error_message": None,
            "consecutive_failures": 0,
            "total_fetches": 0,
            "total_failures": 0,
        }
        health = FeedHealth.from_row(row)
        assert health.last_success_at is None
        assert health.last_error_at is None


class TestUpsertFeedHealthSuccess:
    """Tests for upsert_feed_health_success."""

    def test_creates_new_record(self, db_conn: sqlite3.Connection) -> None:
        upsert_feed_health_success(db_conn, "https://example.com/feed.xml")
        db_conn.commit()

        health = get_feed_health(db_conn, "https://example.com/feed.xml")
        assert health is not None
        assert health.feed_url == "https://example.com/feed.xml"
        assert health.last_success_at is not None
        assert health.consecutive_failures == 0
        assert health.total_fetches == 1
        assert health.total_failures == 0
        assert health.last_error_at is None

    def test_updates_existing_record(self, db_conn: sqlite3.Connection) -> None:
        url = "https://example.com/feed.xml"
        upsert_feed_health_success(db_conn, url)
        upsert_feed_health_success(db_conn, url)
        db_conn.commit()

        health = get_feed_health(db_conn, url)
        assert health is not None
        assert health.total_fetches == 2
        assert health.consecutive_failures == 0


class TestUpsertFeedHealthFailure:
    """Tests for upsert_feed_health_failure."""

    def test_creates_new_record(self, db_conn: sqlite3.Connection) -> None:
        url = "https://broken.example.com/feed.xml"
        upsert_feed_health_failure(db_conn, url, "HTTP 404")
        db_conn.commit()

        health = get_feed_health(db_conn, url)
        assert health is not None
        assert health.feed_url == url
        assert health.last_error_at is not None
        assert health.last_error_message == "HTTP 404"
        assert health.consecutive_failures == 1
        assert health.total_fetches == 1
        assert health.total_failures == 1
        assert health.last_success_at is None

    def test_increments_counters_on_repeat(self, db_conn: sqlite3.Connection) -> None:
        url = "https://broken.example.com/feed.xml"
        upsert_feed_health_failure(db_conn, url, "timeout")
        upsert_feed_health_failure(db_conn, url, "timeout again")
        db_conn.commit()

        health = get_feed_health(db_conn, url)
        assert health is not None
        assert health.consecutive_failures == 2
        assert health.total_fetches == 2
        assert health.total_failures == 2
        assert health.last_error_message == "timeout again"


class TestSuccessAfterFailure:
    """Tests for success resetting consecutive_failures."""

    def test_success_resets_consecutive_failures(self, db_conn: sqlite3.Connection) -> None:
        url = "https://flaky.example.com/feed.xml"

        upsert_feed_health_failure(db_conn, url, "timeout")
        upsert_feed_health_failure(db_conn, url, "timeout")
        upsert_feed_health_success(db_conn, url)
        db_conn.commit()

        health = get_feed_health(db_conn, url)
        assert health is not None
        assert health.consecutive_failures == 0
        assert health.total_fetches == 3
        assert health.total_failures == 2
        assert health.last_success_at is not None


class TestGetFeedHealth:
    """Tests for get_feed_health."""

    def test_returns_none_for_unknown_url(self, db_conn: sqlite3.Connection) -> None:
        result = get_feed_health(db_conn, "https://never-seen.example.com/feed.xml")
        assert result is None

    def test_returns_model_for_known_url(self, db_conn: sqlite3.Connection) -> None:
        url = "https://example.com/feed.xml"
        upsert_feed_health_success(db_conn, url)
        db_conn.commit()

        health = get_feed_health(db_conn, url)
        assert health is not None
        assert health.feed_url == url


class TestListAllFeedHealth:
    """Tests for list_all_feed_health."""

    def test_returns_empty_list_when_no_records(self, db_conn: sqlite3.Connection) -> None:
        result = list_all_feed_health(db_conn)
        assert result == []

    def test_returns_all_records(self, db_conn: sqlite3.Connection) -> None:
        upsert_feed_health_success(db_conn, "https://a.example.com/feed.xml")
        upsert_feed_health_success(db_conn, "https://b.example.com/feed.xml")
        db_conn.commit()

        result = list_all_feed_health(db_conn)
        assert len(result) == 2

    def test_ordered_by_consecutive_failures_desc(self, db_conn: sqlite3.Connection) -> None:
        url_ok = "https://healthy.example.com/feed.xml"
        url_bad = "https://failing.example.com/feed.xml"

        upsert_feed_health_success(db_conn, url_ok)
        upsert_feed_health_failure(db_conn, url_bad, "error")
        upsert_feed_health_failure(db_conn, url_bad, "error")
        upsert_feed_health_failure(db_conn, url_bad, "error")
        db_conn.commit()

        result = list_all_feed_health(db_conn)
        assert result[0].feed_url == url_bad
        assert result[0].consecutive_failures == 3
        assert result[1].feed_url == url_ok
        assert result[1].consecutive_failures == 0

    def test_secondary_sort_by_feed_url(self, db_conn: sqlite3.Connection) -> None:
        upsert_feed_health_success(db_conn, "https://z.example.com/feed.xml")
        upsert_feed_health_success(db_conn, "https://a.example.com/feed.xml")
        db_conn.commit()

        result = list_all_feed_health(db_conn)
        assert result[0].feed_url == "https://a.example.com/feed.xml"
        assert result[1].feed_url == "https://z.example.com/feed.xml"


_SAMPLE_RSS = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Test Feed</title>
    <item>
      <title>Article One</title>
      <link>https://example.com/article-1</link>
    </item>
  </channel>
</rss>"""

_EMPTY_RSS = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel><title>Empty</title></channel>
</rss>"""


def _mock_transport(responses: dict[str, tuple[int, str]]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        for pattern, (status, body) in responses.items():
            if pattern in url:
                return httpx.Response(status, text=body)
        return httpx.Response(404, text="Not found")

    return httpx.MockTransport(handler)


class TestFetchFeedCallback:
    """Tests for callback integration with fetch_feed()."""

    async def test_callback_called_on_success(self) -> None:
        """health_callback reports an OK outcome when the feed is fetched."""
        from app.scraping.rss import fetch_feed

        callback = MagicMock()
        transport = _mock_transport({"example.com/feed.xml": (200, _SAMPLE_RSS)})

        async with httpx.AsyncClient(transport=transport) as client:
            entries = await fetch_feed(
                "https://example.com/feed.xml",
                client=client,
                health_callback=callback,
            )

        assert len(entries) == 1
        callback.assert_called_once_with(
            FeedHealthOutcome("https://example.com/feed.xml", FetchStatus.OK, None, None, None)
        )

    async def test_callback_called_on_http_error(self) -> None:
        """health_callback reports a FAILED outcome on HTTP error."""
        from app.scraping.rss import fetch_feed

        callback = MagicMock()
        transport = _mock_transport({"example.com/feed.xml": (404, "Not found")})

        async with httpx.AsyncClient(transport=transport) as client:
            entries = await fetch_feed(
                "https://example.com/feed.xml",
                client=client,
                health_callback=callback,
            )

        assert entries == []
        callback.assert_called_once()
        outcome = callback.call_args[0][0]
        assert outcome.feed_url == "https://example.com/feed.xml"
        assert outcome.status is FetchStatus.FAILED
        assert outcome.error_msg is not None
        assert "404" in outcome.error_msg

    async def test_no_callback_does_not_error(self) -> None:
        """fetch_feed works fine without a callback."""
        from app.scraping.rss import fetch_feed

        transport = _mock_transport({"example.com/feed.xml": (200, _EMPTY_RSS)})

        async with httpx.AsyncClient(transport=transport) as client:
            entries = await fetch_feed("https://example.com/feed.xml", client=client)

        assert entries == []


class TestFeedValidators:
    """Conditional-GET validator storage on feed_health (Phase 1)."""

    def test_feed_health_persists_validators(self, db_conn: sqlite3.Connection) -> None:
        """etag / last_modified round-trip through the feed_health row."""
        db_conn.execute(
            "INSERT INTO feed_health (feed_url, etag, last_modified) VALUES (?, ?, ?)",
            ("https://ex.com/feed", 'W/"abc"', "Wed, 21 Oct 2025 07:28:00 GMT"),
        )
        db_conn.commit()
        health = get_feed_health(db_conn, "https://ex.com/feed")
        assert health is not None
        assert health.etag == 'W/"abc"'
        assert health.last_modified == "Wed, 21 Oct 2025 07:28:00 GMT"

    def test_success_stores_and_preserves_validators(self, db_conn: sqlite3.Connection) -> None:
        """Validators persist on success; a 304 (None,None) preserves them via COALESCE."""
        upsert_feed_health_success(db_conn, "https://ex.com/feed", etag='W/"v1"', last_modified="LM1")
        db_conn.commit()
        h = get_feed_health(db_conn, "https://ex.com/feed")
        assert h is not None and h.etag == 'W/"v1"' and h.last_modified == "LM1"

        # A 304 success passes None, None — existing validators must be preserved.
        upsert_feed_health_success(db_conn, "https://ex.com/feed", etag=None, last_modified=None)
        db_conn.commit()
        h = get_feed_health(db_conn, "https://ex.com/feed")
        assert h is not None and h.etag == 'W/"v1"' and h.last_modified == "LM1"
        assert h.total_fetches == 2 and h.consecutive_failures == 0

        # A fresh 200 with a new validator overwrites.
        upsert_feed_health_success(db_conn, "https://ex.com/feed", etag='W/"v2"', last_modified="LM2")
        db_conn.commit()
        h = get_feed_health(db_conn, "https://ex.com/feed")
        assert h is not None and h.etag == 'W/"v2"' and h.last_modified == "LM2"


class TestValidatorReplaceVsPreserve:
    """AUG-152: only a 304 speaks for validators it does not carry."""

    def test_a_200_clears_a_validator_the_feed_stopped_sending(self, db_conn: sqlite3.Connection) -> None:
        url = "https://ex.com/feed"
        upsert_feed_health_success(db_conn, url, etag='W/"v1"', last_modified="LM1", replace_validators=True)
        db_conn.commit()

        # The feed now serves Last-Modified only. The obsolete ETag must go, or it
        # is sent as If-None-Match forever against a source that no longer issues it.
        upsert_feed_health_success(db_conn, url, etag=None, last_modified="LM2", replace_validators=True)
        db_conn.commit()

        h = get_feed_health(db_conn, url)
        assert h is not None and h.etag is None and h.last_modified == "LM2"

    def test_a_304_preserves_what_it_does_not_carry(self, db_conn: sqlite3.Connection) -> None:
        url = "https://ex.com/feed"
        upsert_feed_health_success(db_conn, url, etag='W/"v1"', last_modified="LM1", replace_validators=True)
        upsert_feed_health_success(db_conn, url, etag=None, last_modified=None)
        db_conn.commit()

        h = get_feed_health(db_conn, url)
        assert h is not None and h.etag == 'W/"v1"' and h.last_modified == "LM1"


class TestAbortedFetchAccounting:
    """An expired attempt budget is recorded, but never charged to the feed."""

    def test_abort_does_not_advance_the_failure_streak(self, db_conn: sqlite3.Connection) -> None:
        from app.crud import upsert_feed_health_aborted

        url = "https://slow.example/feed"
        upsert_feed_health_failure(db_conn, url, "HTTP 500")
        upsert_feed_health_aborted(db_conn, url, "Source deadline exceeded during the feed request")
        db_conn.commit()

        h = get_feed_health(db_conn, url)
        assert h is not None
        # One real failure, one abandoned attempt: backoff still reflects one failure.
        assert h.consecutive_failures == 1
        assert h.total_failures == 1
        assert h.total_fetches == 2
        assert h.last_error_message == "Source deadline exceeded during the feed request"


class TestBlockedUrlHealth:
    """AUG-177: a URL we refuse to fetch is a failed fetch, not a missing one."""

    async def test_blocked_url_records_a_failure(self) -> None:
        from app.scraping.rss import fetch_feed_outcome

        callback = MagicMock()
        # A real RFC-1918 literal rather than a patched verdict, so the fetch runs
        # the classification it actually ships with.
        result = await fetch_feed_outcome("https://192.168.1.10/feed.xml", health_callback=callback)

        assert result.status is FetchStatus.FAILED
        outcome = callback.call_args[0][0]
        assert outcome.feed_url == "https://192.168.1.10/feed.xml"
        assert outcome.status is FetchStatus.FAILED
        assert outcome.error_msg and "Blocked" in outcome.error_msg

    async def test_saturated_resolver_is_aborted_not_failed(self) -> None:
        """A busy resolver pool is the check's problem, never the feed's (AUG-013).

        ``FAILED`` here advances the per-feed streak, and three of them put an
        untouched healthy feed into exponential backoff for up to 24 hours.
        """
        from app.scraping.rss import fetch_feed_outcome
        from app.url_validation import ResolverSaturatedError

        callback = MagicMock()
        # Raised where the real saturation is detected, so the whole chain --
        # classification, safe_send, the fetch's own handlers -- runs unpatched.
        with patch("app.url_validation._getaddrinfo_bounded", side_effect=ResolverSaturatedError("saturated")):
            result = await fetch_feed_outcome("https://healthy.example/feed.xml", health_callback=callback)

        assert result.status is FetchStatus.ABORTED
        outcome = callback.call_args[0][0]
        assert outcome.status is FetchStatus.ABORTED
        assert outcome.error_msg and "resolver" in outcome.error_msg.lower()


class TestAllRejectedEntries:
    """AUG-178: a feed that publishes entries and yields none is not quiet."""

    async def test_every_entry_rejected_is_a_failure(self) -> None:
        from app.scraping.rss import fetch_feed_outcome

        # Well-formed RSS whose every item has a link this pipeline must refuse.
        body = (
            '<?xml version="1.0"?><rss version="2.0"><channel><title>T</title>'
            "<item><title>A</title><link>javascript:alert(1)</link></item>"
            "<item><title>B</title><link>javascript:alert(2)</link></item>"
            "</channel></rss>"
        )
        callback = MagicMock()
        transport = _mock_transport({"example.com/feed.xml": (200, body)})
        async with httpx.AsyncClient(transport=transport) as client:
            result = await fetch_feed_outcome("https://example.com/feed.xml", client=client, health_callback=callback)

        assert result.entries == []
        assert result.status is FetchStatus.FAILED
        assert callback.call_args[0][0].status is FetchStatus.FAILED

    async def test_a_feed_with_no_entries_at_all_stays_healthy(self) -> None:
        from app.scraping.rss import fetch_feed_outcome

        callback = MagicMock()
        transport = _mock_transport({"example.com/feed.xml": (200, _EMPTY_RSS)})
        async with httpx.AsyncClient(transport=transport) as client:
            result = await fetch_feed_outcome("https://example.com/feed.xml", client=client, health_callback=callback)

        assert result.status is FetchStatus.EMPTY
        assert callback.call_args[0][0].status is FetchStatus.EMPTY

    async def test_a_partly_rejected_feed_keeps_its_good_entries(self) -> None:
        from app.scraping.rss import fetch_feed_outcome

        body = (
            '<?xml version="1.0"?><rss version="2.0"><channel><title>T</title>'
            "<item><title>Bad</title><link>javascript:alert(1)</link></item>"
            "<item><title>Good</title><link>https://example.com/good</link></item>"
            "</channel></rss>"
        )
        transport = _mock_transport({"example.com/feed.xml": (200, body)})
        async with httpx.AsyncClient(transport=transport) as client:
            result = await fetch_feed_outcome("https://example.com/feed.xml", client=client)

        assert [e.title for e in result.entries] == ["Good"]
        assert result.status is FetchStatus.OK


class TestReplaySafeValidators:
    """TW-AUD-020: a shared 304 must not cost a topic its articles."""

    async def test_no_validators_when_the_topic_holds_nothing_from_the_feed(self) -> None:
        """A second topic on a shared feed asks for the full body, not a 304."""
        from app.models import FeedMode, Topic
        from app.scraping.rss import fetch_feeds_for_topic

        sent: list[dict[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            sent.append(dict(request.headers))
            return httpx.Response(200, text=_SAMPLE_RSS)

        topic = Topic(
            id=2,
            name="T",
            description="d",
            feed_mode=FeedMode.MANUAL,
            feed_urls=["https://shared.example/feed.xml"],
        )
        stored = FeedHealth(feed_url="https://shared.example/feed.xml", etag='W/"shared"', last_modified="LM")

        original_init = httpx.AsyncClient.__init__

        def patched_init(self_client, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            original_init(self_client, **kwargs)

        with patch.object(httpx.AsyncClient, "__init__", patched_init):
            response = await fetch_feeds_for_topic(
                topic,
                feed_state_loader=lambda url: stored,
                topic_holds_feed_articles=lambda url: False,
            )

        assert len(response.entries) == 1
        assert "if-none-match" not in sent[0]
        assert "if-modified-since" not in sent[0]

    async def test_validators_are_sent_once_the_topic_holds_articles(self) -> None:
        from app.models import FeedMode, Topic
        from app.scraping.rss import fetch_feeds_for_topic

        sent: list[dict[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            sent.append(dict(request.headers))
            return httpx.Response(304)

        topic = Topic(
            id=2,
            name="T",
            description="d",
            feed_mode=FeedMode.MANUAL,
            feed_urls=["https://shared.example/feed.xml"],
        )
        stored = FeedHealth(feed_url="https://shared.example/feed.xml", etag='W/"shared"', last_modified="LM")

        original_init = httpx.AsyncClient.__init__

        def patched_init(self_client, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            original_init(self_client, **kwargs)

        with patch.object(httpx.AsyncClient, "__init__", patched_init):
            await fetch_feeds_for_topic(
                topic,
                feed_state_loader=lambda url: stored,
                topic_holds_feed_articles=lambda url: True,
            )

        assert sent[0].get("if-none-match") == 'W/"shared"'

    def test_the_article_check_is_topic_scoped(self, db_conn: sqlite3.Connection) -> None:
        from app.crud import create_article, create_topic, topic_has_articles_from_feed
        from app.models import Article, FeedMode, Topic

        feed = "https://shared.example/feed.xml"
        owner = create_topic(db_conn, Topic(name="Owner", description="d", feed_mode=FeedMode.MANUAL, feed_urls=[feed]))
        newcomer = create_topic(
            db_conn, Topic(name="New", description="d", feed_mode=FeedMode.MANUAL, feed_urls=[feed])
        )
        assert owner.id is not None and newcomer.id is not None
        create_article(
            db_conn,
            Article(topic_id=owner.id, title="A", url="https://ex/a", content_hash="h1", source_feed=feed),
        )
        db_conn.commit()

        assert topic_has_articles_from_feed(db_conn, owner.id, feed) is True
        assert topic_has_articles_from_feed(db_conn, newcomer.id, feed) is False


class TestAutoCascadePolicy:
    """What AUTO does with each fetch outcome (AUG-172, AUG-173)."""

    @staticmethod
    def _auto_topic():
        from app.models import FeedMode, Topic

        return Topic(name="T", description="d", feed_mode=FeedMode.AUTO, feed_urls=[])

    async def _run(self, results, router=None):
        """Drive one AUTO fetch with canned per-provider outcomes."""
        from app.scraping.routing import ProviderRouter
        from app.scraping.rss import fetch_feeds_for_topic

        calls: list[str] = []

        async def fake_fetch(url, client, **kwargs):
            calls.append(url)
            return results[len(calls) - 1]

        with patch("app.scraping.rss.fetch_feed_outcome", side_effect=fake_fetch):
            response = await fetch_feeds_for_topic(self._auto_topic(), router=router or ProviderRouter())
        return calls, response

    async def test_not_modified_does_not_cascade(self) -> None:
        """A 304 means the provider already gave us everything (AUG-172)."""
        from app.scraping.source import FeedFetchResult

        calls, response = await self._run([FeedFetchResult(status=FetchStatus.NOT_MODIFIED)])

        assert len(calls) == 1  # the fallback provider is never queried
        assert response.feeds_total == 1
        assert response.feeds_failed == 0

    async def test_empty_result_still_cascades(self) -> None:
        """An empty 200 is a real "nothing here", so the fallback is still worth asking."""
        from app.scraping.source import FeedFetchResult

        calls, _ = await self._run(
            [FeedFetchResult(status=FetchStatus.EMPTY), FeedFetchResult(status=FetchStatus.EMPTY)]
        )

        assert len(calls) == 2

    async def test_an_empty_success_resets_the_failure_streak(self) -> None:
        """AUG-173: failure, failure, empty-success, failure must not reach the threshold."""
        from app.scraping.routing import _FAILURE_THRESHOLD, ProviderRouter
        from app.scraping.source import FeedFetchResult

        router = ProviderRouter()
        primary = router.providers[0].name
        router.mark_unhealthy(primary)
        router.mark_unhealthy(primary)

        await self._run(
            [FeedFetchResult(status=FetchStatus.EMPTY), FeedFetchResult(status=FetchStatus.EMPTY)], router=router
        )
        assert router.mark_healthy(primary) is False  # nothing left to recover from

        # One more failure after that reset is failure 1 of 3, not failure 3 of 3.
        await self._run(
            [FeedFetchResult(status=FetchStatus.FAILED), FeedFetchResult(status=FetchStatus.FAILED)], router=router
        )
        assert router.get_provider().name == primary
        assert _FAILURE_THRESHOLD == 3

    async def test_a_304_also_resets_the_failure_streak(self) -> None:
        from app.scraping.routing import ProviderRouter
        from app.scraping.source import FeedFetchResult

        router = ProviderRouter()
        primary = router.providers[0].name
        router.mark_unhealthy(primary)

        await self._run([FeedFetchResult(status=FetchStatus.NOT_MODIFIED)], router=router)

        assert router.mark_healthy(primary) is False  # the streak was already cleared

    async def test_an_aborted_fetch_touches_neither_side_of_the_streak(self) -> None:
        """Our budget expiring says nothing about the provider, either way."""
        from app.scraping.routing import ProviderRouter
        from app.scraping.source import FeedFetchResult

        router = ProviderRouter()
        primary = router.providers[0].name
        router.mark_unhealthy(primary)
        router.mark_unhealthy(primary)

        calls, response = await self._run([FeedFetchResult(status=FetchStatus.ABORTED)], router=router)

        assert len(calls) == 1  # no cascade on a budget we no longer have
        assert router.mark_healthy(primary) is True  # the two failures are still there
        assert response.feeds_failed == 1  # still a degraded check

    async def test_a_shared_cooldown_stops_the_fetch_entirely(self) -> None:
        """AUG-306: a cooldown that still lets both providers be tried is no cooldown."""
        from app.scraping.routing import _FAILURE_THRESHOLD, ProviderRouter

        router = ProviderRouter()
        for _ in range(_FAILURE_THRESHOLD):
            for provider in router.providers:
                router.mark_unhealthy(provider.name)

        calls, response = await self._run([], router=router)

        assert calls == []  # neither provider is queried, so neither deadline slides
        assert response.feeds_total == 1 and response.feeds_failed == 1
