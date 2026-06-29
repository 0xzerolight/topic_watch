"""Tests for per-feed health tracking."""

import sqlite3
from unittest.mock import MagicMock

import httpx

from app.crud import (
    get_feed_health,
    list_all_feed_health,
    upsert_feed_health_failure,
    upsert_feed_health_success,
)
from app.models import FeedHealth


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
        """health_callback is called with success=True when feed is fetched."""
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
        callback.assert_called_once_with("https://example.com/feed.xml", True, None)

    async def test_callback_called_on_http_error(self) -> None:
        """health_callback is called with success=False on HTTP error."""
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
        args = callback.call_args[0]
        assert args[0] == "https://example.com/feed.xml"
        assert args[1] is False
        assert args[2] is not None
        assert "404" in args[2]

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
