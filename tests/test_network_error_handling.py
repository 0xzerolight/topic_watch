"""Tests for network error handling in fetch_feed()."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.scraping.rss import fetch_feed

VALID_RSS = """<?xml version="1.0"?>
<rss version="2.0">
  <channel>
    <title>Test</title>
    <item>
      <title>Test Article</title>
      <link>https://example.com/article</link>
    </item>
  </channel>
</rss>"""


def _make_response(text: str, status_code: int = 200) -> MagicMock:
    response = MagicMock()
    response.text = text
    response.status_code = status_code
    response.raise_for_status = MagicMock()
    return response


def _make_client(side_effects) -> AsyncMock:
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get = AsyncMock(side_effect=side_effects)
    return client


@pytest.mark.asyncio
async def test_connect_error_retries_and_succeeds():
    """ConnectError on first attempt triggers retry; second attempt succeeds."""
    client = _make_client(
        [
            httpx.ConnectError("Connection refused"),
            _make_response(VALID_RSS),
        ]
    )
    with patch("asyncio.sleep", new_callable=AsyncMock):
        entries = await fetch_feed("https://example.com/feed.xml", client=client)
    assert len(entries) == 1
    assert entries[0].title == "Test Article"
    assert client.get.call_count == 2


@pytest.mark.asyncio
async def test_connect_error_all_attempts_returns_empty():
    """ConnectError on all attempts returns empty list."""
    client = _make_client(
        [
            httpx.ConnectError("Connection refused"),
            httpx.ConnectError("Connection refused"),
        ]
    )
    with patch("asyncio.sleep", new_callable=AsyncMock):
        entries = await fetch_feed("https://example.com/feed.xml", client=client)
    assert entries == []
    assert client.get.call_count == 2


@pytest.mark.asyncio
async def test_dns_failure_handled():
    """DNS resolution failure (ConnectError with DNS message) is handled."""
    client = _make_client(
        [
            httpx.ConnectError("Name or service not known"),
            httpx.ConnectError("Name or service not known"),
        ]
    )
    with patch("asyncio.sleep", new_callable=AsyncMock):
        entries = await fetch_feed("https://nonexistent.invalid/feed.xml", client=client)
    assert entries == []


@pytest.mark.asyncio
async def test_read_error_retried():
    """ReadError (subclass of NetworkError) is retried."""
    client = _make_client(
        [
            httpx.ReadError("Connection reset by peer"),
            _make_response(VALID_RSS),
        ]
    )
    with patch("asyncio.sleep", new_callable=AsyncMock):
        entries = await fetch_feed("https://example.com/feed.xml", client=client)
    assert len(entries) == 1
    assert client.get.call_count == 2


@pytest.mark.asyncio
async def test_timeout_exception_still_handled():
    """Existing TimeoutException handling still works."""
    client = _make_client(
        [
            httpx.TimeoutException("Request timed out"),
            _make_response(VALID_RSS),
        ]
    )
    with patch("asyncio.sleep", new_callable=AsyncMock):
        entries = await fetch_feed("https://example.com/feed.xml", client=client)
    assert len(entries) == 1
    assert client.get.call_count == 2


@pytest.mark.asyncio
async def test_timeout_exception_all_attempts_returns_empty():
    """TimeoutException on all attempts returns empty list."""
    client = _make_client(
        [
            httpx.TimeoutException("Request timed out"),
            httpx.TimeoutException("Request timed out"),
        ]
    )
    with patch("asyncio.sleep", new_callable=AsyncMock):
        entries = await fetch_feed("https://example.com/feed.xml", client=client)
    assert entries == []


@pytest.mark.asyncio
async def test_http_status_error_5xx_retried():
    """Existing HTTPStatusError 5xx handling still retries."""
    error_response = MagicMock()
    error_response.status_code = 503

    def raise_5xx():
        raise httpx.HTTPStatusError("503", request=MagicMock(), response=error_response)

    client = _make_client(
        [
            httpx.HTTPStatusError("503", request=MagicMock(), response=error_response),
            _make_response(VALID_RSS),
        ]
    )
    with patch("asyncio.sleep", new_callable=AsyncMock):
        entries = await fetch_feed("https://example.com/feed.xml", client=client)
    assert len(entries) == 1
    assert client.get.call_count == 2


@pytest.mark.asyncio
async def test_http_status_error_4xx_not_retried():
    """Existing HTTPStatusError 4xx handling does not retry."""
    error_response = MagicMock()
    error_response.status_code = 404

    client = _make_client(
        [
            httpx.HTTPStatusError("404", request=MagicMock(), response=error_response),
        ]
    )
    with patch("asyncio.sleep", new_callable=AsyncMock):
        entries = await fetch_feed("https://example.com/feed.xml", client=client)
    assert entries == []
    assert client.get.call_count == 1


@pytest.mark.asyncio
async def test_network_error_logs_debug_on_retry(caplog):
    """Debug log is emitted on first network error attempt."""
    import logging

    client = _make_client(
        [
            httpx.ConnectError("Connection refused"),
            _make_response(VALID_RSS),
        ]
    )
    with patch("asyncio.sleep", new_callable=AsyncMock), caplog.at_level(logging.DEBUG, logger="app.scraping.rss"):
        entries = await fetch_feed("https://example.com/feed.xml", client=client)
    assert len(entries) == 1
    assert any("Network error fetching feed (attempt 1)" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_network_error_logs_warning_on_final_failure(caplog):
    """Warning log is emitted when all network error attempts exhausted."""
    import logging

    client = _make_client(
        [
            httpx.ConnectError("Connection refused"),
            httpx.ConnectError("Connection refused"),
        ]
    )
    with patch("asyncio.sleep", new_callable=AsyncMock), caplog.at_level(logging.WARNING, logger="app.scraping.rss"):
        entries = await fetch_feed("https://example.com/feed.xml", client=client)
    assert entries == []
    assert any("Network error fetching feed after 2 attempts" in r.message for r in caplog.records)
