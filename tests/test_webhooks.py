"""Tests for the webhook delivery module."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.analysis.llm import NoveltyResult
from app.config import LLMSettings, NotificationSettings, Settings
from app.webhooks import _build_webhook_payload, send_webhook, send_webhooks


def _make_settings(**overrides) -> Settings:
    defaults = {
        "llm": LLMSettings(model="openai/gpt-4o-mini", api_key="test-key"),
        "notifications": NotificationSettings(urls=[]),
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _make_novelty(**overrides) -> NoveltyResult:
    defaults = {
        "has_new_info": True,
        "summary": "New milestone reached",
        "key_facts": ["Fact one", "Fact two"],
        "source_urls": ["https://example.com/article"],
        "confidence": 0.85,
    }
    defaults.update(overrides)
    return NoveltyResult(**defaults)


# --- _build_webhook_payload ---


class TestBuildWebhookPayload:
    """Tests for payload construction."""

    def test_payload_contains_topic_name(self) -> None:
        novelty = _make_novelty()
        payload = _build_webhook_payload("My Topic", novelty)
        assert payload["topic"] == "My Topic"

    def test_payload_contains_summary(self) -> None:
        novelty = _make_novelty(summary="Something happened")
        payload = _build_webhook_payload("T", novelty)
        assert payload["summary"] == "Something happened"

    def test_payload_summary_defaults_to_empty_string_when_none(self) -> None:
        novelty = _make_novelty(summary=None)
        payload = _build_webhook_payload("T", novelty)
        assert payload["summary"] == ""

    def test_payload_contains_key_facts(self) -> None:
        novelty = _make_novelty(key_facts=["Fact A", "Fact B"])
        payload = _build_webhook_payload("T", novelty)
        assert payload["key_facts"] == ["Fact A", "Fact B"]

    def test_payload_contains_source_urls(self) -> None:
        novelty = _make_novelty(source_urls=["https://a.com", "https://b.com"])
        payload = _build_webhook_payload("T", novelty)
        assert payload["source_urls"] == ["https://a.com", "https://b.com"]

    def test_payload_contains_confidence(self) -> None:
        novelty = _make_novelty(confidence=0.72)
        payload = _build_webhook_payload("T", novelty)
        assert payload["confidence"] == pytest.approx(0.72)

    def test_payload_contains_timestamp(self) -> None:
        novelty = _make_novelty()
        payload = _build_webhook_payload("T", novelty)
        assert "timestamp" in payload
        assert isinstance(payload["timestamp"], str)
        # Should be an ISO 8601 string with timezone info
        assert "T" in payload["timestamp"]
        assert "+00:00" in payload["timestamp"] or "Z" in payload["timestamp"]

    def test_payload_has_all_expected_fields(self) -> None:
        novelty = _make_novelty()
        payload = _build_webhook_payload("T", novelty)
        expected_keys = {"topic", "reasoning", "summary", "key_facts", "source_urls", "confidence", "timestamp"}
        assert set(payload.keys()) == expected_keys

    def test_key_facts_is_list_type(self) -> None:
        novelty = _make_novelty(key_facts=[])
        payload = _build_webhook_payload("T", novelty)
        assert isinstance(payload["key_facts"], list)

    def test_source_urls_is_list_type(self) -> None:
        novelty = _make_novelty(source_urls=[])
        payload = _build_webhook_payload("T", novelty)
        assert isinstance(payload["source_urls"], list)

    def test_payload_contains_reasoning(self) -> None:
        novelty = _make_novelty(reasoning="Article [1] mentions a new date.")
        payload = _build_webhook_payload("T", novelty)
        assert payload["reasoning"] == "Article [1] mentions a new date."

    def test_payload_reasoning_defaults_to_empty(self) -> None:
        novelty = _make_novelty()
        payload = _build_webhook_payload("T", novelty)
        assert payload["reasoning"] == ""

    def test_confidence_is_float_type(self) -> None:
        novelty = _make_novelty(confidence=1.0)
        payload = _build_webhook_payload("T", novelty)
        assert isinstance(payload["confidence"], float)


# --- send_webhook ---


class TestSendWebhook:
    """Tests for the individual send_webhook function."""

    async def test_returns_true_on_success(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("app.webhooks.httpx.AsyncClient", return_value=mock_client):
            result = await send_webhook("https://example.com/hook", {"key": "value"})

        assert result is True

    async def test_returns_false_on_timeout(self) -> None:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("app.webhooks.httpx.AsyncClient", return_value=mock_client):
            result = await send_webhook("https://example.com/hook", {})

        assert result is False

    async def test_returns_false_on_http_error(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "Server Error",
                request=MagicMock(),
                response=mock_response,
            )
        )

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("app.webhooks.httpx.AsyncClient", return_value=mock_client):
            result = await send_webhook("https://example.com/hook", {})

        assert result is False

    async def test_returns_false_on_connection_error(self) -> None:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("Connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("app.webhooks.httpx.AsyncClient", return_value=mock_client):
            result = await send_webhook("https://example.com/hook", {})

        assert result is False

    async def test_blocks_private_url(self) -> None:
        result = await send_webhook("http://localhost:9200/hook", {"key": "value"})
        assert result is False

    async def test_blocks_internal_ip(self) -> None:
        result = await send_webhook("http://169.254.169.254/metadata", {})
        assert result is False

    async def test_blocks_loopback_ip(self) -> None:
        result = await send_webhook("http://127.0.0.1:8080/hook", {})
        assert result is False

    async def test_posts_to_correct_url(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        url = "https://hooks.example.com/trigger"
        payload = {"topic": "test"}

        with patch("app.webhooks.httpx.AsyncClient", return_value=mock_client):
            await send_webhook(url, payload)

        mock_client.post.assert_called_once_with(url, json=payload)


# --- send_webhooks ---


class TestSendWebhooks:
    """Tests for the send_webhooks orchestrator."""

    async def test_returns_zero_with_no_webhook_urls(self) -> None:
        settings = _make_settings(notifications=NotificationSettings(urls=[], webhook_urls=[]))
        novelty = _make_novelty()

        result = await send_webhooks("My Topic", novelty, settings)

        assert result == 0

    async def test_returns_count_of_successful_deliveries(self) -> None:
        settings = _make_settings(
            notifications=NotificationSettings(
                urls=[],
                webhook_urls=["https://a.com/hook", "https://b.com/hook", "https://c.com/hook"],
            )
        )
        novelty = _make_novelty()

        # First two succeed, third fails
        async def fake_send_webhook(url: str, payload: dict, timeout: float = 10.0) -> bool:
            return url != "https://c.com/hook"

        with patch("app.webhooks.send_webhook", side_effect=fake_send_webhook):
            result = await send_webhooks("My Topic", novelty, settings)

        assert result == 2

    async def test_returns_full_count_when_all_succeed(self) -> None:
        settings = _make_settings(
            notifications=NotificationSettings(
                urls=[],
                webhook_urls=["https://a.com/hook", "https://b.com/hook"],
            )
        )
        novelty = _make_novelty()

        with patch("app.webhooks.send_webhook", return_value=True):
            result = await send_webhooks("My Topic", novelty, settings)

        assert result == 2

    async def test_returns_zero_when_all_fail(self) -> None:
        settings = _make_settings(
            notifications=NotificationSettings(
                urls=[],
                webhook_urls=["https://a.com/hook", "https://b.com/hook"],
            )
        )
        novelty = _make_novelty()

        with patch("app.webhooks.send_webhook", return_value=False):
            result = await send_webhooks("My Topic", novelty, settings)

        assert result == 0

    async def test_sends_to_all_configured_urls(self) -> None:
        urls = ["https://a.com/hook", "https://b.com/hook"]
        settings = _make_settings(notifications=NotificationSettings(urls=[], webhook_urls=urls))
        novelty = _make_novelty()
        called_urls: list[str] = []

        async def capture_url(url: str, payload: dict, timeout: float = 10.0) -> bool:
            called_urls.append(url)
            return True

        with patch("app.webhooks.send_webhook", side_effect=capture_url):
            await send_webhooks("My Topic", novelty, settings)

        assert sorted(called_urls) == sorted(urls)

    async def test_passes_correct_payload_fields(self) -> None:
        settings = _make_settings(
            notifications=NotificationSettings(
                urls=[],
                webhook_urls=["https://hook.example.com"],
            )
        )
        novelty = _make_novelty(
            summary="Big news",
            key_facts=["fact1"],
            source_urls=["https://src.com"],
            confidence=0.9,
        )
        captured_payloads: list[dict] = []

        async def capture_payload(url: str, payload: dict, timeout: float = 10.0) -> bool:
            captured_payloads.append(payload)
            return True

        with patch("app.webhooks.send_webhook", side_effect=capture_payload):
            await send_webhooks("My Topic", novelty, settings)

        assert len(captured_payloads) == 1
        payload = captured_payloads[0]
        assert payload["topic"] == "My Topic"
        assert payload["summary"] == "Big news"
        assert payload["key_facts"] == ["fact1"]
        assert payload["source_urls"] == ["https://src.com"]
        assert payload["confidence"] == pytest.approx(0.9)
        assert "timestamp" in payload
