"""Tests for rate-limit-aware retry with exponential backoff in LLM functions."""

from unittest.mock import AsyncMock, MagicMock, patch

import litellm
import pytest

from app.analysis.llm import (
    KnowledgeStateUpdate,
    NoveltyResult,
    _call_with_rate_limit_retry,
    analyze_articles,
    generate_initial_knowledge,
    generate_knowledge_update,
)
from app.config import LLMSettings, Settings
from app.models import Article, Topic

# --- Helpers ---


def _make_settings(**overrides) -> Settings:
    defaults = {
        "llm": LLMSettings(model="openai/gpt-4o-mini", api_key="test-key"),
        "knowledge_state_max_tokens": 2000,
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _make_topic(**overrides) -> Topic:
    defaults = {
        "id": 1,
        "name": "Test Topic",
        "description": "A test topic",
        "feed_urls": ["https://example.com/feed.xml"],
    }
    defaults.update(overrides)
    return Topic(**defaults)


def _make_article(**overrides) -> Article:
    defaults = {
        "id": 1,
        "topic_id": 1,
        "title": "Test Article",
        "url": "https://example.com/article-1",
        "content_hash": "abc123",
        "raw_content": "Article content.",
        "source_feed": "https://example.com/feed.xml",
    }
    defaults.update(overrides)
    return Article(**defaults)


def _make_rate_limit_error() -> litellm.RateLimitError:
    return litellm.RateLimitError(
        message="Rate limit exceeded",
        llm_provider="openai",
        model="gpt-4",
    )


def _mock_instructor_client(return_value):
    """Create a mock instructor client."""
    mock_create = AsyncMock(return_value=return_value)
    mock_completions = MagicMock()
    mock_completions.create = mock_create
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions
    mock_client = MagicMock()
    mock_client.chat = mock_chat
    return mock_client, mock_create


# ============================================================
# TestCallWithRateLimitRetry
# ============================================================


class TestCallWithRateLimitRetry:
    async def test_succeeds_on_first_try(self) -> None:
        """When no error occurs, returns the result immediately."""
        call_func = AsyncMock(return_value="ok")

        with patch("app.analysis.llm.asyncio.sleep") as mock_sleep:
            result = await _call_with_rate_limit_retry(call_func)

        assert result == "ok"
        call_func.assert_called_once()
        mock_sleep.assert_not_called()

    async def test_retries_on_rate_limit_and_succeeds(self) -> None:
        """Retries after RateLimitError and returns result on success."""
        rate_error = _make_rate_limit_error()
        call_func = AsyncMock(side_effect=[rate_error, rate_error, "success"])

        with patch("app.analysis.llm.asyncio.sleep") as mock_sleep:
            result = await _call_with_rate_limit_retry(call_func)

        assert result == "success"
        assert call_func.call_count == 3
        assert mock_sleep.call_count == 2

    async def test_uses_exponential_backoff_delays(self) -> None:
        """Verifies backoff delays follow base_delay * (multiplier ** attempt)."""
        rate_error = _make_rate_limit_error()
        call_func = AsyncMock(side_effect=[rate_error, rate_error, "ok"])

        with patch("app.analysis.llm.asyncio.sleep") as mock_sleep:
            await _call_with_rate_limit_retry(call_func, base_delay=5.0, backoff_multiplier=3.0)

        delays = [call.args[0] for call in mock_sleep.call_args_list]
        # attempt=0: 5 * 3^0 = 5, attempt=1: 5 * 3^1 = 15
        assert delays == [5.0, 15.0]

    async def test_raises_after_exhausting_retries(self) -> None:
        """Re-raises the last RateLimitError after max_retries attempts."""
        rate_error = _make_rate_limit_error()
        # max_retries=3 means 4 total calls (initial + 3 retries)
        call_func = AsyncMock(side_effect=[rate_error] * 4)

        with patch("app.analysis.llm.asyncio.sleep"), pytest.raises(litellm.RateLimitError):
            await _call_with_rate_limit_retry(call_func, max_retries=3)

        assert call_func.call_count == 4

    async def test_does_not_retry_non_rate_limit_errors(self) -> None:
        """Non-RateLimitError exceptions are re-raised immediately without retry."""
        call_func = AsyncMock(side_effect=ValueError("bad input"))

        with patch("app.analysis.llm.asyncio.sleep") as mock_sleep, pytest.raises(ValueError, match="bad input"):
            await _call_with_rate_limit_retry(call_func)

        call_func.assert_called_once()
        mock_sleep.assert_not_called()

    async def test_logs_each_retry_attempt(self) -> None:
        """A warning is logged for each retry."""
        rate_error = _make_rate_limit_error()
        call_func = AsyncMock(side_effect=[rate_error, "ok"])

        with (
            patch("app.analysis.llm.asyncio.sleep"),
            patch("app.analysis.llm.logger") as mock_logger,
        ):
            await _call_with_rate_limit_retry(call_func, max_retries=3)

        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args.args[0]
        assert "Rate limit" in warning_msg or "rate limit" in warning_msg.lower()


# ============================================================
# TestAnalyzeArticlesRateLimit
# ============================================================


class TestAnalyzeArticlesRateLimit:
    async def test_retries_on_rate_limit_and_returns_result(self) -> None:
        """analyze_articles retries on RateLimitError and returns result on success."""
        rate_error = _make_rate_limit_error()
        expected = NoveltyResult(has_new_info=True, confidence=0.9)
        mock_client, mock_create = _mock_instructor_client(expected)
        mock_create.side_effect = [rate_error, expected]
        settings = _make_settings()

        with (
            patch("app.analysis.llm._get_client", return_value=mock_client),
            patch("app.analysis.llm.asyncio.sleep"),
        ):
            result = await analyze_articles([_make_article()], "Known facts.", _make_topic(), settings)

        assert result.has_new_info is True
        assert result.confidence == 0.9
        assert mock_create.call_count == 2

    async def test_returns_safe_default_after_exhausting_rate_limit_retries(
        self,
    ) -> None:
        """analyze_articles returns safe default when rate limit retries are exhausted."""
        rate_error = _make_rate_limit_error()
        mock_client, mock_create = _mock_instructor_client(None)
        # 4 errors = initial call + 3 retries
        mock_create.side_effect = [rate_error] * 4
        settings = _make_settings()

        with (
            patch("app.analysis.llm._get_client", return_value=mock_client),
            patch("app.analysis.llm.asyncio.sleep"),
        ):
            result = await analyze_articles([_make_article()], "Known facts.", _make_topic(), settings)

        assert result.has_new_info is False
        assert result.confidence == 0.0

    async def test_returns_safe_default_on_generic_error(self) -> None:
        """analyze_articles still returns safe default for non-rate-limit errors."""
        mock_client, mock_create = _mock_instructor_client(None)
        mock_create.side_effect = Exception("generic LLM error")
        settings = _make_settings()

        with (
            patch("app.analysis.llm._get_client", return_value=mock_client),
            patch("app.analysis.llm.asyncio.sleep"),
        ):
            result = await analyze_articles([_make_article()], "Known facts.", _make_topic(), settings)

        assert result.has_new_info is False
        assert result.confidence == 0.0
        mock_create.assert_called_once()  # no retries for non-rate-limit


# ============================================================
# TestGenerateInitialKnowledgeRateLimit
# ============================================================


class TestGenerateInitialKnowledgeRateLimit:
    async def test_retries_on_rate_limit_and_succeeds(self) -> None:
        """generate_initial_knowledge retries on RateLimitError and succeeds."""
        rate_error = _make_rate_limit_error()
        expected = KnowledgeStateUpdate(sufficient_data=True, confidence=0.9, updated_summary="Summary.", token_count=0)
        mock_client, mock_create = _mock_instructor_client(expected)
        mock_create.side_effect = [rate_error, expected]
        settings = _make_settings()

        with (
            patch("app.analysis.llm._get_client", return_value=mock_client),
            patch("app.analysis.llm.asyncio.sleep"),
            patch("app.analysis.llm.count_tokens", return_value=10),
        ):
            result = await generate_initial_knowledge([_make_article()], _make_topic(), settings)

        assert result.updated_summary == "Summary."
        assert result.token_count == 10
        assert mock_create.call_count == 2

    async def test_raises_after_exhausting_rate_limit_retries(self) -> None:
        """generate_initial_knowledge propagates RateLimitError after max retries."""
        rate_error = _make_rate_limit_error()
        mock_client, mock_create = _mock_instructor_client(None)
        mock_create.side_effect = [rate_error] * 4
        settings = _make_settings()

        with (
            patch("app.analysis.llm._get_client", return_value=mock_client),
            patch("app.analysis.llm.asyncio.sleep"),
            pytest.raises(litellm.RateLimitError),
        ):
            await generate_initial_knowledge([_make_article()], _make_topic(), settings)

    async def test_raises_on_non_rate_limit_error(self) -> None:
        """generate_initial_knowledge propagates non-rate-limit errors immediately."""
        mock_client, mock_create = _mock_instructor_client(None)
        mock_create.side_effect = RuntimeError("unexpected failure")
        settings = _make_settings()

        with (
            patch("app.analysis.llm._get_client", return_value=mock_client),
            patch("app.analysis.llm.asyncio.sleep") as mock_sleep,
            pytest.raises(RuntimeError, match="unexpected failure"),
        ):
            await generate_initial_knowledge([_make_article()], _make_topic(), settings)

        mock_create.assert_called_once()
        mock_sleep.assert_not_called()


# ============================================================
# TestGenerateKnowledgeUpdateRateLimit
# ============================================================


class TestGenerateKnowledgeUpdateRateLimit:
    async def test_retries_on_rate_limit_and_succeeds(self) -> None:
        """generate_knowledge_update retries on RateLimitError and succeeds."""
        rate_error = _make_rate_limit_error()
        expected = KnowledgeStateUpdate(sufficient_data=True, confidence=0.9, updated_summary="Updated.", token_count=0)
        mock_client, mock_create = _mock_instructor_client(expected)
        mock_create.side_effect = [rate_error, expected]
        novelty = NoveltyResult(has_new_info=True, summary="New fact.", confidence=0.85)
        settings = _make_settings()

        with (
            patch("app.analysis.llm._get_client", return_value=mock_client),
            patch("app.analysis.llm.asyncio.sleep"),
            patch("app.analysis.llm.count_tokens", return_value=20),
        ):
            result = await generate_knowledge_update("Old summary.", novelty, _make_topic(), settings)

        assert result.updated_summary == "Updated."
        assert result.token_count == 20
        assert mock_create.call_count == 2

    async def test_raises_after_exhausting_rate_limit_retries(self) -> None:
        """generate_knowledge_update propagates RateLimitError after max retries."""
        rate_error = _make_rate_limit_error()
        mock_client, mock_create = _mock_instructor_client(None)
        mock_create.side_effect = [rate_error] * 4
        novelty = NoveltyResult(has_new_info=True, summary="X.", confidence=0.7)
        settings = _make_settings()

        with (
            patch("app.analysis.llm._get_client", return_value=mock_client),
            patch("app.analysis.llm.asyncio.sleep"),
            pytest.raises(litellm.RateLimitError),
        ):
            await generate_knowledge_update("Old.", novelty, _make_topic(), settings)

    async def test_raises_on_non_rate_limit_error(self) -> None:
        """generate_knowledge_update propagates non-rate-limit errors immediately."""
        mock_client, mock_create = _mock_instructor_client(None)
        mock_create.side_effect = ConnectionError("network down")
        novelty = NoveltyResult(has_new_info=True, summary="X.", confidence=0.7)
        settings = _make_settings()

        with (
            patch("app.analysis.llm._get_client", return_value=mock_client),
            patch("app.analysis.llm.asyncio.sleep") as mock_sleep,
            pytest.raises(ConnectionError, match="network down"),
        ):
            await generate_knowledge_update("Old.", novelty, _make_topic(), settings)

        mock_create.assert_called_once()
        mock_sleep.assert_not_called()
