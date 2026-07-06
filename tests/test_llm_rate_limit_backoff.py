"""Tests for rate-limit-aware retry with exponential backoff in LLM functions."""

import json
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import instructor
import litellm
import pytest
from litellm import ModelResponse
from litellm.types.utils import Choices, Message, Usage

import app.analysis.llm as llm_module
from app.analysis.llm import (
    InstructorRetryException,
    KnowledgeStateUpdate,
    NoveltyResult,
    _call_with_rate_limit_retry,
    _fallback_mode,
    _unwrap_bad_request,
    analyze_articles,
    generate_initial_knowledge,
    generate_knowledge_update,
)
from app.analysis.prompts import build_novelty_messages
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


def _make_bad_request_error() -> litellm.BadRequestError:
    # Shape of the real issue #53 400: litellm auto-injects the model's full
    # max_tokens (64000 for claude-haiku-4-5), which Anthropic rejects.
    return litellm.BadRequestError(
        message="AnthropicException - max_tokens: 64000 > 8192, the maximum allowed",
        llm_provider="anthropic",
        model="claude-haiku-4-5",
    )


class _FakeUsage:
    def __init__(self, prompt_tokens: int = 11, completion_tokens: int = 7) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeCompletion:
    def __init__(self, usage: _FakeUsage | None = None) -> None:
        self.usage = usage if usage is not None else _FakeUsage()


def _mock_instructor_client(return_value):
    """Create a mock instructor client.

    ``create_with_completion`` is the primary seam (analyze/init/update use it);
    the returned mock wraps the model in a ``(model, completion)`` tuple unless a
    ``side_effect`` is set by the test (e.g. to raise RateLimitError).
    """
    fake_completion = _FakeCompletion()

    async def _cwc(*_args, **_kwargs):
        return return_value, fake_completion

    mock_create = AsyncMock(side_effect=_cwc)
    mock_completions = MagicMock()
    mock_completions.create_with_completion = mock_create
    mock_completions.create = AsyncMock(return_value=return_value)
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
        mock_create.side_effect = [rate_error, (expected, _FakeCompletion())]
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
        mock_create.side_effect = [rate_error, (expected, _FakeCompletion())]
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
        mock_create.side_effect = [rate_error, (expected, _FakeCompletion())]
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


# ============================================================
# TestRealInstructorStackBackoff (OVH-008)
# ============================================================
#
# The mock-client tests above feed a bare ``RateLimitError`` straight out of
# ``create_with_completion`` — a shape production can never produce. In reality
# instructor wraps the underlying ``litellm.acompletion`` call in its own
# retry/error layer, so a 429 surfaces as ``InstructorRetryException`` (its
# ``__cause__`` is a tenacity ``RetryError``, not a ``RateLimitError``). These
# tests drive the REAL instructor stack via a fake ``acompletion`` that raises
# ``RateLimitError`` from inside the instructor call, so they catch the dead-code
# regression the unit tests above structurally cannot.


@contextmanager
def _real_instructor_raising(exc_factory):
    """Patch ``_get_client`` to a REAL instructor client over a fake acompletion.

    ``instructor.from_litellm`` is given a fake completion coroutine that raises
    whatever ``exc_factory()`` returns on every call, so the genuine instructor
    retry/wrapping layer runs (the bug surface), but no network call happens.
    Yields a ``{"calls": int}`` dict counting how many times the fake completion
    was invoked.
    """
    import app.analysis.llm as llm_module

    counter = {"calls": 0}

    async def _fake_acompletion(*_args, **_kwargs):
        counter["calls"] += 1
        raise exc_factory()

    real_client = instructor.from_litellm(_fake_acompletion)
    prev = dict(llm_module._clients)
    llm_module._clients.clear()
    try:
        with patch("app.analysis.llm._get_client", return_value=real_client):
            yield counter
    finally:
        llm_module._clients.clear()
        llm_module._clients.update(prev)


class TestRealInstructorStackBackoff:
    async def test_backoff_fires_through_real_instructor_wrapping(self) -> None:
        """A 429 raised inside the real instructor stack triggers the backoff.

        Regression guard for OVH-008: instructor wraps RateLimitError, so the
        operator-facing 'Rate limit hit ... retrying in Ns' warning and the
        asyncio.sleep between attempts must still fire on the real path.
        """
        settings = _make_settings(llm_max_retries=2)

        sleeps: list[float] = []

        async def _record_sleep(delay: float) -> None:
            sleeps.append(delay)

        with (
            _real_instructor_raising(_make_rate_limit_error),
            patch("app.analysis.llm.asyncio.sleep", side_effect=_record_sleep),
            patch("app.analysis.llm.logger") as mock_logger,
        ):
            result = await analyze_articles([_make_article()], "Known facts.", _make_topic(), settings)

        # analyze_articles stays fail-safe (settled decision #3).
        assert result.has_new_info is False
        assert result.confidence == 0.0
        assert result.error is not None

        # Backoff actually slept between attempts (llm_max_retries=2 -> 2 sleeps).
        assert len(sleeps) == 2
        assert all(d > 0 for d in sleeps)

        # The operator-facing rate-limit warning fired (was dead before the fix).
        warning_msgs = [c.args[0] for c in mock_logger.warning.call_args_list if c.args]
        assert any("Rate limit" in m for m in warning_msgs)

    async def test_instructor_does_not_immediately_hammer_on_rate_limit(self) -> None:
        """Each backoff attempt makes exactly one provider call, not max_retries.

        Before the fix, instructor retried the 429 immediately ``max_retries``
        times per attempt (zero delay, hammering the throttled provider). After
        the fix, instructor must NOT retry on RateLimitError, so the call count
        equals the number of backoff attempts (initial + retries), not a product.
        """
        settings = _make_settings(llm_max_retries=2)

        with (
            _real_instructor_raising(_make_rate_limit_error) as counter,
            patch("app.analysis.llm.asyncio.sleep", new=AsyncMock()),
        ):
            await analyze_articles([_make_article()], "Known facts.", _make_topic(), settings)

        # llm_max_retries=2 -> 3 backoff attempts (initial + 2 retries), one
        # provider call each. NOT 3 * (validation retries) — instructor must not
        # immediately re-fire on the 429.
        assert counter["calls"] == 3

    async def test_validation_retry_still_works_for_non_rate_limit(self) -> None:
        """Instructor still retries genuine validation failures (not 429s).

        Disabling instructor's retry-on-RateLimitError must not disable its
        structured-output validation retries. A persistent validation failure
        should still be attempted ``llm_max_retries + 1`` times by instructor
        within a SINGLE backoff attempt (no rate-limit sleep involved).
        """
        settings = _make_settings(llm_max_retries=2)

        def _validation_error() -> Exception:
            # Not a RateLimitError: instructor owns the retry for this one.
            return litellm.APIError(
                status_code=400,
                message="bad output",
                llm_provider="openai",
                model="gpt-4",
            )

        sleeps: list[float] = []

        async def _record_sleep(delay: float) -> None:
            sleeps.append(delay)

        with (
            _real_instructor_raising(_validation_error) as counter,
            patch("app.analysis.llm.asyncio.sleep", side_effect=_record_sleep),
        ):
            result = await analyze_articles([_make_article()], "Known facts.", _make_topic(), settings)

        # Fail-safe result, and no rate-limit *backoff* delay (this is not a 429).
        # Instructor's own between-retry waits are 0.0 (no wait policy); our
        # rate-limit backoff would inject a positive base_delay, which it must not.
        assert result.has_new_info is False
        assert all(d == 0 for d in sleeps)
        # Instructor retried the non-429 itself: max_retries + 1 = 3 attempts.
        # NB: a bare ``APIError(status_code=400)`` is NOT a ``BadRequestError``, so
        # the permanent-4xx exclusion (issue #53) does not apply — it stays retried.
        # This pins the exclusion to exception TYPE, not HTTP status code.
        assert counter["calls"] == 3


# ============================================================
# TestPermanentClientErrorNotRetried (issue #53)
# ============================================================
#
# A permanent client error (litellm 4xx, e.g. BadRequestError) must NOT be
# retried and must NOT be masked. Issue #53: a new topic's first check surfaced
# the opaque ``RetryError[<Future ... raised BadRequestError>]`` because a
# permanent 400 was treated as retryable — fired ``max_retries+1`` times, then
# wrapped so the provider's real message was hidden. The retry policy now
# excludes permanent litellm 4xx by TYPE, so the call happens once and the real
# error surfaces (bare on ``__cause__``; readable in ``str``).
#
# These also pin the MODE-INVARIANT side of the structured-output fallback: the
# ``max_tokens`` 400 (``_make_bad_request_error``) is sent identically in every
# mode, so ``_fallback_mode`` returns None — one call, no TOOLS->JSON->MD_JSON
# hop (contrast ``TestStructuredOutputModeFallback``, where a fixable 400 makes
# the same helper try up to 3 modes: the 1-vs-3 pair pins the real invariant).


class TestPermanentClientErrorNotRetried:
    """Mode-invariant 400s (e.g. max_tokens): retried once, never fall back."""

    async def test_init_surfaces_real_error_without_retry(self) -> None:
        """generate_initial_knowledge: one call, real message, no RetryError mask."""
        settings = _make_settings(llm_max_retries=2)

        with (
            _real_instructor_raising(_make_bad_request_error) as counter,
            patch("app.analysis.llm.asyncio.sleep", new=AsyncMock()),
            pytest.raises(InstructorRetryException) as exc_info,
        ):
            await generate_initial_knowledge([_make_article()], _make_topic(), settings)

        # Called exactly once — a permanent 400 is not retried (was max_retries+1=3).
        assert counter["calls"] == 1
        # The provider's real message surfaces; the opaque RetryError wrapper is gone,
        # and the underlying BadRequestError is preserved on __cause__.
        assert "RetryError" not in str(exc_info.value)
        assert "max_tokens" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, litellm.BadRequestError)

    async def test_analyze_articles_reports_real_error_not_wrapper(self) -> None:
        """analyze_articles stays fail-safe but stores the real error, not RetryError[...]."""
        settings = _make_settings(llm_max_retries=2)

        with (
            _real_instructor_raising(_make_bad_request_error) as counter,
            patch("app.analysis.llm.asyncio.sleep", new=AsyncMock()),
        ):
            result = await analyze_articles([_make_article()], "Known facts.", _make_topic(), settings)

        # Fail-safe result preserved (settled decision #3), no 3x hammering.
        assert result.has_new_info is False
        assert result.confidence == 0.0
        assert counter["calls"] == 1
        # Stored error is the real provider message (truncated 200 chars), not RetryError[...].
        assert result.error is not None
        assert "RetryError" not in result.error
        assert "max_tokens" in result.error

    async def test_genuine_validation_error_still_retried(self) -> None:
        """Excluding permanent 4xx must NOT disable structured-output validation retries."""
        settings = _make_settings(llm_max_retries=2)

        def _pydantic_validation_error() -> Exception:
            from pydantic import BaseModel

            class _Req(BaseModel):
                x: int

            try:
                _Req.model_validate({})
            except Exception as e:  # pydantic.ValidationError
                return e
            raise AssertionError("expected a ValidationError")

        with (
            _real_instructor_raising(_pydantic_validation_error) as counter,
            patch("app.analysis.llm.asyncio.sleep", new=AsyncMock()),
            pytest.raises(InstructorRetryException),
        ):
            await generate_initial_knowledge([_make_article()], _make_topic(), settings)

        # Not a permanent 4xx -> instructor still retries it max_retries + 1 = 3 times.
        assert counter["calls"] == 3


# ============================================================
# TestStructuredOutputModeFallback (issue #53 reopen: DeepSeek thinking mode)
# ============================================================
#
# DeepSeek thinking mode (deepseek-reasoner) rejects instructor's default TOOLS
# mode, which sends a forced named ``tool_choice``, with an HTTP 400. The fix
# falls back TOOLS -> JSON -> MD_JSON per call. These drive the REAL instructor
# stack: they patch ``litellm.acompletion`` (NOT a pre-built client) so the real
# ``_get_client`` builds a distinct client per mode over the fake — the only way
# to observe the mode switch (a fixed client bakes in one mode and keeps sending
# ``tool_choice`` forever, so tests 1/2/5 are unreachable with it).

# Verbatim provider message from the issue #53 reopen report.
_DEEPSEEK_TOOL_CHOICE_400 = (
    "litellm.BadRequestError: DeepseekException - "
    '{"error":{"message":"Error from provider (DeepSeek): '
    'Thinking mode does not support this tool_choice","type":"invalid_request_error"}}'
)


def _tool_choice_error() -> litellm.BadRequestError:
    return litellm.BadRequestError(
        message=_DEEPSEEK_TOOL_CHOICE_400, llm_provider="deepseek", model="deepseek-reasoner"
    )


def _json_mode_error() -> litellm.BadRequestError:
    # Synthetic and heuristic-only: whether DeepSeek thinking mode also rejects
    # response_format is unverified until the reporter retests. The broad
    # (fall-back-on-any-fixable-400) design does not depend on the wording.
    return litellm.BadRequestError(
        message="DeepseekException - response_format json_object not supported in thinking mode",
        llm_provider="deepseek",
        model="deepseek-reasoner",
    )


def _md_json_error() -> litellm.BadRequestError:
    return litellm.BadRequestError(
        message="DeepseekException - structured output rejected in all modes",
        llm_provider="deepseek",
        model="deepseek-reasoner",
    )


def _completion_for(model_instance) -> ModelResponse:
    """A litellm completion whose content is RAW ``json.dumps`` of the model.

    Raw JSON (no ```json fence) parses in BOTH json_mode and markdown_json_mode; a
    fence would make json_mode raise a ValidationError that instructor
    validation-retries, silently inflating the call count. The usage block lets
    ``_extract_usage`` populate token counts.
    """
    content = json.dumps(model_instance.model_dump())
    message = Message(content=content, role="assistant")
    choice = Choices(message=message, index=0, finish_reason="stop")
    return ModelResponse(choices=[choice], usage=Usage(prompt_tokens=11, completion_tokens=7))


@contextmanager
def _fake_acompletion(handler):
    """Patch ``litellm.acompletion`` with ``handler`` and reset the per-mode cache.

    ``handler(kwargs)`` returns a ``ModelResponse`` (success) or raises (provider
    error). Clearing ``_clients`` forces the real ``_get_client`` to rebuild one
    instructor client per mode over the fake, so the mode fallback runs for real.
    Yields the list of per-call kwargs for assertions on ``tool_choice`` /
    ``response_format`` and the (instructor-mutated) messages.
    """
    calls: list[dict] = []

    async def _acompletion(*_args, **kwargs):
        calls.append(kwargs)
        return handler(kwargs)

    prev = dict(llm_module._clients)
    llm_module._clients.clear()
    try:
        with patch("app.analysis.llm.litellm.acompletion", new=_acompletion):
            yield calls
    finally:
        llm_module._clients.clear()
        llm_module._clients.update(prev)


class TestStructuredOutputModeFallback:
    def setup_method(self) -> None:
        # Net-new reset: the per-mode cache is module-global, so wipe it before
        # each test so a client built over a prior fake never leaks in.
        llm_module._clients.clear()

    async def test_tools_400_falls_back_to_json(self) -> None:
        """Test 1: TOOLS tool_choice 400 -> retry in JSON mode -> success."""
        expected = NoveltyResult(has_new_info=True, confidence=0.9)

        def handler(kwargs: dict) -> ModelResponse:
            if "tool_choice" in kwargs:
                raise _tool_choice_error()
            return _completion_for(expected)

        settings = _make_settings(llm=LLMSettings(model="deepseek/deepseek-reasoner", api_key="k"))

        with _fake_acompletion(handler) as calls:
            result = await analyze_articles([_make_article()], "Known facts.", _make_topic(), settings)

        assert result.has_new_info is True
        assert len(calls) == 2  # TOOLS (400) then JSON (success)
        assert "tool_choice" not in calls[1]
        assert calls[1]["response_format"] == {"type": "json_object"}

    async def test_json_400_falls_back_to_md_json(self) -> None:
        """Test 2: TOOLS + JSON both 400 -> MD_JSON succeeds; fresh-build guards."""
        expected = NoveltyResult(has_new_info=True, confidence=0.8)

        def handler(kwargs: dict) -> ModelResponse:
            if "tool_choice" in kwargs:
                raise _tool_choice_error()
            if kwargs.get("response_format"):
                raise _json_mode_error()
            return _completion_for(expected)

        settings = _make_settings(llm=LLMSettings(model="deepseek/deepseek-reasoner", api_key="k"))
        build_spy = MagicMock(side_effect=build_novelty_messages)

        with (
            _fake_acompletion(handler) as calls,
            patch("app.analysis.llm.build_novelty_messages", build_spy),
        ):
            result = await analyze_articles([_make_article()], "Known facts.", _make_topic(), settings)

        assert result.has_new_info is True
        assert len(calls) == 3  # TOOLS -> JSON -> MD_JSON
        # (a) Messages are rebuilt fresh once per attempt (immune to instructor
        # internals): the factory ran exactly once per mode tried.
        assert build_spy.call_count == 3
        # (b) No mutation leak: on the fresh MD_JSON build, instructor's injected
        # schema marker appears exactly once in the single system message (a reused
        # list would double it — the system-message count stays 1 either way).
        system_msgs = [m for m in calls[2]["messages"] if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0]["content"].lower().count("json_schema") == 1

    async def test_all_modes_rejected_reraises_and_fails_safe(self) -> None:
        """Test 2b: every mode 400 -> init re-raises real message; analyze fails safe."""

        def handler(kwargs: dict) -> ModelResponse:
            if "tool_choice" in kwargs:
                raise _tool_choice_error()
            if kwargs.get("response_format"):
                raise _json_mode_error()
            raise _md_json_error()

        settings = _make_settings(llm=LLMSettings(model="deepseek/deepseek-reasoner", api_key="k"))

        # Critical path re-raises with the real MD_JSON message, no RetryError mask.
        with (
            _fake_acompletion(handler) as calls,
            pytest.raises(InstructorRetryException) as exc_info,
        ):
            await generate_initial_knowledge([_make_article()], _make_topic(), settings)

        assert len(calls) == 3  # all three modes tried once each
        assert "RetryError" not in str(exc_info.value)
        assert "structured output rejected" in str(exc_info.value)

        # analyze_articles stays fail-safe but stores the real message.
        with _fake_acompletion(handler) as calls:
            result = await analyze_articles([_make_article()], "Known facts.", _make_topic(), settings)

        assert result.has_new_info is False
        assert result.confidence == 0.0
        assert len(calls) == 3
        assert result.error is not None
        assert "structured output rejected" in result.error

    async def test_rate_limit_mid_chain_reruns_from_tools(self) -> None:
        """Test 5: TOOLS 400 -> JSON 429 -> backoff -> rerun TOOLS 400 -> JSON ok."""
        expected = NoveltyResult(has_new_info=True, confidence=0.7)
        state = {"json_calls": 0}

        def handler(kwargs: dict) -> ModelResponse:
            if "tool_choice" in kwargs:
                raise _tool_choice_error()
            if kwargs.get("response_format"):
                state["json_calls"] += 1
                if state["json_calls"] == 1:
                    raise _make_rate_limit_error()
                return _completion_for(expected)
            return _completion_for(expected)  # MD_JSON not reached

        settings = _make_settings(llm=LLMSettings(model="deepseek/deepseek-reasoner", api_key="k"))
        sleeps: list[float] = []

        async def _record_sleep(delay: float) -> None:
            sleeps.append(delay)

        with (
            _fake_acompletion(handler) as calls,
            patch("app.analysis.llm.asyncio.sleep", side_effect=_record_sleep),
        ):
            result = await analyze_articles([_make_article()], "Known facts.", _make_topic(), settings)

        assert result.has_new_info is True
        assert result.error is None
        # run1: TOOLS(400) + JSON(429); run2: TOOLS(400) + JSON(ok) = 4 calls, 1 sleep.
        assert len(calls) == 4
        assert len(sleeps) == 1


# ============================================================
# TestFallbackPredicates (unit tests, no instructor stack)
# ============================================================


class TestFallbackPredicates:
    """Direct tests for ``_unwrap_bad_request`` / ``_fallback_mode``."""

    def test_unwrap_returns_bare_bad_request(self) -> None:
        # The bare-instance branch integration tests can't reach: instructor
        # always wraps, so only a direct call ever sees an unwrapped 400.
        err = _make_bad_request_error()
        assert _unwrap_bad_request(err) is err

    def test_unwrap_returns_none_for_rate_limit(self) -> None:
        assert _unwrap_bad_request(_make_rate_limit_error()) is None

    def test_plain_400_falls_back_to_next_mode(self) -> None:
        err = litellm.BadRequestError(message="forced tool_choice rejected", llm_provider="x", model="m")
        assert _fallback_mode(instructor.Mode.TOOLS, err) is instructor.Mode.JSON
        assert _fallback_mode(instructor.Mode.JSON, err) is instructor.Mode.MD_JSON

    def test_md_json_is_terminal(self) -> None:
        err = litellm.BadRequestError(message="still rejected", llm_provider="x", model="m")
        assert _fallback_mode(instructor.Mode.MD_JSON, err) is None

    def test_context_window_error_does_not_fall_back(self) -> None:
        err = litellm.ContextWindowExceededError(message="prompt too long", model="m", llm_provider="x")
        assert _fallback_mode(instructor.Mode.TOOLS, err) is None

    def test_max_tokens_400_does_not_fall_back(self) -> None:
        # Mode-invariant: _bounded_max_tokens is identical in every mode.
        assert _fallback_mode(instructor.Mode.TOOLS, _make_bad_request_error()) is None

    def test_rate_limit_does_not_fall_back(self) -> None:
        assert _fallback_mode(instructor.Mode.TOOLS, _make_rate_limit_error()) is None
