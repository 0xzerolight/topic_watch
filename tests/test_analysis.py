"""Tests for the LLM analysis module: prompts, structured output, knowledge management."""

import sqlite3
from datetime import UTC
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
import pytest
from pydantic import ValidationError

from app.analysis.knowledge import initialize_knowledge, update_knowledge
from app.analysis.llm import (
    CompressedKnowledge,
    KnowledgeStateUpdate,
    NoveltyResult,
    analyze_articles,
    compress_knowledge_summary,
    count_tokens,
    generate_initial_knowledge,
    generate_knowledge_update,
)
from app.analysis.prompts import (
    _content_quality_tag,
    _format_articles,
    build_knowledge_compress_messages,
    build_knowledge_init_messages,
    build_knowledge_update_messages,
    build_novelty_messages,
)
from app.config import LLMSettings, Settings
from app.crud import create_knowledge_state, create_topic, get_knowledge_state
from app.models import Article, KnowledgeState, Topic

# --- Fixtures ---


def _make_settings(**overrides) -> Settings:
    """Create a Settings instance for testing (bypasses YAML loading)."""
    defaults = {
        "llm": LLMSettings(model="openai/gpt-4o-mini", api_key="test-key"),
        "knowledge_state_max_tokens": 2000,
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _make_topic(**overrides) -> Topic:
    """Create a Topic instance for testing."""
    defaults = {
        "id": 1,
        "name": "Test Topic",
        "description": "A test topic for unit tests",
        "feed_urls": ["https://example.com/feed.xml"],
    }
    defaults.update(overrides)
    return defaults if overrides.get("_raw") else Topic(**defaults)


def _make_article(**overrides) -> Article:
    """Create an Article instance for testing."""
    defaults = {
        "id": 1,
        "topic_id": 1,
        "title": "Test Article",
        "url": "https://example.com/article-1",
        "content_hash": "abc123",
        "raw_content": "This is the article content about important news.",
        "source_feed": "https://example.com/feed.xml",
    }
    defaults.update(overrides)
    return Article(**defaults)


class _FakeUsage:
    def __init__(self, prompt_tokens: int = 11, completion_tokens: int = 7) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeCompletion:
    def __init__(self, usage: _FakeUsage | None = None) -> None:
        self.usage = usage if usage is not None else _FakeUsage()


def _mock_instructor_client(return_value, *, completion=None):
    """Create a mock instructor client.

    The returned ``mock_create`` is the single seam tests assert on. It backs
    ``client.chat.completions.create`` directly (used by ``compress_knowledge_summary``)
    AND ``create_with_completion`` (used by analyze/init/update), where its
    return value is wrapped as ``(model, completion)``. So ``mock_create.call_args``
    and ``mock_create.call_count`` work for either path.
    """
    fake_completion = completion if completion is not None else _FakeCompletion()
    mock_create = AsyncMock(return_value=return_value)

    async def _cwc(*args, **kwargs):
        model = await mock_create(*args, **kwargs)
        return model, fake_completion

    mock_completions = MagicMock()
    mock_completions.create = mock_create
    mock_completions.create_with_completion = AsyncMock(side_effect=_cwc)
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions
    mock_client = MagicMock()
    mock_client.chat = mock_chat
    return mock_client, mock_create


# ============================================================
# TestNoveltyResult
# ============================================================


class TestNoveltyResult:
    def test_valid_construction(self) -> None:
        result = NoveltyResult(
            has_new_info=True,
            summary="New release date announced",
            key_facts=["Release date: June 2025"],
            source_urls=["https://example.com/1"],
            confidence=0.95,
        )
        assert result.has_new_info is True
        assert result.summary == "New release date announced"
        assert len(result.key_facts) == 1
        assert result.confidence == 0.95

    def test_confidence_bounds(self) -> None:
        with pytest.raises(ValidationError):
            NoveltyResult(has_new_info=False, confidence=1.5)
        with pytest.raises(ValidationError):
            NoveltyResult(has_new_info=False, confidence=-0.1)

    def test_default_values(self) -> None:
        result = NoveltyResult(has_new_info=False, confidence=0.8)
        assert result.summary is None
        assert result.key_facts == []
        assert result.source_urls == []
        assert result.reasoning == ""
        assert result.relevance == 0.0

    def test_no_summary_when_no_new_info(self) -> None:
        result = NoveltyResult(has_new_info=False, confidence=0.9)
        assert result.has_new_info is False
        assert result.summary is None

    def test_reasoning_field(self) -> None:
        result = NoveltyResult(
            has_new_info=True,
            reasoning="Article [1] mentions a new date not in the knowledge state.",
            summary="New date found",
            confidence=0.9,
        )
        assert "new date" in result.reasoning

    def test_relevance_field(self) -> None:
        result = NoveltyResult(has_new_info=True, confidence=0.9, relevance=0.85)
        assert result.relevance == 0.85

    def test_relevance_bounds(self) -> None:
        with pytest.raises(ValidationError):
            NoveltyResult(has_new_info=False, confidence=0.5, relevance=1.5)
        with pytest.raises(ValidationError):
            NoveltyResult(has_new_info=False, confidence=0.5, relevance=-0.1)


# ============================================================
# TestKnowledgeStateUpdate
# ============================================================


class TestKnowledgeStateUpdate:
    def test_valid_construction(self) -> None:
        update = KnowledgeStateUpdate(
            sufficient_data=True,
            confidence=0.9,
            updated_summary="Summary of known facts.",
            token_count=150,
        )
        assert update.updated_summary == "Summary of known facts."
        assert update.token_count == 150
        assert update.sufficient_data is True
        assert update.confidence == 0.9

    def test_token_count_defaults_to_zero(self) -> None:
        update = KnowledgeStateUpdate(
            sufficient_data=True,
            confidence=0.8,
            updated_summary="text",
        )
        assert update.token_count == 0

    def test_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            KnowledgeStateUpdate(sufficient_data=True, confidence=0.9)  # missing updated_summary

    def test_insufficient_data_construction(self) -> None:
        update = KnowledgeStateUpdate(
            sufficient_data=False,
            confidence=0.3,
            updated_summary="Articles did not contain relevant release date information.",
        )
        assert update.sufficient_data is False
        assert update.confidence == 0.3

    def test_confidence_bounds(self) -> None:
        with pytest.raises(ValidationError):
            KnowledgeStateUpdate(sufficient_data=True, confidence=1.5, updated_summary="x")
        with pytest.raises(ValidationError):
            KnowledgeStateUpdate(sufficient_data=True, confidence=-0.1, updated_summary="x")


# ============================================================
# TestContentQualityTag
# ============================================================


class TestContentQualityTag:
    def test_no_content(self) -> None:
        assert _content_quality_tag(None) == "[NO CONTENT]"
        assert _content_quality_tag("") == "[NO CONTENT]"

    def test_stub_content(self) -> None:
        tag = _content_quality_tag("Short snippet only.")
        assert "[STUB" in tag

    def test_sufficient_content(self) -> None:
        tag = _content_quality_tag("x" * 200)
        assert tag == ""


# ============================================================
# TestFormatArticles
# ============================================================


class TestFormatArticles:
    def test_formats_multiple_articles(self) -> None:
        articles = [
            _make_article(id=1, title="First", url="https://example.com/1"),
            _make_article(id=2, title="Second", url="https://example.com/2"),
        ]
        result = _format_articles(articles)
        assert "[1] First" in result
        assert "[2] Second" in result
        assert "URL: https://example.com/1" in result
        assert "URL: https://example.com/2" in result

    def test_includes_source_feed(self) -> None:
        article = _make_article(source_feed="https://news.google.com/rss/search?q=test")
        result = _format_articles([article])
        assert "Source: https://news.google.com/rss/search?q=test" in result

    def test_handles_none_content(self) -> None:
        article = _make_article(raw_content=None)
        result = _format_articles([article])
        assert "(no content available)" in result
        assert "[NO CONTENT]" in result

    def test_stub_content_tagged(self) -> None:
        article = _make_article(raw_content="Short.")
        result = _format_articles([article])
        assert "[STUB" in result

    def test_truncates_long_content(self) -> None:
        long_content = "word " * 400
        article = _make_article(raw_content=long_content)
        result = _format_articles([article], max_content_chars=100)
        # Content should be truncated — not the full original
        assert len(result) < len(long_content)

    def test_sentence_boundary_truncation(self) -> None:
        # Content with clear sentence boundaries
        content = "First sentence here. Second sentence here. Third sentence here. " + "x" * 1500
        article = _make_article(raw_content=content)
        result = _format_articles([article], max_content_chars=80)
        # Truncated content (inside the untrusted fence) should prefer a sentence
        # boundary over a hard cut.
        body = result.split("instructions) ---\n", 1)[1].split("\n    --- END UNTRUSTED", 1)[0]
        assert "..." in body or body.rstrip().endswith(".")

    def test_published_at_emitted_in_header_when_set(self) -> None:
        from datetime import datetime as dt

        pub = dt(2026, 5, 28, 12, 0, 0, tzinfo=UTC)
        article = _make_article(published_at=pub)
        result = _format_articles([article])
        assert "Published: 2026-05-28" in result

    def test_published_at_omitted_when_none(self) -> None:
        article = _make_article(published_at=None)
        result = _format_articles([article])
        assert "Published:" not in result

    def test_published_at_mixed_batch_one_line(self) -> None:
        """A mixed batch (one dated, one None) formats without error; exactly one Published line."""
        from datetime import datetime as dt

        pub = dt(2026, 5, 28, tzinfo=UTC)
        articles = [
            _make_article(id=1, title="Dated", published_at=pub),
            _make_article(id=2, title="Undated", published_at=None),
        ]
        result = _format_articles(articles)
        assert result.count("Published:") == 1
        assert "Published: 2026-05-28" in result

    def test_published_at_in_header_not_fenced_body(self) -> None:
        """Published line must appear in the trusted header block, not inside the untrusted fence."""
        from datetime import datetime as dt

        pub = dt(2026, 5, 28, tzinfo=UTC)
        article = _make_article(published_at=pub)
        result = _format_articles([article])
        # Split on the begin marker to isolate header vs. fenced body
        header_block, fenced_and_after = result.split("--- BEGIN UNTRUSTED", 1)
        assert "Published: 2026-05-28" in header_block
        assert "Published:" not in fenced_and_after


# ============================================================
# TestBuildNoveltyMessages
# ============================================================


class TestBuildNoveltyMessages:
    def test_returns_system_and_user_messages(self) -> None:
        topic = _make_topic()
        articles = [_make_article()]
        messages = build_novelty_messages(articles, "Known facts here.", topic)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_system_message_contains_grounding(self) -> None:
        topic = _make_topic()
        articles = [_make_article()]
        messages = build_novelty_messages(articles, "Known facts.", topic)
        system_msg = messages[0]["content"]
        assert "CRITICAL RULES" in system_msg
        assert "ONLY" in system_msg
        assert "training data" in system_msg

    def test_user_message_includes_context(self) -> None:
        topic = _make_topic(name="Elden Ring DLC")
        articles = [_make_article(title="DLC Release Date")]
        messages = build_novelty_messages(articles, "DLC announced.", topic)
        user_msg = messages[1]["content"]
        assert "Elden Ring DLC" in user_msg
        assert "DLC announced." in user_msg
        assert "DLC Release Date" in user_msg

    def test_handles_empty_knowledge_state(self) -> None:
        topic = _make_topic()
        articles = [_make_article()]
        messages = build_novelty_messages(articles, "", topic)
        user_msg = messages[1]["content"]
        assert "No existing knowledge state." in user_msg

    def test_system_message_contains_calibration_scale(self) -> None:
        topic = _make_topic()
        articles = [_make_article()]
        messages = build_novelty_messages(articles, "Known.", topic)
        system_msg = messages[0]["content"]
        assert "0.9-1.0" in system_msg
        assert "0.3-0.4" in system_msg
        assert "Do NOT default to 0.7-0.8" in system_msg

    def test_system_message_contains_scope_instruction(self) -> None:
        topic = _make_topic()
        articles = [_make_article()]
        messages = build_novelty_messages(articles, "Known.", topic)
        system_msg = messages[0]["content"]
        assert "SCOPE to the topic description" in system_msg

    def test_system_message_rejects_speculation(self) -> None:
        topic = _make_topic()
        articles = [_make_article()]
        messages = build_novelty_messages(articles, "Known.", topic)
        system_msg = messages[0]["content"]
        assert "Rumors or unverified claims" in system_msg

    def test_system_message_contains_relevance_instruction(self) -> None:
        topic = _make_topic()
        articles = [_make_article()]
        messages = build_novelty_messages(articles, "Known.", topic)
        system_msg = messages[0]["content"]
        assert "Set relevance" in system_msg

    def test_system_message_instructs_summary_population(self) -> None:
        # OVH-026: the novelty summary feeds the knowledge-update prompt's
        # "New Findings to Incorporate" block, so the system prompt must tell
        # the model to write it (null only when there is no new info).
        topic = _make_topic()
        articles = [_make_article()]
        messages = build_novelty_messages(articles, "Known.", topic)
        system_msg = messages[0]["content"]
        lowered = system_msg.lower()
        assert "summary" in lowered
        # An explicit instruction to populate the summary field.
        assert "in `summary`" in lowered or "in summary" in lowered
        assert "has_new_info is false" in lowered

    def test_novelty_result_summary_field_has_description(self) -> None:
        # OVH-026: instructor surfaces Field descriptions to the model, so the
        # summary field must carry one (the prompt instruction alone is not
        # enough — instructor builds the schema from the model).
        field = NoveltyResult.model_fields["summary"]
        assert field.description, "summary Field must have a description for instructor"
        assert "summary" in field.description.lower()

    def test_novelty_system_contains_date_anchoring_instruction(self) -> None:
        """_NOVELTY_SYSTEM must contain the date-anchoring rule."""
        from app.analysis.prompts import _NOVELTY_SYSTEM

        assert "resolve it to an absolute date" in _NOVELTY_SYSTEM


# ============================================================
# TestBuildKnowledgeInitMessages
# ============================================================


class TestBuildKnowledgeInitMessages:
    def test_includes_max_tokens_in_system(self) -> None:
        topic = _make_topic()
        articles = [_make_article()]
        messages = build_knowledge_init_messages(articles, topic, max_tokens=2000)
        system_msg = messages[0]["content"]
        assert "2000" in system_msg

    def test_system_message_contains_grounding(self) -> None:
        topic = _make_topic()
        articles = [_make_article()]
        messages = build_knowledge_init_messages(articles, topic, max_tokens=2000)
        system_msg = messages[0]["content"]
        assert "CRITICAL RULES" in system_msg
        assert "ONLY" in system_msg
        assert "Do NOT add facts" in system_msg

    def test_formats_articles_in_user_message(self) -> None:
        topic = _make_topic()
        articles = [_make_article(title="Important Article")]
        messages = build_knowledge_init_messages(articles, topic, max_tokens=2000)
        user_msg = messages[1]["content"]
        assert "Important Article" in user_msg

    def test_system_message_scoped_to_description(self) -> None:
        topic = _make_topic()
        articles = [_make_article()]
        messages = build_knowledge_init_messages(articles, topic, max_tokens=2000)
        system_msg = messages[0]["content"]
        assert "relevant to the topic description" in system_msg

    def test_init_prompt_teaches_absence_as_baseline(self) -> None:
        """The init prompt must teach the LLM that a not-yet-occurred / negative
        current state IS a sufficient baseline for forward-looking topics."""
        topic = _make_topic()
        articles = [_make_article()]
        messages = build_knowledge_init_messages(articles, topic, max_tokens=2000)
        system_msg = messages[0]["content"]
        # Must explicitly call out forward-looking descriptions
        assert "forward-looking" in system_msg
        # Must name the not-yet-occurred / negative state concept
        assert "not-yet-occurred" in system_msg or "not yet occurred" in system_msg

    def test_init_prompt_narrows_sufficient_data_false_condition(self) -> None:
        """sufficient_data=false must be narrowed: only for off-topic articles or
        those establishing NO current state — not merely because the event hasn't happened."""
        topic = _make_topic()
        articles = [_make_article()]
        messages = build_knowledge_init_messages(articles, topic, max_tokens=2000)
        system_msg = messages[0]["content"]
        # The narrowed condition must mention off-topic as one trigger
        assert "off-topic" in system_msg
        # Must clarify: sufficient_data=false is NOT just because event hasn't happened
        assert "has not happened" in system_msg or "hasn't happened" in system_msg or "not yet occurred" in system_msg

    def test_init_prompt_output_format_includes_current_status(self) -> None:
        """The output format section must include a Current Status category so the
        LLM knows where to capture not-yet-occurred / awaiting states."""
        topic = _make_topic()
        articles = [_make_article()]
        messages = build_knowledge_init_messages(articles, topic, max_tokens=2000)
        system_msg = messages[0]["content"]
        assert "Current Status" in system_msg

    def test_sufficient_data_field_description_reflects_narrowed_semantics(self) -> None:
        """The Pydantic Field description for sufficient_data must document the narrowed
        semantics: false only when off-topic or no current state can be established."""
        field = KnowledgeStateUpdate.model_fields["sufficient_data"]
        desc = field.description or ""
        # Must mention off-topic or the narrowed condition
        assert "off-topic" in desc or "no current state" in desc
        # Must not remain the old, too-broad wording
        assert "lack enough relevant information" not in desc

    def test_knowledge_init_system_contains_date_anchoring_instruction(self) -> None:
        """_KNOWLEDGE_INIT_SYSTEM must contain the date-anchoring rule."""
        from app.analysis.prompts import _KNOWLEDGE_INIT_SYSTEM

        assert "resolve it to an absolute date" in _KNOWLEDGE_INIT_SYSTEM

    def test_stub_rule_forbids_confirmed_facts_placement(self) -> None:
        """STUB/NO-CONTENT claims must be barred from Confirmed Facts unless corroborated."""
        from app.analysis.prompts import _KNOWLEDGE_INIT_SYSTEM

        assert "NEVER under **Confirmed Facts**" in _KNOWLEDGE_INIT_SYSTEM

    def test_stub_rule_requires_reported_claimed_for_stub_only_claims(self) -> None:
        """STUB-only claims must be directed to Reported/Claimed category."""
        from app.analysis.prompts import _KNOWLEDGE_INIT_SYSTEM

        assert "must go under **Reported/Claimed**" in _KNOWLEDGE_INIT_SYSTEM

    def test_stub_rule_still_contains_weigh_lower_phrasing(self) -> None:
        """The original weigh-lower phrasing must still be present (appended, not replaced)."""
        from app.analysis.prompts import _KNOWLEDGE_INIT_SYSTEM

        assert "weigh them lower and rely primarily on their titles" in _KNOWLEDGE_INIT_SYSTEM


# ============================================================
# TestBuildKnowledgeUpdateMessages
# ============================================================


class TestBuildKnowledgeUpdateMessages:
    def test_includes_current_summary_and_findings(self) -> None:
        topic = _make_topic()
        messages = build_knowledge_update_messages(
            current_summary="Old summary.",
            novelty_summary="New price announced.",
            key_facts=["Price: $39.99"],
            topic=topic,
            max_tokens=2000,
        )
        user_msg = messages[1]["content"]
        assert "Old summary." in user_msg
        assert "New price announced." in user_msg
        assert "Price: $39.99" in user_msg

    def test_includes_max_tokens_budget(self) -> None:
        topic = _make_topic()
        messages = build_knowledge_update_messages(
            current_summary="Summary.",
            novelty_summary="Update.",
            key_facts=[],
            topic=topic,
            max_tokens=1500,
        )
        system_msg = messages[0]["content"]
        assert "1500" in system_msg

    def test_system_message_contains_grounding(self) -> None:
        topic = _make_topic()
        messages = build_knowledge_update_messages(
            current_summary="Summary.",
            novelty_summary="Update.",
            key_facts=[],
            topic=topic,
            max_tokens=2000,
        )
        system_msg = messages[0]["content"]
        assert "CRITICAL RULES" in system_msg
        assert "training data" in system_msg


# ============================================================
# TestBuildKnowledgeCompressMessages
# ============================================================


class TestBuildKnowledgeCompressMessages:
    def test_includes_current_summary(self) -> None:
        topic = _make_topic()
        messages = build_knowledge_compress_messages(
            current_summary="Verbose old knowledge state with many facts.",
            topic=topic,
            max_tokens=1500,
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "Verbose old knowledge state" in messages[1]["content"]

    def test_includes_max_tokens_budget(self) -> None:
        topic = _make_topic()
        messages = build_knowledge_compress_messages(current_summary="Summary.", topic=topic, max_tokens=1234)
        assert "1234" in messages[0]["content"]

    def test_system_message_preserves_facts_and_grounding(self) -> None:
        topic = _make_topic()
        messages = build_knowledge_compress_messages(current_summary="Summary.", topic=topic, max_tokens=2000)
        system_msg = messages[0]["content"]
        assert "CRITICAL RULES" in system_msg
        assert "PRESERVE" in system_msg
        assert "training data" in system_msg


# ============================================================
# TestCompressedKnowledge
# ============================================================


class TestCompressedKnowledge:
    def test_valid_construction(self) -> None:
        result = CompressedKnowledge(compressed_summary="Dense facts.", token_count=12)
        assert result.compressed_summary == "Dense facts."
        assert result.token_count == 12

    def test_token_count_defaults_to_zero(self) -> None:
        result = CompressedKnowledge(compressed_summary="text")
        assert result.token_count == 0

    def test_required_field(self) -> None:
        with pytest.raises(ValidationError):
            CompressedKnowledge()  # missing compressed_summary


# ============================================================
# TestCompressKnowledgeSummary (async, mocked LLM)
# ============================================================


class TestCompressKnowledgeSummary:
    async def test_passes_correct_args_and_recomputes_tokens(self) -> None:
        expected = CompressedKnowledge(compressed_summary="Condensed knowledge.", token_count=0)
        mock_client, mock_create = _mock_instructor_client(expected)
        settings = _make_settings()
        topic = _make_topic(name="Compress Topic")

        with (
            patch("app.analysis.llm._get_client", return_value=mock_client),
            patch("app.analysis.llm.count_tokens", return_value=11),
        ):
            result = await compress_knowledge_summary("Verbose summary.", topic, settings)

        assert result.token_count == 11  # recomputed, not the LLM's 0
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["model"] == "openai/gpt-4o-mini"
        assert call_kwargs["response_model"] is CompressedKnowledge
        assert call_kwargs["temperature"] == 0.2
        assert "Verbose summary." in call_kwargs["messages"][1]["content"]
        assert "Compress Topic" in call_kwargs["messages"][1]["content"]

    async def test_exposes_completion_usage(self) -> None:
        """OVH-129: compression captures the completion's token usage so its cost
        is not invisible in the per-check totals."""
        expected = CompressedKnowledge(compressed_summary="Condensed.", token_count=0)
        completion = _FakeCompletion(_FakeUsage(prompt_tokens=321, completion_tokens=54))
        mock_client, _ = _mock_instructor_client(expected, completion=completion)
        settings = _make_settings()

        with (
            patch("app.analysis.llm._get_client", return_value=mock_client),
            patch("app.analysis.llm.count_tokens", return_value=11),
        ):
            result = await compress_knowledge_summary("Verbose summary.", _make_topic(), settings)

        assert result.prompt_tokens == 321
        assert result.completion_tokens == 54

    async def test_raises_on_llm_error(self) -> None:
        mock_client, mock_create = _mock_instructor_client(None)
        mock_create.side_effect = Exception("compress failed")
        settings = _make_settings()

        with (
            patch("app.analysis.llm._get_client", return_value=mock_client),
            pytest.raises(Exception, match="compress failed"),
        ):
            await compress_knowledge_summary("text", _make_topic(), settings)


# ============================================================
# TestCountTokens
# ============================================================


class TestCountTokens:
    def test_returns_token_count(self) -> None:
        # litellm.token_counter should return an int for known models
        count = count_tokens("Hello, world!", "openai/gpt-4o-mini")
        assert isinstance(count, int)
        assert count > 0

    @patch("app.analysis.llm.litellm.token_counter", side_effect=Exception("fail"))
    def test_fallback_on_error(self, mock_counter) -> None:
        text = "a" * 400
        count = count_tokens(text, "unknown/model")
        assert count == 100  # len(400) // 4

    @patch("app.analysis.llm.litellm.token_counter", side_effect=Exception("fail"))
    def test_fallback_warns_once_per_model(self, mock_counter, caplog) -> None:
        """OVH-136: the char/4 fallback diverges from the model tokenizer, so the
        first fallback for a model WARNs (budget decisions on a wrong unit are
        observable); later fallbacks for the same model stay quiet (no log flood).
        """
        import logging

        from app.analysis import llm as llm_mod

        llm_mod._token_fallback_warned.discard("flaky/model")
        with caplog.at_level(logging.WARNING, logger="app.analysis.llm"):
            count_tokens("some text", "flaky/model")
            count_tokens("more text", "flaky/model")

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "flaky/model" in r.getMessage()]
        assert len(warnings) == 1
        assert "fallback" in warnings[0].getMessage().lower()


class TestEffectiveBaseUrl:
    """Tests for _effective_base_url safety net."""

    def test_cloud_provider_ignores_base_url(self) -> None:
        from app.analysis.llm import _effective_base_url

        settings = _make_settings(
            llm=LLMSettings(model="anthropic/claude-haiku-4-5", api_key="k", base_url="http://localhost:11434")
        )
        assert _effective_base_url(settings) is None

    def test_local_provider_preserves_base_url(self) -> None:
        from app.analysis.llm import _effective_base_url

        settings = _make_settings(
            llm=LLMSettings(model="ollama/llama3", api_key="k", base_url="http://localhost:11434")
        )
        assert _effective_base_url(settings) == "http://localhost:11434"

    def test_no_base_url_returns_none(self) -> None:
        from app.analysis.llm import _effective_base_url

        settings = _make_settings(llm=LLMSettings(model="openai/gpt-4", api_key="k"))
        assert _effective_base_url(settings) is None


# ============================================================
# TestAnalyzeArticles (async, mocked LLM)
# ============================================================


class TestAnalyzeArticles:
    async def test_passes_correct_args_to_llm(self) -> None:
        """Verify analyze_articles passes the right model, response_model, and messages."""
        expected = NoveltyResult(
            has_new_info=True,
            summary="New release date",
            key_facts=["June 2025"],
            source_urls=["https://example.com/1"],
            confidence=0.9,
        )
        mock_client, mock_create = _mock_instructor_client(expected)
        settings = _make_settings()
        topic = _make_topic(name="My Topic")

        with patch("app.analysis.llm._get_client", return_value=mock_client):
            await analyze_articles([_make_article()], "Known facts.", topic, settings)

        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["model"] == "openai/gpt-4o-mini"
        assert call_kwargs["response_model"] is NoveltyResult
        assert call_kwargs["temperature"] == 0.2
        # Verify messages include the knowledge state and topic info
        messages = call_kwargs["messages"]
        assert len(messages) == 2
        assert "Known facts." in messages[1]["content"]
        assert "My Topic" in messages[1]["content"]

    async def test_returns_safe_default_on_error(self) -> None:
        mock_client, mock_create = _mock_instructor_client(None)
        mock_create.side_effect = Exception("LLM API error")
        settings = _make_settings()

        with patch("app.analysis.llm._get_client", return_value=mock_client):
            result = await analyze_articles([_make_article()], "Known facts.", _make_topic(), settings)

        assert result.has_new_info is False
        assert result.confidence == 0.0

    async def test_passes_model_and_api_key(self) -> None:
        expected = NoveltyResult(has_new_info=False, confidence=0.5)
        mock_client, mock_create = _mock_instructor_client(expected)
        settings = _make_settings(llm=LLMSettings(model="anthropic/claude-haiku", api_key="sk-test-123"))

        with patch("app.analysis.llm._get_client", return_value=mock_client):
            await analyze_articles([_make_article()], "", _make_topic(), settings)

        call_kwargs = mock_create.call_args
        assert call_kwargs.kwargs["model"] == "anthropic/claude-haiku"
        assert call_kwargs.kwargs["api_key"] == "sk-test-123"

    async def test_passes_temperature(self) -> None:
        expected = NoveltyResult(has_new_info=False, confidence=0.5)
        mock_client, mock_create = _mock_instructor_client(expected)
        settings = _make_settings(llm_temperature=0.0)

        with patch("app.analysis.llm._get_client", return_value=mock_client):
            await analyze_articles([_make_article()], "", _make_topic(), settings)

        assert mock_create.call_args.kwargs["temperature"] == 0.0

    async def test_forces_error_none_on_success(self) -> None:
        """A successful call must clear ``error`` even if the model populated it.

        ``error`` is part of the structured-output schema, so a model could set
        it on a clean run. Only the except-branch is allowed to set ``error``;
        otherwise the checker mis-stamps a healthy run as ``analysis_failed``.
        """
        rogue = NoveltyResult(has_new_info=True, summary="X", confidence=0.9, error="model populated this on success")
        mock_client, _ = _mock_instructor_client(rogue)
        settings = _make_settings()

        with patch("app.analysis.llm._get_client", return_value=mock_client):
            result = await analyze_articles([_make_article()], "Known facts.", _make_topic(), settings)

        assert result.has_new_info is True
        assert result.error is None


# ============================================================
# TestGenerateInitialKnowledge (async, mocked LLM)
# ============================================================


class TestGenerateInitialKnowledge:
    async def test_passes_correct_args_and_recomputes_tokens(self) -> None:
        """Verify correct model/messages are passed and token count is recomputed."""
        expected = KnowledgeStateUpdate(
            sufficient_data=True,
            confidence=0.9,
            updated_summary="Initial knowledge summary.",
            token_count=0,
        )
        mock_client, mock_create = _mock_instructor_client(expected)
        settings = _make_settings()
        topic = _make_topic(name="Init Topic")

        with (
            patch("app.analysis.llm._get_client", return_value=mock_client),
            patch("app.analysis.llm.count_tokens", return_value=42),
        ):
            result = await generate_initial_knowledge([_make_article()], topic, settings)

        # Token count is recomputed, not the LLM's guess
        assert result.token_count == 42
        # Verify correct args passed to LLM
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["model"] == "openai/gpt-4o-mini"
        assert call_kwargs["response_model"] is KnowledgeStateUpdate
        assert call_kwargs["temperature"] == 0.2
        messages = call_kwargs["messages"]
        assert "Init Topic" in messages[1]["content"]

    async def test_recomputes_token_count(self) -> None:
        expected = KnowledgeStateUpdate(
            sufficient_data=True,
            confidence=0.9,
            updated_summary="Some text here.",
            token_count=999,
        )
        mock_client, _ = _mock_instructor_client(expected)
        settings = _make_settings()

        with (
            patch("app.analysis.llm._get_client", return_value=mock_client),
            patch("app.analysis.llm.count_tokens", return_value=5),
        ):
            result = await generate_initial_knowledge([_make_article()], _make_topic(), settings)

        assert result.token_count == 5  # our count, not the LLM's 999

    async def test_raises_on_llm_error(self) -> None:
        mock_client, mock_create = _mock_instructor_client(None)
        mock_create.side_effect = Exception("LLM API error")
        settings = _make_settings()

        with (
            patch("app.analysis.llm._get_client", return_value=mock_client),
            pytest.raises(Exception, match="LLM API error"),
        ):
            await generate_initial_knowledge([_make_article()], _make_topic(), settings)


# ============================================================
# TestGenerateKnowledgeUpdate (async, mocked LLM)
# ============================================================


class TestGenerateKnowledgeUpdate:
    async def test_passes_correct_args_and_recomputes_tokens(self) -> None:
        """Verify current summary and novelty are passed in messages, tokens recomputed."""
        expected = KnowledgeStateUpdate(
            sufficient_data=True,
            confidence=0.9,
            updated_summary="Updated summary with new facts.",
            token_count=0,
        )
        mock_client, mock_create = _mock_instructor_client(expected)
        novelty = NoveltyResult(
            has_new_info=True,
            summary="New price announced",
            key_facts=["$39.99"],
            confidence=0.9,
        )
        settings = _make_settings()

        with (
            patch("app.analysis.llm._get_client", return_value=mock_client),
            patch("app.analysis.llm.count_tokens", return_value=60),
        ):
            result = await generate_knowledge_update("Old summary.", novelty, _make_topic(), settings)

        assert result.token_count == 60
        # Verify the messages include current summary and new findings
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["response_model"] is KnowledgeStateUpdate
        assert call_kwargs["temperature"] == 0.2
        messages = call_kwargs["messages"]
        user_msg = messages[1]["content"]
        assert "Old summary." in user_msg
        assert "New price announced" in user_msg
        assert "$39.99" in user_msg

    async def test_recomputes_token_count(self) -> None:
        expected = KnowledgeStateUpdate(sufficient_data=True, confidence=0.9, updated_summary="Text.", token_count=999)
        mock_client, _ = _mock_instructor_client(expected)
        novelty = NoveltyResult(has_new_info=True, summary="X", confidence=0.8)
        settings = _make_settings()

        with (
            patch("app.analysis.llm._get_client", return_value=mock_client),
            patch("app.analysis.llm.count_tokens", return_value=3),
        ):
            result = await generate_knowledge_update("Old.", novelty, _make_topic(), settings)

        assert result.token_count == 3

    async def test_raises_on_llm_error(self) -> None:
        mock_client, mock_create = _mock_instructor_client(None)
        mock_create.side_effect = Exception("LLM down")
        novelty = NoveltyResult(has_new_info=True, summary="X", confidence=0.8)
        settings = _make_settings()

        with (
            patch("app.analysis.llm._get_client", return_value=mock_client),
            pytest.raises(Exception, match="LLM down"),
        ):
            await generate_knowledge_update("Old.", novelty, _make_topic(), settings)


# ============================================================
# TestInitializeKnowledge (async, db_conn)
# ============================================================


class TestInitializeKnowledge:
    async def test_creates_and_stores_knowledge(self, db_conn: sqlite3.Connection) -> None:
        topic = Topic(name="Test", description="Desc", feed_urls=[])
        topic = create_topic(db_conn, topic)
        db_conn.commit()
        articles = [_make_article(topic_id=topic.id)]
        settings = _make_settings()

        llm_result = KnowledgeStateUpdate(
            sufficient_data=True, confidence=0.9, updated_summary="Initial summary.", token_count=30
        )
        with patch(
            "app.analysis.knowledge.generate_initial_knowledge",
            new_callable=AsyncMock,
            return_value=llm_result,
        ):
            result = await initialize_knowledge(topic, articles, db_conn, settings)

        state = result.state
        assert state.id is not None
        assert state.topic_id == topic.id
        assert state.summary_text == "Initial summary."
        assert state.token_count == 30
        assert result.sufficient_data is True

        # Verify persisted in DB
        stored = get_knowledge_state(db_conn, topic.id)
        assert stored is not None
        assert stored.summary_text == "Initial summary."

    async def test_returns_state_with_id(self, db_conn: sqlite3.Connection) -> None:
        topic = Topic(name="T2", description="D2", feed_urls=[])
        topic = create_topic(db_conn, topic)
        db_conn.commit()
        settings = _make_settings()

        llm_result = KnowledgeStateUpdate(sufficient_data=True, confidence=0.9, updated_summary="S", token_count=5)
        with patch(
            "app.analysis.knowledge.generate_initial_knowledge",
            new_callable=AsyncMock,
            return_value=llm_result,
        ):
            result = await initialize_knowledge(topic, [], db_conn, settings)

        state = result.state
        assert isinstance(state.id, int)
        assert state.id > 0

    async def test_propagates_llm_error(self, db_conn: sqlite3.Connection) -> None:
        topic = Topic(name="T3", description="D3", feed_urls=[])
        topic = create_topic(db_conn, topic)
        db_conn.commit()
        settings = _make_settings()

        with (
            patch(
                "app.analysis.knowledge.generate_initial_knowledge",
                new_callable=AsyncMock,
                side_effect=Exception("LLM failed"),
            ),
            pytest.raises(Exception, match="LLM failed"),
        ):
            await initialize_knowledge(topic, [], db_conn, settings)

        # Verify nothing was stored
        assert get_knowledge_state(db_conn, topic.id) is None

    async def test_insufficient_data_still_stores(self, db_conn: sqlite3.Connection) -> None:
        """When LLM reports insufficient data, we still store the explanation."""
        topic = Topic(name="Thin", description="Desc", feed_urls=[])
        topic = create_topic(db_conn, topic)
        db_conn.commit()
        settings = _make_settings()

        llm_result = KnowledgeStateUpdate(
            sufficient_data=False,
            confidence=0.2,
            updated_summary="Articles did not contain relevant information about the topic.",
            token_count=10,
        )
        with patch(
            "app.analysis.knowledge.generate_initial_knowledge",
            new_callable=AsyncMock,
            return_value=llm_result,
        ):
            result = await initialize_knowledge(topic, [], db_conn, settings)

        state = result.state
        assert state.id is not None
        assert "did not contain" in state.summary_text
        # Task 3: insufficient-data signal is programmatic, not string-parsed.
        assert result.sufficient_data is False

        stored = get_knowledge_state(db_conn, topic.id)
        assert stored is not None


# ============================================================
# TestUpdateKnowledge (async, db_conn)
# ============================================================


class TestUpdateKnowledge:
    async def test_updates_existing_state(self, db_conn: sqlite3.Connection) -> None:
        topic = Topic(name="U1", description="D1", feed_urls=[])
        topic = create_topic(db_conn, topic)
        db_conn.commit()

        # Create initial knowledge state
        initial = KnowledgeState(topic_id=topic.id, summary_text="Old summary.", token_count=20)
        create_knowledge_state(db_conn, initial)
        db_conn.commit()

        novelty = NoveltyResult(
            has_new_info=True,
            summary="New fact found",
            key_facts=["Fact 1"],
            confidence=0.9,
        )
        llm_result = KnowledgeStateUpdate(
            sufficient_data=True,
            confidence=0.9,
            updated_summary="Updated summary with Fact 1.",
            token_count=35,
        )
        settings = _make_settings()

        with patch(
            "app.analysis.knowledge.generate_knowledge_update",
            new_callable=AsyncMock,
            return_value=llm_result,
        ):
            result = await update_knowledge(topic, novelty, db_conn, settings)

        state = result.state
        assert state.summary_text == "Updated summary with Fact 1."
        assert state.token_count == 35
        assert result.sufficient_data is True

        # Verify persisted
        stored = get_knowledge_state(db_conn, topic.id)
        assert stored.summary_text == "Updated summary with Fact 1."

    async def test_raises_if_no_existing_state(self, db_conn: sqlite3.Connection) -> None:
        topic = Topic(name="U2", description="D2", feed_urls=[])
        topic = create_topic(db_conn, topic)
        db_conn.commit()
        settings = _make_settings()

        novelty = NoveltyResult(has_new_info=True, summary="X", confidence=0.8)

        with pytest.raises(ValueError, match="No knowledge state found"):
            await update_knowledge(topic, novelty, db_conn, settings)

    async def test_propagates_llm_error(self, db_conn: sqlite3.Connection) -> None:
        topic = Topic(name="U3", description="D3", feed_urls=[])
        topic = create_topic(db_conn, topic)
        db_conn.commit()

        initial = KnowledgeState(topic_id=topic.id, summary_text="Old.", token_count=10)
        create_knowledge_state(db_conn, initial)
        db_conn.commit()

        novelty = NoveltyResult(has_new_info=True, summary="X", confidence=0.8)
        settings = _make_settings()

        with (
            patch(
                "app.analysis.knowledge.generate_knowledge_update",
                new_callable=AsyncMock,
                side_effect=Exception("LLM failed"),
            ),
            pytest.raises(Exception, match="LLM failed"),
        ):
            await update_knowledge(topic, novelty, db_conn, settings)

        # Verify original state unchanged
        stored = get_knowledge_state(db_conn, topic.id)
        assert stored.summary_text == "Old."

    async def test_insufficient_data_preserves_existing(self, db_conn: sqlite3.Connection) -> None:
        """When LLM reports insufficient data on update, preserve the existing state."""
        topic = Topic(name="U4", description="D4", feed_urls=[])
        topic = create_topic(db_conn, topic)
        db_conn.commit()

        initial = KnowledgeState(topic_id=topic.id, summary_text="Existing knowledge.", token_count=15)
        create_knowledge_state(db_conn, initial)
        db_conn.commit()

        novelty = NoveltyResult(has_new_info=True, summary="Vague update", confidence=0.5)
        llm_result = KnowledgeStateUpdate(
            sufficient_data=False,
            confidence=0.2,
            updated_summary="Findings too vague to incorporate.",
        )
        settings = _make_settings()

        with patch(
            "app.analysis.knowledge.generate_knowledge_update",
            new_callable=AsyncMock,
            return_value=llm_result,
        ):
            result = await update_knowledge(topic, novelty, db_conn, settings)

        # Should return the original state unchanged
        state = result.state
        assert state.summary_text == "Existing knowledge."
        assert state.token_count == 15
        assert result.sufficient_data is False

        stored = get_knowledge_state(db_conn, topic.id)
        assert stored.summary_text == "Existing knowledge."


# ============================================================
# TestTokenUsage (Task 1 — usage extraction + population)
# ============================================================


class TestExtractUsage:
    def test_extracts_attribute_style_usage(self) -> None:
        from app.analysis.llm import _extract_usage

        usage = _extract_usage(_FakeCompletion(_FakeUsage(prompt_tokens=42, completion_tokens=17)))
        assert usage.prompt_tokens == 42
        assert usage.completion_tokens == 17

    def test_missing_usage_returns_zeros(self) -> None:
        from app.analysis.llm import _extract_usage

        class _NoUsage:
            usage = None

        usage = _extract_usage(_NoUsage())
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0

    def test_dict_style_usage(self) -> None:
        from app.analysis.llm import _extract_usage

        class _DictComp:
            usage = {"prompt_tokens": 5, "completion_tokens": 9}

        usage = _extract_usage(_DictComp())
        assert usage.prompt_tokens == 5
        assert usage.completion_tokens == 9

    def test_non_integer_usage_coerces_to_zero(self) -> None:
        from app.analysis.llm import _extract_usage

        usage = _extract_usage(_FakeCompletion(_FakeUsage(prompt_tokens=None, completion_tokens="bad")))  # type: ignore[arg-type]
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0


class TestAnalyzeArticlesTokenUsage:
    async def test_populates_token_usage_from_completion(self) -> None:
        expected = NoveltyResult(has_new_info=True, summary="New thing", confidence=0.9)
        completion = _FakeCompletion(_FakeUsage(prompt_tokens=123, completion_tokens=45))
        mock_client, _ = _mock_instructor_client(expected, completion=completion)
        settings = _make_settings()

        with patch("app.analysis.llm._get_client", return_value=mock_client):
            result = await analyze_articles([_make_article()], "Known facts.", _make_topic(), settings)

        assert result.prompt_tokens == 123
        assert result.completion_tokens == 45

    async def test_error_path_token_usage_is_zero(self) -> None:
        mock_client, mock_create = _mock_instructor_client(None)
        mock_create.side_effect = Exception("LLM API error")
        settings = _make_settings()

        with patch("app.analysis.llm._get_client", return_value=mock_client):
            result = await analyze_articles([_make_article()], "Known facts.", _make_topic(), settings)

        assert result.has_new_info is False
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0


class TestGenerateKnowledgeTokenUsage:
    async def test_initial_knowledge_exposes_usage(self) -> None:
        expected = KnowledgeStateUpdate(sufficient_data=True, confidence=0.9, updated_summary="Init.", token_count=0)
        completion = _FakeCompletion(_FakeUsage(prompt_tokens=200, completion_tokens=60))
        mock_client, _ = _mock_instructor_client(expected, completion=completion)
        settings = _make_settings()

        with (
            patch("app.analysis.llm._get_client", return_value=mock_client),
            patch("app.analysis.llm.count_tokens", return_value=10),
        ):
            result = await generate_initial_knowledge([_make_article()], _make_topic(), settings)

        assert result.prompt_tokens == 200
        assert result.completion_tokens == 60

    async def test_knowledge_update_exposes_usage(self) -> None:
        expected = KnowledgeStateUpdate(sufficient_data=True, confidence=0.9, updated_summary="Upd.", token_count=0)
        completion = _FakeCompletion(_FakeUsage(prompt_tokens=80, completion_tokens=30))
        mock_client, _ = _mock_instructor_client(expected, completion=completion)
        novelty = NoveltyResult(has_new_info=True, summary="X", confidence=0.8)
        settings = _make_settings()

        with (
            patch("app.analysis.llm._get_client", return_value=mock_client),
            patch("app.analysis.llm.count_tokens", return_value=10),
        ):
            result = await generate_knowledge_update("Old.", novelty, _make_topic(), settings)

        assert result.prompt_tokens == 80
        assert result.completion_tokens == 30


class TestKnowledgeWriteResultUsage:
    """initialize_knowledge / update_knowledge expose usage via KnowledgeWriteResult."""

    async def test_initialize_exposes_usage(self, db_conn: sqlite3.Connection) -> None:
        topic = Topic(name="UsageInit", description="D", feed_urls=[])
        topic = create_topic(db_conn, topic)
        db_conn.commit()
        settings = _make_settings()

        llm_result = KnowledgeStateUpdate(
            sufficient_data=True,
            confidence=0.9,
            updated_summary="Initial.",
            token_count=10,
            prompt_tokens=150,
            completion_tokens=40,
        )
        with patch(
            "app.analysis.knowledge.generate_initial_knowledge",
            new_callable=AsyncMock,
            return_value=llm_result,
        ):
            result = await initialize_knowledge(topic, [], db_conn, settings)

        assert result.usage.prompt_tokens == 150
        assert result.usage.completion_tokens == 40

    async def test_update_exposes_usage(self, db_conn: sqlite3.Connection) -> None:
        topic = Topic(name="UsageUpd", description="D", feed_urls=[])
        topic = create_topic(db_conn, topic)
        db_conn.commit()
        initial = KnowledgeState(topic_id=topic.id, summary_text="Old.", token_count=10)
        create_knowledge_state(db_conn, initial)
        db_conn.commit()
        settings = _make_settings()

        novelty = NoveltyResult(has_new_info=True, summary="X", confidence=0.8)
        llm_result = KnowledgeStateUpdate(
            sufficient_data=True,
            confidence=0.9,
            updated_summary="Updated.",
            token_count=12,
            prompt_tokens=90,
            completion_tokens=25,
        )
        with patch(
            "app.analysis.knowledge.generate_knowledge_update",
            new_callable=AsyncMock,
            return_value=llm_result,
        ):
            result = await update_knowledge(topic, novelty, db_conn, settings)

        assert result.usage.prompt_tokens == 90
        assert result.usage.completion_tokens == 25

    async def test_update_insufficient_still_exposes_usage(self, db_conn: sqlite3.Connection) -> None:
        topic = Topic(name="UsageUpdIns", description="D", feed_urls=[])
        topic = create_topic(db_conn, topic)
        db_conn.commit()
        initial = KnowledgeState(topic_id=topic.id, summary_text="Old.", token_count=10)
        create_knowledge_state(db_conn, initial)
        db_conn.commit()
        settings = _make_settings()

        novelty = NoveltyResult(has_new_info=True, summary="X", confidence=0.5)
        llm_result = KnowledgeStateUpdate(
            sufficient_data=False,
            confidence=0.2,
            updated_summary="Too vague.",
            token_count=5,
            prompt_tokens=33,
            completion_tokens=11,
        )
        with patch(
            "app.analysis.knowledge.generate_knowledge_update",
            new_callable=AsyncMock,
            return_value=llm_result,
        ):
            result = await update_knowledge(topic, novelty, db_conn, settings)

        assert result.sufficient_data is False
        assert result.usage.prompt_tokens == 33
        assert result.usage.completion_tokens == 11

    async def test_update_folds_compression_usage_into_total(self, db_conn: sqlite3.Connection) -> None:
        """OVH-129: when an over-budget update triggers compression, the
        compression round-trip's tokens are added to the reported usage."""
        topic = Topic(name="UsageCompress", description="D", feed_urls=[])
        topic = create_topic(db_conn, topic)
        db_conn.commit()
        create_knowledge_state(db_conn, KnowledgeState(topic_id=topic.id, summary_text="Old.", token_count=10))
        db_conn.commit()
        settings = _make_settings(knowledge_state_max_tokens=500)

        novelty = NoveltyResult(has_new_info=True, summary="X", confidence=0.8)
        # Update call: 90/25, and reports an over-budget token_count to trigger compression.
        llm_result = KnowledgeStateUpdate(
            sufficient_data=True,
            confidence=0.9,
            updated_summary="Long over-budget summary.",
            token_count=501,
            prompt_tokens=90,
            completion_tokens=25,
        )
        # Compression call: 200/40, fits the budget.
        compressed = CompressedKnowledge(
            compressed_summary="Tight.",
            token_count=5,
            prompt_tokens=200,
            completion_tokens=40,
        )
        with (
            patch(
                "app.analysis.knowledge.generate_knowledge_update",
                new_callable=AsyncMock,
                return_value=llm_result,
            ),
            patch(
                "app.analysis.knowledge.compress_knowledge_summary",
                new_callable=AsyncMock,
                return_value=compressed,
            ),
        ):
            result = await update_knowledge(topic, novelty, db_conn, settings)

        # Update (90/25) + compression (200/40) folded together.
        assert result.usage.prompt_tokens == 290
        assert result.usage.completion_tokens == 65


# ============================================================
# TestKeyFactsRestatementFilter (Task 2)
# ============================================================


class TestRestatementFilter:
    def test_verbatim_restatement_is_filtered(self) -> None:
        from app.analysis.llm import _filter_restated_key_facts

        summary = "Confirmed Facts: Release date is March 2026. Price is $59.99."
        facts = ["Release date is March 2026", "A brand new collector's edition was announced"]
        kept = _filter_restated_key_facts(facts, summary)
        assert "Release date is March 2026" not in kept
        assert "A brand new collector's edition was announced" in kept

    def test_genuinely_new_fact_is_kept(self) -> None:
        from app.analysis.llm import _filter_restated_key_facts

        summary = "Confirmed Facts: The game was announced in 2024."
        facts = ["The studio confirmed a Q3 2026 release window with cross-play support"]
        kept = _filter_restated_key_facts(facts, summary)
        assert kept == facts

    def test_empty_summary_keeps_all(self) -> None:
        from app.analysis.llm import _filter_restated_key_facts

        facts = ["Fact A", "Fact B"]
        assert _filter_restated_key_facts(facts, "") == facts

    def test_short_new_fact_with_scattered_words_is_kept(self) -> None:
        """Regression: a short genuinely-new fact whose words merely appear
        scattered (non-contiguously) across a long summary must NOT be dropped.

        Old bag-of-words overlap scored 1.0 here and silently hid the fact.
        """
        from app.analysis.llm import _filter_restated_key_facts

        summary = (
            "Confirmed Facts: The price of the standard edition is significant. "
            "Earlier reports said the value had dropped sharply. "
            "Separately, a number close to 49 was mentioned regarding subscriber counts."
        )
        facts = ["Price dropped to $49"]
        kept = _filter_restated_key_facts(facts, summary)
        assert kept == facts

    def test_contiguous_phrase_restatement_is_filtered(self) -> None:
        from app.analysis.llm import _filter_restated_key_facts

        summary = "Confirmed Facts: The official launch event is scheduled for next quarter in Berlin."
        facts = ["the official launch event is scheduled for next quarter"]
        kept = _filter_restated_key_facts(facts, summary)
        assert kept == []

    # --- Boundary seam tests for the two tuned constants (OVH-078) ---
    # _RESTATEMENT_MIN_FACT_WORDS = 4 ; _RESTATEMENT_PHRASE_OVERLAP_THRESHOLD = 0.8.
    # These pin where retuning either constant (or a >= /< slip) silently changes
    # which genuinely-new facts get hidden.

    def test_three_word_fact_fully_contained_is_kept_below_min_words(self) -> None:
        """A 3-word fact whose words all appear as a contiguous run is KEPT — it
        is below the 4-word minimum, so it is never auto-dropped (commas keep it
        off the verbatim-substring branch, so only the word-run branch could fire)."""
        from app.analysis.llm import _filter_restated_key_facts

        summary = "Confirmed: alpha, beta, gamma were noted."
        facts = ["alpha beta gamma"]  # 3 content words, run == 3, but < min 4
        assert _filter_restated_key_facts(facts, summary) == facts

    def test_five_word_fact_with_four_word_run_at_080_is_dropped(self) -> None:
        """A 5-word fact whose longest contiguous run is exactly 4 (4/5 == 0.80)
        is DROPPED — at the threshold. Word order differs from the summary so the
        full phrase is not a verbatim substring; only the word-run branch fires."""
        from app.analysis.llm import _filter_restated_key_facts, _normalize_for_match

        fact = "the quick brown fox today"  # 5 content words
        summary = "Note: the quick brown fox jumped over things."  # contains the 4-word run
        # Guard the test's own premise: the full phrase is NOT a verbatim substring,
        # so a pass here proves the contiguous-run branch (not substring) did it.
        assert _normalize_for_match(fact) not in _normalize_for_match(summary)
        assert _filter_restated_key_facts([fact], summary) == []

    def test_five_word_fact_with_three_word_run_at_060_is_kept(self) -> None:
        """A 5-word fact whose longest contiguous run is 3 (3/5 == 0.60) is KEPT —
        below the 0.80 overlap threshold."""
        from app.analysis.llm import _filter_restated_key_facts

        fact = "alpha beta gamma delta epsilon"  # 5 content words
        summary = "Note: alpha beta gamma appeared but delta and epsilon were elsewhere entirely."
        assert _filter_restated_key_facts([fact], summary) == [fact]

    async def test_analyze_articles_filters_restated_key_facts(self) -> None:
        knowledge = "Confirmed Facts: The release date is March 2026."
        expected = NoveltyResult(
            has_new_info=True,
            summary="A delay was announced",
            key_facts=[
                "The release date is March 2026",  # restatement -> dropped
                "The release was delayed to September 2026 due to a recall",  # new -> kept
            ],
            confidence=0.9,
        )
        mock_client, _ = _mock_instructor_client(expected)
        settings = _make_settings()

        with patch("app.analysis.llm._get_client", return_value=mock_client):
            result = await analyze_articles([_make_article()], knowledge, _make_topic(), settings)

        assert "The release date is March 2026" not in result.key_facts
        assert any("delayed to September 2026" in f for f in result.key_facts)

    async def test_all_facts_filtered_keeps_has_new_info(self) -> None:
        knowledge = "Confirmed Facts: The release date is March 2026 and the price is fifty nine dollars."
        expected = NoveltyResult(
            has_new_info=True,
            summary="Still novel via summary",
            key_facts=["The release date is March 2026"],
            confidence=0.9,
        )
        mock_client, _ = _mock_instructor_client(expected)
        settings = _make_settings()

        with patch("app.analysis.llm._get_client", return_value=mock_client):
            result = await analyze_articles([_make_article()], knowledge, _make_topic(), settings)

        assert result.has_new_info is True
        assert result.key_facts == []


class TestLongestContiguousRun:
    """OVH-079: direct unit tests for the longest-common-substring DP that backs
    restatement filtering. Exercised only end-to-end before, so an off-by-one
    (len-1) or a repeated-adjacent-token bug was invisible."""

    def test_empty_fact_returns_zero(self) -> None:
        from app.analysis.llm import _longest_contiguous_run

        assert _longest_contiguous_run([], ["a", "b"]) == 0

    def test_empty_summary_returns_zero(self) -> None:
        from app.analysis.llm import _longest_contiguous_run

        assert _longest_contiguous_run(["a"], []) == 0

    def test_both_empty_returns_zero(self) -> None:
        from app.analysis.llm import _longest_contiguous_run

        assert _longest_contiguous_run([], []) == 0

    def test_full_match_returns_len(self) -> None:
        from app.analysis.llm import _longest_contiguous_run

        assert _longest_contiguous_run(["a", "b", "c"], ["a", "b", "c"]) == 3

    def test_embedded_run_returns_run_length(self) -> None:
        from app.analysis.llm import _longest_contiguous_run

        assert _longest_contiguous_run(["a", "b", "c"], ["x", "a", "b", "c", "y"]) == 3

    def test_repeated_tokens_returns_two(self) -> None:
        """['a','a'] inside ['a','a','a'] is a contiguous run of length 2, not 1 or 3."""
        from app.analysis.llm import _longest_contiguous_run

        assert _longest_contiguous_run(["a", "a"], ["a", "a", "a"]) == 2

    def test_no_overlap_returns_zero(self) -> None:
        from app.analysis.llm import _longest_contiguous_run

        assert _longest_contiguous_run(["a", "b"], ["x", "y", "z"]) == 0

    def test_partial_then_restart_returns_two(self) -> None:
        """A false start ('a' alone) must not block the later full run ('a','b')."""
        from app.analysis.llm import _longest_contiguous_run

        assert _longest_contiguous_run(["a", "b"], ["a", "x", "a", "b"]) == 2


class TestRateLimitRetry:
    """The rate-limit backoff loop honors settings.llm_max_retries."""

    async def test_retry_count_honors_max_retries(self) -> None:
        from app.analysis import llm as llm_module

        attempts = 0

        async def _always_rate_limited() -> None:
            nonlocal attempts
            attempts += 1
            raise litellm.RateLimitError.__new__(litellm.RateLimitError)

        with (
            patch("app.analysis.llm.asyncio.sleep", new=AsyncMock()),
            pytest.raises(litellm.RateLimitError),
        ):
            await llm_module._call_with_rate_limit_retry(_always_rate_limited, max_retries=4)

        assert attempts == 5  # initial attempt + 4 retries

    async def test_succeeds_after_transient_rate_limit(self) -> None:
        from app.analysis import llm as llm_module

        attempts = 0

        async def _flaky() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                raise litellm.RateLimitError.__new__(litellm.RateLimitError)
            return "ok"

        with patch("app.analysis.llm.asyncio.sleep", new=AsyncMock()):
            result = await llm_module._call_with_rate_limit_retry(_flaky, max_retries=3)

        assert result == "ok"
        assert attempts == 2


class TestClientCaching:
    """_get_client returns a single cached client instead of rebuilding per call."""

    def test_client_is_cached(self) -> None:
        from app.analysis import llm as llm_module

        llm_module._client = None
        settings = _make_settings()
        first = llm_module._get_client(settings)
        second = llm_module._get_client(settings)
        assert first is second
