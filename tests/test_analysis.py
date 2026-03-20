"""Tests for the LLM analysis module: prompts, structured output, knowledge management."""

import sqlite3
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from app.analysis.knowledge import initialize_knowledge, update_knowledge
from app.analysis.llm import (
    KnowledgeStateUpdate,
    NoveltyResult,
    analyze_articles,
    count_tokens,
    generate_initial_knowledge,
    generate_knowledge_update,
)
from app.analysis.prompts import (
    _format_articles,
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


def _mock_instructor_client(return_value):
    """Create a mock instructor client that returns the given value from create()."""
    mock_create = AsyncMock(return_value=return_value)
    mock_completions = MagicMock()
    mock_completions.create = mock_create
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

    def test_no_summary_when_no_new_info(self) -> None:
        result = NoveltyResult(has_new_info=False, confidence=0.9)
        assert result.has_new_info is False
        assert result.summary is None


# ============================================================
# TestKnowledgeStateUpdate
# ============================================================


class TestKnowledgeStateUpdate:
    def test_valid_construction(self) -> None:
        update = KnowledgeStateUpdate(
            updated_summary="Summary of known facts.",
            token_count=150,
        )
        assert update.updated_summary == "Summary of known facts."
        assert update.token_count == 150

    def test_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            KnowledgeStateUpdate(updated_summary="text")  # missing token_count
        with pytest.raises(ValidationError):
            KnowledgeStateUpdate(token_count=100)  # missing updated_summary


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

    def test_handles_none_content(self) -> None:
        article = _make_article(raw_content=None)
        result = _format_articles([article])
        assert "(no content available)" in result

    def test_truncates_long_content(self) -> None:
        long_content = "x" * 2000
        article = _make_article(raw_content=long_content)
        result = _format_articles([article], max_content_chars=100)
        # Content should be truncated to 100 chars + "..."
        assert "x" * 100 + "..." in result
        assert "x" * 101 not in result


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

    def test_formats_articles_in_user_message(self) -> None:
        topic = _make_topic()
        articles = [_make_article(title="Important Article")]
        messages = build_knowledge_init_messages(articles, topic, max_tokens=2000)
        user_msg = messages[1]["content"]
        assert "Important Article" in user_msg


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


# ============================================================
# TestGenerateInitialKnowledge (async, mocked LLM)
# ============================================================


class TestGenerateInitialKnowledge:
    async def test_passes_correct_args_and_recomputes_tokens(self) -> None:
        """Verify correct model/messages are passed and token count is recomputed."""
        expected = KnowledgeStateUpdate(
            updated_summary="Initial knowledge summary.",
            token_count=0,  # will be recomputed
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
        messages = call_kwargs["messages"]
        assert "Init Topic" in messages[1]["content"]

    async def test_recomputes_token_count(self) -> None:
        expected = KnowledgeStateUpdate(
            updated_summary="Some text here.",
            token_count=999,  # LLM's guess
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
        messages = call_kwargs["messages"]
        user_msg = messages[1]["content"]
        assert "Old summary." in user_msg
        assert "New price announced" in user_msg
        assert "$39.99" in user_msg

    async def test_recomputes_token_count(self) -> None:
        expected = KnowledgeStateUpdate(updated_summary="Text.", token_count=999)
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

        llm_result = KnowledgeStateUpdate(updated_summary="Initial summary.", token_count=30)
        with patch(
            "app.analysis.knowledge.generate_initial_knowledge",
            new_callable=AsyncMock,
            return_value=llm_result,
        ):
            state = await initialize_knowledge(topic, articles, db_conn, settings)

        assert state.id is not None
        assert state.topic_id == topic.id
        assert state.summary_text == "Initial summary."
        assert state.token_count == 30

        # Verify persisted in DB
        stored = get_knowledge_state(db_conn, topic.id)
        assert stored is not None
        assert stored.summary_text == "Initial summary."

    async def test_returns_state_with_id(self, db_conn: sqlite3.Connection) -> None:
        topic = Topic(name="T2", description="D2", feed_urls=[])
        topic = create_topic(db_conn, topic)
        db_conn.commit()
        settings = _make_settings()

        llm_result = KnowledgeStateUpdate(updated_summary="S", token_count=5)
        with patch(
            "app.analysis.knowledge.generate_initial_knowledge",
            new_callable=AsyncMock,
            return_value=llm_result,
        ):
            state = await initialize_knowledge(topic, [], db_conn, settings)

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
            updated_summary="Updated summary with Fact 1.",
            token_count=35,
        )
        settings = _make_settings()

        with patch(
            "app.analysis.knowledge.generate_knowledge_update",
            new_callable=AsyncMock,
            return_value=llm_result,
        ):
            state = await update_knowledge(topic, novelty, db_conn, settings)

        assert state.summary_text == "Updated summary with Fact 1."
        assert state.token_count == 35

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
