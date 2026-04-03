"""Tests for the token-budget truncation safety net in knowledge.py."""

import sqlite3
from unittest.mock import AsyncMock, patch

from app.analysis.knowledge import _truncate_to_budget, initialize_knowledge, update_knowledge
from app.analysis.llm import KnowledgeStateUpdate, NoveltyResult
from app.config import LLMSettings, Settings
from app.crud import create_knowledge_state, create_topic, get_knowledge_state
from app.models import Article, KnowledgeState, Topic

# --- Helpers ---


def _make_settings(max_tokens: int = 500, **overrides) -> Settings:
    defaults = {
        "llm": LLMSettings(model="openai/gpt-4o-mini", api_key="test-key"),
        "knowledge_state_max_tokens": max_tokens,
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


# Mock count_tokens to return word count for predictable tests
def _word_count_tokens(text: str, model: str) -> int:
    return len(text.split())


# ============================================================
# TestTruncateToBudget
# ============================================================


class TestTruncateToBudget:
    def test_under_budget_returns_unchanged(self) -> None:
        text = "Short sentence."
        with patch(
            "app.analysis.knowledge.count_tokens",
            side_effect=_word_count_tokens,
        ):
            result_text, result_count = _truncate_to_budget(text, max_tokens=100, model="m")
        assert result_text == text
        assert result_count == _word_count_tokens(text, "m")

    def test_at_budget_returns_unchanged(self) -> None:
        # "One two." → 2 words
        text = "One two."
        with patch(
            "app.analysis.knowledge.count_tokens",
            side_effect=_word_count_tokens,
        ):
            result_text, result_count = _truncate_to_budget(text, max_tokens=2, model="m")
        assert result_text == text
        assert result_count == 2

    def test_over_budget_truncates_trailing_sentences(self) -> None:
        # Three sentences: 2 + 2 + 2 = 6 words, budget = 4 → drop last sentence
        text = "Fact one. Fact two. Fact three."
        with patch(
            "app.analysis.knowledge.count_tokens",
            side_effect=_word_count_tokens,
        ):
            result_text, result_count = _truncate_to_budget(text, max_tokens=4, model="m")
        # Should drop "Fact three." and return "Fact one. Fact two."
        assert "Fact three." not in result_text
        assert "Fact one." in result_text
        assert result_count <= 4

    def test_single_sentence_over_budget_returns_as_is(self) -> None:
        # One long sentence, budget too small — must not truncate to empty
        text = "This is a very long single sentence with many words."
        with patch(
            "app.analysis.knowledge.count_tokens",
            side_effect=_word_count_tokens,
        ):
            result_text, result_count = _truncate_to_budget(text, max_tokens=1, model="m")
        assert result_text == text

    def test_removes_multiple_sentences_until_fits(self) -> None:
        # "A b. C d. E f. G h." → each 2 words, 8 total, budget = 2
        text = "A b. C d. E f. G h."
        with patch(
            "app.analysis.knowledge.count_tokens",
            side_effect=_word_count_tokens,
        ):
            result_text, result_count = _truncate_to_budget(text, max_tokens=2, model="m")
        assert result_count <= 2
        assert "A b." in result_text

    def test_returns_updated_token_count(self) -> None:
        text = "First sentence. Second sentence. Third sentence."
        with patch(
            "app.analysis.knowledge.count_tokens",
            side_effect=_word_count_tokens,
        ):
            _, result_count = _truncate_to_budget(text, max_tokens=4, model="m")
        # Verify token count matches the truncated text's actual word count
        assert result_count <= 4


# ============================================================
# TestInitializeKnowledgeTruncation (async, db_conn)
# ============================================================


class TestInitializeKnowledgeTruncation:
    async def test_truncates_when_llm_returns_over_budget(self, db_conn: sqlite3.Connection) -> None:
        topic = Topic(name="Budget Topic", description="Desc", feed_urls=[])
        topic = create_topic(db_conn, topic)
        db_conn.commit()

        # Three sentences; mock treats each word as 100 tokens.
        # Budget=500 → only "Sentence one here." (3 words = 300 tokens) fits.
        long_summary = "Sentence one here. Sentence two here. Sentence three here."
        llm_result = KnowledgeStateUpdate(
            sufficient_data=True,
            confidence=0.9,
            updated_summary=long_summary,
            token_count=9999,  # LLM's (wrong) self-reported count
        )
        settings = _make_settings(max_tokens=500)

        def _heavy_word_count(text: str, model: str) -> int:
            return len(text.split()) * 100

        with (
            patch(
                "app.analysis.knowledge.generate_initial_knowledge",
                new_callable=AsyncMock,
                return_value=llm_result,
            ),
            patch(
                "app.analysis.knowledge.count_tokens",
                side_effect=_heavy_word_count,
            ),
        ):
            state = await initialize_knowledge(topic, [], db_conn, settings)

        assert state.token_count <= 500
        assert "Sentence three here." not in state.summary_text

    async def test_no_truncation_when_within_budget(self, db_conn: sqlite3.Connection) -> None:
        topic = Topic(name="Within Budget", description="Desc", feed_urls=[])
        topic = create_topic(db_conn, topic)
        db_conn.commit()

        # 2 words → 2 tokens (with word-count mock), well under 500
        short_summary = "Short summary."
        llm_result = KnowledgeStateUpdate(
            sufficient_data=True,
            confidence=0.9,
            updated_summary=short_summary,
            token_count=2,
        )
        settings = _make_settings(max_tokens=500)

        with (
            patch(
                "app.analysis.knowledge.generate_initial_knowledge",
                new_callable=AsyncMock,
                return_value=llm_result,
            ),
            patch(
                "app.analysis.knowledge.count_tokens",
                side_effect=_word_count_tokens,
            ),
        ):
            state = await initialize_knowledge(topic, [], db_conn, settings)

        assert state.summary_text == short_summary

    async def test_truncated_text_is_persisted(self, db_conn: sqlite3.Connection) -> None:
        topic = Topic(name="Persist Topic", description="Desc", feed_urls=[])
        topic = create_topic(db_conn, topic)
        db_conn.commit()

        long_summary = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota."
        llm_result = KnowledgeStateUpdate(
            sufficient_data=True,
            confidence=0.9,
            updated_summary=long_summary,
            token_count=9999,
        )
        settings = _make_settings(max_tokens=500)

        def _heavy_word_count(text: str, model: str) -> int:
            return len(text.split()) * 100

        with (
            patch(
                "app.analysis.knowledge.generate_initial_knowledge",
                new_callable=AsyncMock,
                return_value=llm_result,
            ),
            patch(
                "app.analysis.knowledge.count_tokens",
                side_effect=_heavy_word_count,
            ),
        ):
            await initialize_knowledge(topic, [], db_conn, settings)

        stored = get_knowledge_state(db_conn, topic.id)
        assert stored is not None
        assert stored.token_count <= 500
        assert "Eta theta iota." not in stored.summary_text


# ============================================================
# TestUpdateKnowledgeTruncation (async, db_conn)
# ============================================================


class TestUpdateKnowledgeTruncation:
    async def test_truncates_when_llm_returns_over_budget(self, db_conn: sqlite3.Connection) -> None:
        topic = Topic(name="Update Budget", description="Desc", feed_urls=[])
        topic = create_topic(db_conn, topic)
        db_conn.commit()

        initial = KnowledgeState(topic_id=topic.id, summary_text="Old summary.", token_count=5)
        create_knowledge_state(db_conn, initial)
        db_conn.commit()

        # Each word = 100 tokens; budget=500 → only first sentence fits
        long_summary = "New fact one. New fact two. New fact three."
        llm_result = KnowledgeStateUpdate(
            sufficient_data=True,
            confidence=0.9,
            updated_summary=long_summary,
            token_count=9999,
        )
        novelty = NoveltyResult(
            has_new_info=True,
            summary="New findings",
            key_facts=["Fact"],
            confidence=0.9,
        )
        settings = _make_settings(max_tokens=500)

        def _heavy_word_count(text: str, model: str) -> int:
            return len(text.split()) * 100

        with (
            patch(
                "app.analysis.knowledge.generate_knowledge_update",
                new_callable=AsyncMock,
                return_value=llm_result,
            ),
            patch(
                "app.analysis.knowledge.count_tokens",
                side_effect=_heavy_word_count,
            ),
        ):
            state = await update_knowledge(topic, novelty, db_conn, settings)

        assert state.token_count <= 500
        assert "New fact three." not in state.summary_text

    async def test_no_truncation_when_within_budget(self, db_conn: sqlite3.Connection) -> None:
        topic = Topic(name="Update No Trunc", description="Desc", feed_urls=[])
        topic = create_topic(db_conn, topic)
        db_conn.commit()

        initial = KnowledgeState(topic_id=topic.id, summary_text="Old.", token_count=1)
        create_knowledge_state(db_conn, initial)
        db_conn.commit()

        # 2 words → 2 tokens, well under 500
        updated_summary = "Short update."
        llm_result = KnowledgeStateUpdate(
            sufficient_data=True,
            confidence=0.9,
            updated_summary=updated_summary,
            token_count=2,
        )
        novelty = NoveltyResult(has_new_info=True, summary="X", confidence=0.8)
        settings = _make_settings(max_tokens=500)

        with (
            patch(
                "app.analysis.knowledge.generate_knowledge_update",
                new_callable=AsyncMock,
                return_value=llm_result,
            ),
            patch(
                "app.analysis.knowledge.count_tokens",
                side_effect=_word_count_tokens,
            ),
        ):
            state = await update_knowledge(topic, novelty, db_conn, settings)

        assert state.summary_text == updated_summary

    async def test_truncated_text_is_persisted(self, db_conn: sqlite3.Connection) -> None:
        topic = Topic(name="Update Persist", description="Desc", feed_urls=[])
        topic = create_topic(db_conn, topic)
        db_conn.commit()

        initial = KnowledgeState(topic_id=topic.id, summary_text="Old.", token_count=1)
        create_knowledge_state(db_conn, initial)
        db_conn.commit()

        long_summary = "One two three. Four five six. Seven eight nine."
        llm_result = KnowledgeStateUpdate(
            sufficient_data=True,
            confidence=0.9,
            updated_summary=long_summary,
            token_count=9999,
        )
        novelty = NoveltyResult(has_new_info=True, summary="X", confidence=0.8)
        settings = _make_settings(max_tokens=500)

        def _heavy_word_count(text: str, model: str) -> int:
            return len(text.split()) * 100

        with (
            patch(
                "app.analysis.knowledge.generate_knowledge_update",
                new_callable=AsyncMock,
                return_value=llm_result,
            ),
            patch(
                "app.analysis.knowledge.count_tokens",
                side_effect=_heavy_word_count,
            ),
        ):
            await update_knowledge(topic, novelty, db_conn, settings)

        stored = get_knowledge_state(db_conn, topic.id)
        assert stored is not None
        assert stored.token_count <= 500
        assert "Seven eight nine." not in stored.summary_text
