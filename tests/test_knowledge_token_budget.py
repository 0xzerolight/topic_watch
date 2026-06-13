"""Tests for the token-budget handling (LLM compression + truncation fallback) in knowledge.py."""

import sqlite3
from unittest.mock import AsyncMock, patch

from app.analysis.knowledge import _truncate_to_budget, compress_knowledge, initialize_knowledge, update_knowledge
from app.analysis.llm import CompressedKnowledge, KnowledgeStateUpdate, NoveltyResult
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


def _heavy_word_count(text: str, model: str) -> int:
    """Each word = 100 tokens — makes multi-sentence summaries blow the 500 budget."""
    return len(text.split()) * 100


# ============================================================
# TestCompressKnowledge (unit, async)
# ============================================================


class TestCompressKnowledge:
    async def test_uses_llm_compression_preserving_all_facts(self) -> None:
        """Over-budget text is compressed by the LLM — no trailing fact is dropped."""
        topic = _make_topic()
        long_summary = "Fact one here. Fact two here. Fact three here."
        # LLM returns a denser summary that keeps ALL three facts.
        compressed = CompressedKnowledge(compressed_summary="F1. F2. F3.", token_count=0)
        settings = _make_settings(max_tokens=500)

        with (
            patch(
                "app.analysis.knowledge.compress_knowledge_summary",
                new_callable=AsyncMock,
                return_value=compressed,
            ),
            patch("app.analysis.knowledge.count_tokens", side_effect=_word_count_tokens),
        ):
            text, count = await compress_knowledge(long_summary, topic, settings)

        assert text == "F1. F2. F3."
        assert count <= 500
        assert count == 3  # recomputed authoritatively, not the LLM's 0

    async def test_falls_back_to_truncation_on_llm_error(self) -> None:
        """If compression raises, degrade to lossy truncation rather than crash."""
        topic = _make_topic()
        long_summary = "Sentence one here. Sentence two here. Sentence three here."
        settings = _make_settings(max_tokens=500)

        with (
            patch(
                "app.analysis.knowledge.compress_knowledge_summary",
                new_callable=AsyncMock,
                side_effect=Exception("LLM compression failed"),
            ),
            patch("app.analysis.knowledge.count_tokens", side_effect=_heavy_word_count),
        ):
            text, count = await compress_knowledge(long_summary, topic, settings)

        # Fell back to truncation: fits the budget, trailing sentence dropped.
        assert count <= 500
        assert "Sentence three here." not in text
        assert "Sentence one here." in text

    async def test_truncates_when_compression_still_over_budget(self) -> None:
        """If the LLM's compression is itself over budget, truncate its output."""
        topic = _make_topic()
        long_summary = "Old verbose summary text."
        # LLM returns something still too long.
        compressed = CompressedKnowledge(compressed_summary="Still one. Still two. Still three.", token_count=0)
        settings = _make_settings(max_tokens=500)

        with (
            patch(
                "app.analysis.knowledge.compress_knowledge_summary",
                new_callable=AsyncMock,
                return_value=compressed,
            ),
            patch("app.analysis.knowledge.count_tokens", side_effect=_heavy_word_count),
        ):
            text, count = await compress_knowledge(long_summary, topic, settings)

        assert count <= 500
        assert "Still three." not in text


# ============================================================
# TestInitializeKnowledgeBudget (async, db_conn)
# ============================================================


class TestInitializeKnowledgeBudget:
    async def test_compresses_when_llm_returns_over_budget(self, db_conn: sqlite3.Connection) -> None:
        """Over-budget init triggers LLM compression that preserves every fact."""
        topic = Topic(name="Budget Topic", description="Desc", feed_urls=[])
        topic = create_topic(db_conn, topic)
        db_conn.commit()

        long_summary = "Sentence one here. Sentence two here. Sentence three here."
        llm_result = KnowledgeStateUpdate(
            sufficient_data=True,
            confidence=0.9,
            updated_summary=long_summary,
            token_count=9999,
        )
        # Compression keeps all three facts but shorter (3 words = 300 tokens).
        compressed = CompressedKnowledge(compressed_summary="S1 S2 S3.", token_count=0)
        settings = _make_settings(max_tokens=500)

        with (
            patch(
                "app.analysis.knowledge.generate_initial_knowledge",
                new_callable=AsyncMock,
                return_value=llm_result,
            ),
            patch(
                "app.analysis.knowledge.compress_knowledge_summary",
                new_callable=AsyncMock,
                return_value=compressed,
            ),
            patch("app.analysis.knowledge.count_tokens", side_effect=_heavy_word_count),
        ):
            state = await initialize_knowledge(topic, [], db_conn, settings)

        assert state.token_count <= 500
        assert state.summary_text == "S1 S2 S3."

        stored = get_knowledge_state(db_conn, topic.id)
        assert stored is not None
        assert stored.summary_text == "S1 S2 S3."
        assert stored.token_count <= 500

    async def test_falls_back_to_truncation_on_compression_error(self, db_conn: sqlite3.Connection) -> None:
        """If compression fails, init degrades to truncation (still no overflow)."""
        topic = Topic(name="Fallback Init", description="Desc", feed_urls=[])
        topic = create_topic(db_conn, topic)
        db_conn.commit()

        long_summary = "Sentence one here. Sentence two here. Sentence three here."
        llm_result = KnowledgeStateUpdate(
            sufficient_data=True,
            confidence=0.9,
            updated_summary=long_summary,
            token_count=9999,
        )
        settings = _make_settings(max_tokens=500)

        with (
            patch(
                "app.analysis.knowledge.generate_initial_knowledge",
                new_callable=AsyncMock,
                return_value=llm_result,
            ),
            patch(
                "app.analysis.knowledge.compress_knowledge_summary",
                new_callable=AsyncMock,
                side_effect=Exception("compression down"),
            ),
            patch("app.analysis.knowledge.count_tokens", side_effect=_heavy_word_count),
        ):
            state = await initialize_knowledge(topic, [], db_conn, settings)

        assert state.token_count <= 500
        assert "Sentence three here." not in state.summary_text

    async def test_no_compression_when_within_budget(self, db_conn: sqlite3.Connection) -> None:
        topic = Topic(name="Within Budget", description="Desc", feed_urls=[])
        topic = create_topic(db_conn, topic)
        db_conn.commit()

        short_summary = "Short summary."
        llm_result = KnowledgeStateUpdate(
            sufficient_data=True,
            confidence=0.9,
            updated_summary=short_summary,
            token_count=2,
        )
        settings = _make_settings(max_tokens=500)
        compress_mock = AsyncMock()

        with (
            patch(
                "app.analysis.knowledge.generate_initial_knowledge",
                new_callable=AsyncMock,
                return_value=llm_result,
            ),
            patch("app.analysis.knowledge.compress_knowledge_summary", compress_mock),
            patch("app.analysis.knowledge.count_tokens", side_effect=_word_count_tokens),
        ):
            state = await initialize_knowledge(topic, [], db_conn, settings)

        assert state.summary_text == short_summary
        compress_mock.assert_not_called()


# ============================================================
# TestUpdateKnowledgeBudget (async, db_conn)
# ============================================================


class TestUpdateKnowledgeBudget:
    async def test_compresses_when_llm_returns_over_budget(self, db_conn: sqlite3.Connection) -> None:
        """Over-budget update compresses (preserving facts) instead of dropping them."""
        topic = Topic(name="Update Budget", description="Desc", feed_urls=[])
        topic = create_topic(db_conn, topic)
        db_conn.commit()

        initial = KnowledgeState(topic_id=topic.id, summary_text="Old summary.", token_count=5)
        create_knowledge_state(db_conn, initial)
        db_conn.commit()

        long_summary = "New fact one. New fact two. New fact three."
        llm_result = KnowledgeStateUpdate(
            sufficient_data=True,
            confidence=0.9,
            updated_summary=long_summary,
            token_count=9999,
        )
        # Compression keeps all three new facts — the one trailing-truncation would lose.
        compressed = CompressedKnowledge(compressed_summary="N1 N2 N3.", token_count=0)
        novelty = NoveltyResult(has_new_info=True, summary="New findings", key_facts=["Fact"], confidence=0.9)
        settings = _make_settings(max_tokens=500)

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
            patch("app.analysis.knowledge.count_tokens", side_effect=_heavy_word_count),
        ):
            state = await update_knowledge(topic, novelty, db_conn, settings)

        assert state.token_count <= 500
        # The third fact survives compression — a trailing truncation would have dropped it.
        assert state.summary_text == "N1 N2 N3."

        stored = get_knowledge_state(db_conn, topic.id)
        assert stored is not None
        assert stored.summary_text == "N1 N2 N3."

    async def test_falls_back_to_truncation_on_compression_error(self, db_conn: sqlite3.Connection) -> None:
        """If compression fails on update, degrade to truncation (still no overflow)."""
        topic = Topic(name="Update Fallback", description="Desc", feed_urls=[])
        topic = create_topic(db_conn, topic)
        db_conn.commit()

        initial = KnowledgeState(topic_id=topic.id, summary_text="Old.", token_count=1)
        create_knowledge_state(db_conn, initial)
        db_conn.commit()

        long_summary = "New fact one. New fact two. New fact three."
        llm_result = KnowledgeStateUpdate(
            sufficient_data=True,
            confidence=0.9,
            updated_summary=long_summary,
            token_count=9999,
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
                "app.analysis.knowledge.compress_knowledge_summary",
                new_callable=AsyncMock,
                side_effect=Exception("compression down"),
            ),
            patch("app.analysis.knowledge.count_tokens", side_effect=_heavy_word_count),
        ):
            state = await update_knowledge(topic, novelty, db_conn, settings)

        assert state.token_count <= 500
        assert "New fact three." not in state.summary_text

    async def test_no_compression_when_within_budget(self, db_conn: sqlite3.Connection) -> None:
        topic = Topic(name="Update No Compress", description="Desc", feed_urls=[])
        topic = create_topic(db_conn, topic)
        db_conn.commit()

        initial = KnowledgeState(topic_id=topic.id, summary_text="Old.", token_count=1)
        create_knowledge_state(db_conn, initial)
        db_conn.commit()

        updated_summary = "Short update."
        llm_result = KnowledgeStateUpdate(
            sufficient_data=True,
            confidence=0.9,
            updated_summary=updated_summary,
            token_count=2,
        )
        novelty = NoveltyResult(has_new_info=True, summary="X", confidence=0.8)
        settings = _make_settings(max_tokens=500)
        compress_mock = AsyncMock()

        with (
            patch(
                "app.analysis.knowledge.generate_knowledge_update",
                new_callable=AsyncMock,
                return_value=llm_result,
            ),
            patch("app.analysis.knowledge.compress_knowledge_summary", compress_mock),
            patch("app.analysis.knowledge.count_tokens", side_effect=_word_count_tokens),
        ):
            state = await update_knowledge(topic, novelty, db_conn, settings)

        assert state.summary_text == updated_summary
        compress_mock.assert_not_called()
