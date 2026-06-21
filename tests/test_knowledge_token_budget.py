"""Tests for the token-budget handling (LLM compression + truncation fallback) in knowledge.py."""

import math
import re
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

    async def test_single_mega_sentence_overflow_persists_over_budget(self) -> None:
        """OVH-164: the one documented path where the result can exceed the budget.

        When the LLM's compression is still over budget AND consists of a single
        sentence (no boundaries to truncate at), ``_truncate_to_budget`` keeps that
        sentence intact rather than returning empty — so the returned token_count
        is legitimately > max_tokens. Pins this overflow contract (and the honest
        docstring caveat) so a future "always fits" refactor that silently drops
        the only sentence is caught.
        """
        topic = _make_topic()
        long_summary = "Old verbose summary text."
        # One sentence, no internal boundaries; _heavy_word_count makes it overflow.
        compressed = CompressedKnowledge(
            compressed_summary="One huge unsplittable mega sentence with many words and no boundaries",
            token_count=0,
        )
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

        # Facts preserved (never truncated to empty), but the budget is exceeded.
        assert text == compressed.compressed_summary
        assert count > 500


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
            state = (await initialize_knowledge(topic, [], db_conn, settings)).state

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
            state = (await initialize_knowledge(topic, [], db_conn, settings)).state

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
            state = (await initialize_knowledge(topic, [], db_conn, settings)).state

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
            state = (await update_knowledge(topic, novelty, db_conn, settings)).state

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
            state = (await update_knowledge(topic, novelty, db_conn, settings)).state

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
            state = (await update_knowledge(topic, novelty, db_conn, settings)).state

        assert state.summary_text == updated_summary
        compress_mock.assert_not_called()


# ============================================================
# TestTruncateToBudgetCharacterization (OVH-049)
# ============================================================
#
# The binary-search rewrite of _truncate_to_budget must be a pure algorithmic
# optimization: identical output to the old keep-leading/drop-trailing impl,
# with O(log n) tokenizer calls instead of O(n).


def _old_truncate_to_budget(text: str, max_tokens: int, count_tokens) -> tuple[str, int]:
    """The pre-OVH-049 O(n^2) reference implementation (oracle for output parity)."""
    token_count = count_tokens(text)
    if token_count <= max_tokens:
        return text, token_count

    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) <= 1:
        return text, token_count

    while len(sentences) > 1:
        sentences.pop()
        truncated = " ".join(sentences)
        token_count = count_tokens(truncated)
        if token_count <= max_tokens:
            return truncated, token_count

    final = sentences[0]
    return final, count_tokens(final)


class _CountingTokenizer:
    """A deterministic word-count tokenizer that records how often it is called."""

    def __init__(self, tokens_per_word: int = 1) -> None:
        self.tokens_per_word = tokens_per_word
        self.calls = 0

    def __call__(self, text: str, model: str = "m") -> int:
        self.calls += 1
        return len(text.split()) * self.tokens_per_word


_SAMPLE_TEXTS = [
    # Many short single-word sentences (worst case for the old impl).
    " ".join(f"S{i}." for i in range(50)),
    # Two-word sentences.
    " ".join(f"Fact {i}." for i in range(30)),
    # Mixed lengths with !/? terminators.
    "Alpha beta gamma. Delta! Epsilon zeta? Eta theta iota kappa. Lambda mu. Nu xi omicron pi rho.",
    # Short, already under any reasonable budget.
    "Just one sentence here.",
    # Single very long sentence (no split points).
    "word " * 40,
]


class TestTruncateToBudgetCharacterization:
    def test_output_identical_to_old_impl(self) -> None:
        """New binary-search impl yields byte-identical output to the old loop."""
        for text in _SAMPLE_TEXTS:
            for max_tokens in (1, 2, 5, 10, 25, 1000):
                old = _CountingTokenizer()
                expected_text, expected_count = _old_truncate_to_budget(text, max_tokens, old)

                new = _CountingTokenizer()
                with patch("app.analysis.knowledge.count_tokens", side_effect=new):
                    got_text, got_count = _truncate_to_budget(text, max_tokens, model="m")

                assert got_text == expected_text, f"text={text!r} budget={max_tokens}"
                assert got_count == expected_count, f"text={text!r} budget={max_tokens}"

    def test_output_identical_with_multi_token_words(self) -> None:
        """Parity holds when tokens != words (each word weighs >1 token)."""
        for text in _SAMPLE_TEXTS:
            for max_tokens in (3, 7, 50, 300):
                old = _CountingTokenizer(tokens_per_word=4)
                expected_text, expected_count = _old_truncate_to_budget(text, max_tokens, old)

                new = _CountingTokenizer(tokens_per_word=4)
                with patch("app.analysis.knowledge.count_tokens", side_effect=new):
                    got_text, got_count = _truncate_to_budget(text, max_tokens, model="m")

                assert got_text == expected_text, f"text={text!r} budget={max_tokens}"
                assert got_count == expected_count, f"text={text!r} budget={max_tokens}"

    def test_uses_fewer_tokenizer_calls_than_old_impl(self) -> None:
        """The new impl makes strictly fewer tokenizer calls on a many-sentence text."""
        text = " ".join(f"S{i}." for i in range(64))  # 64 single-word sentences
        max_tokens = 4

        old = _CountingTokenizer()
        _old_truncate_to_budget(text, max_tokens, old)

        new = _CountingTokenizer()
        with patch("app.analysis.knowledge.count_tokens", side_effect=new):
            _truncate_to_budget(text, max_tokens, model="m")

        assert new.calls < old.calls
        # O(log n) bound: bisection over n sentences plus a small constant of
        # bookkeeping calls (initial full count, final recount).
        n = 64
        assert new.calls <= 3 * math.ceil(math.log2(n + 1)) + 5

    def test_tokenizer_call_count_scales_logarithmically(self) -> None:
        """Doubling the sentence count adds only a constant number of calls."""

        def calls_for(num_sentences: int) -> int:
            text = " ".join(f"S{i}." for i in range(num_sentences))
            tok = _CountingTokenizer()
            with patch("app.analysis.knowledge.count_tokens", side_effect=tok):
                _truncate_to_budget(text, max_tokens=4, model="m")
            return tok.calls

        small = calls_for(16)
        large = calls_for(256)  # 16x more sentences
        # Linear would be ~16x; logarithmic adds a small constant per doubling.
        assert large <= small + 12


# TestCompressionTriggerBoundary (OVH-077)
# ============================================================


class TestCompressionTriggerBoundary:
    """OVH-077: the compression trigger is a strict ``>`` against the budget.

    A summary of EXACTLY max_tokens must NOT compress (stored verbatim);
    max_tokens + 1 MUST compress. A regression flipping ``>`` to ``>=`` (needless
    at-budget compression, extra LLM cost) or to a slack threshold (silent
    over-budget persistence) would change behavior at exactly these two cells.
    The LLM-reported ``token_count`` drives the trigger, so it is set directly.
    """

    _BUDGET = 500

    async def test_init_at_budget_not_compressed(self, db_conn: sqlite3.Connection) -> None:
        """token_count == max_tokens: no compression, summary stored verbatim."""
        topic = create_topic(db_conn, Topic(name="Init At Budget", description="D", feed_urls=[]))
        db_conn.commit()

        summary = "Exactly at budget summary."
        llm_result = KnowledgeStateUpdate(
            sufficient_data=True,
            confidence=0.9,
            updated_summary=summary,
            token_count=self._BUDGET,  # == budget
        )
        compress_mock = AsyncMock()
        settings = _make_settings(max_tokens=self._BUDGET)

        with (
            patch(
                "app.analysis.knowledge.generate_initial_knowledge",
                new_callable=AsyncMock,
                return_value=llm_result,
            ),
            patch("app.analysis.knowledge.compress_knowledge_summary", compress_mock),
        ):
            state = (await initialize_knowledge(topic, [], db_conn, settings)).state

        compress_mock.assert_not_called()
        assert state.summary_text == summary
        assert state.token_count == self._BUDGET

    async def test_init_one_over_budget_compressed(self, db_conn: sqlite3.Connection) -> None:
        """token_count == max_tokens + 1: compression runs."""
        topic = create_topic(db_conn, Topic(name="Init Over Budget", description="D", feed_urls=[]))
        db_conn.commit()

        llm_result = KnowledgeStateUpdate(
            sufficient_data=True,
            confidence=0.9,
            updated_summary="Over budget summary needing compression.",
            token_count=self._BUDGET + 1,  # == budget + 1
        )
        compressed = CompressedKnowledge(compressed_summary="Tight.", token_count=0)
        compress_mock = AsyncMock(return_value=compressed)
        settings = _make_settings(max_tokens=self._BUDGET)

        with (
            patch(
                "app.analysis.knowledge.generate_initial_knowledge",
                new_callable=AsyncMock,
                return_value=llm_result,
            ),
            patch("app.analysis.knowledge.compress_knowledge_summary", compress_mock),
            # Recompute of the compressed output fits the budget (1 word = 1 token).
            patch("app.analysis.knowledge.count_tokens", side_effect=_word_count_tokens),
        ):
            state = (await initialize_knowledge(topic, [], db_conn, settings)).state

        compress_mock.assert_awaited_once()
        assert state.summary_text == "Tight."

    async def test_update_at_budget_not_compressed(self, db_conn: sqlite3.Connection) -> None:
        """token_count == max_tokens on update: no compression, summary verbatim."""
        topic = create_topic(db_conn, Topic(name="Update At Budget", description="D", feed_urls=[]))
        db_conn.commit()
        create_knowledge_state(db_conn, KnowledgeState(topic_id=topic.id, summary_text="Old.", token_count=1))
        db_conn.commit()

        summary = "Update exactly at budget."
        llm_result = KnowledgeStateUpdate(
            sufficient_data=True,
            confidence=0.9,
            updated_summary=summary,
            token_count=self._BUDGET,  # == budget
        )
        compress_mock = AsyncMock()
        novelty = NoveltyResult(has_new_info=True, summary="X", confidence=0.8)
        settings = _make_settings(max_tokens=self._BUDGET)

        with (
            patch(
                "app.analysis.knowledge.generate_knowledge_update",
                new_callable=AsyncMock,
                return_value=llm_result,
            ),
            patch("app.analysis.knowledge.compress_knowledge_summary", compress_mock),
        ):
            state = (await update_knowledge(topic, novelty, db_conn, settings)).state

        compress_mock.assert_not_called()
        assert state.summary_text == summary
        assert state.token_count == self._BUDGET

    async def test_update_one_over_budget_compressed(self, db_conn: sqlite3.Connection) -> None:
        """token_count == max_tokens + 1 on update: compression runs."""
        topic = create_topic(db_conn, Topic(name="Update Over Budget", description="D", feed_urls=[]))
        db_conn.commit()
        create_knowledge_state(db_conn, KnowledgeState(topic_id=topic.id, summary_text="Old.", token_count=1))
        db_conn.commit()

        llm_result = KnowledgeStateUpdate(
            sufficient_data=True,
            confidence=0.9,
            updated_summary="Update over budget needing compression.",
            token_count=self._BUDGET + 1,  # == budget + 1
        )
        compressed = CompressedKnowledge(compressed_summary="Tight.", token_count=0)
        compress_mock = AsyncMock(return_value=compressed)
        novelty = NoveltyResult(has_new_info=True, summary="X", confidence=0.8)
        settings = _make_settings(max_tokens=self._BUDGET)

        with (
            patch(
                "app.analysis.knowledge.generate_knowledge_update",
                new_callable=AsyncMock,
                return_value=llm_result,
            ),
            patch("app.analysis.knowledge.compress_knowledge_summary", compress_mock),
            patch("app.analysis.knowledge.count_tokens", side_effect=_word_count_tokens),
        ):
            state = (await update_knowledge(topic, novelty, db_conn, settings)).state

        compress_mock.assert_awaited_once()
        assert state.summary_text == "Tight."
