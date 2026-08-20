"""Tests for the token-budget handling: knowledge.py's compression/truncation
fallback, and the aggregate input budget the LLM gateway enforces per request."""

import math
import re
import sqlite3
from unittest.mock import AsyncMock, MagicMock, patch

import app.analysis.llm as llm_module
from app.analysis.knowledge import _truncate_to_budget, compress_knowledge
from app.analysis.llm import (
    CompressedKnowledge,
    KnowledgeStateUpdate,
    NoveltyResult,
    _fit_article_prompt,
    _input_token_budget,
    analyze_articles,
    generate_initial_knowledge,
)
from app.analysis.prompts import build_knowledge_init_messages, build_novelty_messages
from app.config import LLMSettings, Settings
from app.crud import create_knowledge_state, create_topic, get_knowledge_state
from app.models import Article, KnowledgeState, Topic
from tests.helpers import init_knowledge, update_knowledge

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
        # LLM returns a denser summary that keeps ALL three facts. token_count is
        # what compress_knowledge_summary already computed via count_tokens
        # (_word_count_tokens("F1. F2. F3.") == 3); compress_knowledge reuses it
        # rather than recounting (OVH-135).
        compressed = CompressedKnowledge(compressed_summary="F1. F2. F3.", token_count=3)
        settings = _make_settings(max_tokens=500)

        with (
            patch(
                "app.analysis.knowledge.compress_knowledge_summary",
                new_callable=AsyncMock,
                return_value=compressed,
            ),
            patch("app.analysis.knowledge.count_tokens", side_effect=_word_count_tokens),
        ):
            text, count, usage = await compress_knowledge(long_summary, topic, settings)

        assert text == "F1. F2. F3."
        assert count <= 500
        assert count == 3  # reused from compress_knowledge_summary, not re-counted

    async def test_reuses_summary_count_without_recount(self) -> None:
        """OVH-135: the in-budget success path does NOT re-tokenize the compressed
        summary — it reuses the count compress_knowledge_summary already set."""
        topic = _make_topic()
        long_summary = "Fact one here. Fact two here. Fact three here."
        compressed = CompressedKnowledge(compressed_summary="F1. F2. F3.", token_count=3)
        settings = _make_settings(max_tokens=500)

        count_mock = MagicMock(side_effect=_word_count_tokens)
        with (
            patch(
                "app.analysis.knowledge.compress_knowledge_summary",
                new_callable=AsyncMock,
                return_value=compressed,
            ),
            patch("app.analysis.knowledge.count_tokens", count_mock),
        ):
            _, count, _ = await compress_knowledge(long_summary, topic, settings)

        # In-budget success path must not call count_tokens at all (reuse only).
        count_mock.assert_not_called()
        assert count == 3

    async def test_propagates_compression_usage(self) -> None:
        """OVH-129: the compression round-trip's token usage is returned so the
        caller can fold it into the persisted per-check totals."""
        topic = _make_topic()
        long_summary = "Fact one here. Fact two here. Fact three here."
        compressed = CompressedKnowledge(
            compressed_summary="F1. F2. F3.",
            token_count=3,
            prompt_tokens=123,
            completion_tokens=45,
        )
        settings = _make_settings(max_tokens=500)

        with (
            patch(
                "app.analysis.knowledge.compress_knowledge_summary",
                new_callable=AsyncMock,
                return_value=compressed,
            ),
            patch("app.analysis.knowledge.count_tokens", side_effect=_word_count_tokens),
        ):
            _, _, usage = await compress_knowledge(long_summary, topic, settings)

        assert usage.prompt_tokens == 123
        assert usage.completion_tokens == 45

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
            text, count, usage = await compress_knowledge(long_summary, topic, settings)

        # Fell back to truncation: fits the budget, trailing sentence dropped.
        assert count <= 500
        assert "Sentence three here." not in text
        assert "Sentence one here." in text
        # No LLM round-trip succeeded, so no compression usage to report.
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0

    async def test_truncates_when_compression_still_over_budget(self) -> None:
        """If the LLM's compression is itself over budget, truncate its output."""
        topic = _make_topic()
        long_summary = "Old verbose summary text."
        # LLM returns something still too long. token_count reflects what
        # compress_knowledge_summary computed (3 words * 100 = 300... but
        # _heavy_word_count counts the whole string; set it to overflow).
        compressed = CompressedKnowledge(compressed_summary="Still one. Still two. Still three.", token_count=9999)
        settings = _make_settings(max_tokens=500)

        with (
            patch(
                "app.analysis.knowledge.compress_knowledge_summary",
                new_callable=AsyncMock,
                return_value=compressed,
            ),
            patch("app.analysis.knowledge.count_tokens", side_effect=_heavy_word_count),
        ):
            text, count, usage = await compress_knowledge(long_summary, topic, settings)

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
        # token_count mirrors what compress_knowledge_summary computed via
        # count_tokens (11 words * 100 = 1100 under _heavy_word_count), which
        # compress_knowledge reuses to decide it is over budget (OVH-135).
        mega = "One huge unsplittable mega sentence with many words and no boundaries"
        compressed = CompressedKnowledge(
            compressed_summary=mega,
            token_count=_heavy_word_count(mega, "m"),
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
            text, count, usage = await compress_knowledge(long_summary, topic, settings)

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
        # token_count mirrors what compress_knowledge_summary computed, which
        # compress_knowledge reuses (OVH-135).
        compressed = CompressedKnowledge(
            compressed_summary="S1 S2 S3.", token_count=_heavy_word_count("S1 S2 S3.", "m")
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
                return_value=compressed,
            ),
            patch("app.analysis.knowledge.count_tokens", side_effect=_heavy_word_count),
        ):
            state = (await init_knowledge(topic, [], db_conn, settings)).state

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
            state = (await init_knowledge(topic, [], db_conn, settings)).state

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
            state = (await init_knowledge(topic, [], db_conn, settings)).state

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
        # token_count mirrors compress_knowledge_summary's count, reused (OVH-135).
        compressed = CompressedKnowledge(
            compressed_summary="N1 N2 N3.", token_count=_heavy_word_count("N1 N2 N3.", "m")
        )
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
            state = (await init_knowledge(topic, [], db_conn, settings)).state

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
        compressed = CompressedKnowledge(
            compressed_summary="Tight.", token_count=1
        )  # _word_count_tokens("Tight.")==1, reused (OVH-135)
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
            state = (await init_knowledge(topic, [], db_conn, settings)).state

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
        compressed = CompressedKnowledge(
            compressed_summary="Tight.", token_count=1
        )  # _word_count_tokens("Tight.")==1, reused (OVH-135)
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


# ============================================================
# TestAggregateInputBudget (TW-AUD-016)
# ============================================================


def _novelty_build(articles, chars, summary, topic):
    return build_novelty_messages(articles, summary, topic, chars)


# gpt-4 has an 8k context window, so a realistic batch genuinely overruns it and
# the degradation ladder actually runs (a 128k model would hide it).
_SMALL_CONTEXT = LLMSettings(model="openai/gpt-4", api_key="test-key")


def _long_article(index: int, body_chars: int = 20_000) -> Article:
    return _make_article(
        id=index,
        title=f"Article {index}",
        url=f"https://example.com/{index}",
        raw_content=" ".join(f"word{index}x{i}" for i in range(body_chars // 8)),
    )


class TestAggregateInputBudget:
    """Per-item caps bounded each article; nothing bounded the SUM of knowledge +
    N articles + schema overhead + requested output against the model's context.

    A request over that window is not a retryable failure: every check safe-fails
    and every knowledge operation raises, permanently, until the configuration
    changes.
    """

    def test_budget_reserves_output_and_schema(self) -> None:
        settings = _make_settings()
        budget = _input_token_budget(settings)
        info = __import__("litellm").get_model_info(settings.llm.model)
        assert budget is not None
        assert budget < info["max_input_tokens"]
        assert budget <= info["max_input_tokens"] - llm_module._bounded_max_tokens(settings)

    def test_unknown_model_has_no_budget(self) -> None:
        """A gateway model string litellm cannot size must not be guessed at."""
        settings = _make_settings(llm=LLMSettings(model="openai/some-private-gateway-model", api_key="k"))
        assert _input_token_budget(settings) is None

    def test_oversized_batch_is_trimmed_to_fit(self) -> None:
        topic = _make_topic()
        settings = _make_settings(llm=_SMALL_CONTEXT)
        articles = [_long_article(i) for i in range(40)]

        messages, kept = _fit_article_prompt(
            lambda arts, chars, summary: _novelty_build(arts, chars, summary, topic),
            articles,
            "Known facts.",
            settings,
            topic.name,
        )

        budget = _input_token_budget(settings)
        assert budget is not None
        assert llm_module._count_messages(messages, settings.llm.model) <= budget
        # The trim is reported, not merely logged: these are the only articles the
        # model actually read, and the caller marks the rest unfinished.
        assert 0 < len(kept) < len(articles)
        assert kept == articles[: len(kept)]

    def test_small_batch_is_untouched(self) -> None:
        topic = _make_topic()
        settings = _make_settings()
        articles = [_make_article(id=1, raw_content="x" * 900)]

        fitted, kept = _fit_article_prompt(
            lambda arts, chars, summary: _novelty_build(arts, chars, summary, topic),
            articles,
            "Known facts.",
            settings,
            topic.name,
        )

        # Full body, no ellipsis: nothing was degraded (the fence nonce is fresh
        # per call, so the messages are compared by their variable part).
        assert "x" * 900 in fitted[1]["content"]
        assert "..." not in fitted[1]["content"]
        assert kept == articles

    def test_articles_are_shrunk_before_any_are_dropped(self) -> None:
        topic = _make_topic()
        settings = _make_settings(llm=_SMALL_CONTEXT)
        articles = [_long_article(i, body_chars=3_000) for i in range(6)]

        messages, kept = _fit_article_prompt(
            lambda arts, chars, summary: _novelty_build(arts, chars, summary, topic),
            articles,
            "",
            settings,
            topic.name,
        )

        user = messages[1]["content"]
        # Every article still present, just with shorter bodies.
        for i in range(6):
            assert f"https://example.com/{i}" in user
        assert kept == articles

    def test_oversized_knowledge_state_is_trimmed_last(self) -> None:
        topic = _make_topic()
        settings = _make_settings(llm=_SMALL_CONTEXT)
        articles = [_make_article(id=1, raw_content="body")]
        huge_summary = " ".join(f"fact{i}" for i in range(200_000))

        messages, kept = _fit_article_prompt(
            lambda arts, chars, summary: _novelty_build(arts, chars, summary, topic),
            articles,
            huge_summary,
            settings,
            topic.name,
        )

        budget = _input_token_budget(settings)
        assert budget is not None
        assert llm_module._count_messages(messages, settings.llm.model) <= budget
        # Trimming the knowledge state costs no article: the one input stays read.
        assert kept == articles

    async def test_analyze_articles_sends_a_fitting_request(self) -> None:
        """The gateway, not the caller, is where the whole request is bounded."""
        captured: dict = {}

        async def _cwc(*_args, **kwargs):
            captured.update(kwargs)
            return NoveltyResult(has_new_info=False, confidence=0.5), MagicMock(usage=None)

        client = MagicMock()
        client.chat.completions.create_with_completion = AsyncMock(side_effect=_cwc)
        settings = _make_settings(llm=_SMALL_CONTEXT)

        with patch("app.analysis.llm._get_client", return_value=client):
            await analyze_articles([_long_article(i) for i in range(40)], "Known.", _make_topic(), settings)

        budget = _input_token_budget(settings)
        assert budget is not None
        assert llm_module._count_messages(captured["messages"], settings.llm.model) <= budget

    async def test_initial_knowledge_sends_a_fitting_request(self) -> None:
        captured: dict = {}

        async def _cwc(*_args, **kwargs):
            captured.update(kwargs)
            return (
                KnowledgeStateUpdate(sufficient_data=True, confidence=0.9, updated_summary="Summary."),
                MagicMock(usage=None),
            )

        client = MagicMock()
        client.chat.completions.create_with_completion = AsyncMock(side_effect=_cwc)
        settings = _make_settings(llm=_SMALL_CONTEXT)

        with patch("app.analysis.llm._get_client", return_value=client):
            await generate_initial_knowledge([_long_article(i) for i in range(40)], _make_topic(), settings)

        budget = _input_token_budget(settings)
        assert budget is not None
        assert llm_module._count_messages(captured["messages"], settings.llm.model) <= budget
        assert build_knowledge_init_messages  # imported builder is the one under test


class TestAnalyzedSubsetIsReported:
    """A prompt fitted by dropping articles has to say which ones survived.

    The drop is what the caller needs to know about: an article the model never
    read must not be marked processed, because nothing would ever offer it again.
    """

    @staticmethod
    def _mock_client(result) -> MagicMock:
        client = MagicMock()
        client.chat.completions.create_with_completion = AsyncMock(
            return_value=(result, MagicMock(usage=None)),
        )
        return client

    async def test_analyze_articles_reports_the_analyzed_subset(self) -> None:
        settings = _make_settings(llm=_SMALL_CONTEXT)
        articles = [_long_article(i) for i in range(40)]
        client = self._mock_client(
            llm_module.NoveltyResponse(has_new_info=False, confidence=0.5, relevance=0.1, importance=1)
        )

        with patch("app.analysis.llm._get_client", return_value=client):
            result = await analyze_articles(articles, "Known.", _make_topic(), settings)

        assert result.analyzed_article_ids is not None
        assert 0 < len(result.analyzed_article_ids) < len(articles)
        assert result.analyzed_article_ids == [a.id for a in articles[: len(result.analyzed_article_ids)]]

    async def test_analyze_articles_reports_every_article_when_nothing_is_dropped(self) -> None:
        settings = _make_settings()
        articles = [_make_article(id=7, raw_content="Short body.")]
        client = self._mock_client(
            llm_module.NoveltyResponse(has_new_info=False, confidence=0.5, relevance=0.1, importance=1)
        )

        with patch("app.analysis.llm._get_client", return_value=client):
            result = await analyze_articles(articles, "Known.", _make_topic(), settings)

        assert result.analyzed_article_ids == [7]

    async def test_initial_knowledge_reports_the_analyzed_subset(self) -> None:
        settings = _make_settings(llm=_SMALL_CONTEXT)
        articles = [_long_article(i) for i in range(40)]
        client = self._mock_client(
            KnowledgeStateUpdate(sufficient_data=True, confidence=0.9, updated_summary="Summary.")
        )

        with patch("app.analysis.llm._get_client", return_value=client):
            result = await generate_initial_knowledge(articles, _make_topic(), settings)

        assert result.analyzed_article_ids is not None
        assert 0 < len(result.analyzed_article_ids) < len(articles)

    def test_the_subset_is_never_asked_of_the_provider(self) -> None:
        """It is filled in after the call, like the token counts — not by the model."""
        for model in (llm_module.NoveltyResponse, KnowledgeStateUpdate, NoveltyResult):
            schema = model.model_json_schema()
            assert "analyzed_article_ids" not in schema["properties"]
            assert "analyzed_article_ids" not in schema.get("required", [])
        assert sorted(llm_module.NoveltyResponse.model_json_schema()["required"]) == [
            "confidence",
            "has_new_info",
            "importance",
            "relevance",
        ]

    def test_the_subset_is_not_persisted_in_the_stored_blob(self) -> None:
        """Stored ``llm_response`` blobs are re-parsed by the force-notify handler."""
        stored = NoveltyResult(has_new_info=False, confidence=0.5, analyzed_article_ids=[1, 2]).model_dump_json()
        assert "analyzed_article_ids" not in stored
        assert NoveltyResult.model_validate_json(stored).analyzed_article_ids is None
