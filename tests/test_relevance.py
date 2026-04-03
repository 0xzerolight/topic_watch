"""Tests for the lightweight relevance scoring module."""

from app.models import Topic
from app.scraping.relevance import _tokenize, score_relevance
from app.scraping.rss import FeedEntry


def _make_topic(name: str = "Test", description: str = "") -> Topic:
    return Topic(name=name, description=description, feed_urls=[])


def _make_entry(title: str = "Article", summary: str = "") -> FeedEntry:
    return FeedEntry(title=title, url="https://example.com", summary=summary, source_feed="https://example.com/feed")


class TestTokenize:
    def test_basic_words(self) -> None:
        assert _tokenize("Hello World") == {"hello", "world"}

    def test_strips_punctuation(self) -> None:
        tokens = _tokenize("C++, news & updates!")
        assert "c" in tokens
        assert "news" in tokens
        assert "updates" in tokens

    def test_empty_string(self) -> None:
        assert _tokenize("") == set()

    def test_numbers_included(self) -> None:
        assert "3" in _tokenize("season 3")


class TestScoreRelevance:
    def test_high_relevance(self) -> None:
        topic = _make_topic("Solo Leveling season 3", "release date anime")
        entry = _make_entry("Solo Leveling Season 3 Gets Official Release Date")
        score = score_relevance(topic, entry)
        assert score > 0.5

    def test_low_relevance(self) -> None:
        topic = _make_topic("Solo Leveling season 3", "release date anime")
        entry = _make_entry("Best Anime Merchandise Deals This Week")
        score = score_relevance(topic, entry)
        assert score < 0.4

    def test_zero_relevance_empty_topic(self) -> None:
        topic = _make_topic("", "")
        entry = _make_entry("Some Article Title")
        assert score_relevance(topic, entry) == 0.0

    def test_zero_relevance_empty_article(self) -> None:
        topic = _make_topic("Solo Leveling")
        entry = _make_entry("", "")
        assert score_relevance(topic, entry) == 0.0

    def test_case_insensitive(self) -> None:
        topic = _make_topic("SOLO LEVELING")
        entry = _make_entry("solo leveling news")
        score = score_relevance(topic, entry)
        assert score > 0.5

    def test_description_contributes(self) -> None:
        topic = _make_topic("Solo Leveling", "release date")
        entry_with_date = _make_entry("Solo Leveling Release Date Announced")
        entry_without = _make_entry("Solo Leveling Fan Art Gallery")
        score_with = score_relevance(topic, entry_with_date)
        score_without = score_relevance(topic, entry_without)
        assert score_with > score_without

    def test_summary_contributes(self) -> None:
        topic = _make_topic("Elden Ring DLC")
        entry = _make_entry("Gaming News Roundup", summary="Elden Ring DLC pricing revealed today")
        score = score_relevance(topic, entry)
        assert score > 0.5
