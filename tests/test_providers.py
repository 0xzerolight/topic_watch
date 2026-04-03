"""Tests for news search provider definitions."""

from urllib.parse import quote_plus

from app.models import Topic
from app.scraping.providers import (
    BingNewsProvider,
    GoogleNewsProvider,
    NewsProvider,
    _build_search_query,
)


class TestBuildSearchQuery:
    def test_name_only(self) -> None:
        topic = Topic(name="AI Safety", description="")
        assert _build_search_query(topic) == "AI Safety"

    def test_name_with_short_description(self) -> None:
        topic = Topic(name="AI Safety", description="alignment research papers")
        assert _build_search_query(topic) == "AI Safety alignment research papers"

    def test_name_with_long_description_truncated(self) -> None:
        topic = Topic(
            name="AI Safety",
            description="alignment research papers from leading institutions around the world today",
        )
        # Only first 6 words of description
        assert _build_search_query(topic) == "AI Safety alignment research papers from leading institutions"

    def test_empty_description_ignored(self) -> None:
        topic = Topic(name="Quantum Computing", description="   ")
        # description.split()[:6] returns [] for whitespace-only
        assert _build_search_query(topic) == "Quantum Computing"


class TestBingNewsProvider:
    def test_url_format(self) -> None:
        provider = BingNewsProvider()
        topic = Topic(name="AI Safety", description="alignment research")
        url = provider.build_feed_url(topic)
        expected_query = quote_plus("AI Safety alignment research")
        assert url == f"https://www.bing.com/news/search?q={expected_query}&format=rss"

    def test_special_chars_url_encoded(self) -> None:
        provider = BingNewsProvider()
        topic = Topic(name="C++ & Rust", description="")
        url = provider.build_feed_url(topic)
        assert "C%2B%2B+%26+Rust" in url

    def test_needs_url_resolution_false(self) -> None:
        provider = BingNewsProvider()
        assert provider.needs_url_resolution() is False

    def test_name(self) -> None:
        provider = BingNewsProvider()
        assert provider.name == "bing_news"

    def test_no_api_key_required(self) -> None:
        provider = BingNewsProvider()
        assert provider.requires_api_key is False


class TestGoogleNewsProvider:
    def test_url_format(self) -> None:
        provider = GoogleNewsProvider()
        topic = Topic(name="AI Safety", description="alignment research")
        url = provider.build_feed_url(topic)
        expected_query = quote_plus("AI Safety alignment research")
        assert url == f"https://news.google.com/rss/search?q={expected_query}&hl=en-US&gl=US&ceid=US:en"

    def test_matches_old_build_google_news_url(self) -> None:
        """Regression: new provider must produce identical URLs to the old function."""
        provider = GoogleNewsProvider()
        topic = Topic(name="Quantum Computing", description="recent breakthroughs in quantum")

        # Manually reproduce the old build_google_news_url logic
        query_parts = [topic.name]
        desc_words = topic.description.split()[:6]
        query_parts.append(" ".join(desc_words))
        query = " ".join(query_parts)
        old_url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"

        assert provider.build_feed_url(topic) == old_url

    def test_needs_url_resolution_true(self) -> None:
        provider = GoogleNewsProvider()
        assert provider.needs_url_resolution() is True

    def test_name(self) -> None:
        provider = GoogleNewsProvider()
        assert provider.name == "google_news"

    def test_no_api_key_required(self) -> None:
        provider = GoogleNewsProvider()
        assert provider.requires_api_key is False


class TestProtocolCompliance:
    def test_bing_satisfies_protocol(self) -> None:
        provider: NewsProvider = BingNewsProvider()
        assert hasattr(provider, "name")
        assert hasattr(provider, "requires_api_key")
        assert hasattr(provider, "build_feed_url")
        assert hasattr(provider, "needs_url_resolution")

    def test_google_satisfies_protocol(self) -> None:
        provider: NewsProvider = GoogleNewsProvider()
        assert hasattr(provider, "name")
        assert hasattr(provider, "requires_api_key")
        assert hasattr(provider, "build_feed_url")
        assert hasattr(provider, "needs_url_resolution")
