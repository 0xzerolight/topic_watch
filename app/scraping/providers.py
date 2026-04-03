"""News search provider definitions.

Each provider knows how to build a feed URL for a topic. The
``NewsProvider`` Protocol defines the interface; concrete classes
implement it for specific news sources.
"""

from typing import Protocol
from urllib.parse import quote_plus

from app.models import Topic


def _build_search_query(topic: Topic) -> str:
    """Build a search query string from topic name and description.

    Shared by all providers that use keyword-based search URLs.
    Includes the topic name plus the first 6 words of the description
    (if any) for additional context.
    """
    query_parts = [topic.name]
    if topic.description:
        desc_words = topic.description.split()[:6]
        if desc_words:
            query_parts.append(" ".join(desc_words))
    return " ".join(query_parts)


class NewsProvider(Protocol):
    """Interface for news search providers."""

    name: str
    requires_api_key: bool

    def build_feed_url(self, topic: Topic) -> str: ...

    def needs_url_resolution(self) -> bool: ...


class BingNewsProvider:
    """Bing News RSS provider. No redirect resolution needed."""

    name = "bing_news"
    requires_api_key = False

    def build_feed_url(self, topic: Topic) -> str:
        query = _build_search_query(topic)
        return f"https://www.bing.com/news/search?q={quote_plus(query)}&format=rss"

    def needs_url_resolution(self) -> bool:
        return False


class GoogleNewsProvider:
    """Google News RSS provider. Requires async URL resolution."""

    name = "google_news"
    requires_api_key = False

    _TEMPLATE = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

    def build_feed_url(self, topic: Topic) -> str:
        query = _build_search_query(topic)
        return self._TEMPLATE.format(query=quote_plus(query))

    def needs_url_resolution(self) -> bool:
        return True
