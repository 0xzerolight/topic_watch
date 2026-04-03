"""Lightweight relevance scoring for feed entries.

Scores articles by keyword overlap between topic metadata and article
metadata. Used as a pre-LLM filter to prioritize relevant articles.
"""

import re

from app.models import Topic
from app.scraping.rss import FeedEntry


def _tokenize(text: str) -> set[str]:
    """Tokenize into lowercase alphanumeric words."""
    return set(re.findall(r"\b[a-z0-9]+\b", text.lower()))


def score_relevance(topic: Topic, entry: FeedEntry) -> float:
    """Score an article's relevance to a topic (0.0 to 1.0) by keyword overlap.

    Returns the fraction of topic keywords found in the article's title and summary.
    """
    topic_tokens = _tokenize(topic.name) | _tokenize(topic.description)
    if not topic_tokens:
        return 0.0
    article_tokens = _tokenize(entry.title) | _tokenize(entry.summary)
    if not article_tokens:
        return 0.0
    overlap = topic_tokens & article_tokens
    return len(overlap) / len(topic_tokens)
