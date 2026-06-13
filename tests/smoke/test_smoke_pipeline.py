"""Hermetic end-to-end smoke test for the real check pipeline.

Exercises ``initialize_new_topic`` and ``check_topic`` against canned feeds and
a stubbed LLM, running the genuine production code in between (scraping, dedup,
content extraction, knowledge persistence, novelty thresholding, notification
queueing). Only the outermost boundaries are stubbed:

    * HTTP  -- via an injected ``httpx.MockTransport`` (RSS + article HTML).
    * LLM   -- via ``stub_llm_boundary`` (no live completion calls).
    * Delivery -- ``send_notification`` / ``send_webhooks`` patched so no real
      network notification is attempted.

Hermeticity notes:
    * Uses the ``db_conn`` fixture (tmp_path SQLite) -- never touches real data/.
    * Uses MANUAL feed mode with explicit feed_urls, so the module-global
      ``app.scraping.routing.router`` singleton is never consulted or mutated.
      A belt-and-suspenders ``_reset_provider_router`` fixture snapshots and
      restores that singleton's health state regardless.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.analysis.llm import NoveltyResult
from app.checker import check_topic, initialize_new_topic
from app.config import LLMSettings, Settings
from app.crud import create_topic, get_knowledge_state, list_articles_for_topic, list_check_results
from app.models import CheckResult, FeedMode, Topic, TopicStatus
from tests.helpers import RssEntry, build_rss_transport, stub_llm_boundary

_FEED_URL = "https://example.com/feed.xml"
_ARTICLE_1 = "https://example.com/article-1"
_ARTICLE_2 = "https://example.com/article-2"

_ARTICLE_HTML = (
    "<html><body><article><h1>{title}</h1>"
    "<p>This is substantial article body text about the topic at hand, long "
    "enough that trafilatura treats it as real extractable content rather than "
    "boilerplate. It discusses the subject in several sentences across the "
    "paragraph so extraction succeeds.</p>"
    "<p>A second paragraph adds further detail and context for the reader so "
    "the content quality heuristics see a genuine article worth analyzing.</p>"
    "</article></body></html>"
)


def _settings() -> Settings:
    """Build offline-safe Settings (mirrors test_analysis _make_settings)."""
    return Settings(
        llm=LLMSettings(model="openai/gpt-4o-mini", api_key="test-key"),
        knowledge_state_max_tokens=2000,
    )


@contextmanager
def _inject_transport(transport: httpx.MockTransport) -> Iterator[None]:
    """Inject a MockTransport into every httpx.AsyncClient created internally.

    The scraping layer constructs its own clients, so we patch __init__ to
    force the transport -- the established pattern from tests/test_scraping.py.
    """
    original_init = httpx.AsyncClient.__init__

    def patched_init(self_client: httpx.AsyncClient, **kwargs: object) -> None:
        kwargs["transport"] = transport
        original_init(self_client, **kwargs)  # type: ignore[arg-type]

    with patch.object(httpx.AsyncClient, "__init__", patched_init):
        yield


@pytest.fixture(autouse=True)
def _reset_provider_router() -> Iterator[None]:
    """Snapshot + restore the module-global ProviderRouter health state.

    The smoke test uses MANUAL mode and never touches the router, but resetting
    it unconditionally guarantees this module can never pollute the shared
    singleton that other tests (test_routing, test_scraping) rely on.
    """
    from app.scraping import routing

    saved = dict(routing.router._health)
    try:
        yield
    finally:
        routing.router._health.clear()
        routing.router._health.update(saved)


def _make_topic(conn: sqlite3.Connection) -> Topic:
    """Create a MANUAL-mode topic with one explicit feed URL."""
    topic = create_topic(
        conn,
        Topic(
            name="Smoke Topic",
            description="release date and pricing news for the smoke-test subject",
            feed_mode=FeedMode.MANUAL,
            feed_urls=[_FEED_URL],
            status=TopicStatus.NEW,
        ),
    )
    conn.commit()
    return topic


def _build_transport(entries: list[RssEntry]) -> httpx.MockTransport:
    return build_rss_transport(
        feeds={"example.com/feed": entries},
        articles={
            _ARTICLE_1: _ARTICLE_HTML.format(title="Article One"),
            _ARTICLE_2: _ARTICLE_HTML.format(title="Article Two"),
        },
    )


async def test_initialize_then_check_pipeline(db_conn: sqlite3.Connection) -> None:
    """Init builds + persists knowledge; a later check records a CheckResult and notifies."""
    settings = _settings()
    topic = _make_topic(db_conn)

    init_entries = [
        RssEntry(
            title="Article One",
            link=_ARTICLE_1,
            summary="Summary of article one.",
            published=datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC),
        ),
    ]

    # --- Phase 1: initialize_new_topic (real pipeline, stubbed edges) ---
    with (
        _inject_transport(_build_transport(init_entries)),
        stub_llm_boundary(),
    ):
        await initialize_new_topic(topic, db_conn, settings)

    # initialize_new_topic mutates `topic` in place, transitioning it to READY.
    assert topic.status == TopicStatus.READY
    assert topic.error_message is None

    # Knowledge state persisted to the tmp_path DB.
    knowledge = get_knowledge_state(db_conn, topic.id)
    assert knowledge is not None
    assert knowledge.summary_text == "Canned knowledge summary."
    assert knowledge.token_count > 0  # recomputed by real count_tokens, not the LLM guess

    # The init article was stored and marked processed.
    init_articles = list_articles_for_topic(db_conn, topic.id)
    assert len(init_articles) == 1
    assert init_articles[0].processed is True

    # --- Phase 2: check_topic against a NEW article (novelty path) ---
    check_entries = [
        RssEntry(
            title="Article Two",
            link=_ARTICLE_2,
            summary="Summary of article two: a brand-new development.",
            published=datetime(2025, 1, 2, 12, 0, 0, tzinfo=UTC),
        ),
    ]
    novelty = NoveltyResult(
        has_new_info=True,
        summary="A new development was announced.",
        key_facts=["New fact: release confirmed"],
        source_urls=[_ARTICLE_2],
        confidence=0.95,
        relevance=0.9,
    )

    with (
        _inject_transport(_build_transport(check_entries)),
        stub_llm_boundary(novelty=novelty),
        patch("app.checker.send_notification", new=AsyncMock(return_value=True)) as mock_notify,
        patch("app.checker.send_webhooks", new=AsyncMock(return_value=0)),
    ):
        result = await check_topic(topic, db_conn, settings)

    # A CheckResult was produced and recorded with the novelty outcome.
    assert isinstance(result, CheckResult)
    assert result.id is not None
    assert result.has_new_info is True
    assert result.articles_new == 1
    assert result.notification_sent is True

    # The notification path was genuinely exercised (real format_notification ran).
    mock_notify.assert_awaited_once()

    # CheckResult persisted to the DB.
    recorded = list_check_results(db_conn, topic.id)
    assert len(recorded) == 1
    assert recorded[0].has_new_info is True

    # Knowledge state was updated by the real update_knowledge path.
    updated_knowledge = get_knowledge_state(db_conn, topic.id)
    assert updated_knowledge is not None
    assert updated_knowledge.summary_text == "Canned knowledge summary."

    # The new article was stored (init's article + this one).
    all_articles = list_articles_for_topic(db_conn, topic.id)
    assert len(all_articles) == 2
