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

from app.analysis.llm import CompressedKnowledge, KnowledgeStateUpdate, NoveltyResult
from app.checker import check_topic, initialize_new_topic
from app.config import LLMSettings, NotificationSettings, Settings
from app.crud import (
    create_topic,
    get_knowledge_state,
    list_articles_for_topic,
    list_check_results,
    list_pending_notifications,
    list_pending_webhooks,
)
from app.models import CheckResult, FeedMode, NotificationDelivery, Topic, TopicStatus
from tests.helpers import RssEntry, build_rss_transport, build_rss_xml, stub_llm_boundary

_FEED_URL = "https://example.com/feed.xml"
_ARTICLE_1 = "https://example.com/article-1"
_ARTICLE_2 = "https://example.com/article-2"
_WEBHOOK_URL = "https://hooks.example.com/topic-watch"

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
        patch(
            "app.checker.send_notification_per_url",
            new=AsyncMock(return_value=[NotificationDelivery(url="json://localhost", ok=True)]),
        ) as mock_notify,
        patch("app.checker.send_webhooks", new=AsyncMock(return_value=0)),
    ):
        result = await check_topic(topic, db_conn, settings)

    # A CheckResult was produced and recorded with the novelty outcome.
    assert isinstance(result, CheckResult)
    assert result.id is not None
    assert result.has_new_info is True
    assert result.articles_new == 1
    assert result.notification_sent is True

    # OVH-163: token usage propagates from the raw completions into CheckResult.
    # The stub supplies 12 prompt / 8 completion tokens per call; the novelty
    # path runs analyze (12/8) then the knowledge update (12/8), so the result
    # accumulates a deterministic 24 prompt / 16 completion across the two calls.
    assert result.prompt_tokens == 24
    assert result.completion_tokens == 16
    persisted = list_check_results(db_conn, topic.id)[0]
    assert persisted.prompt_tokens == 24
    assert persisted.completion_tokens == 16

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


def _build_webhook_failing_transport(entries: list[RssEntry]) -> httpx.MockTransport:
    """Serve the RSS feed + article HTML, but fail the webhook POST with 500.

    The webhook POST is identified by its URL; everything else is the canned
    feed/article content so the real scraping pipeline runs unchanged.
    """
    feed_xml = build_rss_xml(entries)
    article_html = _ARTICLE_HTML.format(title="Article Two")

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if _WEBHOOK_URL in url:
            # Real send_webhooks path: a 5xx makes send_webhook return False, so
            # the delivery is enqueued to pending_webhooks via the held conn.
            return httpx.Response(500, text="webhook receiver down")
        if _ARTICLE_2 in url:
            return httpx.Response(200, text=article_html, headers={"content-type": "text/html"})
        if "example.com/feed" in url:
            return httpx.Response(200, text=feed_xml, headers={"content-type": "application/rss+xml"})
        return httpx.Response(404, text="Not found")

    return httpx.MockTransport(handler)


async def test_check_topic_queues_webhook_through_held_conn_on_500(db_conn: sqlite3.Connection) -> None:
    """OVH-073: a failed webhook delivery is queued through check_topic's held conn.

    Unlike the other smoke test, ``send_webhooks`` is NOT patched — it runs for
    real against a MockTransport that returns 500 for the webhook POST. This
    exercises the constitution-sensitive path where check_topic threads ONE
    connection through HTTP + LLM awaits and then a webhook send that itself does
    DB writes (enqueue to pending_webhooks). The assertions pin that:

      * a ``pending_webhooks`` row is written via the held connection, correlated
        to the committed CheckResult (``check_result_id`` non-NULL, OVH-101), and
      * check_topic still committed its CheckResult.

    A regression leaving the held connection in a bad transaction state (or a
    nested writer deadlocking under WAL) would surface here.
    """
    settings = _settings()
    settings.notifications = NotificationSettings(webhook_urls=[_WEBHOOK_URL])
    topic = _make_topic(db_conn)

    # --- Phase 1: initialize the topic (stub LLM + delivery; no webhooks yet) ---
    init_entries = [
        RssEntry(
            title="Article One",
            link=_ARTICLE_1,
            summary="Summary of article one.",
            published=datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC),
        ),
    ]
    with (
        _inject_transport(_build_transport(init_entries)),
        stub_llm_boundary(),
    ):
        await initialize_new_topic(topic, db_conn, settings)
    assert topic.status == TopicStatus.READY

    # --- Phase 2: check against a new article; the webhook POST fails (500) ---
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
        _inject_transport(_build_webhook_failing_transport(check_entries)),
        stub_llm_boundary(novelty=novelty),
        # Stub only the Apprise delivery — the webhook queue path stays real.
        patch(
            "app.checker.send_notification_per_url",
            new=AsyncMock(return_value=[NotificationDelivery(url="json://localhost", ok=True)]),
        ),
        # Let the real webhook POST reach the MockTransport (skip the SSRF DNS check).
        patch("app.webhooks.is_private_url", return_value=False),
    ):
        result = await check_topic(topic, db_conn, settings)

    # The CheckResult was created and committed (queued webhook correlates to it).
    assert isinstance(result, CheckResult)
    assert result.id is not None
    recorded = list_check_results(db_conn, topic.id)
    assert len(recorded) == 1
    assert recorded[0].id == result.id

    # The failed webhook was queued via the held connection — readable on db_conn.
    pending = list_pending_webhooks(db_conn)
    assert len(pending) == 1
    queued = pending[0]
    assert queued["topic_id"] == topic.id
    assert queued["url"] == _WEBHOOK_URL
    # check_result_id is populated (created before the send), not NULL (OVH-101).
    assert queued["check_result_id"] == result.id


async def _init_ready_topic(db_conn: sqlite3.Connection, settings: Settings) -> Topic:
    """Initialize a topic to READY through the real pipeline (shared smoke setup)."""
    topic = _make_topic(db_conn)
    init_entries = [
        RssEntry(
            title="Article One",
            link=_ARTICLE_1,
            summary="Summary of article one.",
            published=datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC),
        ),
    ]
    with (
        _inject_transport(_build_transport(init_entries)),
        stub_llm_boundary(),
    ):
        await initialize_new_topic(topic, db_conn, settings)
    assert topic.status == TopicStatus.READY
    return topic


async def test_below_threshold_suppresses_notification_end_to_end(db_conn: sqlite3.Connection) -> None:
    """OVH-161: has_new_info=True but below the confidence threshold => no notify,
    no knowledge update, but the article is still marked processed.

    Drives a real NoveltyResult through the real thresholding gate (not a mocked
    branch): the integration seam where the per-topic/global threshold flows into
    ``check_topic``. A wiring bug (threshold not applied, or processed-flag skipped
    on the suppressed branch) would surface here but stays green in the fully
    mocked unit tests.
    """
    settings = _settings()
    settings.min_confidence_threshold = 0.9
    topic = await _init_ready_topic(db_conn, settings)

    knowledge_before = get_knowledge_state(db_conn, topic.id)
    assert knowledge_before is not None

    check_entries = [
        RssEntry(
            title="Article Two",
            link=_ARTICLE_2,
            summary="Summary of article two: a marginal development.",
            published=datetime(2025, 1, 2, 12, 0, 0, tzinfo=UTC),
        ),
    ]
    # New info, but confidence (0.5) is below the 0.9 threshold => suppressed.
    novelty = NoveltyResult(
        has_new_info=True,
        summary="A low-confidence development.",
        key_facts=["Maybe a release"],
        source_urls=[_ARTICLE_2],
        confidence=0.5,
        relevance=0.9,
    )

    with (
        _inject_transport(_build_transport(check_entries)),
        stub_llm_boundary(novelty=novelty),
        patch("app.checker.send_notification_per_url", new=AsyncMock()) as mock_notify,
        patch("app.checker.send_webhooks", new=AsyncMock()) as mock_webhooks,
    ):
        result = await check_topic(topic, db_conn, settings)

    # Detected new info, but no delivery was attempted at all.
    assert result.has_new_info is True
    assert result.notification_sent is False
    mock_notify.assert_not_awaited()
    mock_webhooks.assert_not_awaited()

    # No pending rows were queued (suppression is not a delivery failure).
    assert list_pending_notifications(db_conn) == []
    assert list_pending_webhooks(db_conn) == []

    # Knowledge state was NOT updated (the suppressed branch skips update_knowledge).
    knowledge_after = get_knowledge_state(db_conn, topic.id)
    assert knowledge_after is not None
    assert knowledge_after.updated_at == knowledge_before.updated_at

    # The article is STILL marked processed so it is never re-analyzed.
    articles = list_articles_for_topic(db_conn, topic.id)
    article_two = next(a for a in articles if a.url == _ARTICLE_2)
    assert article_two.processed is True


async def test_delivery_failure_queues_pending_notification_end_to_end(db_conn: sqlite3.Connection) -> None:
    """OVH-161: an above-threshold notify whose delivery fails is queued for retry.

    ``send_notification_per_url`` returns a failed ``NotificationDelivery`` (the
    real Apprise call is the only thing stubbed); the rest of ``check_topic``
    runs for real. Pins that the failed URL lands in ``pending_notifications``
    with its scoped url, correlated to the same committed CheckResult.
    """
    settings = _settings()
    topic = await _init_ready_topic(db_conn, settings)

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
    failed_url = "tgram://token/chatid"

    with (
        _inject_transport(_build_transport(check_entries)),
        stub_llm_boundary(novelty=novelty),
        patch(
            "app.checker.send_notification_per_url",
            new=AsyncMock(return_value=[NotificationDelivery(url=failed_url, ok=False, error="boom")]),
        ),
        patch("app.checker.send_webhooks", new=AsyncMock(return_value=0)),
    ):
        result = await check_topic(topic, db_conn, settings)

    # The check committed a CheckResult flagging the delivery failure.
    assert result.id is not None
    assert result.notification_sent is False
    assert result.notification_error is not None

    # The failed URL was queued for retry, scoped to that URL only.
    pending = list_pending_notifications(db_conn)
    assert len(pending) == 1
    queued = pending[0]
    assert queued.url == failed_url
    assert queued.topic_id == topic.id


async def test_compression_path_runs_offline_end_to_end(db_conn: sqlite3.Connection) -> None:
    """OVH-162: a tiny knowledge budget drives the real over-budget compression
    branch through the full pipeline, served entirely by the stub (no live call).

    Before the stub learned ``CompressedKnowledge`` this scenario AssertionError-ed
    inside the stub. Here init produces an over-budget summary, the real
    ``compress_knowledge`` calls ``compress_knowledge_summary`` (served by the
    stub), and the persisted state fits the budget.
    """
    settings = _settings()
    # Tiny budget so the canned init summary overflows and triggers compression.
    settings.knowledge_state_max_tokens = 3
    topic = _make_topic(db_conn)

    init_entries = [
        RssEntry(
            title="Article One",
            link=_ARTICLE_1,
            summary="Summary of article one.",
            published=datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC),
        ),
    ]
    # Verbose init summary (overflows budget=3); compression returns 2 short words.
    init_update = KnowledgeStateUpdate(
        sufficient_data=True,
        confidence=0.9,
        updated_summary="One two three four five six seven eight nine ten eleven twelve.",
        token_count=0,
    )
    compressed = CompressedKnowledge(compressed_summary="Dense facts", token_count=0)

    with (
        _inject_transport(_build_transport(init_entries)),
        stub_llm_boundary(knowledge_init=init_update, compressed=compressed),
    ):
        await initialize_new_topic(topic, db_conn, settings)

    assert topic.status == TopicStatus.READY
    knowledge = get_knowledge_state(db_conn, topic.id)
    assert knowledge is not None
    # Compression branch ran: the stored summary is the compressed text, and the
    # recomputed token_count fits the (tiny) budget.
    assert knowledge.summary_text == "Dense facts"
    assert knowledge.token_count <= settings.knowledge_state_max_tokens
