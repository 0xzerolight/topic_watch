"""Core check loop: scrape, analyze, notify, record.

Orchestrates the full pipeline for checking topics for new information.
Each check cycle fetches articles, analyzes them against the knowledge
state, sends notifications for genuine updates, and records the outcome.
"""

import asyncio
import logging
import secrets
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.analysis.knowledge import (
    KnowledgeUpdatePlan,
    apply_knowledge_update,
    prepare_initial_knowledge,
    prepare_knowledge_update,
    reported_article_ids,
)
from app.analysis.llm import analyze_articles
from app.check_context import check_id_var, generate_check_id
from app.config import Settings
from app.crud import (
    HEARTBEAT_INTENT_KINDS,
    MAX_ANALYSIS_ATTEMPTS,
    abandon_expired_notifications,
    apply_notification_outcome,
    claim_heartbeat_alert,
    claim_notification_intent,
    claim_topic_for_init,
    clear_heartbeat_alert,
    create_check_result,
    create_notification_intents,
    create_webhook_intents,
    get_heartbeat_latch_raw,
    get_knowledge_state,
    get_topic,
    get_topics_due_for_check,
    list_articles_for_topic,
    list_due_notification_intents,
    list_sent_heartbeat_alert_targets,
    mark_articles_processed,
    record_article_analysis_failure,
    release_stale_notification_claims,
    revoke_heartbeat_intents,
    topic_generation_matches,
    update_check_result_delivery,
    update_topic_init_status,
)
from app.database import get_db, short_conn
from app.heartbeat import HeartbeatDecision, evaluate_heartbeat
from app.models import (
    Article,
    CheckResult,
    KnowledgeRevisionSource,
    NotificationDelivery,
    NotificationKind,
    NotifyDisposition,
    PendingNotification,
    PendingWebhook,
    Topic,
    TopicStatus,
    next_attempt_at,
    to_db_utc,
)
from app.notifications import format_notification, redact_url, send_single_notification
from app.scraping import FetchResult, all_sources_failed, fetch_new_articles_for_topic
from app.web.state import _checking_state
from app.webhooks import build_webhook_intents, deliver_webhook_intents, retry_pending_webhooks

logger = logging.getLogger(__name__)

# Single-flight guard: serializes notification drains within this process so
# two overlapping drains (scheduler tick vs. a UI/CLI check-all) cannot both
# walk the queue at once. The cross-process case is covered by the atomic
# per-row claim below. (OVH-017)
_notification_retry_lock = asyncio.Lock()

# Claims older than this are treated as stale — a drainer crashed mid-send, or a
# send timed out with an unknown outcome — and the intent is re-armed. Measured
# against the stored wall-clock claim stamp, so a clock jump can re-arm a live
# claim early; that is harmless because the apply is fenced by the immutable
# claim token rather than by elapsed time (AUG-277).
_CLAIM_STALE_AFTER = timedelta(minutes=10)

# Per-batch send fan-out. Matches the webhook drain's bound: enough to stop a
# backlog serializing at one timeout each, gentle enough not to hammer a channel.
_DELIVERY_CONCURRENCY = 5

# AUG-027: intents per drain. Bounds how long a queued backlog can hold a
# scheduler cycle before due-topic work starts; the rest wait for the next tick.
_RETRY_DRAIN_LIMIT = 20

# Delivery failures that will not change on a retry: the target is an unedited
# example URL, one Apprise cannot even parse, or one the SSRF gate refuses.
# Three more attempts only delay the abandonment (AUG-245).
_TERMINAL_DELIVERY_ERRORS = frozenset(
    {
        "placeholder notification URL",
        "invalid notification URL",
        "blocked notification target",
    }
)


def _summarize_exc(exc: BaseException, *, limit: int = 200) -> str:
    """One-line, length-bounded exception summary for the stored stage_error."""
    summary = f"{type(exc).__name__}: {exc}".replace("\n", " ").strip()
    return summary[:limit]


class TopicInitRefused(Exception):
    """Initialization never started because this caller does not own the topic.

    Raised instead of silently returning so every entry point (CLI exit code, web
    response, scheduler log) can say what happened: the topic was already being
    initialized by someone else, is paused, or disappeared. A refusal writes
    nothing — in particular it never takes a RESEARCHING claim it cannot release.
    """


class CheckTransitionAborted(Exception):
    """The durable transition was refused because the world moved underneath it.

    Raised inside the C3 transaction when the topic was deleted and its rowid
    recycled, or when the knowledge state advanced while this check's LLM phase
    was running. Both mean the work in hand was computed against state that no
    longer exists, so nothing is written and no CheckResult is recorded — the
    next cycle re-runs cleanly against the live state.
    """


@dataclass
class TopicSnapshot:
    """Everything the pipeline reads from the database before it goes offline.

    Taken under a short connection at the start of a check and carried through
    every network/LLM phase, so no connection has to stay open to answer "what
    did the row say?". ``generation`` and ``knowledge_version`` are the two values
    the durable transition re-checks before writing: the first proves the topic is
    still the same topic, the second proves the knowledge summary this check built
    on is still current.
    """

    topic: Topic
    generation: str
    knowledge_version: int
    knowledge_summary: str


@dataclass
class CheckOutcome:
    """The complete set of durable changes one check wants to make.

    Accumulated with no connection open, then applied by
    ``_commit_check_transition`` in a single transaction. Keeping it a plain value
    is what makes the transition atomic: nothing here is written until every part
    of it can be.
    """

    result: CheckResult
    knowledge_plan: KnowledgeUpdatePlan | None = None
    knowledge_source: KnowledgeRevisionSource = KnowledgeRevisionSource.UPDATE
    knowledge_change_note: str | None = None
    article_ids: list[int] = field(default_factory=list)
    """Articles this check evaluated, to be marked processed."""
    failed_article_ids: list[int] = field(default_factory=list)
    """Articles this check did not finish with — the analysis failed, or the
    knowledge merge they justified never landed. They stay unprocessed and take an
    attempt, so the next cycle re-analyzes them and a hopeless one is eventually
    abandoned (see ``crud.record_article_analysis_failure``)."""
    notify_disposition: str | None = None
    intents: list[PendingNotification] = field(default_factory=list)
    """Per-target notification delivery intents, inserted in the same commit as the
    CheckResult that justifies them and stamped with its id (TW-AUD-004)."""
    webhook_intents: list[PendingWebhook] = field(default_factory=list)
    """The webhook half of the same thing."""


def _snapshot_topic(conn: sqlite3.Connection, topic_id: int) -> TopicSnapshot | None:
    """Read the live topic row and its knowledge state. ``None`` if the topic is gone.

    Deliberately re-reads rather than trusting the ``Topic`` the caller passed in:
    that object may have been loaded minutes earlier (a queued background task, a
    scheduler snapshot taken before a slow sibling check), so its status,
    active flag and thresholds can all be stale by the time the check runs.
    """
    topic = get_topic(conn, topic_id)
    if topic is None:
        return None
    knowledge = get_knowledge_state(conn, topic_id)
    return TopicSnapshot(
        topic=topic,
        generation=topic.generation,
        knowledge_version=knowledge.version if knowledge else 0,
        knowledge_summary=knowledge.summary_text if knowledge else "",
    )


def _no_source_detail(fetch_result: FetchResult) -> str:
    """Say why no source ran: everything backed off, or nothing configured."""
    skipped = fetch_result.feeds_skipped
    return f"{skipped} feed(s) in backoff" if skipped else "no source configured or enabled"


def _init_empty_error(fetch_result: FetchResult) -> str:
    """Name the reason a first initialization fetch came back empty."""
    if all_sources_failed(fetch_result.feeds_total, fetch_result.feeds_failed):
        return "All feed source(s) failed during initialization (check credentials/connectivity; see logs)"
    if fetch_result.feeds_total == 0:
        return f"No source attempted during initialization ({_no_source_detail(fetch_result)})"
    return "No articles found during initialization"


def _analysis_batch(
    db_path: Path | None,
    topic_id: int,
    new_articles: list[Article],
    max_articles: int,
) -> list[Article]:
    """This cycle's fetch plus whatever earlier cycles left unfinished.

    An article is stored before it is analyzed, and the scraper deduplicates
    against stored hashes — so once a cycle stores an article and then fails to
    finish with it (LLM outage, failed knowledge merge), no feed will ever offer
    it again. Re-selecting the unprocessed rows here is what makes that work
    resumable instead of silently lost (TW-AUD-001).

    Stranded rows come first: they are bounded by the per-article attempt cap and
    drain within a few cycles, while a busy feed that fills ``max_articles`` every
    cycle would otherwise starve them indefinitely.
    """
    fresh_ids = {a.id for a in new_articles if a.id is not None}
    with get_db(db_path) as conn:
        stored = list_articles_for_topic(conn, topic_id, unprocessed_only=True, limit=max_articles)
    stranded = [a for a in stored if a.id not in fresh_ids]
    return (stranded + new_articles)[:max_articles]


def _split_batch(batch: list[Article], analyzed_ids: list[int] | None) -> tuple[list[int], list[int]]:
    """Split a batch into the articles the LLM read and the ones it never saw.

    ``analyzed_ids is None`` means the analysis layer reported nothing, so the
    whole batch counts as read — the behaviour before the signal existed.
    """
    ids = [article.id for article in batch if article.id is not None]
    if analyzed_ids is None:
        return ids, []
    seen = set(analyzed_ids)
    return [i for i in ids if i in seen], [i for i in ids if i not in seen]


def _commit_check_transition(
    conn: sqlite3.Connection,
    snapshot: TopicSnapshot,
    outcome: CheckOutcome,
    *,
    settings: Settings,
) -> CheckResult:
    """Apply a whole check's durable state in ONE transaction. The C3 boundary.

    Knowledge write, revision append, article disposition and the CheckResult were
    previously three independent commits spread across the pipeline, so a crash or
    a rollback between any two left a partial transition: knowledge advanced with
    no check recorded, or a check recorded against knowledge that never landed.
    Here they either all happen or none do.

    ``BEGIN IMMEDIATE`` takes the write lock up front so the generation and
    version guards below are evaluated against state no concurrent writer can
    change before this transaction commits. Raises ``CheckTransitionAborted``
    when either guard fails; the caller's connection context rolls back.
    """
    topic_id = snapshot.topic.id
    if topic_id is None:
        raise ValueError("Topic must have an ID")

    conn.execute("BEGIN IMMEDIATE")

    if not topic_generation_matches(conn, topic_id, snapshot.generation):
        raise CheckTransitionAborted(f"topic_id={topic_id} was deleted or replaced during this check")

    if outcome.knowledge_plan is not None:
        applied = apply_knowledge_update(
            conn,
            topic_id,
            outcome.knowledge_plan,
            expected_version=snapshot.knowledge_version,
            source=outcome.knowledge_source,
            settings=settings,
            change_note=outcome.knowledge_change_note,
        )
        if not applied:
            raise CheckTransitionAborted(
                f"knowledge for topic_id={topic_id} moved past version {snapshot.knowledge_version}"
            )

    mark_articles_processed(conn, outcome.article_ids)
    abandoned = record_article_analysis_failure(conn, outcome.failed_article_ids)
    if abandoned:
        logger.warning(
            "Topic id=%d: abandoning %d article(s) after %d failed analysis attempt(s)",
            topic_id,
            abandoned,
            MAX_ANALYSIS_ATTEMPTS,
        )
    outcome.result.notify_disposition = outcome.notify_disposition
    created = create_check_result(conn, outcome.result)

    # The delivery intents ride the same commit. Stamped with the CheckResult id
    # only now that it exists, so an abandoned delivery is always attributable to
    # the check that wanted it (OVH-040).
    for intent in outcome.intents:
        intent.check_result_id = created.id
    for webhook_intent in outcome.webhook_intents:
        webhook_intent.check_result_id = created.id
    create_notification_intents(conn, outcome.intents)
    create_webhook_intents(conn, outcome.webhook_intents)

    conn.commit()
    return created


async def check_topic(
    topic: Topic,
    settings: Settings,
    *,
    db_path: Path | None = None,
    guard: bool = True,
) -> CheckResult:
    """Run the full check pipeline for a single topic.

    Phases (AUG-136): the pipeline never holds a database connection across a
    network or LLM await. It snapshots what it needs under a short connection,
    runs fetch/analysis/knowledge-generation with nothing open, applies every
    durable change in one transaction, and only then performs the irreversible
    sends.

        P0  snapshot (short connection)
        P1  fetch new articles (connection-free; owns its own short phases)
        P2  analyze against the snapshotted knowledge (connection-free)
        P3  generate the knowledge update (connection-free)
        C3  ONE transaction: knowledge + revision + article disposition + result
        P4  notification and webhook sends (connection-free)
        C4  record the delivery outcome; run the Silence Heartbeat

    Concurrency: ``check_topic`` is the single per-topic funnel, so it acquires
    the process-wide ``_checking_state`` guard itself (OVH-096). The scheduler
    ``check_all_topics`` loop, the UI check-all, the JSON API, and the CLI all
    reach the pipeline through here, so a same-topic check already in flight is
    skipped (returns a CheckResult with ``stage_error='skipped: already in
    flight'`` and no LLM/notification work). Callers that already hold the guard
    (the manual web ``/check`` path, which acquires it synchronously so it can
    return the current row immediately) pass ``guard=False`` to avoid
    self-blocking on the entry they already own.

    Args:
        topic: The topic to check. Must have an id; its status is re-read from
            the database at P0, so a stale in-memory copy is safe to pass.
        settings: Application settings.
        db_path: Database path used to open each phase's short-lived connection.
        guard: When True (default), acquire/release the per-topic in-flight
            guard. Pass False when the caller already holds it.

    Returns:
        CheckResult recording the outcome of this check.
    """
    if topic.id is None:
        raise ValueError("Topic must have an ID")
    topic_id: int = topic.id

    if not guard:
        return await _check_topic_guarded(topic, settings, db_path)

    owner = await _checking_state.start_check(topic_id)
    if owner is None:
        logger.info("Topic '%s' (id=%d) already being checked; skipping", topic.name, topic_id)
        return CheckResult(topic_id=topic_id, stage_error="skipped: already in flight")
    try:
        return await _check_topic_guarded(topic, settings, db_path)
    finally:
        await _checking_state.finish_check(topic_id, owner)


async def _check_topic_guarded(
    topic: Topic,
    settings: Settings,
    db_path: Path | None,
) -> CheckResult:
    """Run the pipeline with a fresh check_id (caller owns the in-flight guard)."""
    if topic.id is None:
        raise ValueError("Topic must have an ID")

    cid = generate_check_id()
    # Token idiom (OVH-103): restore whatever id the caller had set, rather than
    # clobbering it to None. A future outer flow that sets its own check_id and
    # then calls check_topic keeps that id intact after this nested run returns.
    token = check_id_var.set(cid)

    try:
        return await _check_topic_inner(topic, settings, db_path, cid)
    finally:
        check_id_var.reset(token)


async def _check_topic_inner(
    topic: Topic,
    settings: Settings,
    db_path: Path | None,
    cid: str,
) -> CheckResult:
    """Inner implementation of check_topic with check_id already set."""
    if topic.id is None:
        raise ValueError("Topic must have an ID to be checked")
    topic_id: int = topic.id
    result = CheckResult(topic_id=topic_id)

    logger.info("Starting check for topic '%s' [check_id=%s]", topic.name, cid)

    # --- P0: snapshot under a short connection, then run offline.
    with get_db(db_path) as conn:
        snapshot = _snapshot_topic(conn, topic_id)
    if snapshot is None:
        logger.warning("Topic id=%d no longer exists; skipping check", topic_id)
        return CheckResult(topic_id=topic_id, stage_error="skipped: topic no longer exists")
    topic = snapshot.topic

    # Only check READY topics
    if topic.status != TopicStatus.READY:
        logger.warning(
            "Skipping topic '%s' — status is '%s', not 'ready'",
            topic.name,
            topic.status,
        )
        # Nothing is recorded: no fetch and no analysis ran, so a stored row would
        # claim monitoring work that never happened — a clean zero-valued check
        # breaks a source-failure streak and its fresh ``checked_at`` postpones the
        # first real check once the topic becomes READY (AUG-134). No heartbeat
        # either: a paused or errored topic must neither alert nor claim recovery.
        return CheckResult(
            topic_id=topic_id,
            stage_error=f"skipped: topic not ready (status: {topic.status.value})",
        )

    # --- P1: fetch new articles. Opens and closes its own short connections.
    try:
        fetch_result = await fetch_new_articles_for_topic(
            topic,
            db_path=db_path,
            max_articles=settings.max_articles_per_check,
            feed_fetch_timeout=settings.feed_fetch_timeout,
            article_fetch_timeout=settings.article_fetch_timeout,
            feed_max_retries=settings.feed_max_retries,
            concurrency=settings.content_fetch_concurrency,
            feed_backoff_base_minutes=settings.feed_backoff_base_minutes,
            feed_backoff_cap_hours=settings.feed_backoff_cap_hours,
            exa_settings=settings.exa,
        )
    except Exception as exc:
        # An exception escaping the fetch is an internal failure, not a source
        # outage: per-feed and per-provider errors are caught inside and reported
        # through ``feeds_failed``/feed health, so what reaches here is storage,
        # deduplication or extraction breaking on our side. Labelling those
        # ``scrape_failed`` made the heartbeat announce failing sources and sent
        # the operator after feeds, network and API keys (AUG-133).
        logger.warning("Fetch pipeline failed for topic '%s'", topic.name, exc_info=True)
        result.stage_error = f"pipeline_failed: {_summarize_exc(exc)}"
        return await _finish_check(db_path, topic, result, settings)

    new_articles = fetch_result.articles
    result.articles_found = fetch_result.total_feed_entries
    result.articles_new = len(new_articles)

    if not new_articles:
        # Distinguish a genuine "nothing new" from a total source failure (e.g. a bad
        # Exa key going stale, or every RSS feed down) so the detail page/`doctor` show
        # the real cause instead of a silent empty check. Mode-agnostic by design.
        if all_sources_failed(fetch_result.feeds_total, fetch_result.feeds_failed):
            result.stage_error = "sources_failed: all feed source(s) failed (see logs)"
        elif fetch_result.feeds_total == 0:
            # Nothing was even attempted: Exa disabled/keyless, a MANUAL topic with no
            # feed URLs, or every feed inside a backoff window. Not a fetch failure —
            # hence not ``sources_failed`` — but equally a check that cannot see news,
            # so it must not read as healthy silence (Silence Heartbeat).
            result.stage_error = f"sources_unavailable: no source attempted ({_no_source_detail(fetch_result)})"

    # --- P1c: the analysis batch is this cycle's fetch PLUS any article an earlier
    # cycle stored but never finished with. The scraper dedups against stored
    # hashes, so those rows can never arrive from a feed again: without this
    # re-select a single failed analysis or knowledge update stranded them forever
    # (TW-AUD-001). Stranded work goes first — it is bounded by the attempt cap and
    # drains, whereas a busy feed filling the cap every cycle would starve it.
    articles = _analysis_batch(db_path, topic_id, new_articles, settings.max_articles_per_check)

    if not articles:
        logger.info("Topic '%s': no new articles found", topic.name)
        return await _finish_check(db_path, topic, result, settings)

    # --- P2: analyze against the knowledge snapshotted at P0, with no connection
    # open (returns a safe default on LLM error — analyze_articles never raises).
    novelty = await analyze_articles(articles, snapshot.knowledge_summary, topic, settings)
    result.has_new_info = novelty.has_new_info
    result.llm_response = novelty.model_dump_json()
    result.prompt_tokens += novelty.prompt_tokens
    result.completion_tokens += novelty.completion_tokens

    # analyze_articles stays fail-safe (never raises), so an LLM failure surfaces
    # as the safe default plus a populated ``error``. Record it distinctly so a
    # broken analysis is not byte-identical to a clean "nothing new" run.
    if novelty.error:
        result.stage_error = f"analysis_failed: {novelty.error}"

    # Why this check will or will not notify, recorded truthfully on the row so
    # "we chose not to send" is never indistinguishable from "we sent" (m026).
    disposition: str = NotifyDisposition.ANALYSIS_FAILED if novelty.error else NotifyDisposition.NO_NEW_INFO

    # Effective thresholds: per-topic override (NULL = inherit global).
    confidence_threshold = (
        topic.confidence_threshold if topic.confidence_threshold is not None else settings.min_confidence_threshold
    )
    relevance_threshold = (
        topic.relevance_threshold if topic.relevance_threshold is not None else settings.min_relevance_threshold
    )
    # Importance is per-topic only (no global setting by design: a global default
    # of 1 would be a functional no-op). NULL = no suppression -> floor of 1,
    # which every importance score passes.
    importance_threshold = topic.importance_threshold if topic.importance_threshold is not None else 1

    # --- P3: if new info clears the thresholds, generate the knowledge update.
    # Still connection-free: this is another multi-second LLM round-trip, and the
    # plan it produces is not written until the C3 transaction below.
    # True when the knowledge state did NOT advance this cycle — the merge raised,
    # or the model refused it as insufficient. Both leave the stored baseline
    # behind the evidence, so both leave the articles unfinished.
    knowledge_stale = False
    should_notify = False
    notification: tuple[str, str] | None = None
    knowledge_plan: KnowledgeUpdatePlan | None = None
    if novelty.has_new_info:
        if novelty.confidence < confidence_threshold:
            disposition = NotifyDisposition.BELOW_CONFIDENCE
            logger.info(
                "Topic '%s': new info detected but confidence %.2f below threshold %.2f, skipping notification",
                topic.name,
                novelty.confidence,
                confidence_threshold,
            )
        elif novelty.relevance < relevance_threshold:
            disposition = NotifyDisposition.BELOW_RELEVANCE
            logger.info(
                "Topic '%s': new info detected but relevance %.2f below threshold %.2f, skipping notification",
                topic.name,
                novelty.relevance,
                relevance_threshold,
            )
        else:
            # Unlike the confidence/relevance gates above (which treat the result
            # as unreliable and skip everything), the importance gate suppresses
            # only the SENDS: the info is genuinely new and on-topic, so the
            # knowledge state still absorbs it — otherwise the same trivial fact
            # would re-flag as "new" every cycle.
            should_notify = novelty.importance >= importance_threshold
            disposition = NotifyDisposition.PENDING if should_notify else NotifyDisposition.SUPPRESSED_IMPORTANCE
            if not should_notify:
                logger.info(
                    "Topic '%s': new info importance %d below threshold %d, updating knowledge without notification",
                    topic.name,
                    novelty.importance,
                    importance_threshold,
                )
            try:
                plan = await prepare_knowledge_update(topic, novelty, snapshot.knowledge_summary, settings)
                result.prompt_tokens += plan.usage.prompt_tokens
                result.completion_tokens += plan.usage.completion_tokens
                if plan.sufficient_data:
                    knowledge_plan = plan
                else:
                    # The merge was refused: the findings were too vague to fold
                    # in, so the stored summary stays as it was. Recording that is
                    # the whole point — a check that alerts on evidence its own
                    # baseline never absorbed will keep re-detecting the same fact
                    # as new, and used to look identical to a clean update
                    # (TW-AUD-003).
                    knowledge_stale = True
                    result.stage_error = (
                        "knowledge_insufficient: findings too vague to merge; knowledge state unchanged"
                    )
                    logger.warning(
                        "Topic '%s': knowledge merge refused as insufficient; baseline unchanged",
                        topic.name,
                    )
            except Exception as exc:
                logger.warning(
                    "Knowledge update failed for topic '%s'",
                    topic.name,
                    exc_info=True,
                )
                # OVH-009: any alert that passed the importance gate still fires,
                # but record the failure distinctly and do NOT mark these
                # new-info-bearing articles processed, so the next cycle
                # re-attempts the knowledge update (no silent drift).
                knowledge_stale = True
                result.stage_error = f"knowledge_update_failed: {_summarize_exc(exc)}"
            if should_notify:
                notification = format_notification(topic.name, novelty)
                if knowledge_stale:
                    disposition = NotifyDisposition.PENDING_KNOWLEDGE_STALE

    # Article disposition. "processed" means "we've evaluated this article" — set
    # even for below-threshold (new-but-not-notified) and not-new articles, so
    # they are never re-analyzed. Leaving them unprocessed would re-fetch +
    # re-analyze them every cycle after retention deletion + feed reappearance,
    # wasting LLM quota.
    #
    # Two exceptions, both meaning "this check did not finish with these
    # articles": the analysis itself failed (nothing was evaluated at all), or the
    # knowledge update did not land, leaving the recorded state stale (OVH-009).
    # Either way they stay unprocessed and carry an attempt, so the next cycle
    # re-analyzes them and a permanently failing article is eventually abandoned
    # instead of retried forever (TW-AUD-001).
    # Articles dropped to fit the model's context window are in the same position:
    # they were paid for and stored, but never read. Marking them processed with
    # the rest is how they used to disappear unanalyzed.
    analyzed_ids, dropped_ids = _split_batch(articles, reported_article_ids(novelty))
    unfinished = bool(novelty.error) or knowledge_stale
    article_ids = [] if unfinished else analyzed_ids
    failed_article_ids = (analyzed_ids + dropped_ids) if unfinished else dropped_ids

    # --- C3: THE durable transition. Knowledge + revision + article disposition
    # + CheckResult in one commit, fenced by the P0 generation and knowledge
    # version, and completed BEFORE any irreversible send — so a failure here
    # leaves nothing behind and the next cycle re-runs cleanly (OVH-066), while a
    # crash after it leaves a complete, self-consistent record.
    #
    # The delivery intents go in with it: one durable row per target, created
    # BEFORE any send begins. A crash between "this check decided to notify" and
    # "the message left" used to be indistinguishable from "no notification was
    # ever intended", so the alert was simply lost; now the next drain finishes
    # it, exactly once, because each intent is claimed atomically (TW-AUD-004).
    intents: list[PendingNotification] = []
    webhook_intents: list[PendingWebhook] = []
    if should_notify and notification is not None:
        title, body = notification
        intents = build_notification_intents(title, body, settings, topic_id)
        webhook_intents = build_webhook_intents(topic.name, novelty, settings, topic_id)

    outcome = CheckOutcome(
        result=result,
        knowledge_plan=knowledge_plan,
        knowledge_source=KnowledgeRevisionSource.UPDATE,
        knowledge_change_note=novelty.summary,
        article_ids=article_ids,
        failed_article_ids=failed_article_ids,
        notify_disposition=disposition,
        intents=intents,
        webhook_intents=webhook_intents,
    )
    try:
        with get_db(db_path) as conn:
            result = _commit_check_transition(conn, snapshot, outcome, settings=settings)
    except CheckTransitionAborted as exc:
        logger.warning("Check for topic '%s' aborted at the durable transition: %s", topic.name, exc)
        result.stage_error = f"transition_aborted: {exc}"
        return result

    # --- P4: irreversible network sends, now that durable state is committed and
    # no connection is open. Same claim/send/apply cycle the retry drain uses, so
    # a message delivered here and one delivered three cycles later go through
    # exactly one code path.
    if intents or webhook_intents:
        deliveries = await deliver_notification_intents(intents, settings, db_path)
        # True only when every target this check owed actually delivered. A
        # timed-out target is not a success: its outcome is unknown.
        result.notification_sent = bool(deliveries) and all(d.ok for d in deliveries)
        failed = [d for d in deliveries if not d.ok]
        if failed:
            # Surface the aggregated reason without leaking a raw URL.
            result.notification_error = _summarize_delivery_failures(failed)

        try:
            await deliver_webhook_intents(webhook_intents, settings, db_path)
        except Exception:
            logger.warning(
                "Webhook delivery failed for topic '%s'",
                topic.name,
                exc_info=True,
            )

        # --- C4: record the post-send delivery outcome onto the committed row.
        if result.id is not None:
            with get_db(db_path) as conn:
                update_check_result_delivery(
                    conn,
                    result.id,
                    notification_sent=result.notification_sent,
                    notification_error=result.notification_error,
                )
                conn.commit()

    # Reaching analysis proves the sources are alive: clear any outstanding
    # Silence Heartbeat (and announce the recovery once). The CheckResult was
    # committed at the C3 boundary above, so the streak query sees it.
    await _run_heartbeat(db_path, topic, result.id, settings)

    logger.info(
        "Topic '%s': %d articles, new_info=%s, notified=%s",
        topic.name,
        len(articles),
        novelty.has_new_info,
        result.notification_sent,
    )

    return result


def _record_result(db_path: Path | None, result: CheckResult, *, generation: str | None) -> CheckResult:
    """Persist a CheckResult on a short connection (the no-send early-return paths).

    Fenced by the topic's generation for the same reason the full transition is: a
    rowid freed by a delete is handed to the next topic created, so a check that
    started before the delete would otherwise file its result against a topic it
    never looked at (TW-AUD-007).
    """
    with get_db(db_path) as conn:
        topic_id = result.topic_id
        if generation is not None and topic_id is not None and not topic_generation_matches(conn, topic_id, generation):
            raise CheckTransitionAborted(f"topic_id={topic_id} was deleted or replaced during this check")
        created = create_check_result(conn, result)
        conn.commit()
    return created


async def _finish_check(
    db_path: Path | None,
    topic: Topic,
    result: CheckResult,
    settings: Settings,
) -> CheckResult:
    """Persist a no-send check result, then run the Silence Heartbeat over it."""
    try:
        recorded = _record_result(db_path, result, generation=topic.generation)
    except CheckTransitionAborted as exc:
        logger.warning("Check for topic '%s' aborted before recording its result: %s", topic.name, exc)
        result.stage_error = f"transition_aborted: {exc}"
        return result
    await _run_heartbeat(db_path, topic, recorded.id, settings)
    return recorded


async def _run_heartbeat(
    db_path: Path | None,
    topic: Topic,
    check_result_id: int | None,
    settings: Settings,
) -> None:
    """Announce (or clear) a source outage for this topic. Never raises.

    The heartbeat is an observability guarantee layered on top of the pipeline, so
    a failure here must not turn a recorded check into a lost one.

    The latch transition, the per-target delivery intents that announce it, and
    the revocation of the messages it supersedes are ONE commit (AUG-019/132).
    Before that boundary the announcement has not happened and the next check
    re-decides from scratch; after it, every target owes a delivery whatever
    happens to this process. The send itself is outside the transaction, so a
    crash mid-send costs at most a retry rather than re-alerting forever.

    Each database interaction runs on its own short connection, so the send below
    never has one open behind it.
    """
    try:
        if topic.id is None:
            return

        if settings.silence_heartbeat_checks <= 0:
            _disable_heartbeat_for_topic(db_path, topic)
            return

        with get_db(db_path) as conn:
            decision = evaluate_heartbeat(conn, topic, settings.silence_heartbeat_checks)
            if decision is None:
                return

            # Targets are resolved BEFORE the latch is touched: with no configured
            # Apprise URL the announcement cannot happen at all, and consuming the
            # one-shot latch for it would suppress the alert a target added during
            # the same outage should still receive (AUG-130).
            alerted_at = datetime.now(UTC)
            intents = _heartbeat_intents(
                conn,
                decision,
                topic.id,
                settings,
                check_result_id,
                alerted_at=alerted_at,
            )
            if decision.kind == "alert" and not intents:
                logger.info(
                    "Silence Heartbeat: topic '%s' has no notification target; leaving the latch unclaimed",
                    topic.name,
                )
                return

            conn.execute("BEGIN IMMEDIATE")
            # Fenced to this topic's generation and to the check the decision was
            # computed from: a check that outlived a delete can reach a replacement
            # topic that recycled the rowid (AUG-020), and a decision from check N
            # must not land once check N+1 exists (AUG-131).
            if decision.kind == "alert":
                won = claim_heartbeat_alert(
                    conn,
                    topic.id,
                    alerted_at,
                    generation=topic.generation,
                    head_check_id=decision.head_check_id,
                )
            else:
                won = clear_heartbeat_alert(
                    conn,
                    topic.id,
                    generation=topic.generation,
                    head_check_id=decision.head_check_id,
                )
            if not won:
                # Another checker (e.g. a CLI run against the live server) already
                # sent this one, or a newer check has superseded the decision.
                conn.rollback()
                return

            # Anything still queued belongs to the state this transition replaces,
            # so it is revoked rather than delivered after the event it
            # contradicts (AUG-132). Revoked first, so the rows inserted next are
            # never caught by it.
            revoke_heartbeat_intents(conn, topic.id, HEARTBEAT_INTENT_KINDS)
            create_notification_intents(conn, intents)
            conn.commit()

        deliveries = await deliver_notification_intents(intents, settings, db_path)
        failed = [d for d in deliveries if not d.ok]
        if failed:
            logger.warning(
                "Silence Heartbeat delivery failed for topic '%s': %s",
                topic.name,
                _summarize_delivery_failures(failed),
            )
    except Exception:
        logger.warning("Silence Heartbeat failed for topic '%s'", topic.name, exc_info=True)


def _disable_heartbeat_for_topic(db_path: Path | None, topic: Topic) -> None:
    """Reset this topic's heartbeat state because the feature is off. One commit.

    Defence in depth beside the global reset the setting's save and startup run
    (AUG-260): a topic latched during an outage would otherwise keep that state
    parked and fire a phantom "recovered" whenever the feature came back, and a
    queued alert would arrive long after the user switched the feature off
    (AUG-132). The latch is read raw so a corrupt value is cleared rather than
    read as already-unset (AUG-144).
    """
    if topic.id is None:
        return
    with get_db(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        if get_heartbeat_latch_raw(conn, topic.id) is not None:
            clear_heartbeat_alert(conn, topic.id, generation=topic.generation)
        revoke_heartbeat_intents(conn, topic.id, HEARTBEAT_INTENT_KINDS)
        conn.commit()


def _heartbeat_intents(
    conn: sqlite3.Connection,
    decision: HeartbeatDecision,
    topic_id: int,
    settings: Settings,
    check_result_id: int | None,
    *,
    alerted_at: datetime,
) -> list[PendingNotification]:
    """The messages this transition owes.

    An alert goes to every configured target, stamped with the latch value it is
    about to claim — the same spelling ``claim_heartbeat_alert`` stores, so the
    recovery can find this outage's deliveries later. A recovery goes only to the
    targets that actually received that outage's alert: an all-clear to somebody
    who was never told about the outage is noise, not reassurance.
    """
    latch_value: str | None
    if decision.kind == "alert":
        latch_value = to_db_utc(alerted_at)
        kind = NotificationKind.HEARTBEAT_ALERT
        targets: set[str] | None = None
    else:
        latch_value = decision.latch_value
        kind = NotificationKind.HEARTBEAT_RECOVERY
        targets = list_sent_heartbeat_alert_targets(conn, topic_id, latch_value)

    intents = build_notification_intents(
        decision.title,
        decision.body,
        settings,
        topic_id,
        kind=kind,
        latch_value=latch_value,
        check_result_id=check_result_id,
    )
    if targets is not None:
        intents = [intent for intent in intents if intent.url in targets]
    return intents


def _summarize_delivery_failures(failed: list[NotificationDelivery]) -> str:
    """Build a redacted, operator-readable summary of failed deliveries.

    Surfaces the per-channel reason (OVH-039) without leaking any raw URL/token
    (OVH-027): each entry is ``scheme://host: reason``.
    """
    parts = [f"{redact_url(d.url)}: {d.error or 'delivery failed'}" for d in failed]
    return "; ".join(parts)


def build_notification_intents(
    title: str,
    body: str,
    settings: Settings,
    topic_id: int,
    *,
    kind: NotificationKind = NotificationKind.NOVELTY,
    latch_value: str | None = None,
    check_result_id: int | None = None,
) -> list[PendingNotification]:
    """One unsaved delivery intent per configured Apprise target. Pure.

    Pure so the intents can be built with no connection open and handed to the
    transaction that commits them alongside the state change they announce.
    ``kind`` and ``latch_value`` are what let a heartbeat transition revoke a
    superseded message instead of sending it after the event it contradicts.
    """
    return [
        PendingNotification(
            topic_id=topic_id,
            check_result_id=check_result_id,
            title=title,
            body=body,
            url=url,
            kind=kind,
            latch_value=latch_value,
        )
        for url in settings.notifications.urls
    ]


async def _deliver_one_notification_intent(
    intent: PendingNotification,
    settings: Settings,
    db_path: Path | None,
    conn: sqlite3.Connection | None = None,
) -> NotificationDelivery | None:
    """Claim, send, apply — for one intent. ``None`` when the claim was lost.

    Three short connections, three commits, no connection open across the send.
    """
    intent_id = intent.id
    if intent_id is None:
        return None

    claim_token = secrets.token_hex(8)
    now_iso = to_db_utc(datetime.now(UTC))
    with short_conn(conn, db_path) as claim_conn:
        won = claim_notification_intent(claim_conn, intent_id, claim_token, now_iso)
        claim_conn.commit()
    if not won:
        logger.debug("Notification intent id=%d not claimable (claimed, exhausted or not due); skipping", intent_id)
        return None

    if intent.url is None:
        # A pre-per-target row names no destination. Re-sending it to every
        # configured URL would re-hit channels that already delivered, so it is
        # abandoned rather than duplicated.
        with short_conn(conn, db_path) as apply_conn:
            apply_notification_outcome(
                apply_conn,
                intent_id,
                claim_token,
                sent=False,
                error="intent has no target URL",
                terminal=True,
            )
            apply_conn.commit()
        return None

    delivery = await send_single_notification(intent.title, intent.body, intent.url, settings.apprise_timeout_seconds)
    delivery = delivery.model_copy(update={"intent_id": intent_id})

    if delivery.timed_out:
        # Unknown, not failed: the Apprise thread cannot be cancelled and may
        # still be delivering. Leaving the intent 'sending' is the honest record
        # (TW-AUD-004); the stale-claim release re-arms it once enough time has
        # passed that the send cannot still be in flight (AUG-071).
        logger.warning(
            "Notification intent id=%d left in flight after its deadline; outcome unknown",
            intent_id,
        )
        return delivery

    terminal = (delivery.error or "") in _TERMINAL_DELIVERY_ERRORS
    due = None if delivery.ok or terminal else next_attempt_at(intent.retry_count)
    with short_conn(conn, db_path) as apply_conn:
        applied = apply_notification_outcome(
            apply_conn,
            intent_id,
            claim_token,
            sent=delivery.ok,
            error=delivery.error,
            next_attempt_at=due,
            terminal=terminal,
        )
        apply_conn.commit()
    if not applied:
        logger.warning("Late apply for notification intent id=%d ignored: the claim is no longer ours", intent_id)
    elif delivery.ok:
        logger.info("Notification intent id=%d delivered to %s", intent_id, redact_url(intent.url))
    elif terminal:
        logger.warning(
            "Abandoning notification intent id=%d without retry (url=%s reason=%s)",
            intent_id,
            redact_url(intent.url),
            delivery.error,
        )
    return delivery


async def deliver_notification_intents(
    intents: list[PendingNotification],
    settings: Settings,
    db_path: Path | None,
    conn: sqlite3.Connection | None = None,
) -> list[NotificationDelivery]:
    """Deliver a batch of already-persisted intents. One outcome per intent sent.

    Shared by the live path and the retry drain: a message delivered seconds after
    its check and one delivered three cycles later take the same code path, so
    there is only one place where a claim, a send and an apply can disagree.

    Intents whose claim was lost (another drainer owns them, they are exhausted,
    or they are not due yet) contribute no outcome. ``return_exceptions=True`` with
    every result inspected keeps ownership until every child has settled (AUG-263).
    """
    if not intents:
        return []

    semaphore = asyncio.Semaphore(_DELIVERY_CONCURRENCY)

    async def _process(intent: PendingNotification) -> NotificationDelivery | None:
        async with semaphore:
            return await _deliver_one_notification_intent(intent, settings, db_path, conn)

    results = await asyncio.gather(*(_process(intent) for intent in intents), return_exceptions=True)

    deliveries: list[NotificationDelivery] = []
    for intent, result in zip(intents, results, strict=True):
        if isinstance(result, BaseException):
            logger.warning(
                "Notification intent id=%s failed unexpectedly",
                intent.id,
                exc_info=(type(result), result, None),
            )
            continue
        if result is not None:
            deliveries.append(result)
    return deliveries


async def retry_pending_notifications(
    conn: sqlite3.Connection | None = None,
    settings: Settings | None = None,
    *,
    db_path: Path | None = None,
) -> None:
    """Drain the notification delivery queue: due intents get one more attempt.

    Args:
        conn: Optional existing connection (back-compat). Reused if given but
            committed per item and never held across a send.
        settings: Application settings (required).
        db_path: Path used to open short-lived connections when ``conn`` is None.
    """
    if settings is None:
        raise ValueError("settings is required")

    # Single-flight: only one drain runs at a time in this process. A second
    # caller skips rather than walking the same queue concurrently (OVH-017).
    if _notification_retry_lock.locked():
        logger.debug("Notification retry already in progress; skipping overlapping drain")
        return

    # OVH-102: run the drain under a generated correlation id so a single drain's
    # snapshot/claim/send/apply log lines are traceable across interleaved ticks
    # (otherwise they all logged '-'). Token idiom (OVH-103): restore the caller's
    # prior id afterwards rather than clobbering it to None.
    async with _notification_retry_lock:
        token = check_id_var.set(generate_check_id())
        try:
            await _drain_notification_intents(conn, settings, db_path)
        finally:
            check_id_var.reset(token)


async def _drain_notification_intents(
    conn: sqlite3.Connection | None,
    settings: Settings,
    db_path: Path | None,
) -> None:
    """Drain the notification queue once (caller holds the retry lock)."""
    now = datetime.now(UTC)
    stale_cutoff = to_db_utc(now - _CLAIM_STALE_AFTER)
    with short_conn(conn, db_path) as snapshot:
        released = release_stale_notification_claims(snapshot, stale_cutoff)
        if released:
            logger.warning("Re-armed %d stale notification claim(s)", released)
        for item in abandon_expired_notifications(snapshot):
            # One WARNING per permanently-dropped delivery so an abandoned
            # notification is observable: identify it by topic/check ids (the
            # body is not logged — it may carry the notified content) (OVH-040).
            logger.warning(
                "Abandoning notification after max retries (topic_id=%s check_result_id=%s title=%r created_at=%s)",
                item.topic_id,
                item.check_result_id,
                item.title,
                item.created_at.isoformat(),
            )
        snapshot.commit()
        pending = list_due_notification_intents(snapshot, to_db_utc(now), _RETRY_DRAIN_LIMIT)

    if not pending:
        return

    logger.info("Retrying %d pending notification(s)", len(pending))
    await deliver_notification_intents(pending, settings, db_path, conn)


async def check_all_topics(
    settings: Settings,
    db_path: Path | None = None,
    *,
    guard: bool = True,
) -> list[CheckResult]:
    """Check all active, ready topics for new information.

    Uses per-topic connection granularity: a single connection held for the
    whole cycle would stay open across every topic's HTTP + LLM awaits,
    blocking concurrent web requests. Instead each phase uses its own
    short-lived connection that is committed and closed promptly:

      * retry passes (notifications + webhooks) — each manages its own
        short-lived connections internally (snapshot, send with none held,
        commit per item)
      * the due-topics query — one connection
      * each topic check — ``check_topic`` opens its own per-phase connections

    Each topic is checked independently. Errors in one topic do not
    affect others.

    Concurrency: this is the single whole-cycle funnel, so it acquires the
    process-wide ``start_check_all`` gate itself (OVH-034). The scheduler tick,
    the UI check-all, and the CLI all share it, so a tick overlapping a UI
    check-all (or vice versa) skips rather than running two full cycles that
    double-drain the retry queues and double-notify. Each per-topic
    ``check_topic`` additionally holds the per-topic guard (OVH-096). Callers
    that already hold the whole-cycle gate (the web handler, which acquires it
    synchronously to decide whether to enqueue) pass ``guard=False``.

    Args:
        settings: Application settings.
        db_path: Optional database path override for testing.
        guard: When True (default), acquire/release the whole-cycle gate. Pass
            False when the caller already holds it.

    Returns:
        List of CheckResults, one per topic checked.
    """
    if not guard:
        return await _check_all_topics_inner(settings, db_path)

    owner = _checking_state.start_check_all()
    if owner is None:
        logger.info("Check-all already in flight; skipping overlapping cycle")
        return []
    try:
        return await _check_all_topics_inner(settings, db_path)
    finally:
        _checking_state.finish_check_all(owner)


async def _check_all_topics_inner(
    settings: Settings,
    db_path: Path | None,
) -> list[CheckResult]:
    """Run one whole check cycle (caller owns the whole-cycle gate)."""

    async def _drain_retries() -> None:
        """Retry failed deliveries from previous cycles.

        Each retry function manages its own short-lived connections: it snapshots
        pending rows, sends with NO connection held, and commits per item.
        """
        await retry_pending_notifications(settings=settings, db_path=db_path)
        await retry_pending_webhooks(settings=settings, db_path=db_path)

    # Snapshot the due topics, then release the connection before the long
    # per-topic HTTP/LLM work begins.
    with get_db(db_path) as conn:
        due_topics = get_topics_due_for_check(conn, settings.check_interval_minutes)

    if not due_topics:
        await _drain_retries()
        return []

    logger.info("Starting check cycle for %d due topics", len(due_topics))

    # Bound per-topic checks so a slow topic does not head-of-line-block the rest
    # within this single tick (OVH-055). This stays inside the one whole-cycle
    # gate (settled #9: one minute-tick job); each per-topic ``check_topic`` still
    # funnels through its own ``_checking_state`` per-topic guard. Mirrors the
    # ``content_fetch_concurrency`` Semaphore precedent. Each topic keeps its own
    # short-lived connection so concurrent checks never share a handle.
    semaphore = asyncio.Semaphore(settings.topic_check_concurrency)

    async def _check_one(topic: Topic) -> CheckResult | None:
        async with semaphore:
            try:
                # No connection is opened here: check_topic owns its own
                # short-lived per-phase connections (AUG-136). Wrapping this call
                # in one held the handle — and, via the feed-health callback, the
                # WAL writer — across every fetch/LLM/send await in the pipeline.
                return await check_topic(topic, settings, db_path=db_path)
            except Exception:
                logger.error(
                    "Unexpected error checking topic '%s'",
                    topic.name,
                    exc_info=True,
                )
                return None

    # The retry backlog runs BESIDE the due topics, not in front of them: a
    # source outage can queue one heartbeat row per topic per target, and each of
    # those can burn the full Apprise deadline before the first topic would
    # otherwise start — long enough for the single-instance scheduler to skip
    # later ticks entirely (AUG-027). The drain is bounded per cycle; the topics
    # no longer wait for it either way. One gather owns all of it, so the cycle
    # never returns while a drain it started is still writing (AUG-263).
    gathered = await asyncio.gather(
        _drain_retries(),
        *(_check_one(topic) for topic in due_topics),
        return_exceptions=True,
    )
    drain_outcome, *topic_outcomes = gathered
    if isinstance(drain_outcome, BaseException):
        logger.error(
            "Retry drain failed during the check cycle",
            exc_info=(type(drain_outcome), drain_outcome, drain_outcome.__traceback__),
        )
    results: list[CheckResult] = [r for r in topic_outcomes if isinstance(r, CheckResult)]

    logger.info(
        "Check cycle complete: %d topics checked, %d with new info",
        len(results),
        sum(1 for r in results if r.has_new_info),
    )
    return results


def _init_corpus(
    db_path: Path | None,
    topic_id: int,
    fetched: list[Article],
    max_articles: int,
) -> list[Article]:
    """The articles a baseline is built from: this fetch plus what is already stored.

    ``fetch_new_articles_for_topic`` returns only entries the topic has never seen
    — that is what makes a routine check cheap. Initialization is not a routine
    check. A Retry after the LLM failed re-fetches feeds that have not moved,
    gets nothing back, and reports the same error again while the batch it failed
    on sits unprocessed in the database; a Re-initialize on a mature topic
    rebuilds its whole understanding from whatever handful of entries appeared
    since the last check and records that as the new baseline (AUG-252).

    Stored rows are appended newest-first behind the fresh ones and the whole
    batch is capped at ``max_articles``, so an over-budget prompt is fitted by
    dropping the oldest stored articles, never the new ones. Marking already-
    processed rows processed again is a no-op.
    """
    seen = {article.id for article in fetched if article.id is not None}
    corpus = list(fetched)
    if len(corpus) >= max_articles:
        return corpus[:max_articles]

    with get_db(db_path) as conn:
        stored = list_articles_for_topic(conn, topic_id, limit=max_articles)
    for article in stored:
        if len(corpus) >= max_articles:
            break
        if article.id is None or article.id in seen:
            continue
        corpus.append(article)
    return corpus


def _commit_init_transition(
    conn: sqlite3.Connection,
    snapshot: TopicSnapshot,
    plan: KnowledgeUpdatePlan,
    article_ids: list[int],
    settings: Settings,
) -> None:
    """Land a whole initialization in ONE transaction: knowledge + articles + READY.

    Mirrors ``_commit_check_transition``. Previously the knowledge write, the
    revision, the article disposition and the READY status were four separate
    commits, so an interruption between any two left a topic that was READY with
    no knowledge, or carried knowledge while still showing RESEARCHING.
    """
    topic_id = snapshot.topic.id
    if topic_id is None:
        raise ValueError("Topic must have an ID")

    conn.execute("BEGIN IMMEDIATE")

    if not topic_generation_matches(conn, topic_id, snapshot.generation):
        raise CheckTransitionAborted(f"topic_id={topic_id} was deleted or replaced during initialization")

    if not apply_knowledge_update(
        conn,
        topic_id,
        plan,
        expected_version=snapshot.knowledge_version,
        source=KnowledgeRevisionSource.INIT,
        settings=settings,
    ):
        raise CheckTransitionAborted(
            f"knowledge for topic_id={topic_id} moved past version {snapshot.knowledge_version}"
        )

    mark_articles_processed(conn, article_ids)
    if not update_topic_init_status(
        conn,
        topic_id,
        status=TopicStatus.READY,
        status_changed_at=datetime.now(UTC),
        error_message=None,
        init_attempts=0,
        expected_status=TopicStatus.RESEARCHING,
    ):
        # Stuck recovery gave up on this initialization and moved the row to ERROR
        # while the LLM phase was running, so a Retry may already be under way. The
        # claim is gone; landing READY here would make the terminal status
        # last-writer-wins between recovery and abandoned work (AUG-139). Abort the
        # whole transition instead — nothing is written and the next attempt is clean.
        raise CheckTransitionAborted(f"topic_id={topic_id} left RESEARCHING during initialization")
    conn.commit()


async def initialize_new_topic(
    topic: Topic,
    settings: Settings,
    *,
    db_path: Path | None = None,
    claimed: bool = False,
) -> None:
    """Initialize a topic's knowledge state from its first batch of articles.

    Transitions: NEW/READY/ERROR → RESEARCHING → READY (or ERROR on failure).
    Called by the web layer (background task), the scheduler (gradual init) and
    the CLI.

    Ownership: the RESEARCHING transition is a conditional claim, not an
    unconditional write (AUG-288). Callers that already won the claim durably —
    the scheduler's ``claim_new_topic_for_init``, the web Retry handler, the
    just-INSERTed row of a topic creation — pass ``claimed=True``; everyone else
    lets this function claim, and gets :class:`TopicInitRefused` when the topic is
    already being initialized, is paused, or is gone. Every terminal write is
    fenced back to that claim, so an initializer stuck recovery has already
    written off cannot overwrite the recovered state (AUG-139).

    Connection invariant (AUG-136): no connection is open across the fetch or LLM
    awaits. Status writes each take a short connection of their own, the fetch and
    ``prepare_initial_knowledge`` phases run offline, and the knowledge state,
    its revision, the article disposition and the READY transition land together
    in one fenced transaction.
    """
    if topic.id is None:
        raise ValueError("Topic must have an ID")
    topic_id: int = topic.id

    # OVH-102: run the whole multi-round init under a generated correlation id so a
    # single topic's NEW->RESEARCHING->READY/ERROR flow is traceable across
    # interleaved scheduler ticks (otherwise every init log line was '-'). Token
    # idiom (OVH-103): restore the caller's prior id afterwards rather than clobber.
    cid = generate_check_id()
    token = check_id_var.set(cid)

    # Status transitions here use ``update_topic_init_status`` (a targeted UPDATE of
    # only status/error/init_attempts) rather than ``update_topic`` so a concurrent
    # UI edit to this topic's feeds/thresholds during the long fetch/LLM await is
    # never clobbered by a stale in-memory snapshot (OVH-100).

    def _set_init_status(status: TopicStatus, *, error_message: str | None, init_attempts: int) -> None:
        """Write a terminal status, but only while this initializer still owns the claim."""
        now = datetime.now(UTC)
        with get_db(db_path) as conn:
            won = update_topic_init_status(
                conn,
                topic_id,
                status=status,
                status_changed_at=now,
                error_message=error_message,
                init_attempts=init_attempts,
                expected_status=TopicStatus.RESEARCHING,
            )
            conn.commit()
        if not won:
            logger.warning(
                "Topic id=%d left RESEARCHING during initialization; not writing %s",
                topic_id,
                status.value,
            )
            return
        topic.status = status
        topic.status_changed_at = now
        topic.error_message = error_message
        topic.init_attempts = init_attempts

    # The status this call took the claim from, so a cancellation can hand it back
    # (AUG-243). ``None`` when the caller claimed on our behalf and we never saw it.
    prior_status: TopicStatus | None = None

    if not claimed:
        # One conditional UPDATE decides who initializes. Reading the status and
        # then writing RESEARCHING let two initializers both pass (AUG-288).
        with get_db(db_path) as conn:
            live = get_topic(conn, topic_id)
            if live is None:
                check_id_var.reset(token)
                raise TopicInitRefused(f"Topic id={topic_id} no longer exists")
            if live.status == TopicStatus.RESEARCHING:
                check_id_var.reset(token)
                raise TopicInitRefused(f"Topic '{live.name}' is already being initialized")
            if not live.is_active:
                check_id_var.reset(token)
                raise TopicInitRefused(f"Topic '{live.name}' is paused; enable it before initializing")
            if not claim_topic_for_init(conn, topic_id, live.status):
                check_id_var.reset(token)
                raise TopicInitRefused(f"Topic '{live.name}' was claimed by another initializer")
            prior_status = live.status
        topic.status = TopicStatus.RESEARCHING

    with get_db(db_path) as conn:
        snapshot = _snapshot_topic(conn, topic_id)
    if snapshot is None:
        logger.warning("Topic id=%d no longer exists; skipping initialization", topic_id)
        check_id_var.reset(token)
        return

    logger.info("Initializing knowledge for topic '%s' (id=%d) [check_id=%s]", topic.name, topic_id, cid)

    try:
        fetch_result = await fetch_new_articles_for_topic(
            topic,
            db_path=db_path,
            max_articles=settings.max_articles_per_check,
            feed_fetch_timeout=settings.feed_fetch_timeout,
            article_fetch_timeout=settings.article_fetch_timeout,
            feed_max_retries=settings.feed_max_retries,
            concurrency=settings.content_fetch_concurrency,
            feed_backoff_base_minutes=settings.feed_backoff_base_minutes,
            feed_backoff_cap_hours=settings.feed_backoff_cap_hours,
            exa_settings=settings.exa,
        )
        articles = _init_corpus(db_path, topic_id, fetch_result.articles, settings.max_articles_per_check)

        if not articles:
            # OVH-001: during a NEW-topic re-init (init_attempts>0) every prior
            # article is already stored, so a feed with no fresh entries yields an
            # empty fetch. That is not a failure — keep waiting in NEW for a later
            # cycle. Only the very first attempt (init_attempts==0) with no articles
            # at all is a genuine initialization error.
            if topic.init_attempts > 0:
                _set_init_status(TopicStatus.NEW, error_message=None, init_attempts=topic.init_attempts)
                logger.info(
                    "Topic '%s': no new articles on re-init (attempt %d) — staying NEW",
                    topic.name,
                    topic.init_attempts,
                )
                return
            # Three different empty results, three different diagnoses — the same
            # vocabulary a normal check uses (AUG-135). A total source failure (bad
            # Exa key, every RSS feed down) is not the same as no source having run
            # at all (Exa disabled or keyless, no feed URLs, every feed in a backoff
            # window), and neither is the same as a healthy source with nothing to
            # say. Only the last deserves the generic message.
            init_error = _init_empty_error(fetch_result)
            _set_init_status(
                TopicStatus.ERROR,
                error_message=init_error,
                init_attempts=topic.init_attempts,
            )
            return

        # LLM phase, still connection-free; the plan is persisted below.
        plan = await prepare_initial_knowledge(topic, articles, settings)

        # Only what the baseline was actually built from is finished with: an
        # over-budget corpus is fitted by dropping trailing articles, and those
        # stay unprocessed for the first check to pick up.
        article_ids, _ = _split_batch(articles, plan.analyzed_article_ids)
        with get_db(db_path) as conn:
            _commit_init_transition(conn, snapshot, plan, article_ids, settings)
        topic.status = TopicStatus.READY
        topic.error_message = None
        topic.init_attempts = 0

        if plan.sufficient_data:
            logger.info("Knowledge initialized for topic '%s' — now READY", topic.name)
        else:
            logger.warning(
                "Topic '%s' READY with thin/insufficient knowledge — baseline stored, will self-heal on future checks",
                topic.name,
            )

    except asyncio.CancelledError:
        # Ctrl-C on the CLI — and task cancellation generally — is not an
        # ``Exception``, so it walked straight past the handler below and left the
        # committed RESEARCHING claim behind. Offline, nothing ever recovers that:
        # every later CLI init refuses the topic until the server is started
        # (AUG-243). Hand the claim back before re-raising. All of this is
        # synchronous on purpose; an await here would be cancelled again.
        restored = prior_status if prior_status is not None else TopicStatus.ERROR
        logger.warning(
            "Initialization of topic '%s' was cancelled; restoring status %s",
            topic.name,
            restored.value,
        )
        _set_init_status(
            restored,
            error_message=("Initialization interrupted. Click Retry." if restored is TopicStatus.ERROR else None),
            init_attempts=topic.init_attempts,
        )
        raise
    except Exception as exc:
        logger.error("Knowledge init failed for topic '%s'", topic.name, exc_info=True)
        _set_init_status(TopicStatus.ERROR, error_message=str(exc), init_attempts=topic.init_attempts)
    finally:
        # Restore the caller's prior correlation id (OVH-103 token idiom).
        check_id_var.reset(token)
