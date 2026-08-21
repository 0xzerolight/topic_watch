"""Neutral source layer: the types every article source shares, and the registry.

A "source" is anything that can answer "what is new about this topic": the RSS
providers (AUTO), a topic's own feed URLs (MANUAL), or the Exa search API (EXA).
They differ only in how they reach the network, so everything they exchange with
the pipeline lives here rather than in any one source's module (TW-AUD-022):

- ``FeedEntry`` / ``FeedResponse`` — the entries a source returns and the
  identity plus capabilities of whatever produced them. ``provider_name`` and
  ``needs_url_resolution`` are response-level because AUTO can cascade between
  providers mid-fetch, so only the response knows which one actually answered.
- ``article_identity`` / ``collapse_duplicate_entries`` — what makes two entries
  the same article. One definition for every source, so dedup does not depend on
  which provider happened to answer.
- ``FetchStatus`` / ``FeedHealthOutcome`` / ``FeedHealthCallback`` /
  ``FeedStateLoader`` — how a fetch went, and the health side-channel that
  carries it.
- ``SourceRequest`` — the per-attempt inputs a fetcher needs beyond the topic.
- ``register_source`` / ``fetch_feeds_for_topic`` — mode-to-fetcher dispatch.

Sources register themselves at import, so adding one means writing its module
and registering it, not editing a dispatcher branch in an unrelated source's
module. This module imports no source, which is what keeps that possible.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Any, TypeVar
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from pydantic import BaseModel

from app.feed_backoff import BACKOFF_BASE_MINUTES, BACKOFF_CAP_HOURS
from app.models import FeedMode, Topic

if TYPE_CHECKING:
    from app.config import ExaSettings
    from app.models import FeedHealth
    from app.scraping.routing import ProviderRouter

logger = logging.getLogger(__name__)

T = TypeVar("T")


class FetchStatus(StrEnum):
    """How one feed fetch went — the single vocabulary every source reports in.

    Streak, cascade and coverage decisions used to be inferred from "did this
    return entries", which conflated four different outcomes: a feed that
    answered with news, one the server said had not changed, one that genuinely
    has nothing today, and one that failed. Each of those wants a different
    answer from provider health and from the fallback cascade (AUG-172/AUG-173),
    so each gets its own value and the callers ask the question they mean.
    """

    OK = "ok"
    """Fetched and parsed, with usable entries."""

    NOT_MODIFIED = "not_modified"
    """A conditional request the source answered 304: our copy is current."""

    EMPTY = "empty"
    """Fetched and parsed fine; the source simply has nothing to offer."""

    FAILED = "failed"
    """The source itself failed us: blocked, unreachable, HTTP error, unusable body."""

    ABORTED = "aborted"
    """We gave up, not the source: the topic's whole attempt budget ran out.

    Kept apart from ``FAILED`` because the usual cause is some *other* feed being
    slow. Charging it to this feed's failure streak would put a healthy feed into
    exponential backoff — or trip a provider cooldown — for something it did not
    do. It still counts against the check's own coverage, so a check that ran out
    of budget never passes for healthy silence.
    """

    @property
    def succeeded(self) -> bool:
        """True when the source answered us, whether or not it had news.

        This is what resets a failure streak: an empty 200 and a 304 are both
        proof the source is reachable and behaving (AUG-173).
        """
        return self in (FetchStatus.OK, FetchStatus.NOT_MODIFIED, FetchStatus.EMPTY)

    @property
    def is_source_failure(self) -> bool:
        """True when the streak should advance and backoff should engage."""
        return self is FetchStatus.FAILED

    @property
    def counts_as_failed_fetch(self) -> bool:
        """True when this fetch yielded nothing *and* that is not normal silence.

        Feeds the check's ``feeds_failed`` counter, which is what tells the
        pipeline the cycle was degraded rather than quiet.
        """
        return self in (FetchStatus.FAILED, FetchStatus.ABORTED)

    @property
    def should_cascade(self) -> bool:
        """True when AUTO should try the other provider.

        A 304 must not: the provider answered, and its answer was "you already
        have it", so cascading buys a second provider's articles for a topic that
        is not missing any (AUG-172).
        """
        return self in (FetchStatus.EMPTY, FetchStatus.FAILED)


@dataclass(frozen=True)
class FeedHealthOutcome:
    """One feed fetch's health verdict, captured in memory during source I/O.

    The fetch layer reports per-feed outcomes through a callback that fires while
    other feeds are still in flight. Writing them to SQLite from inside that
    callback opened a WAL write transaction that stayed open across every
    remaining feed await — in MANUAL mode a fast feed could own the single writer
    while ``gather`` waited on a slow one, and in AUTO mode the primary's outcome
    could own it across the fallback fetch. Past the 5-second busy timeout that
    fails concurrent UI and scheduler writes outright, and a cancellation
    discarded health outcomes that had already been observed (AUG-171).

    Collecting them as plain values and applying the batch afterwards, in one
    short no-await phase, removes both failure modes.
    """

    feed_url: str
    status: FetchStatus
    error_msg: str | None = None
    etag: str | None = None
    last_modified: str | None = None

    @property
    def success(self) -> bool:
        return self.status.succeeded


FeedHealthCallback = Callable[[FeedHealthOutcome], None]
"""Reports one fetch's verdict. One record rather than five positional values,
so a new distinction (304 vs empty vs aborted) reaches every consumer at once."""

FeedStateLoader = Callable[[str], "FeedHealth | None"]
"""Returns the stored health row for a feed URL, or None when untracked."""

FeedArticleCheck = Callable[[str], bool]
"""True when the calling topic already stores articles that came from this feed.

Conditional-GET validators are keyed by feed URL alone while articles are owned
per topic, so a second topic subscribing to a feed another topic already polls
inherits validators for a representation it has never seen: the server answers
304, the topic stores nothing, and it stays empty for as long as the feed is
unchanged (TW-AUD-020). Asking this before sending validators makes them
replay-safe — a topic holding nothing from a feed asks for the full body."""


SOURCE_ATTEMPT_BUDGET_SECONDS = 300.0
"""Total wall-clock a single topic's source work may occupy (TW-AUD-018).

Per-request timeouts bound one transport wait each, and nothing bounded their
sum: retries, redirect hops, DNS checks, Google resolution and content
extraction each drew a fresh budget, so a source that is merely slow rather than
broken could hold a topic slot indefinitely and starve the next scheduler tick.
Five minutes is far above a healthy check (seconds) and far below an interval,
so it only ever truncates work that was already failing.
"""

DEADLINE_ERROR = "Source deadline exceeded"
"""Health-row text for the typed timeout outcome, so an expired budget is
distinguishable from an ordinary per-request timeout on the Feed Health page."""


class SourceDeadlineExceeded(Exception):
    """One logical source attempt ran out of its total budget."""


@dataclass(frozen=True)
class Deadline:
    """An absolute monotonic deadline shared by every stage of one attempt.

    Monotonic by construction: ``at`` is a ``time.monotonic()`` reference, never
    a wall-clock timestamp, so a clock adjustment mid-check cannot extend or
    collapse the budget (wave-A clock policy).

    Passing the same instance down through fetch, retry, resolution and
    extraction is what makes the budget one budget rather than one per stage.
    """

    at: float

    @classmethod
    def after(cls, budget: float = SOURCE_ATTEMPT_BUDGET_SECONDS) -> Deadline:
        """A deadline ``budget`` seconds from now."""
        return cls(at=time.monotonic() + budget)

    def remaining(self) -> float:
        """Seconds left, clamped at zero."""
        return max(0.0, self.at - time.monotonic())

    def expired(self) -> bool:
        return self.remaining() <= 0.0

    def slice(self, timeout: float) -> float:
        """The per-request timeout, shortened to what is left of the budget."""
        return min(timeout, self.remaining())

    def check(self, stage: str) -> None:
        """Raise :class:`SourceDeadlineExceeded` when nothing is left to spend."""
        if self.expired():
            raise SourceDeadlineExceeded(f"{DEADLINE_ERROR} before {stage}")


async def bounded(deadline: Deadline, stage: str, coro: Coroutine[Any, Any, T]) -> T:
    """Await ``coro`` under what is left of ``deadline``.

    Raises :class:`SourceDeadlineExceeded` instead of letting a stage run past
    the budget, and never starts one that has nothing left to spend.
    """
    if deadline.expired():
        coro.close()
        raise SourceDeadlineExceeded(f"{DEADLINE_ERROR} before {stage}")
    try:
        async with asyncio.timeout(deadline.remaining()):
            return await coro
    except TimeoutError as exc:
        raise SourceDeadlineExceeded(f"{DEADLINE_ERROR} during {stage}") from exc


def url_hostname(url: str) -> str:
    """The lowercased hostname of a URL, or ``""`` when it has none.

    ``urlparse`` already strips userinfo and port and lowercases the host, so a
    crafted ``https://news.google.com@evil.example/`` yields ``evil.example``.
    A trailing root dot is dropped so ``example.com.`` and ``example.com`` are
    the same host.
    """
    try:
        host = urlparse(url).hostname or ""
    except ValueError:  # malformed netloc (bad port, unbalanced IPv6 brackets)
        return ""
    return host.lower().rstrip(".")


def host_matches(host: str, domain: str) -> bool:
    """True when ``host`` is ``domain`` itself or a subdomain of it.

    Source identity is a host fact, so it is decided on the parsed hostname and
    never on a substring of the raw URL (TW-AUD-031): ``news.google.com`` in a
    path or query belongs to whoever owns the host, and ``news.google.com.evil``
    is a different host that happens to start with the same labels.
    """
    return host == domain or host.endswith(f".{domain}")


class FeedEntry(BaseModel):
    """A single entry parsed from an RSS/Atom feed."""

    title: str
    url: str
    published: datetime | None = None
    updated: datetime | None = None
    """When the source says this article was last revised, if it says so distinctly.

    Kept apart from ``published`` because it is the one signal a feed gives that the
    story at a stable URL is not the story it told last time (AUG-320): it feeds
    ``article_identity`` and picks the winner when two entries describe one URL."""
    summary: str = ""
    source_feed: str
    content: str | None = None
    """Pre-extracted full text, when the source already provides it (e.g. Exa search).
    ``None`` for RSS entries, whose text is fetched during content extraction. When set
    and non-empty, it short-circuits the network fetch in ``extract_article_content``."""


# --- Article identity ---------------------------------------------------------

_TRACKING_PARAMS = frozenset(
    {"fbclid", "gclid", "gbraid", "wbraid", "msclkid", "dclid", "yclid", "igshid", "mc_cid", "mc_eid"}
)
"""Query parameters that identify the click, not the article. Plus any ``utm_*``."""

_DEFAULT_PORTS = {"http": 80, "https": 443}

_MAX_CLOCK_SKEW = timedelta(hours=1)
"""How far ahead of us a publisher's clock may plausibly be (AUG-184)."""

_UNDATED = datetime.min.replace(tzinfo=UTC)


def as_utc(value: datetime | None) -> datetime | None:
    """A feed datetime in UTC; a naive one is read as UTC rather than local time."""
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def normalize_published(value: datetime | None, *, now: datetime | None = None) -> datetime | None:
    """UTC-normalize a publication date, discarding impossible ones (AUG-184).

    A timestamp further ahead than a plausible clock difference is a publisher
    defect, not a scoop. Trusting it puts that entry permanently at the head of
    the recency ranking, where it displaces real stories from the article cap on
    every check; returning ``None`` files it with the undated entries instead,
    which are ranked by retrieval order.
    """
    stamp = as_utc(value)
    if stamp is None:
        return None
    if stamp > (now or datetime.now(UTC)) + _MAX_CLOCK_SKEW:
        logger.debug("Discarding impossible publication date %s", stamp.isoformat())
        return None
    return stamp


def _is_tracking_param(name: str) -> bool:
    key = name.lower()
    return key.startswith("utm_") or key in _TRACKING_PARAMS


def canonical_url(url: str) -> str:
    """The comparable form of an article URL.

    Case is folded only where the URL grammar says it is insignificant — the
    scheme and the host — because a path or query IS case-sensitive on many
    publishers, and folding it made ``/A`` and ``/a`` the same article (AUG-183).
    Beyond that: userinfo and default ports are dropped, the fragment is dropped
    (it never reaches the server), and click-tracking parameters are removed so
    the same story shared through two campaigns is one story (AUG-180).

    A URL too malformed to parse is returned stripped, so it still compares
    equal to itself.
    """
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
        port = parsed.port
    except ValueError:  # malformed netloc (bad port, unbalanced IPv6 brackets)
        return url.strip()
    if not host:
        return url.strip()
    scheme = parsed.scheme.lower()
    netloc = host.rstrip(".")
    if port is not None and port != _DEFAULT_PORTS.get(scheme):
        netloc = f"{netloc}:{port}"
    query = urlencode([(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True) if not _is_tracking_param(k)])
    return urlunparse((scheme, netloc, parsed.path, parsed.params, query, ""))


def _identity_digest(*parts: str) -> str:
    """Hash a fixed field list so no two different field sets serialize alike.

    Each field is length-prefixed rather than joined by a delimiter: with a plain
    ``|`` join, a title containing a pipe could produce the same string as a
    different URL/title pair (AUG-183).
    """
    payload = "\n".join(f"{len(part)}:{part}" for part in parts)
    return hashlib.sha256(payload.encode()).hexdigest()


def revision_marker(entry: FeedEntry) -> str:
    """What distinguishes this representation of a story from an earlier one.

    Empty unless the source itself said the article changed: an ``updated`` stamp
    that differs from ``published``, or prefetched full text (Exa) whose digest
    moved. The entry summary is deliberately NOT in here — aggregator summaries
    carry related-coverage lists and blurbs that churn on their own, and every
    churn would cost a content fetch, a row and an LLM analysis for an article
    nobody revised.
    """
    parts: list[str] = []
    revised = as_utc(entry.updated)
    if revised is not None and revised != as_utc(entry.published):
        parts.append(revised.isoformat())
    text = " ".join((entry.content or "").split())
    if text:
        parts.append(hashlib.sha256(text.encode()).hexdigest())
    return "\x1f".join(parts)


def article_identity(entry: FeedEntry) -> str:
    """The dedup key for one article representation — the single definition.

    Every source path keys off this: what is stored in ``articles.content_hash``,
    what same-topic dedup skips on, and what cross-topic content reuse matches.

    Identity is the story it is (``story_identity``) AND the revision the source is
    currently serving. Keying on the story alone meant a correction, retraction or
    expansion published at the same URL under the same headline was indistinguishable
    from the copy already stored, so it was skipped before anything could read it and
    the knowledge state kept the superseded facts (AUG-320). With the revision in the
    key, an unchanged article still deduplicates silently and only a changed one
    reaches novelty analysis.
    """
    return _identity_digest(canonical_url(entry.url), entry.title.casefold(), revision_marker(entry))


def story_identity(entry: FeedEntry) -> str:
    """Which article this is, regardless of which revision of it.

    The coarser half of the pair: two entries share a story when they name the
    same canonical URL under the same headline, whatever wrapper, tracking
    parameters or provider they arrived through.
    """
    return compute_article_hash(entry.url, entry.title)


def compute_article_hash(url: str, title: str) -> str:
    """``story_identity`` for callers holding only a URL and a title.

    Same serialization, no revision marker — one recipe, not a second one.
    """
    return _identity_digest(canonical_url(url), title.casefold(), "")


def _representation_rank(entry: FeedEntry) -> tuple[datetime, int]:
    """Sort key deciding which of two entries for one URL is the better copy."""
    stamps = [s for s in (as_utc(entry.updated), as_utc(entry.published)) if s is not None]
    return (max(stamps) if stamps else _UNDATED, 1 if (entry.content or entry.summary).strip() else 0)


def collapse_duplicate_entries(entries: list[FeedEntry]) -> list[FeedEntry]:
    """One entry per story, keeping the best copy, in first-seen order.

    Repeats reach the pipeline routinely — two configured feeds carrying one
    story, an aggregator listing it twice. Left in, they occupy the bounded
    article slots that unique stories needed and pay for the same content fetch
    twice (AUG-179).

    The merge key is ``story_identity``, the same key dedup uses, and not the URL
    alone. A publisher republishing a correction at the story's own URL under a
    new headline is two stories to everything downstream, so keying the merge on
    the URL discarded one of them here — before dedup, extraction or analysis
    could see it — and which one survived was decided by feed order (AUG-322).

    Among copies that really are one story, the survivor is decided by the
    entries themselves — newest revision first, then the one that actually
    carries text — never by which feed was configured first.
    """
    best: dict[str, FeedEntry] = {}
    for entry in entries:
        key = story_identity(entry)
        current = best.get(key)
        # dict keeps the first insertion's position when a value is replaced,
        # so the surviving copy stays where the story first appeared.
        if current is None or _representation_rank(entry) > _representation_rank(current):
            best[key] = entry
    return list(best.values())


@dataclass(frozen=True)
class FeedFetchResult:
    """What one feed URL gave us: its entries and how the fetch went.

    Returned instead of ``(entries, ok)`` so callers stop re-deriving the outcome
    from ``len(entries)`` — the inference that made an unchanged feed look like an
    empty one (AUG-172) and an empty one look like nothing at all (AUG-173).
    """

    entries: list[FeedEntry] = field(default_factory=list)
    status: FetchStatus = FetchStatus.EMPTY


@dataclass(frozen=True)
class SourceIdentity:
    """Which source answered, and what the pipeline must do with its entries.

    ``needs_url_resolution`` is the one capability the pipeline branches on: only
    Google News hands back opaque redirects that need an async resolution pass.
    """

    name: str
    needs_url_resolution: bool = False


@dataclass
class FeedResponse:
    """Result of fetching feeds for a topic.

    Wraps the parsed entries with metadata about which provider was
    used, so downstream code can make provider-specific decisions
    (e.g. Google News URL resolution) without importing provider classes.

    ``feeds_total`` / ``feeds_failed`` expose per-fetch health so the check
    pipeline can distinguish a healthy partial yield from a degraded check where
    some sources silently dropped out (OVH-130). For AUTO mode a single provider
    is fetched (with at most one cascade), so the counts reflect that attempt;
    for MANUAL mode they count the topic's explicit feed URLs.
    """

    entries: list[FeedEntry] = field(default_factory=list)
    provider_name: str | None = None
    needs_url_resolution: bool = False
    feeds_total: int = 0
    feeds_failed: int = 0
    feeds_skipped: int = 0
    """MANUAL mode: feeds skipped this cycle because they are in a backoff window
    (persistently failing). For MANUAL mode ``feeds_total`` counts feeds ATTEMPTED
    (skipped feeds are excluded and surface here), so a backed-off feed is never
    miscounted as a partial failure."""

    @classmethod
    def from_source(
        cls,
        identity: SourceIdentity,
        *,
        entries: list[FeedEntry] | None = None,
        feeds_total: int = 0,
        feeds_failed: int = 0,
        feeds_skipped: int = 0,
        needs_url_resolution: bool | None = None,
    ) -> FeedResponse:
        """Build a response stamped with the identity and capabilities of its source.

        ``needs_url_resolution`` defaults to the source's own capability; a
        response with no entries overrides it to False, since there is nothing
        left to resolve.
        """
        return cls(
            entries=entries if entries is not None else [],
            provider_name=identity.name,
            needs_url_resolution=(
                identity.needs_url_resolution if needs_url_resolution is None else needs_url_resolution
            ),
            feeds_total=feeds_total,
            feeds_failed=feeds_failed,
            feeds_skipped=feeds_skipped,
        )


@dataclass(frozen=True)
class SourceRequest:
    """Everything a source fetcher needs for one attempt, besides the topic.

    One request object per logical fetch keeps every source's signature identical,
    which is what lets the registry dispatch without knowing who it is calling.
    Fields a given source does not use are simply ignored by it.
    """

    timeout: float
    deadline: Deadline = field(default_factory=Deadline.after)
    max_attempts: int = 2
    max_results: int = 10
    health_callback: FeedHealthCallback | None = None
    feed_state_loader: FeedStateLoader | None = None
    topic_holds_feed_articles: FeedArticleCheck | None = None
    backoff_base_minutes: int = BACKOFF_BASE_MINUTES
    backoff_cap_hours: int = BACKOFF_CAP_HOURS
    exa_settings: ExaSettings | None = None
    router: ProviderRouter | None = None


SourceFetcher = Callable[[Topic, SourceRequest], Awaitable[FeedResponse]]

_SOURCES: dict[FeedMode, SourceFetcher] = {}


def register_source(mode: FeedMode, fetcher: SourceFetcher) -> None:
    """Bind a feed mode to the coroutine that fetches it (called at import)."""
    _SOURCES[mode] = fetcher


async def fetch_feeds_for_topic(
    topic: Topic,
    timeout: float = 15.0,
    max_attempts: int = 2,
    health_callback: FeedHealthCallback | None = None,
    router: ProviderRouter | None = None,
    feed_state_loader: FeedStateLoader | None = None,
    topic_holds_feed_articles: FeedArticleCheck | None = None,
    backoff_base_minutes: int = BACKOFF_BASE_MINUTES,
    backoff_cap_hours: int = BACKOFF_CAP_HOURS,
    exa_settings: ExaSettings | None = None,
    max_results: int = 10,
    deadline: Deadline | None = None,
) -> FeedResponse:
    """Fetch all entries for a topic from whichever source its feed mode names.

    For AUTO mode: uses the router to select a provider, with within-cycle
    fallback (max 1 retry with the next provider). For MANUAL mode: fetches all
    explicit feed URLs concurrently, skipping any in a backoff window. For EXA
    mode: queries the Exa search API (``exa_settings`` required; ``max_results``
    bounds the paid result count).

    ``feed_state_loader`` supplies the stored ``FeedHealth`` per URL — used to
    send conditional-GET validators (both RSS modes) and to skip backed-off feeds
    (MANUAL only; AUTO provider backoff is owned by ``ProviderRouter``).
    ``topic_holds_feed_articles`` gates those validators on this topic actually
    holding something from the feed, so a shared 304 cannot leave it empty
    (TW-AUD-020).

    ``deadline`` bounds the whole attempt. Callers that own a larger unit of work
    (a topic check) pass theirs so the budget spans it; otherwise a fresh one is
    started here, because an unbounded source attempt is never wanted.
    """
    request = SourceRequest(
        timeout=timeout,
        deadline=deadline if deadline is not None else Deadline.after(),
        max_attempts=max_attempts,
        max_results=max_results,
        health_callback=health_callback,
        feed_state_loader=feed_state_loader,
        topic_holds_feed_articles=topic_holds_feed_articles,
        backoff_base_minutes=backoff_base_minutes,
        backoff_cap_hours=backoff_cap_hours,
        exa_settings=exa_settings,
        router=router,
    )
    return await _SOURCES[topic.feed_mode](topic, request)
