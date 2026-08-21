"""Pydantic data models for Topic Watch.

These models represent the core data structures used for validation,
data transfer between layers, and serialization to/from SQLite rows.
"""

import json
import logging
import re
import secrets
import sqlite3
import unicodedata
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import ClassVar, Self

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# Characters that draw nothing but keep two visually identical tags apart: the
# bidi formatting set, zero-width marks and the BOM. Stripped rather than
# escaped — a tag is a label the user picked, not untrusted output.
_TAG_INVISIBLE = re.compile(
    "["
    "؜"  # Arabic letter mark
    "​-‏"  # zero-width space/joiners, LRM, RLM
    "‪-‮"  # bidi embeddings and overrides
    "⁠-⁤"  # word joiner and invisible operators
    "⁦-⁩"  # bidi isolates
    "﻿"  # BOM / zero-width no-break space
    "]"
)


def normalize_tag(value: str) -> str:
    """Canonical form of one tag.

    NFC-normalized, invisible formatting characters removed, every remaining
    control or separator character folded to a space, and whitespace collapsed.
    Case is deliberately preserved: casefolding would rewrite every chip the user
    sees, and it is not needed to make canonically-equivalent variants of one tag
    resolve to the same identity (AUG-338).
    """
    text = _TAG_INVISIBLE.sub("", unicodedata.normalize("NFC", value))
    text = "".join(" " if unicodedata.category(ch) in ("Cc", "Cf", "Zl", "Zp") else ch for ch in text)
    return " ".join(text.split())


def normalize_tags(values: Iterable[str]) -> list[str]:
    """Canonicalize a tag list, drop blanks, and stable-deduplicate it.

    Form input, OPML folders and stored rows all pass through here, so one
    logical tag can no longer exist as several indistinguishable labels that
    filter differently (AUG-338). Order is the order first seen.
    """
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        tag = normalize_tag(value)
        if not tag or tag in seen:
            continue
        seen.add(tag)
        out.append(tag)
    return out


def to_utc(dt: datetime) -> datetime:
    """Return ``dt`` as an aware UTC datetime.

    The one rule for what a stored timestamp means, shared by the write side
    (``to_db_utc``) and the read side (``_coerce_dt``): a naive value is assumed
    to already be UTC, matching how the DB stores them, and an offset-carrying
    value names an instant that is re-expressed in UTC.
    """
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def to_db_utc(dt: datetime) -> str:
    """Serialize a datetime as canonical UTC TEXT with an explicit ``+00:00`` offset.

    Durable due-times and timestamps are compared and ordered as TEXT (``<=`` on
    a bare column so SQLite can use the index — see ``_DELETE_OLD_ARTICLES_SQL``).
    That is only correct when every writer spells the same instant identically, so
    a value carrying a local offset (``...+02:00``) would sort and compare wrong
    against a ``+00:00`` sibling despite naming the same moment.
    """
    return to_utc(dt).isoformat()


def new_generation() -> str:
    """Mint a topic's never-reused lifecycle identity.

    ``topics.id`` is a recyclable SQLite rowid: deleting a topic frees its id for
    the next INSERT, so a long-running check that captured only the id can wake up
    and write its results onto an unrelated replacement topic. The generation is
    captured alongside the id at snapshot time and re-checked before any durable
    write, which makes that stale apply a no-op. 8 random bytes, matching the
    ``lower(hex(randomblob(8)))`` backfill in migration 026.
    """
    return secrets.token_hex(8)


_RETRY_BASE_DELAY_SECONDS = 60.0
_RETRY_MAX_DELAY_SECONDS = 3600.0


def retry_delay_seconds(retry_count: int, *, hint_s: float | None = None) -> float:
    """How long to wait before the next delivery attempt.

    A receiver that stated its own recovery time (``Retry-After``) is believed,
    clamped to the same ceiling as the computed backoff so a hostile or mistaken
    header cannot park a delivery for a week. Otherwise: capped exponential
    backoff with jitter, so a whole batch of intents failing against the same
    down endpoint does not march back in lockstep.
    """
    if hint_s is not None and hint_s >= 0:
        return float(min(hint_s, _RETRY_MAX_DELAY_SECONDS))
    base = min(_RETRY_BASE_DELAY_SECONDS * (2.0 ** max(retry_count, 0)), _RETRY_MAX_DELAY_SECONDS)
    # secrets rather than random: not for secrecy, but because the S-rules ban
    # the non-cryptographic generator outright and this needs no seeding story.
    jitter = base * (secrets.randbelow(1000) / 10000.0)
    return base + jitter


def next_attempt_at(retry_count: int, *, hint_s: float | None = None, now: datetime | None = None) -> str:
    """Canonical UTC due-time for the next delivery attempt of a failed intent."""
    moment = now or datetime.now(UTC)
    return to_db_utc(moment + timedelta(seconds=retry_delay_seconds(retry_count, hint_s=hint_s)))


class CorruptTimestampError(ValueError):
    """A NOT NULL timestamp column holds something that is not a timestamp.

    Raised by the row->model boundary instead of substituting a value (TW-AUD-013).
    """


def _coerce_dt(value: object, field: str | None = None) -> datetime | None:
    """Parse a *nullable* DB datetime cell defensively, as aware UTC.

    Empty/whitespace-only strings and unparseable values become ``None`` rather
    than reaching Pydantic as a raw string and raising ``ValidationError`` on
    legacy/migrated/corrupt rows — a nullable column already has "no value" as a
    meaning it can carry, so degrading to it invents nothing. Non-empty text that
    does not parse is logged: it is stored state that reads as unset here while
    SQL still sees it as set, which is exactly the divergence AUG-144 describes.
    Code whose write path compares against the stored cell (the heartbeat latch)
    must read that cell raw rather than trust this value.

    Whatever comes back is aware UTC (``to_utc``): rows written before the
    canonical spelling existed hold naive text, and a naive datetime raises
    ``TypeError`` the moment it is compared with or added to an aware one
    (``feed_backoff_until`` vs ``datetime.now(UTC)``).
    """
    if isinstance(value, datetime):
        return to_utc(value)
    if isinstance(value, str):
        if not value.strip():
            return None
        try:
            return to_utc(datetime.fromisoformat(value))
        except (ValueError, TypeError):
            logger.warning("Unparseable datetime in column %r; reading it as unset", field or "<unknown>")
            return None
    return None


def _coerce_required_dt(value: object, field: str) -> datetime:
    """Parse a *required* datetime cell as aware UTC, or raise.

    A NOT NULL timestamp has no safe default: substituting ``now(UTC)`` gives the
    row a ``created_at``/``checked_at``/``fetched_at`` it never had, and that
    invented instant then drives scheduling, ordering, retention and what the UI
    reports — silently, and for good once anything writes the row back. Corruption
    here is a data state to surface, not one to paper over (TW-AUD-013).
    """
    parsed = _coerce_dt(value, field)
    if parsed is None:
        raise CorruptTimestampError(
            f"Required timestamp column {field!r} holds {value!r}, which is not a timestamp. "
            "The row is corrupt; repair or remove it (a backup lives beside the database)."
        )
    return parsed


def _safe_json(value: object, default: object, field: str) -> object:
    """Parse a JSON TEXT cell, returning ``default`` on malformed/empty input.

    Corruption is logged (mirroring ``_coerce_required_dt``) so a column that
    silently coerces to its empty default — e.g. ``feed_urls`` becoming ``[]``
    and quietly halting a topic's monitoring — leaves a diagnosable trace.
    Empty/NULL cells are treated as a benign default and are not logged.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return default
    if not isinstance(value, str):
        logger.warning("Non-string JSON cell for %s (%r); using default", field, type(value).__name__)
        return default
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError):
        logger.warning("Corrupt JSON in %s cell (%r); using default", field, value)
        return default
    if type(parsed) is not type(default):
        logger.warning("Unexpected JSON type %s for %s; using default", type(parsed).__name__, field)
        return default
    return parsed


class SQLiteModel(BaseModel):
    """Base for models persisted to SQLite, factoring out the row<->model interop.

    SQLite stores booleans as INTEGER (0/1), datetimes as ISO-8601 TEXT, and JSON
    arrays/objects as TEXT. Every persisted model otherwise re-implemented the
    same coercion boilerplate in its own ``from_row`` / ``to_insert_dict`` (OVH-150).
    Subclasses declare the columns needing each coercion; the shared
    ``_coerce_row`` / ``_dump_for_insert`` helpers apply them, with custom per-model
    logic (e.g. Topic's check-interval backcompat, CheckResult's derived
    confidence) layered on top.

    Class-level declarations (override per subclass as needed):

    * ``_bool_fields``: columns stored as 0/1 INTEGER <-> ``bool``.
    * ``_required_dt_fields``: NOT NULL datetime columns (corrupt/empty raises
      ``CorruptTimestampError``, via ``_coerce_required_dt``).
    * ``_optional_dt_fields``: nullable datetime columns (corrupt/empty -> None,
      via ``_coerce_dt``). Both hydrate as aware UTC.
    * ``_json_fields``: mapping of column name -> empty default (list/dict) for
      JSON TEXT columns coerced via ``_safe_json``.
    * ``_insert_exclude``: extra field names dropped from ``to_insert_dict`` beyond
      the always-excluded ``id``.
    """

    _bool_fields: ClassVar[tuple[str, ...]] = ()
    _required_dt_fields: ClassVar[tuple[str, ...]] = ()
    _optional_dt_fields: ClassVar[tuple[str, ...]] = ()
    _json_fields: ClassVar[dict[str, object]] = {}
    _insert_exclude: ClassVar[frozenset[str]] = frozenset()

    @classmethod
    def _coerce_row(cls, row: sqlite3.Row) -> dict:
        """Return a model-ready dict from a DB row, applying the declared coercions.

        Operates on a copy of the row (never the row itself). Subclasses needing
        extra handling call this first, then adjust the dict before constructing.
        """
        data = dict(row)
        for field in cls._json_fields:
            data[field] = _safe_json(data.get(field), cls._json_fields[field], field)
        for field in cls._bool_fields:
            if field in data:
                data[field] = bool(data[field])
        for field in cls._required_dt_fields:
            data[field] = _coerce_required_dt(data.get(field), field)
        for field in cls._optional_dt_fields:
            data[field] = _coerce_dt(data.get(field), field)
        return data

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Self:
        """Construct the model from a database row using the declared coercions."""
        return cls(**cls._coerce_row(row))

    def _dump_for_insert(self) -> dict:
        """Return a model_dump dict ready for SQL INSERT (shared serialization).

        Excludes ``id`` plus any ``_insert_exclude`` fields, then serializes the
        declared bool/datetime/JSON columns back to their SQLite storage forms.
        StrEnum values are emitted as their ``.value`` string.
        """
        d = self.model_dump(exclude={"id", *self._insert_exclude})
        for field in self._json_fields:
            if field in d:
                d[field] = json.dumps(d[field])
        for field in self._bool_fields:
            if field in d:
                d[field] = int(d[field])
        for field in (*self._required_dt_fields, *self._optional_dt_fields):
            if d.get(field) is not None:
                d[field] = d[field].isoformat()
        for field, value in list(d.items()):
            if isinstance(value, StrEnum):
                d[field] = value.value
        return d

    def to_insert_dict(self) -> dict:
        """Return a dict for SQL INSERT (excludes auto-generated id)."""
        return self._dump_for_insert()


class TopicStatus(StrEnum):
    """Status of a topic's lifecycle."""

    NEW = "new"
    RESEARCHING = "researching"
    READY = "ready"
    ERROR = "error"


class FeedMode(StrEnum):
    """How a topic resolves its feed URLs."""

    AUTO = "auto"
    MANUAL = "manual"
    EXA = "exa"


class KnowledgeRevisionSource(StrEnum):
    """What produced a knowledge revision.

    ``UNKNOWN`` is never written: it is what a stored value this version does not
    recognise degrades to, so a row from a newer version, a restored backup or a
    hand edit is labelled honestly instead of being passed off as an ordinary
    update (AUG-155).
    """

    INIT = "init"
    UPDATE = "update"
    UNKNOWN = "unknown"


# Cap for the per-topic novelty instruction (free text injected into the novelty
# prompt). Single source of truth: enforced at the form boundary
# (``parse_novelty_instruction``) and rendered as the textarea ``maxlength``.
NOVELTY_INSTRUCTION_MAX_CHARS = 500


class Topic(SQLiteModel):
    """A monitored topic with associated feed URLs."""

    _bool_fields = ("is_active",)
    _required_dt_fields = ("created_at",)
    _optional_dt_fields = ("status_changed_at", "heartbeat_alerted_at")
    _json_fields = {"feed_urls": [], "tags": []}  # noqa: RUF012 - declarative

    id: int | None = None
    name: str
    description: str
    feed_urls: list[str] = Field(default_factory=list)
    feed_mode: FeedMode = FeedMode.AUTO
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    status_changed_at: datetime | None = None
    is_active: bool = True
    status: TopicStatus = TopicStatus.RESEARCHING
    error_message: str | None = None
    check_interval_minutes: int | None = None
    tags: list[str] = Field(default_factory=list)
    confidence_threshold: float | None = None
    relevance_threshold: float | None = None
    novelty_instruction: str | None = None
    importance_threshold: int | None = None
    init_attempts: int = 0
    # Silence Heartbeat latch: when the checker last announced that this topic's
    # sources are failing. NULL = no outstanding alert. Written only by
    # ``crud.claim_heartbeat_alert`` / ``clear_heartbeat_alert`` and intentionally
    # excluded from the create_topic/update_topic column lists (mirroring
    # CheckResult.seen_at), so a topic edit carrying a stale Topic cannot reset it.
    heartbeat_alerted_at: datetime | None = None
    # Never-reused lifecycle identity (see ``new_generation``). Written once by
    # ``crud.create_topic`` and deliberately absent from ``update_topic``'s column
    # list, so an edit carrying a stale Topic can never rotate or blank it.
    generation: str = Field(default_factory=new_generation)

    @field_validator("tags", mode="after")
    @classmethod
    def _canonicalize_tags(cls, value: list[str]) -> list[str]:
        """Canonicalize and deduplicate tags on every construction (AUG-338).

        Applied on the read path too, so a row written before this existed still
        renders and filters as one identity. Never raises: like the clamping
        validators above, a bad stored value degrades instead of 500-ing a page.
        """
        return normalize_tags(value)

    @field_validator("confidence_threshold", "relevance_threshold", mode="before")
    @classmethod
    def _clamp_threshold(cls, value: object, info: object) -> object:
        """Clamp per-topic thresholds into [0.0, 1.0] (OVH-107).

        Validation otherwise lives only at the form boundary (``parse_threshold``).
        A value outside [0.0, 1.0] reaching a topic row — via a manual DB edit,
        restore, or a future write path that skips that helper — would make
        ``novelty.confidence < confidence_threshold`` always true (or always
        false), silently suppressing ALL notifications for the topic. Clamp (and
        warn) rather than raise so loading a corrupt row degrades gracefully
        instead of 500-ing the route, matching the defensive ``from_row`` layer.
        """
        if value is None:
            return None
        try:
            parsed = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return value  # let Pydantic raise its standard type error
        if parsed < 0.0 or parsed > 1.0:
            field_name = getattr(info, "field_name", "threshold")
            clamped = min(max(parsed, 0.0), 1.0)
            logger.warning("Out-of-range %s %r clamped to %s", field_name, parsed, clamped)
            return clamped
        return parsed

    @field_validator("importance_threshold", mode="before")
    @classmethod
    def _clamp_importance_threshold(cls, value: object) -> object:
        """Clamp the per-topic importance threshold into [1, 5].

        Mirrors ``_clamp_threshold``: an out-of-range value reaching a topic row
        (manual DB edit, restore, or a write path that skips ``parse_importance``)
        would make ``novelty.importance >= importance_threshold`` always false,
        silently suppressing ALL notifications for the topic. Clamp and warn
        rather than raise so loading a corrupt row degrades gracefully.

        A FRACTIONAL value is the one thing not clamped but rejected (AUG-156).
        SQLite's INTEGER affinity keeps a REAL 4.9 as 4.9, and ``int()`` turned it
        into a silent 4 — a different, broader threshold than anything a user
        could have chosen, which a later unrelated save then persisted for good.
        Importance is a whole-number 1-to-5 scale, so there is no correct value to
        clamp a fraction to; handing it back untouched lets Pydantic reject it the
        same way it rejects any other non-integer.
        """
        if value is None:
            return None
        try:
            numeric = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return value  # let Pydantic raise its standard type error
        if not numeric.is_integer():
            logger.warning("Fractional importance_threshold %r rejected (the scale is whole numbers 1-5)", value)
            return value  # let Pydantic raise its standard integer error
        parsed = int(numeric)
        if parsed < 1 or parsed > 5:
            clamped = min(max(parsed, 1), 5)
            logger.warning("Out-of-range importance_threshold %r clamped to %s", parsed, clamped)
            return clamped
        return parsed

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Self:
        """Construct a Topic from a database row."""
        data = cls._coerce_row(row)
        # Backwards compatibility for a row shape that predates the minute column:
        # convert the legacy hours only when the current column is ABSENT, never
        # when it is present and NULL (AUG-145). m008 copied the legacy value into
        # minutes but kept the old column populated, so treating NULL as "missing"
        # made an intentionally CLEARED override resurrect the stale hours — and a
        # later unrelated edit wrote that obsolete schedule back. NULL in the
        # current column means "inherit the global interval", which is the only
        # value a cleared field can have.
        if "check_interval_minutes" not in data and data.get("check_interval_hours") is not None:
            data["check_interval_minutes"] = data["check_interval_hours"] * 60
        data.pop("check_interval_hours", None)
        return cls(**data)


class Article(SQLiteModel):
    """A fetched article associated with a topic."""

    _bool_fields = ("processed",)
    _required_dt_fields = ("fetched_at",)
    _optional_dt_fields = ("published_at",)

    id: int | None = None
    topic_id: int
    title: str
    url: str
    content_hash: str
    raw_content: str | None = None
    source_feed: str
    source_provider: str | None = None
    published_at: datetime | None = None
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    processed: bool = False
    # How many checks have tried and failed to analyze this article. ``processed``
    # says the article is done with; this says how much of that was wasted effort,
    # and caps the retries so one undecodable row cannot ride along in every future
    # prompt (see ``crud.record_article_analysis_failure``).
    analysis_attempts: int = 0


class KnowledgeState(SQLiteModel):
    """Rolling summary of everything known about a topic."""

    _required_dt_fields = ("updated_at",)

    id: int | None = None
    topic_id: int
    summary_text: str
    token_count: int = 0
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # Compare-and-swap counter, bumped by every knowledge write. A check snapshots
    # the version before its (conn-free) LLM phase and the write is rejected when
    # the row moved meanwhile, so two overlapping checks cannot lose an update.
    version: int = 0


class KnowledgeRevision(SQLiteModel):
    """One recorded snapshot of a topic's knowledge state.

    History beside ``knowledge_states``: rows are appended on write and pruned
    oldest-first, never rewritten. The checker never reads them, so a corrupt or
    pruned revision cannot affect novelty detection. ``change_note`` is the
    novelty summary that prompted an update (NULL for 'init' revisions and for
    updates where the LLM returned no summary).
    """

    _required_dt_fields = ("created_at",)

    id: int | None = None
    topic_id: int
    summary_text: str
    token_count: int = 0
    source: KnowledgeRevisionSource = KnowledgeRevisionSource.UPDATE
    change_note: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # Provenance, NULL on every row written before migration 029 and on the m025
    # backfill. ``model`` is the configured LLM that wrote this summary and, being
    # the same string ``count_tokens`` was called with, the identity of the unit
    # ``token_count`` is measured in — two revisions counted under different
    # models are not subtractable (AUG-255). ``basis_hash`` fingerprints the
    # topic scope the summary was derived from, so a later scope edit is
    # detectable rather than silently baked in (TW-AUD-017).
    model: str | None = None
    basis_hash: str | None = None

    @field_validator("source", mode="before")
    @classmethod
    def _coerce_source(cls, value: object) -> object:
        """Degrade an unrecognised stored ``source`` to UNKNOWN instead of raising.

        Mirrors ``Topic._clamp_importance_threshold``: a value written by a
        future version, restored from an old backup, or hand-edited in sqlite3
        must not raise ValidationError and 500 the topic detail page over a
        badge label. It must not be relabelled ``update`` either — that is a
        plausible, false lineage the diff view would then compare adjacently
        (AUG-155). The ``isinstance`` check is load-bearing for mypy, whose enum
        plugin types ``StrEnum.__call__`` as ``(value: str)``.
        """
        if isinstance(value, KnowledgeRevisionSource):
            return value
        if not isinstance(value, str):
            logger.warning("Non-string knowledge revision source %r; treating as 'unknown'", value)
            return KnowledgeRevisionSource.UNKNOWN
        try:
            return KnowledgeRevisionSource(value)
        except ValueError:
            logger.warning("Unrecognised knowledge revision source %r; treating as 'unknown'", value)
            return KnowledgeRevisionSource.UNKNOWN


# ``check_results.stage_error`` vocabulary, written by app/checker.py. Collected
# here so the pipeline, the Silence Heartbeat and the templates classify a check
# the same way instead of each re-deriving the prefixes.
#
# These three mean "this check could not see the news".
# ``analysis_failed`` / ``knowledge_update_failed`` are deliberately excluded:
# articles were fetched, so the sources themselves are healthy.
SOURCE_FAILURE_PREFIXES: tuple[str, ...] = (
    "sources_failed:",
    "scrape_failed:",
    "sources_unavailable:",
)

# The check broke inside Topic Watch — storage, deduplication, extraction — so it
# never learned anything about the sources at all. Visible on the row, but neutral
# to the Silence Heartbeat: counting it as an outage sends the operator hunting
# for a dead feed or a bad API key, and counting it as health would announce a
# recovery nobody observed (AUG-133).
INTERNAL_FAILURE_PREFIXES: tuple[str, ...] = ("pipeline_failed:",)


def is_source_failure(stage_error: str | None) -> bool:
    """True when a recorded stage_error means no source produced usable results."""
    return stage_error is not None and stage_error.startswith(SOURCE_FAILURE_PREFIXES)


def is_internal_failure(stage_error: str | None) -> bool:
    """True when a check failed inside the pipeline without observing its sources."""
    return stage_error is not None and stage_error.startswith(INTERNAL_FAILURE_PREFIXES)


class NotifyDisposition(StrEnum):
    """Why a check did (or did not) notify — ``check_results.notify_disposition``.

    Distinct from ``notification_error``, which only says a delivery failed: this
    records the pipeline's own decision, so "we chose not to send" is never
    indistinguishable from "we sent and it worked".
    """

    SENT = "sent"
    PENDING = "pending"
    PENDING_KNOWLEDGE_STALE = "pending_knowledge_stale"
    """Notifying, but the knowledge state behind the alert did not advance.

    The merge was refused as too vague to apply (``knowledge_insufficient``) or it
    raised (``knowledge_update_failed``) — ``stage_error`` says which. The alert is
    still worth sending, so recording it as an ordinary send would claim the
    baseline absorbed evidence it never saw (TW-AUD-003).
    """
    NO_NEW_INFO = "no_new_info"
    BELOW_CONFIDENCE = "below_confidence"
    BELOW_RELEVANCE = "below_relevance"
    SUPPRESSED_IMPORTANCE = "suppressed_importance"
    ANALYSIS_FAILED = "analysis_failed"


class CheckResult(SQLiteModel):
    """Record of a single check cycle for a topic."""

    _bool_fields = ("has_new_info", "notification_sent")
    _required_dt_fields = ("checked_at",)
    # ``seen_at`` is nullable: registering it here makes the shared ``_coerce_row``
    # populate it on the ``from_row`` path, so BOTH render paths — the dashboard
    # (``from_dashboard_row``) and the HTMX row re-render (``_topic_row_context`` ->
    # ``list_check_results`` -> ``from_row``) — honor the badge gate. Do not drop it.
    _optional_dt_fields = ("seen_at",)
    # ``confidence``/``importance`` are derived from llm_response, not real
    # columns — never persist them (OVH-052, AUG-037).
    _insert_exclude = frozenset({"confidence", "importance"})

    id: int | None = None
    topic_id: int
    checked_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    articles_found: int = 0
    articles_new: int = 0
    has_new_info: bool = False
    llm_response: str | None = None
    notification_sent: bool = False
    notification_error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Machine-distinguishable failure stage for an otherwise-recorded check:
    # 'sources_failed' / 'sources_unavailable' / 'pipeline_failed' /
    # 'analysis_failed' / 'knowledge_insufficient' / 'knowledge_update_failed'
    # (+ a short exception summary). NULL on clean runs. Distinct from
    # notification_error, which only covers delivery. 'scrape_failed' is retained
    # in SOURCE_FAILURE_PREFIXES for rows written before AUG-133 split internal
    # failures out of it.
    stage_error: str | None = None
    # Why this check did or did not notify (see ``NotifyDisposition``). NULL on
    # rows written before migration 026.
    notify_disposition: str | None = None
    # When the user first opened a topic whose latest check carried new info. NULL
    # = unseen. Gates only the dashboard "new info" badge (has_new_info AND
    # seen_at IS NULL); ``has_new_info`` itself is never mutated, so the detail-page
    # history column and Notify button are unaffected. Intentionally omitted from
    # the create_check_result INSERT so new rows are born NULL/unseen.
    seen_at: datetime | None = None
    # Non-persisted: confidence scalar extracted from llm_response. The dashboard
    # listing populates this via SQL ``json_extract`` so it can render the
    # confidence badge WITHOUT shipping/parsing the full llm_response blob per
    # topic (OVH-052). Never written back to the DB (excluded from inserts).
    confidence: float | None = None
    # Non-persisted: the 1-5 importance scalar, derived from the same single blob
    # decode as ``confidence``. ``None`` when the blob is missing, unparseable, or
    # predates m023. The dashboard listing never reads it.
    importance: int | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Self:
        """Construct a CheckResult from a database row."""
        data = cls._coerce_row(row)
        # ``confidence`` is a derived, non-column field; drop any stray DB key so
        # it is only ever set explicitly here from the loaded blob.
        data.pop("confidence", None)
        # Derive confidence from the already-loaded blob on the single-row paths
        # (detail/history) so the badge renders without a second parse. The
        # dashboard path skips the blob entirely and sets ``confidence`` via SQL
        # json_extract (OVH-052).
        data.pop("importance", None)
        data["confidence"], data["importance"] = cls._scalars_from_blob(data.get("llm_response"))
        return cls(**data)

    @staticmethod
    def _scalars_from_blob(llm_response: object) -> tuple[float | None, int | None]:
        """Extract the confidence and importance scalars from one blob decode.

        Both are rendered per row by the check-history table. Decoding the blob
        once here and handing the template two scalars replaces the three full
        ``json.loads`` calls a displayed row used to cost — model hydration plus
        one template filter each (AUG-037).
        """
        if not isinstance(llm_response, str) or not llm_response:
            return None, None
        try:
            data = json.loads(llm_response)
        except json.JSONDecodeError:
            return None, None
        # json.loads() also accepts arrays, scalars, booleans and null; only a
        # dict has keys to read (AUG-215).
        if not isinstance(data, dict):
            return None, None

        confidence: float | None
        try:
            confidence = float(data["confidence"])
        except (KeyError, TypeError, ValueError):
            confidence = None
        importance: int | None
        try:
            importance = int(data["importance"])
        except (KeyError, TypeError, ValueError):
            importance = None
        return confidence, importance

    @classmethod
    def from_dashboard_row(cls, row: sqlite3.Row, topic_id: int) -> Self:
        """Build the partial CheckResult the dashboard listing carries (OVH-151).

        The dashboard SELECT joins each topic to its latest check via ``cr_``-
        prefixed aliases and pre-extracts ``confidence`` with SQL ``json_extract``,
        so the full ``llm_response`` blob is never shipped/parsed per topic
        (OVH-052). This maps those aliases to the model, routing the required
        ``checked_at`` through the same coercion ``from_row`` uses (OVH-108) — a
        legacy/naive cell normalizes, a corrupt one raises rather than reporting an
        invented check time (TW-AUD-013). ``llm_response`` is intentionally left
        ``None`` on this path.
        """
        return cls(
            id=row["cr_id"],
            topic_id=topic_id,
            checked_at=_coerce_required_dt(row["cr_checked_at"], "checked_at"),
            articles_found=row["cr_articles_found"],
            articles_new=row["cr_articles_new"],
            has_new_info=bool(row["cr_has_new_info"]),
            llm_response=None,
            confidence=row["cr_confidence"],
            notification_sent=bool(row["cr_notification_sent"]),
            notification_error=row["cr_notification_error"],
            # Paired with the ``cr.stage_error AS cr_stage_error`` alias in
            # _DASHBOARD_SELECT; drives the dashboard's failing-sources badge.
            stage_error=row["cr_stage_error"],
            # Paired with the ``cr.seen_at AS cr_seen_at`` alias in _DASHBOARD_SELECT;
            # one without the other 500s the dashboard.
            seen_at=_coerce_dt(row["cr_seen_at"], "seen_at"),
        )

    @property
    def sources_failing(self) -> bool:
        """True when this check saw no usable source (fetch failed, or none attempted)."""
        return is_source_failure(self.stage_error)


class FeedHealth(SQLiteModel):
    """Health tracking for a single feed URL.

    OVH-150: the nullable datetime cells (``last_success_at`` / ``last_error_at``)
    are coerced through the shared ``_coerce_dt`` path like every other model
    instead of an inlined copy that had drifted from it.
    """

    _optional_dt_fields = ("last_success_at", "last_error_at")

    id: int | None = None
    feed_url: str
    last_success_at: datetime | None = None
    last_error_at: datetime | None = None
    last_error_message: str | None = None
    consecutive_failures: int = 0
    total_fetches: int = 0
    total_failures: int = 0
    etag: str | None = None
    last_modified: str | None = None


class DashboardStats(BaseModel):
    """Aggregate statistics for the dashboard."""

    total_topics: int = 0
    active_topics: int = 0
    checks_24h: int = 0
    checks_total: int = 0
    new_info_24h: int = 0
    new_info_total: int = 0
    last_notification_at: datetime | None = None


class DeliveryStatus(StrEnum):
    """Lifecycle of one delivery intent.

    ``pending`` -> ``sending`` (claimed) -> ``sent`` | ``abandoned``, or
    ``revoked`` when the event the intent announces stopped being true before it
    left. ``sent`` rows are kept: they are the durable delivery ledger the
    dashboard's "last notified" reads, and retention prunes them on the daily
    maintenance tick like any other aged row.
    """

    PENDING = "pending"
    SENDING = "sending"
    SENT = "sent"
    ABANDONED = "abandoned"
    REVOKED = "revoked"


class NotificationKind(StrEnum):
    """What a notification intent announces.

    The kind travels with the row so the drain can tell a novelty alert from a
    heartbeat transition, and so a superseded outage alert can be revoked by kind
    instead of being sent after the recovery it contradicts.
    """

    NOVELTY = "novelty"
    HEARTBEAT_ALERT = "heartbeat_alert"
    HEARTBEAT_RECOVERY = "heartbeat_recovery"


class PendingNotification(SQLiteModel):
    """One durable per-target notification delivery intent.

    Created inside the check's durable transaction, BEFORE any send begins
    (TW-AUD-004): a crash between "we decided to notify" and "the message left"
    used to be indistinguishable from "we never tried", so the alert was simply
    lost. The intent survives that window and the next drain finishes it.

    Scoped to a single ``url`` (OVH-039) so retry re-hits exactly the target that
    owes a delivery. ``url`` is NULL on legacy whole-batch rows, in which case the
    drain falls back to every configured URL. ``claim_token`` is the immutable
    owner stamped by the winning claim; every result apply is fenced by it, so a
    worker whose claim was released as stale cannot mutate the row its successor
    now owns (AUG-277). ``next_attempt_at`` is a canonical UTC due-time, so a
    backoff survives a restart instead of collapsing to "retry immediately".
    """

    _required_dt_fields = ("created_at",)
    _insert_exclude = frozenset({"claimed_at", "claim_token", "delivered_at"})

    id: int | None = None
    topic_id: int
    check_result_id: int | None = None
    title: str
    body: str
    url: str | None = None
    last_error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    retry_count: int = 0
    max_retries: int = 3
    claimed_at: str | None = None
    status: DeliveryStatus = DeliveryStatus.PENDING
    kind: NotificationKind = NotificationKind.NOVELTY
    claim_token: str | None = None
    next_attempt_at: str | None = None
    latch_value: str | None = None
    """The heartbeat latch this intent belongs to (A5); NULL for novelty alerts."""
    delivered_at: str | None = None


class NotificationDelivery(BaseModel):
    """Per-URL outcome of a notification delivery attempt.

    Lets the pipeline re-queue only the targets that failed (OVH-039) and
    surface a per-channel reason without leaking the raw URL (OVH-027).

    ``timed_out`` is the third outcome, and it is not a failure: a thread running
    Apprise cannot be cancelled, so a target that outlived its deadline may still
    be delivering. Recording it as failed would queue a retry for a message the
    user is about to receive (AUG-071); the intent is left claimed instead, and
    the stale-claim release re-arms it once the deadline for that has passed.
    """

    url: str
    ok: bool
    error: str | None = None
    timed_out: bool = False
    intent_id: int | None = None
    """Delivery intent this outcome belongs to, when it came from one."""


class PendingWebhook(SQLiteModel):
    """A webhook delivery that failed to send and should be retried.

    Mirrors ``PendingNotification`` for the webhook retry queue. The outbound
    ``payload`` is stored as a JSON TEXT column.

    OVH-110: ``payload`` is coerced via the shared ``_safe_json`` path, which
    falls back to ``{}`` (with a warning) not only on malformed JSON but also on
    valid JSON of the wrong type — an array/scalar/string whose ``type(parsed) is
    not type(default)``. So a payload that was ever a non-dict (manual edit,
    partial corruption, future path) degrades here instead of raising
    ValidationError and 500-ing the retry-queue view or crashing the retry worker.
    """

    _required_dt_fields = ("created_at",)
    _json_fields = {"payload": {}}  # noqa: RUF012 - declarative
    _insert_exclude = frozenset({"claimed_at", "claim_token", "delivered_at"})

    id: int | None = None
    topic_id: int
    check_result_id: int | None = None
    url: str
    payload: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    retry_count: int = 0
    max_retries: int = 3
    claimed_at: str | None = None
    status: DeliveryStatus = DeliveryStatus.PENDING
    claim_token: str | None = None
    next_attempt_at: str | None = None
    last_error: str | None = None
    delivered_at: str | None = None
