"""Pydantic data models for Topic Watch.

These models represent the core data structures used for validation,
data transfer between layers, and serialization to/from SQLite rows.
"""

import json
import logging
import sqlite3
from datetime import UTC, datetime
from enum import StrEnum
from typing import ClassVar, Self

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


def _coerce_dt(value: object) -> datetime | None:
    """Parse a DB datetime cell defensively.

    Mirrors ``FeedHealth.from_row``: empty/whitespace-only strings and
    unparseable values become ``None`` rather than reaching Pydantic as a raw
    string and raising ``ValidationError`` on legacy/migrated/corrupt rows.
    Already-parsed ``datetime`` instances pass through untouched.
    """
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        if not value.strip():
            return None
        try:
            return datetime.fromisoformat(value)
        except (ValueError, TypeError):
            return None
    return None


def _coerce_required_dt(value: object) -> datetime:
    """Parse a *required* datetime cell, defaulting to now(UTC) when corrupt.

    Required datetime columns cannot be ``None``. A single bad/empty cell must
    still not 500 the route that loads it, so fall back to the current time.
    """
    parsed = _coerce_dt(value)
    if parsed is None:
        if value is None:
            logger.warning("Required datetime is NULL in DB row; defaulting to now(UTC)")
        elif value == "":
            logger.warning("Required datetime is empty string in DB row; defaulting to now(UTC)")
        else:
            logger.warning("Corrupt required datetime %r in DB row; defaulting to now(UTC)", value)
        return datetime.now(UTC)
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
    * ``_required_dt_fields``: NOT NULL datetime columns (corrupt/empty -> now(UTC),
      via ``_coerce_required_dt``).
    * ``_optional_dt_fields``: nullable datetime columns (corrupt/empty -> None,
      via ``_coerce_dt``).
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
            data[field] = _coerce_required_dt(data.get(field))
        for field in cls._optional_dt_fields:
            data[field] = _coerce_dt(data.get(field))
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


class Topic(SQLiteModel):
    """A monitored topic with associated feed URLs."""

    _bool_fields = ("is_active",)
    _required_dt_fields = ("created_at",)
    _optional_dt_fields = ("status_changed_at",)
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
    init_attempts: int = 0

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

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Self:
        """Construct a Topic from a database row."""
        data = cls._coerce_row(row)
        # Backwards compatibility: if check_interval_minutes is absent but
        # check_interval_hours is present, convert hours to minutes.
        if data.get("check_interval_minutes") is None and data.get("check_interval_hours") is not None:
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


class KnowledgeState(SQLiteModel):
    """Rolling summary of everything known about a topic."""

    _required_dt_fields = ("updated_at",)

    id: int | None = None
    topic_id: int
    summary_text: str
    token_count: int = 0
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CheckResult(SQLiteModel):
    """Record of a single check cycle for a topic."""

    _bool_fields = ("has_new_info", "notification_sent")
    _required_dt_fields = ("checked_at",)
    # ``seen_at`` is nullable: registering it here makes the shared ``_coerce_row``
    # populate it on the ``from_row`` path, so BOTH render paths — the dashboard
    # (``from_dashboard_row``) and the HTMX row re-render (``_topic_row_context`` ->
    # ``list_check_results`` -> ``from_row``) — honor the badge gate. Do not drop it.
    _optional_dt_fields = ("seen_at",)
    # ``confidence`` is derived from llm_response, not a real column — never persist it (OVH-052).
    _insert_exclude = frozenset({"confidence"})

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
    # 'scrape_failed' / 'analysis_failed' / 'knowledge_update_failed' (+ a short
    # exception summary). NULL on clean runs. Distinct from notification_error,
    # which only covers delivery.
    stage_error: str | None = None
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
        data["confidence"] = cls._confidence_from_blob(data.get("llm_response"))
        return cls(**data)

    @staticmethod
    def _confidence_from_blob(llm_response: object) -> float | None:
        """Extract the confidence scalar from an llm_response JSON blob."""
        if not isinstance(llm_response, str) or not llm_response:
            return None
        try:
            value = json.loads(llm_response).get("confidence")
        except (json.JSONDecodeError, AttributeError):
            return None
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    @classmethod
    def from_dashboard_row(cls, row: sqlite3.Row, topic_id: int) -> Self:
        """Build the partial CheckResult the dashboard listing carries (OVH-151).

        The dashboard SELECT joins each topic to its latest check via ``cr_``-
        prefixed aliases and pre-extracts ``confidence`` with SQL ``json_extract``,
        so the full ``llm_response`` blob is never shipped/parsed per topic
        (OVH-052). This maps those aliases to the model, routing the required
        ``checked_at`` through the same defensive coercion ``from_row`` uses
        (OVH-108) so a corrupt/legacy cell degrades to now(UTC) instead of 500-ing
        the dashboard. ``llm_response`` is intentionally left ``None`` on this path.
        """
        return cls(
            id=row["cr_id"],
            topic_id=topic_id,
            checked_at=_coerce_required_dt(row["cr_checked_at"]),
            articles_found=row["cr_articles_found"],
            articles_new=row["cr_articles_new"],
            has_new_info=bool(row["cr_has_new_info"]),
            llm_response=None,
            confidence=row["cr_confidence"],
            notification_sent=bool(row["cr_notification_sent"]),
            notification_error=row["cr_notification_error"],
            # Paired with the ``cr.seen_at AS cr_seen_at`` alias in _DASHBOARD_SELECT;
            # one without the other 500s the dashboard.
            seen_at=_coerce_dt(row["cr_seen_at"]),
        )


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


class PendingNotification(SQLiteModel):
    """A notification that failed to send and should be retried.

    Scoped to a single ``url`` when that target failed (OVH-039): a partial
    batch failure queues one row per failed URL so retry never re-hits the
    targets that already delivered. ``url`` is NULL on legacy/whole-batch rows,
    in which case the drain falls back to every configured URL. ``last_error``
    records the most recent failure reason for operator diagnostics.
    """

    _required_dt_fields = ("created_at",)
    _insert_exclude = frozenset({"claimed_at"})

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


class NotificationDelivery(BaseModel):
    """Per-URL outcome of a notification delivery attempt.

    Lets the pipeline re-queue only the targets that failed (OVH-039) and
    surface a per-channel reason without leaking the raw URL (OVH-027).
    """

    url: str
    ok: bool
    error: str | None = None


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
    _insert_exclude = frozenset({"claimed_at"})

    id: int | None = None
    topic_id: int
    check_result_id: int | None = None
    url: str
    payload: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    retry_count: int = 0
    max_retries: int = 3
    claimed_at: str | None = None
