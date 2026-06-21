"""Pydantic data models for Topic Watch.

These models represent the core data structures used for validation,
data transfer between layers, and serialization to/from SQLite rows.
"""

import json
import logging
import sqlite3
from datetime import UTC, datetime
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, Field

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


def _safe_json(value: object, default: object) -> object:
    """Parse a JSON TEXT cell, returning ``default`` on malformed/empty input."""
    if not isinstance(value, str) or not value.strip():
        return default
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return default


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


class Topic(BaseModel):
    """A monitored topic with associated feed URLs."""

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

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Self:
        """Construct a Topic from a database row."""
        data = dict(row)
        data["feed_urls"] = _safe_json(data.get("feed_urls"), [])
        data["is_active"] = bool(data["is_active"])
        data["tags"] = _safe_json(data.get("tags"), [])
        data["created_at"] = _coerce_required_dt(data.get("created_at"))
        data["status_changed_at"] = _coerce_dt(data.get("status_changed_at"))
        # Backwards compatibility: if check_interval_minutes is absent but
        # check_interval_hours is present, convert hours to minutes.
        if data.get("check_interval_minutes") is None and data.get("check_interval_hours") is not None:
            data["check_interval_minutes"] = data["check_interval_hours"] * 60
        data.pop("check_interval_hours", None)
        return cls(**data)

    def to_insert_dict(self) -> dict:
        """Return a dict for SQL INSERT (excludes auto-generated id)."""
        d = self.model_dump(exclude={"id"})
        d["feed_urls"] = json.dumps(d["feed_urls"])
        d["tags"] = json.dumps(d["tags"])
        d["feed_mode"] = d["feed_mode"].value
        d["created_at"] = d["created_at"].isoformat()
        d["status"] = d["status"].value
        d["is_active"] = int(d["is_active"])
        if d["status_changed_at"] is not None:
            d["status_changed_at"] = d["status_changed_at"].isoformat()
        return d


class Article(BaseModel):
    """A fetched article associated with a topic."""

    id: int | None = None
    topic_id: int
    title: str
    url: str
    content_hash: str
    raw_content: str | None = None
    source_feed: str
    source_provider: str | None = None
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    processed: bool = False

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Self:
        """Construct an Article from a database row."""
        data = dict(row)
        data["processed"] = bool(data["processed"])
        data["fetched_at"] = _coerce_required_dt(data.get("fetched_at"))
        return cls(**data)

    def to_insert_dict(self) -> dict:
        """Return a dict for SQL INSERT (excludes auto-generated id)."""
        d = self.model_dump(exclude={"id"})
        d["fetched_at"] = d["fetched_at"].isoformat()
        d["processed"] = int(d["processed"])
        return d


class KnowledgeState(BaseModel):
    """Rolling summary of everything known about a topic."""

    id: int | None = None
    topic_id: int
    summary_text: str
    token_count: int = 0
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Self:
        """Construct a KnowledgeState from a database row."""
        data = dict(row)
        data["updated_at"] = _coerce_required_dt(data.get("updated_at"))
        return cls(**data)

    def to_insert_dict(self) -> dict:
        """Return a dict for SQL INSERT (excludes auto-generated id)."""
        d = self.model_dump(exclude={"id"})
        d["updated_at"] = d["updated_at"].isoformat()
        return d


class CheckResult(BaseModel):
    """Record of a single check cycle for a topic."""

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

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Self:
        """Construct a CheckResult from a database row."""
        data = dict(row)
        data["has_new_info"] = bool(data["has_new_info"])
        data["notification_sent"] = bool(data["notification_sent"])
        data["checked_at"] = _coerce_required_dt(data.get("checked_at"))
        return cls(**data)

    def to_insert_dict(self) -> dict:
        """Return a dict for SQL INSERT (excludes auto-generated id)."""
        d = self.model_dump(exclude={"id"})
        d["checked_at"] = d["checked_at"].isoformat()
        d["has_new_info"] = int(d["has_new_info"])
        d["notification_sent"] = int(d["notification_sent"])
        return d


class FeedHealth(BaseModel):
    """Health tracking for a single feed URL."""

    id: int | None = None
    feed_url: str
    last_success_at: datetime | None = None
    last_error_at: datetime | None = None
    last_error_message: str | None = None
    consecutive_failures: int = 0
    total_fetches: int = 0
    total_failures: int = 0

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Self:
        data = dict(row)
        for field in ("last_success_at", "last_error_at"):
            value = data.get(field)
            # Treat empty/whitespace-only strings as missing (None); a falsy
            # "" would otherwise skip parsing and reach Pydantic as a raw
            # string, raising ValidationError on legacy/migrated rows.
            if isinstance(value, str) and not value.strip():
                data[field] = None
            elif value:
                try:
                    data[field] = datetime.fromisoformat(value)
                except (ValueError, TypeError):
                    data[field] = None
        return cls(**data)


class DashboardStats(BaseModel):
    """Aggregate statistics for the dashboard."""

    total_topics: int = 0
    active_topics: int = 0
    checks_24h: int = 0
    checks_total: int = 0
    new_info_24h: int = 0
    new_info_total: int = 0
    last_notification_at: datetime | None = None


class PendingNotification(BaseModel):
    """A notification that failed to send and should be retried."""

    id: int | None = None
    topic_id: int
    check_result_id: int | None = None
    title: str
    body: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    retry_count: int = 0
    max_retries: int = 3
    claimed_at: str | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Self:
        """Construct a PendingNotification from a database row."""
        data = dict(row)
        data["created_at"] = _coerce_required_dt(data.get("created_at"))
        return cls(**data)

    def to_insert_dict(self) -> dict:
        """Return a dict for SQL INSERT (excludes auto-generated id)."""
        d = self.model_dump(exclude={"id", "claimed_at"})
        d["created_at"] = d["created_at"].isoformat()
        return d


class PendingWebhook(BaseModel):
    """A webhook delivery that failed to send and should be retried.

    Mirrors ``PendingNotification`` for the webhook retry queue. The outbound
    ``payload`` is stored as a JSON TEXT column.
    """

    id: int | None = None
    topic_id: int
    check_result_id: int | None = None
    url: str
    payload: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    retry_count: int = 0
    max_retries: int = 3
    claimed_at: str | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Self:
        """Construct a PendingWebhook from a database row."""
        data = dict(row)
        data["payload"] = _safe_json(data.get("payload"), {})
        data["created_at"] = _coerce_required_dt(data.get("created_at"))
        return cls(**data)

    def to_insert_dict(self) -> dict:
        """Return a dict for SQL INSERT (excludes auto-generated id)."""
        d = self.model_dump(exclude={"id", "claimed_at"})
        d["payload"] = json.dumps(d["payload"])
        d["created_at"] = d["created_at"].isoformat()
        return d
