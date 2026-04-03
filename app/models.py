"""Pydantic data models for Topic Watch.

These models represent the core data structures used for validation,
data transfer between layers, and serialization to/from SQLite rows.
"""

import json
import sqlite3
from datetime import UTC, datetime
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, Field


class TopicStatus(StrEnum):
    """Status of a topic's lifecycle."""

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

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Self:
        """Construct a Topic from a database row."""
        data = dict(row)
        data["feed_urls"] = json.loads(data["feed_urls"])
        data["is_active"] = bool(data["is_active"])
        data["tags"] = json.loads(data.get("tags") or "[]")
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
        return cls(**dict(row))

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

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Self:
        """Construct a CheckResult from a database row."""
        data = dict(row)
        data["has_new_info"] = bool(data["has_new_info"])
        data["notification_sent"] = bool(data["notification_sent"])
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
            if data.get(field):
                try:
                    data[field] = datetime.fromisoformat(data[field])
                except (ValueError, TypeError):
                    data[field] = None
        return cls(**data)


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

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Self:
        """Construct a PendingNotification from a database row."""
        return cls(**dict(row))

    def to_insert_dict(self) -> dict:
        """Return a dict for SQL INSERT (excludes auto-generated id)."""
        d = self.model_dump(exclude={"id"})
        d["created_at"] = d["created_at"].isoformat()
        return d
