"""Defensive-loading tests for ``Model.from_row`` methods.

A single malformed/empty cell (from a migration bug, a manual DB edit, or a
future code bug) must NOT crash the route that loads the row. Each ``from_row``
must coerce bad JSON/datetime cells to a safe default instead of raising.
"""

from datetime import datetime

from app.models import (
    Article,
    CheckResult,
    KnowledgeState,
    PendingNotification,
    Topic,
)


class TestTopicFromRow:
    """Topic.from_row defensive handling of JSON + datetime cells."""

    def _base_row(self) -> dict:
        return {
            "id": 1,
            "name": "Topic",
            "description": "desc",
            "feed_urls": '["https://example.com/feed.xml"]',
            "feed_mode": "auto",
            "created_at": "2026-06-13T12:00:00+00:00",
            "status_changed_at": "2026-06-13T12:00:00+00:00",
            "is_active": 1,
            "status": "ready",
            "error_message": None,
            "check_interval_minutes": 60,
            "tags": '["news"]',
        }

    def test_valid_row_parses_identically(self) -> None:
        topic = Topic.from_row(self._base_row())
        assert topic.feed_urls == ["https://example.com/feed.xml"]
        assert topic.tags == ["news"]
        assert topic.created_at.year == 2026
        assert topic.status_changed_at is not None
        assert topic.status_changed_at.month == 6

    def test_malformed_feed_urls_json_becomes_empty_list(self) -> None:
        row = self._base_row()
        row["feed_urls"] = "{not valid json"
        topic = Topic.from_row(row)
        assert topic.feed_urls == []

    def test_empty_feed_urls_string_becomes_empty_list(self) -> None:
        row = self._base_row()
        row["feed_urls"] = ""
        topic = Topic.from_row(row)
        assert topic.feed_urls == []

    def test_malformed_tags_json_becomes_empty_list(self) -> None:
        row = self._base_row()
        row["tags"] = "}}}bad"
        topic = Topic.from_row(row)
        assert topic.tags == []

    def test_empty_created_at_does_not_raise(self) -> None:
        row = self._base_row()
        row["created_at"] = ""
        topic = Topic.from_row(row)
        # created_at is required; corrupt -> default-now rather than crash.
        assert isinstance(topic.created_at, datetime)

    def test_malformed_created_at_does_not_raise(self) -> None:
        row = self._base_row()
        row["created_at"] = "not-a-date"
        topic = Topic.from_row(row)
        assert isinstance(topic.created_at, datetime)

    def test_empty_status_changed_at_becomes_none(self) -> None:
        row = self._base_row()
        row["status_changed_at"] = ""
        topic = Topic.from_row(row)
        assert topic.status_changed_at is None

    def test_malformed_status_changed_at_becomes_none(self) -> None:
        row = self._base_row()
        row["status_changed_at"] = "garbage"
        topic = Topic.from_row(row)
        assert topic.status_changed_at is None


class TestArticleFromRow:
    """Article.from_row defensive handling of fetched_at."""

    def _base_row(self) -> dict:
        return {
            "id": 1,
            "topic_id": 1,
            "title": "t",
            "url": "https://example.com/a",
            "content_hash": "abc",
            "raw_content": None,
            "source_feed": "https://example.com/feed.xml",
            "source_provider": None,
            "fetched_at": "2026-06-13T12:00:00+00:00",
            "processed": 0,
        }

    def test_valid_row_parses_identically(self) -> None:
        article = Article.from_row(self._base_row())
        assert article.fetched_at.year == 2026
        assert article.processed is False

    def test_empty_fetched_at_does_not_raise(self) -> None:
        row = self._base_row()
        row["fetched_at"] = ""
        article = Article.from_row(row)
        assert isinstance(article.fetched_at, datetime)

    def test_malformed_fetched_at_does_not_raise(self) -> None:
        row = self._base_row()
        row["fetched_at"] = "nope"
        article = Article.from_row(row)
        assert isinstance(article.fetched_at, datetime)


class TestCheckResultFromRow:
    """CheckResult.from_row defensive handling of checked_at."""

    def _base_row(self) -> dict:
        return {
            "id": 1,
            "topic_id": 1,
            "checked_at": "2026-06-13T12:00:00+00:00",
            "articles_found": 3,
            "articles_new": 1,
            "has_new_info": 1,
            "llm_response": None,
            "notification_sent": 0,
            "notification_error": None,
        }

    def test_valid_row_parses_identically(self) -> None:
        result = CheckResult.from_row(self._base_row())
        assert result.checked_at.year == 2026
        assert result.has_new_info is True
        assert result.notification_sent is False

    def test_empty_checked_at_does_not_raise(self) -> None:
        row = self._base_row()
        row["checked_at"] = ""
        result = CheckResult.from_row(row)
        assert isinstance(result.checked_at, datetime)

    def test_malformed_checked_at_does_not_raise(self) -> None:
        row = self._base_row()
        row["checked_at"] = "bad"
        result = CheckResult.from_row(row)
        assert isinstance(result.checked_at, datetime)


class TestKnowledgeStateFromRow:
    """KnowledgeState.from_row defensive handling of updated_at."""

    def _base_row(self) -> dict:
        return {
            "id": 1,
            "topic_id": 1,
            "summary_text": "summary",
            "token_count": 10,
            "updated_at": "2026-06-13T12:00:00+00:00",
        }

    def test_valid_row_parses_identically(self) -> None:
        state = KnowledgeState.from_row(self._base_row())
        assert state.updated_at.year == 2026
        assert state.summary_text == "summary"

    def test_empty_updated_at_does_not_raise(self) -> None:
        row = self._base_row()
        row["updated_at"] = ""
        state = KnowledgeState.from_row(row)
        assert isinstance(state.updated_at, datetime)

    def test_malformed_updated_at_does_not_raise(self) -> None:
        row = self._base_row()
        row["updated_at"] = "xyz"
        state = KnowledgeState.from_row(row)
        assert isinstance(state.updated_at, datetime)


class TestPendingNotificationFromRow:
    """PendingNotification.from_row defensive handling of created_at."""

    def _base_row(self) -> dict:
        return {
            "id": 1,
            "topic_id": 1,
            "check_result_id": None,
            "title": "title",
            "body": "body",
            "created_at": "2026-06-13T12:00:00+00:00",
            "retry_count": 0,
            "max_retries": 3,
        }

    def test_valid_row_parses_identically(self) -> None:
        notif = PendingNotification.from_row(self._base_row())
        assert notif.created_at.year == 2026
        assert notif.title == "title"

    def test_empty_created_at_does_not_raise(self) -> None:
        row = self._base_row()
        row["created_at"] = ""
        notif = PendingNotification.from_row(row)
        assert isinstance(notif.created_at, datetime)

    def test_malformed_created_at_does_not_raise(self) -> None:
        row = self._base_row()
        row["created_at"] = "???"
        notif = PendingNotification.from_row(row)
        assert isinstance(notif.created_at, datetime)
