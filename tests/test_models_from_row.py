"""Defensive-loading tests for ``Model.from_row`` methods.

A single malformed/empty cell (from a migration bug, a manual DB edit, or a
future code bug) must NOT crash the route that loads the row. Each ``from_row``
must coerce bad JSON/datetime cells to a safe default instead of raising.
"""

import logging
from datetime import datetime

import pytest

from app.models import (
    Article,
    CheckResult,
    FeedMode,
    KnowledgeState,
    PendingNotification,
    PendingWebhook,
    Topic,
    TopicStatus,
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


class TestTopicThresholdValidation:
    """OVH-107: per-topic thresholds must stay within [0.0, 1.0].

    A value >1.0 reaching a topic row (manual DB edit, restore, or a future write
    path that skips ``parse_threshold``) would make ``novelty.confidence <
    confidence_threshold`` always true, silently suppressing ALL notifications for
    that topic. The model clamps out-of-range values to the valid range (and
    warns) rather than either raising — which would 500 a route loading a corrupt
    row, violating the defensive-load contract — or letting the bad value through.
    """

    def _base_row(self) -> dict:
        return {
            "id": 1,
            "name": "Topic",
            "description": "desc",
            "feed_urls": '["https://example.com/feed.xml"]',
            "feed_mode": "auto",
            "created_at": "2026-06-13T12:00:00+00:00",
            "status_changed_at": None,
            "is_active": 1,
            "status": "ready",
            "error_message": None,
            "check_interval_minutes": 60,
            "tags": "[]",
            "confidence_threshold": None,
            "relevance_threshold": None,
        }

    def test_in_range_values_pass_through(self) -> None:
        topic = Topic(name="T", description="d", confidence_threshold=0.7, relevance_threshold=0.0)
        assert topic.confidence_threshold == 0.7
        assert topic.relevance_threshold == 0.0
        boundary = Topic(name="T", description="d", confidence_threshold=1.0, relevance_threshold=1.0)
        assert boundary.confidence_threshold == 1.0
        assert boundary.relevance_threshold == 1.0

    def test_none_thresholds_pass_through(self) -> None:
        topic = Topic(name="T", description="d")
        assert topic.confidence_threshold is None
        assert topic.relevance_threshold is None

    def test_above_one_is_clamped_to_one(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="app.models"):
            topic = Topic(name="T", description="d", confidence_threshold=1.5)
        assert topic.confidence_threshold == 1.0
        assert any("confidence_threshold" in r.message for r in caplog.records)

    def test_below_zero_is_clamped_to_zero(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="app.models"):
            topic = Topic(name="T", description="d", relevance_threshold=-0.5)
        assert topic.relevance_threshold == 0.0
        assert any("relevance_threshold" in r.message for r in caplog.records)

    def test_from_row_clamps_out_of_range_db_value(self, caplog: pytest.LogCaptureFixture) -> None:
        """A corrupt >1.0 value in the DB must not survive to suppress all alerts."""
        row = self._base_row()
        row["confidence_threshold"] = 9.0
        with caplog.at_level(logging.WARNING, logger="app.models"):
            topic = Topic.from_row(row)
        assert topic.confidence_threshold == 1.0


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

    def test_published_at_iso_string_round_trips(self) -> None:
        """published_at ISO string deserializes and re-serializes correctly."""
        row = self._base_row()
        row["published_at"] = "2025-01-15T12:00:00+00:00"
        article = Article.from_row(row)
        assert article.published_at is not None
        assert article.published_at.year == 2025
        assert article.published_at.month == 1
        d = article.to_insert_dict()
        assert "published_at" in d
        assert d["published_at"] == "2025-01-15T12:00:00+00:00"

    def test_published_at_null_coerces_to_none(self) -> None:
        """published_at NULL in DB row becomes None on the model."""
        row = self._base_row()
        row["published_at"] = None
        article = Article.from_row(row)
        assert article.published_at is None

    def test_published_at_empty_string_coerces_to_none(self) -> None:
        """published_at empty string in DB row becomes None (legacy rows)."""
        row = self._base_row()
        row["published_at"] = ""
        article = Article.from_row(row)
        assert article.published_at is None

    def test_to_insert_dict_includes_published_at_key(self) -> None:
        """to_insert_dict() always includes published_at (None -> None)."""
        row = self._base_row()
        row["published_at"] = None
        article = Article.from_row(row)
        d = article.to_insert_dict()
        assert "published_at" in d
        assert d["published_at"] is None


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


class TestPendingWebhookFromRow:
    """PendingWebhook.from_row / to_insert_dict defensive handling + round-trip."""

    def _base_row(self) -> dict:
        return {
            "id": 1,
            "topic_id": 1,
            "check_result_id": None,
            "url": "https://example.com/hook",
            "payload": '{"topic": "Hooked", "count": 2}',
            "created_at": "2026-06-13T12:00:00+00:00",
            "retry_count": 0,
            "max_retries": 3,
        }

    def test_valid_row_parses_identically(self) -> None:
        hook = PendingWebhook.from_row(self._base_row())
        assert hook.url == "https://example.com/hook"
        assert hook.payload == {"topic": "Hooked", "count": 2}
        assert hook.created_at.year == 2026

    def test_malformed_payload_json_becomes_empty_dict(self) -> None:
        row = self._base_row()
        row["payload"] = "{not valid json"
        hook = PendingWebhook.from_row(row)
        assert hook.payload == {}

    def test_valid_json_array_payload_becomes_empty_dict(self) -> None:
        """OVH-110: valid JSON of the wrong type (array) must not raise."""
        row = self._base_row()
        row["payload"] = "[1, 2, 3]"
        hook = PendingWebhook.from_row(row)
        assert hook.payload == {}

    def test_valid_json_scalar_payload_becomes_empty_dict(self) -> None:
        """OVH-110: valid JSON of the wrong type (scalar) must not raise."""
        row = self._base_row()
        row["payload"] = "5"
        hook = PendingWebhook.from_row(row)
        assert hook.payload == {}

    def test_valid_json_string_payload_becomes_empty_dict(self) -> None:
        """OVH-110: valid JSON of the wrong type (string) must not raise."""
        row = self._base_row()
        row["payload"] = '"just a string"'
        hook = PendingWebhook.from_row(row)
        assert hook.payload == {}

    def test_wrong_type_payload_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """OVH-110: a type-mismatched (but valid JSON) payload warns naming the field."""
        row = self._base_row()
        row["payload"] = "[1, 2]"
        with caplog.at_level(logging.WARNING, logger="app.models"):
            hook = PendingWebhook.from_row(row)
        assert hook.payload == {}
        assert any("payload" in r.message for r in caplog.records)

    def test_empty_created_at_does_not_raise(self) -> None:
        row = self._base_row()
        row["created_at"] = ""
        hook = PendingWebhook.from_row(row)
        assert isinstance(hook.created_at, datetime)

    def test_round_trip_from_row_to_insert_dict(self) -> None:
        hook = PendingWebhook.from_row(self._base_row())
        data = hook.to_insert_dict()
        # id excluded; payload + created_at serialized back to TEXT.
        assert "id" not in data
        assert data["url"] == "https://example.com/hook"
        assert data["created_at"] == "2026-06-13T12:00:00+00:00"
        # Re-loading the insert dict reproduces the model (sans id).
        reloaded = PendingWebhook.from_row({**data, "id": None})
        assert reloaded.payload == hook.payload
        assert reloaded.url == hook.url
        assert reloaded.created_at == hook.created_at
        assert reloaded.max_retries == hook.max_retries


class TestRequiredDatetimeWarnings:
    """_coerce_required_dt must log a WARNING for every corrupt/empty/None value."""

    def _topic_row(self, created_at_value: object) -> dict:
        return {
            "id": 1,
            "name": "Topic",
            "description": "desc",
            "feed_urls": "[]",
            "feed_mode": "auto",
            "created_at": created_at_value,
            "status_changed_at": None,
            "is_active": 1,
            "status": "ready",
            "error_message": None,
            "check_interval_minutes": 60,
            "tags": "[]",
        }

    def test_empty_string_required_datetime_emits_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Empty-string in a required datetime column must log a WARNING."""
        with caplog.at_level(logging.WARNING, logger="app.models"):
            topic = Topic.from_row(self._topic_row(""))
        assert isinstance(topic.created_at, datetime)
        assert any("empty string" in r.message for r in caplog.records)

    def test_none_required_datetime_emits_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """None in a required datetime column must log a WARNING."""
        with caplog.at_level(logging.WARNING, logger="app.models"):
            topic = Topic.from_row(self._topic_row(None))
        assert isinstance(topic.created_at, datetime)
        assert any("NULL" in r.message for r in caplog.records)

    def test_malformed_required_datetime_emits_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """An unparseable string in a required datetime column must log a WARNING."""
        with caplog.at_level(logging.WARNING, logger="app.models"):
            topic = Topic.from_row(self._topic_row("not-a-date"))
        assert isinstance(topic.created_at, datetime)
        assert any("Corrupt required datetime" in r.message for r in caplog.records)


class TestSafeJsonWarnings:
    """OVH-023: _safe_json must log a WARNING (with the field name) on corruption."""

    def _topic_row(self) -> dict:
        return {
            "id": 1,
            "name": "Topic",
            "description": "desc",
            "feed_urls": "[]",
            "feed_mode": "auto",
            "created_at": "2026-06-13T12:00:00+00:00",
            "status_changed_at": None,
            "is_active": 1,
            "status": "ready",
            "error_message": None,
            "check_interval_minutes": 60,
            "tags": "[]",
        }

    def test_corrupt_feed_urls_logs_warning_and_yields_empty(self, caplog: pytest.LogCaptureFixture) -> None:
        """Corrupt feed_urls JSON logs a warning naming the field and yields []."""
        row = self._topic_row()
        row["feed_urls"] = "{not valid json"
        with caplog.at_level(logging.WARNING, logger="app.models"):
            topic = Topic.from_row(row)
        assert topic.feed_urls == []
        assert any("feed_urls" in r.message for r in caplog.records)

    def test_corrupt_tags_logs_warning_with_field_name(self, caplog: pytest.LogCaptureFixture) -> None:
        """Corrupt tags JSON logs a warning naming the field and yields []."""
        row = self._topic_row()
        row["tags"] = "}}}bad"
        with caplog.at_level(logging.WARNING, logger="app.models"):
            topic = Topic.from_row(row)
        assert topic.tags == []
        assert any("tags" in r.message for r in caplog.records)

    def test_wrong_type_feed_urls_logs_warning_and_yields_default(self, caplog: pytest.LogCaptureFixture) -> None:
        """Valid JSON of the wrong type (e.g. a number) is rejected with a warning."""
        row = self._topic_row()
        row["feed_urls"] = "42"
        with caplog.at_level(logging.WARNING, logger="app.models"):
            topic = Topic.from_row(row)
        assert topic.feed_urls == []
        assert any("feed_urls" in r.message for r in caplog.records)

    def test_corrupt_payload_logs_warning_with_field_name(self, caplog: pytest.LogCaptureFixture) -> None:
        """Corrupt PendingWebhook payload JSON logs a warning naming the field."""
        row = {
            "id": 1,
            "topic_id": 1,
            "check_result_id": None,
            "url": "https://example.com/hook",
            "payload": "{not valid json",
            "created_at": "2026-06-13T12:00:00+00:00",
            "retry_count": 0,
            "max_retries": 3,
        }
        with caplog.at_level(logging.WARNING, logger="app.models"):
            hook = PendingWebhook.from_row(row)
        assert hook.payload == {}
        assert any("payload" in r.message for r in caplog.records)

    def test_valid_json_does_not_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        """A well-formed JSON cell of the correct type emits no warning."""
        row = self._topic_row()
        row["feed_urls"] = '["https://example.com/feed.xml"]'
        with caplog.at_level(logging.WARNING, logger="app.models"):
            topic = Topic.from_row(row)
        assert topic.feed_urls == ["https://example.com/feed.xml"]
        assert not any("feed_urls" in r.message for r in caplog.records)


class TestSQLiteModelSharedInterop:
    """OVH-150: the shared SQLiteModel base coercions every persisted model uses.

    Characterization of the centralized row<->model interop so the per-model
    ``from_row``/``to_insert_dict`` keep emitting the documented SQLite storage
    forms (0/1 INTEGER bools, ISO-8601 datetimes, JSON TEXT, StrEnum ``.value``).
    """

    def test_subclasses_share_one_base(self) -> None:
        from app.models import SQLiteModel

        for model in (Topic, Article, CheckResult, KnowledgeState, PendingNotification, PendingWebhook):
            assert issubclass(model, SQLiteModel)

    def test_bool_serialized_as_int(self) -> None:
        """bool fields round-trip to 0/1 INTEGER (not Python True/False)."""
        topic = Topic(name="T", description="d", is_active=False)
        data = topic.to_insert_dict()
        assert data["is_active"] == 0
        assert isinstance(data["is_active"], int) and not isinstance(data["is_active"], bool)

    def test_strenum_serialized_as_value(self) -> None:
        """StrEnum fields (feed_mode, status) serialize to their ``.value``."""
        topic = Topic(name="T", description="d", status=TopicStatus.READY, feed_mode=FeedMode.MANUAL)
        data = topic.to_insert_dict()
        assert data["status"] == "ready"
        assert data["feed_mode"] == "manual"

    def test_datetime_serialized_as_isoformat(self) -> None:
        from datetime import UTC

        topic = Topic(name="T", description="d", created_at=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC))
        data = topic.to_insert_dict()
        assert data["created_at"] == "2026-01-02T03:04:05+00:00"

    def test_optional_datetime_none_stays_none(self) -> None:
        topic = Topic(name="T", description="d", status_changed_at=None)
        data = topic.to_insert_dict()
        assert data["status_changed_at"] is None

    def test_json_field_serialized_as_text(self) -> None:
        topic = Topic(name="T", description="d", feed_urls=["a", "b"], tags=["x"])
        data = topic.to_insert_dict()
        assert data["feed_urls"] == '["a", "b"]'
        assert data["tags"] == '["x"]'

    def test_id_always_excluded_from_insert(self) -> None:
        topic = Topic(id=99, name="T", description="d")
        assert "id" not in topic.to_insert_dict()

    def test_insert_exclude_drops_extra_fields(self) -> None:
        """CheckResult drops ``confidence``; PendingWebhook/Notification drop ``claimed_at``."""
        cr = CheckResult(topic_id=1, confidence=0.5)
        assert "confidence" not in cr.to_insert_dict()
        pn = PendingNotification(topic_id=1, title="t", body="b", claimed_at="x")
        assert "claimed_at" not in pn.to_insert_dict()


class TestCheckResultFromDashboardRow:
    """OVH-151: CheckResult.from_dashboard_row maps the cr_-prefixed join aliases."""

    def _dash_row(self, **overrides: object) -> dict:
        row = {
            "cr_id": 7,
            "cr_checked_at": "2026-06-13T12:00:00+00:00",
            "cr_articles_found": 4,
            "cr_articles_new": 2,
            "cr_has_new_info": 1,
            "cr_confidence": 0.75,
            "cr_notification_sent": 0,
            "cr_notification_error": None,
            "cr_seen_at": None,
        }
        row.update(overrides)
        return row

    def test_maps_aliases_to_model(self) -> None:
        cr = CheckResult.from_dashboard_row(self._dash_row(), topic_id=3)
        assert cr.id == 7
        assert cr.topic_id == 3
        assert cr.checked_at.year == 2026
        assert cr.articles_found == 4
        assert cr.articles_new == 2
        assert cr.has_new_info is True
        assert cr.notification_sent is False
        # Confidence is pre-extracted by SQL on this path; blob never shipped.
        assert cr.confidence == 0.75
        assert cr.llm_response is None

    def test_corrupt_checked_at_degrades_to_now(self) -> None:
        cr = CheckResult.from_dashboard_row(self._dash_row(cr_checked_at="garbage"), topic_id=1)
        assert isinstance(cr.checked_at, datetime)

    def test_null_confidence_stays_none(self) -> None:
        cr = CheckResult.from_dashboard_row(self._dash_row(cr_confidence=None), topic_id=1)
        assert cr.confidence is None

    def test_null_seen_at_stays_none(self) -> None:
        cr = CheckResult.from_dashboard_row(self._dash_row(cr_seen_at=None), topic_id=1)
        assert cr.seen_at is None

    def test_seen_at_populates_from_alias(self) -> None:
        cr = CheckResult.from_dashboard_row(self._dash_row(cr_seen_at="2026-06-14T09:30:00+00:00"), topic_id=1)
        assert isinstance(cr.seen_at, datetime)
        assert cr.seen_at.year == 2026
        assert cr.seen_at.month == 6

    def test_corrupt_seen_at_degrades_to_none(self) -> None:
        cr = CheckResult.from_dashboard_row(self._dash_row(cr_seen_at="garbage"), topic_id=1)
        assert cr.seen_at is None
