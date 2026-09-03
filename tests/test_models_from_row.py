"""Defensive-loading tests for ``Model.from_row`` methods.

A single malformed/empty *optional* cell (from a migration bug, a manual DB edit,
or a future code bug) must NOT crash the route that loads the row: bad JSON and
bad nullable datetimes coerce to a safe default.

A corrupt *required* timestamp is the exception (TW-AUD-013). There is no safe
default for it — substituting ``now()`` invents a checked_at/created_at the row
never had, which then drives scheduling, ordering and the UI. Those raise
``CorruptTimestampError`` instead. Every hydrated datetime is aware UTC, so a
naive stored value can never reach aware timestamp arithmetic.
"""

import logging
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from app.models import (
    Article,
    CheckIntent,
    CheckIntentStatus,
    CheckResult,
    CorruptTimestampError,
    FeedHealth,
    FeedMode,
    KnowledgeRevision,
    KnowledgeRevisionSource,
    KnowledgeState,
    PendingNotification,
    PendingWebhook,
    Topic,
    TopicStatus,
    is_source_failure,
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

    def test_empty_created_at_raises(self) -> None:
        row = self._base_row()
        row["created_at"] = ""
        with pytest.raises(CorruptTimestampError, match="created_at"):
            Topic.from_row(row)

    def test_malformed_created_at_raises(self) -> None:
        row = self._base_row()
        row["created_at"] = "not-a-date"
        with pytest.raises(CorruptTimestampError, match="created_at"):
            Topic.from_row(row)

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


class TestTopicImportanceThresholdClamping:
    """Per-topic importance threshold must stay within [1, 5].

    Mirrors the confidence/relevance clamp: a corrupt out-of-range value (e.g. 9)
    reaching a topic row would make ``novelty.importance >= importance_threshold``
    always false, silently suppressing ALL notifications for that topic.
    """

    def test_in_range_and_none_pass_through(self) -> None:
        topic = Topic(name="T", description="d", importance_threshold=3)
        assert topic.importance_threshold == 3
        assert Topic(name="T", description="d").importance_threshold is None
        assert Topic(name="T", description="d", importance_threshold=1).importance_threshold == 1
        assert Topic(name="T", description="d", importance_threshold=5).importance_threshold == 5

    def test_above_five_is_clamped(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="app.models"):
            topic = Topic(name="T", description="d", importance_threshold=9)
        assert topic.importance_threshold == 5
        assert any("importance_threshold" in r.message for r in caplog.records)

    def test_below_one_is_clamped(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="app.models"):
            topic = Topic(name="T", description="d", importance_threshold=0)
        assert topic.importance_threshold == 1
        assert any("importance_threshold" in r.message for r in caplog.records)

    @pytest.mark.parametrize("value", [4.9, 1.5, "4.9", -0.5])
    def test_fractional_values_are_rejected_not_truncated(self, value: object) -> None:
        """AUG-156: SQLite INTEGER affinity keeps a REAL 4.9, ``int()`` made it a 4.

        Truncating broadens notification eligibility (>= 4 passes far more than
        >= 5) with nothing in the logs, and a later unrelated save writes the
        fabricated value back permanently. The scale is whole numbers, so a
        fractional threshold is corrupt data and says so.
        """
        with pytest.raises(ValidationError):
            Topic(name="T", description="d", importance_threshold=value)

    def test_integral_floats_still_load(self) -> None:
        """A REAL 4.0 is the same threshold as a 4 — only the fraction is corrupt."""
        assert Topic(name="T", description="d", importance_threshold=4.0).importance_threshold == 4
        assert Topic(name="T", description="d", importance_threshold="4").importance_threshold == 4
        # Out of range but integral: still clamped, as before.
        assert Topic(name="T", description="d", importance_threshold=9.0).importance_threshold == 5


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

    def test_empty_fetched_at_raises(self) -> None:
        row = self._base_row()
        row["fetched_at"] = ""
        with pytest.raises(CorruptTimestampError, match="fetched_at"):
            Article.from_row(row)

    def test_malformed_fetched_at_raises(self) -> None:
        row = self._base_row()
        row["fetched_at"] = "nope"
        with pytest.raises(CorruptTimestampError, match="fetched_at"):
            Article.from_row(row)

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

    def test_empty_checked_at_raises(self) -> None:
        row = self._base_row()
        row["checked_at"] = ""
        with pytest.raises(CorruptTimestampError, match="checked_at"):
            CheckResult.from_row(row)

    def test_malformed_checked_at_raises(self) -> None:
        row = self._base_row()
        row["checked_at"] = "bad"
        with pytest.raises(CorruptTimestampError, match="checked_at"):
            CheckResult.from_row(row)

    def test_one_decode_yields_both_scalars(self) -> None:
        """AUG-037: the history table used to parse each blob three times — model
        hydration plus one template filter per scalar."""
        import json
        from unittest.mock import patch

        row = self._base_row()
        row["llm_response"] = json.dumps({"has_new_info": True, "confidence": 0.82, "importance": 4})

        with patch("app.models.json.loads", wraps=json.loads) as loads:
            result = CheckResult.from_row(row)

        assert result.confidence == 0.82
        assert result.importance == 4
        assert loads.call_count == 1

    def test_missing_and_unusable_scalars_are_none(self) -> None:
        import json

        for blob in (None, "", "not json {{{", "[1, 2, 3]", "42", json.dumps({"importance": "high"})):
            row = self._base_row()
            row["llm_response"] = blob
            result = CheckResult.from_row(row)
            assert result.confidence is None
            assert result.importance is None

    def test_float_importance_truncates_like_the_filter_did(self) -> None:
        import json

        row = self._base_row()
        row["llm_response"] = json.dumps({"importance": 4.0})
        assert CheckResult.from_row(row).importance == 4

    def test_derived_scalars_are_never_persisted(self) -> None:
        insertable = CheckResult(topic_id=1, confidence=0.5, importance=3).to_insert_dict()
        assert "confidence" not in insertable
        assert "importance" not in insertable


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

    def test_empty_updated_at_raises(self) -> None:
        row = self._base_row()
        row["updated_at"] = ""
        with pytest.raises(CorruptTimestampError, match="updated_at"):
            KnowledgeState.from_row(row)

    def test_malformed_updated_at_raises(self) -> None:
        row = self._base_row()
        row["updated_at"] = "xyz"
        with pytest.raises(CorruptTimestampError, match="updated_at"):
            KnowledgeState.from_row(row)


class TestKnowledgeRevisionFromRow:
    """KnowledgeRevision.from_row defensive handling of created_at and source."""

    def _base_row(self) -> dict:
        return {
            "id": 1,
            "topic_id": 1,
            "summary_text": "body",
            "token_count": 10,
            "source": "init",
            "change_note": None,
            "created_at": "2026-06-13T12:00:00+00:00",
        }

    def test_valid_row_parses(self) -> None:
        revision = KnowledgeRevision.from_row(self._base_row())
        assert revision.source == KnowledgeRevisionSource.INIT
        assert revision.created_at.year == 2026

    def test_empty_created_at_raises(self) -> None:
        row = self._base_row()
        row["created_at"] = ""
        with pytest.raises(CorruptTimestampError, match="created_at"):
            KnowledgeRevision.from_row(row)

    def test_unknown_source_degrades_to_unknown(self, caplog: pytest.LogCaptureFixture) -> None:
        """AUG-155: a value written by a future version must not 500 the detail
        page — and must not be relabelled ``update`` either, which is a plausible,
        false lineage the diff view would then compare adjacently."""
        row = self._base_row()
        row["source"] = "from-the-future"
        with caplog.at_level(logging.WARNING, logger="app.models"):
            revision = KnowledgeRevision.from_row(row)
        assert revision.source == KnowledgeRevisionSource.UNKNOWN
        assert any("source" in r.message for r in caplog.records)

    def test_non_string_source_degrades_to_unknown(self) -> None:
        row = self._base_row()
        row["source"] = 42
        assert KnowledgeRevision.from_row(row).source == KnowledgeRevisionSource.UNKNOWN


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

    def test_empty_created_at_raises(self) -> None:
        row = self._base_row()
        row["created_at"] = ""
        with pytest.raises(CorruptTimestampError, match="created_at"):
            PendingNotification.from_row(row)

    def test_malformed_created_at_raises(self) -> None:
        row = self._base_row()
        row["created_at"] = "???"
        with pytest.raises(CorruptTimestampError, match="created_at"):
            PendingNotification.from_row(row)


class TestCheckIntentFromRow:
    """CheckIntent.from_row / to_insert_dict defensive handling + round-trip."""

    def _base_row(self) -> dict:
        return {
            "id": 1,
            "request_id": "req-abc",
            "topic_id": 7,
            "baseline_check_id": 42,
            "status": "pending",
            "created_at": "2026-06-13T12:00:00+00:00",
            "attempts": 0,
            "max_attempts": 3,
            "next_attempt_at": None,
            "claimed_at": None,
            "claim_token": None,
            "check_result_id": None,
            "last_error": None,
        }

    def test_valid_row_parses_identically(self) -> None:
        intent = CheckIntent.from_row(self._base_row())
        assert intent.request_id == "req-abc"
        assert intent.topic_id == 7
        assert intent.baseline_check_id == 42
        assert intent.status is CheckIntentStatus.PENDING
        assert intent.created_at.year == 2026

    def test_malformed_created_at_raises(self) -> None:
        row = self._base_row()
        row["created_at"] = "???"
        with pytest.raises(CorruptTimestampError, match="created_at"):
            CheckIntent.from_row(row)

    def test_round_trip_from_row_to_insert_dict(self) -> None:
        """Only the admission columns are inserted; the claim/outcome ones are the runner's."""
        intent = CheckIntent.from_row(self._base_row())
        data = intent.to_insert_dict()
        assert set(data) == {
            "request_id",
            "topic_id",
            "baseline_check_id",
            "status",
            "created_at",
            "attempts",
            "max_attempts",
        }
        assert data["status"] == "pending"
        assert data["created_at"] == "2026-06-13T12:00:00+00:00"


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

    def test_empty_created_at_raises(self) -> None:
        row = self._base_row()
        row["created_at"] = ""
        with pytest.raises(CorruptTimestampError, match="created_at"):
            PendingWebhook.from_row(row)

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


class TestRequiredDatetimeFailsLoud:
    """TW-AUD-013: a required timestamp cell is never replaced with now(UTC).

    The old behaviour warned and substituted the current time, which is a value
    the row never held: a check_results row then claims to have run just now, a
    topic claims to have been created just now, and the scheduler, the ordering
    and the UI all act on the invented value. The error names the column so the
    operator can find the row.
    """

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

    def test_empty_string_required_datetime_raises(self) -> None:
        with pytest.raises(CorruptTimestampError) as excinfo:
            Topic.from_row(self._topic_row(""))
        assert "created_at" in str(excinfo.value)

    def test_null_required_datetime_raises(self) -> None:
        with pytest.raises(CorruptTimestampError) as excinfo:
            Topic.from_row(self._topic_row(None))
        assert "created_at" in str(excinfo.value)

    def test_malformed_required_datetime_raises(self) -> None:
        with pytest.raises(CorruptTimestampError) as excinfo:
            Topic.from_row(self._topic_row("not-a-date"))
        assert "not-a-date" in str(excinfo.value)

    def test_valid_required_datetime_still_loads(self) -> None:
        topic = Topic.from_row(self._topic_row("2026-06-13T12:00:00+00:00"))
        assert topic.created_at == datetime(2026, 6, 13, 12, 0, tzinfo=UTC)


class TestTimestampsHydrateAsAwareUtc:
    """TW-AUD-013: hydration never yields a naive or non-UTC datetime.

    Timestamp columns are compared and ordered as TEXT, and compared in Python
    against ``datetime.now(UTC)``. A naive value hydrated as-is raises
    ``TypeError`` the moment it meets an aware one (feed backoff did exactly
    that), and a value carrying a local offset sorts wrong against its ``+00:00``
    siblings even though it names the same instant.
    """

    def _topic_row(self, created_at: str, status_changed_at: object = None) -> dict:
        return {
            "id": 1,
            "name": "Topic",
            "description": "desc",
            "feed_urls": "[]",
            "feed_mode": "auto",
            "created_at": created_at,
            "status_changed_at": status_changed_at,
            "is_active": 1,
            "status": "ready",
            "error_message": None,
            "check_interval_minutes": 60,
            "tags": "[]",
        }

    def test_naive_required_timestamp_becomes_aware_utc(self) -> None:
        topic = Topic.from_row(self._topic_row("2026-06-13T12:00:00"))
        assert topic.created_at == datetime(2026, 6, 13, 12, 0, tzinfo=UTC)
        assert topic.created_at.tzinfo is not None

    def test_naive_optional_timestamp_becomes_aware_utc(self) -> None:
        topic = Topic.from_row(self._topic_row("2026-06-13T12:00:00+00:00", "2026-06-13T13:00:00"))
        assert topic.status_changed_at == datetime(2026, 6, 13, 13, 0, tzinfo=UTC)

    def test_offset_timestamp_is_converted_to_utc(self) -> None:
        """A row written with a local offset must hydrate as the same instant in UTC."""
        topic = Topic.from_row(self._topic_row("2026-06-13T14:00:00+02:00"))
        assert topic.created_at == datetime(2026, 6, 13, 12, 0, tzinfo=UTC)
        assert topic.created_at.utcoffset() == timedelta(0)
        # And it re-serializes in the canonical +00:00 spelling the columns sort on.
        assert topic.to_insert_dict()["created_at"] == "2026-06-13T12:00:00+00:00"

    def test_naive_feed_health_timestamp_survives_backoff_arithmetic(self) -> None:
        """The concrete break: naive last_error_at + aware now() raised TypeError."""
        from app.feed_backoff import feed_backoff_until

        health = FeedHealth.from_row(
            {
                "id": 1,
                "feed_url": "https://example.com/feed.xml",
                "last_success_at": None,
                "last_error_at": datetime.now(UTC).replace(tzinfo=None).isoformat(),
                "last_error_message": "boom",
                "consecutive_failures": 4,
                "total_fetches": 10,
                "total_failures": 4,
                "etag": None,
                "last_modified": None,
            }
        )
        until = feed_backoff_until(health)
        assert until is not None
        assert until > datetime.now(UTC)


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
            "cr_stage_error": None,
            "cr_seen_at": None,
            "cr_notify_disposition": None,
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

    def test_corrupt_checked_at_raises(self) -> None:
        """The dashboard path uses the same required-timestamp contract (TW-AUD-013)."""
        with pytest.raises(CorruptTimestampError, match="checked_at"):
            CheckResult.from_dashboard_row(self._dash_row(cr_checked_at="garbage"), topic_id=1)

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

    def test_stage_error_alias_maps_through(self) -> None:
        cr = CheckResult.from_dashboard_row(self._dash_row(cr_stage_error="sources_failed: x"), topic_id=3)
        assert cr.stage_error == "sources_failed: x"
        assert cr.sources_failing

    def test_null_stage_error_stays_none(self) -> None:
        cr = CheckResult.from_dashboard_row(self._dash_row(), topic_id=3)
        assert cr.stage_error is None
        assert not cr.sources_failing


class TestStageErrorClassification:
    """The stage_error vocabulary the Silence Heartbeat and the templates share."""

    def test_source_failure_prefixes(self) -> None:
        assert is_source_failure("sources_failed: all feed source(s) failed (see logs)")
        assert is_source_failure("scrape_failed: TimeoutError: boom")
        assert is_source_failure("sources_unavailable: no source attempted (2 feed(s) in backoff)")

    def test_non_source_failures(self) -> None:
        assert not is_source_failure(None)
        assert not is_source_failure("")
        assert not is_source_failure("analysis_failed: LLM timeout")
        assert not is_source_failure("knowledge_update_failed: ValueError: nope")
        assert not is_source_failure("skipped: already in flight")

    def test_check_result_property(self) -> None:
        assert CheckResult(topic_id=1, stage_error="sources_failed: x").sources_failing
        assert not CheckResult(topic_id=1, stage_error="analysis_failed: x").sources_failing
        assert not CheckResult(topic_id=1).sources_failing


class TestTagCanonicalization:
    """AUG-338: one logical tag has exactly one stored and filtered identity."""

    def test_whitespace_and_case_of_a_single_tag(self) -> None:
        from app.models import normalize_tag

        assert normalize_tag("  Tech   News \n") == "Tech News"
        # Case is preserved on purpose: it is what the chip displays.
        assert normalize_tag("Policy") == "Policy"

    def test_nfd_and_nfc_forms_collapse(self) -> None:
        from app.models import normalize_tags

        assert normalize_tags(["Café", "Café"]) == ["Café"]

    def test_invisible_characters_are_stripped(self) -> None:
        from app.models import normalize_tag

        assert normalize_tag("Tech​News") == "TechNews"
        assert normalize_tag("‮Policy") == "Policy"

    def test_blanks_dropped_and_order_preserved(self) -> None:
        from app.models import normalize_tags

        assert normalize_tags(["b", "  ", "a", "b", "​"]) == ["b", "a"]

    def test_topic_canonicalizes_on_construction_and_load(self) -> None:
        import json

        row = {
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
            "tags": json.dumps(["  Tech   News ", "Tech News", ""]),
        }
        assert Topic.from_row(row).tags == ["Tech News"]
        assert Topic(name="n", description="d", tags=[" A ", "A"]).tags == ["A"]
