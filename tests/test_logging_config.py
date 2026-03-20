"""Tests for app/logging_config.py — JSON and plain text logging setup."""

import io
import json
import logging

import pytest

from app.logging_config import JSONFormatter, setup_logging


def _reset_root_logger():
    """Remove all handlers from the root logger and reset level."""
    root = logging.root
    root.handlers.clear()
    root.setLevel(logging.WARNING)


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset root logger before and after each test."""
    _reset_root_logger()
    yield
    _reset_root_logger()


class TestSetupLoggingPlainText:
    def test_no_env_var_uses_plain_text_formatter(self, monkeypatch):
        monkeypatch.delenv("TOPIC_WATCH_LOG_FORMAT", raising=False)
        setup_logging()
        root = logging.root
        assert root.handlers, "Root logger should have at least one handler"
        handler = root.handlers[0]
        assert not isinstance(handler.formatter, JSONFormatter)

    def test_non_json_value_uses_plain_text_formatter(self, monkeypatch):
        monkeypatch.setenv("TOPIC_WATCH_LOG_FORMAT", "text")
        setup_logging()
        root = logging.root
        assert root.handlers, "Root logger should have at least one handler"
        handler = root.handlers[0]
        assert not isinstance(handler.formatter, JSONFormatter)


class TestSetupLoggingJSON:
    def test_json_env_var_uses_json_formatter(self, monkeypatch):
        monkeypatch.setenv("TOPIC_WATCH_LOG_FORMAT", "json")
        setup_logging()
        root = logging.root
        assert root.handlers, "Root logger should have at least one handler"
        handler = root.handlers[0]
        assert isinstance(handler.formatter, JSONFormatter)

    def test_json_output_is_valid_json(self, monkeypatch):
        monkeypatch.setenv("TOPIC_WATCH_LOG_FORMAT", "json")

        stream = io.StringIO()
        setup_logging()
        # Replace the handler's stream with our StringIO
        root = logging.root
        root.handlers[0].stream = stream

        logger = logging.getLogger("test.json_output")
        logger.info("hello world")

        output = stream.getvalue().strip()
        assert output, "Expected log output"
        parsed = json.loads(output)
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test.json_output"
        assert parsed["message"] == "hello world"
        assert "timestamp" in parsed

    def test_json_timestamp_is_iso_format(self, monkeypatch):
        monkeypatch.setenv("TOPIC_WATCH_LOG_FORMAT", "json")

        stream = io.StringIO()
        setup_logging()
        logging.root.handlers[0].stream = stream

        logger = logging.getLogger("test.timestamp")
        logger.info("ts test")

        parsed = json.loads(stream.getvalue().strip())
        # ISO format timestamps contain 'T' and '+' or 'Z'
        assert "T" in parsed["timestamp"]
        assert "+00:00" in parsed["timestamp"] or parsed["timestamp"].endswith("Z")


class TestJSONFormatterException:
    def test_exception_info_included(self, monkeypatch):
        monkeypatch.setenv("TOPIC_WATCH_LOG_FORMAT", "json")

        stream = io.StringIO()
        setup_logging()
        logging.root.handlers[0].stream = stream

        logger = logging.getLogger("test.exception")
        try:
            raise ValueError("boom")
        except ValueError:
            logger.exception("caught an error")

        parsed = json.loads(stream.getvalue().strip())
        assert "exception" in parsed
        assert "ValueError" in parsed["exception"]
        assert "boom" in parsed["exception"]

    def test_no_exception_key_when_no_exc_info(self, monkeypatch):
        monkeypatch.setenv("TOPIC_WATCH_LOG_FORMAT", "json")

        stream = io.StringIO()
        setup_logging()
        logging.root.handlers[0].stream = stream

        logger = logging.getLogger("test.no_exception")
        logger.info("no error here")

        parsed = json.loads(stream.getvalue().strip())
        assert "exception" not in parsed


class TestJSONFormatterExtraFields:
    def test_extra_fields_included_in_json(self, monkeypatch):
        monkeypatch.setenv("TOPIC_WATCH_LOG_FORMAT", "json")

        stream = io.StringIO()
        setup_logging()
        logging.root.handlers[0].stream = stream

        logger = logging.getLogger("test.extra_fields")
        logger.info("test", extra={"my_custom_field": "abc123"})

        parsed = json.loads(stream.getvalue().strip())
        assert "extra" in parsed
        assert parsed["extra"]["my_custom_field"] == "abc123"

    def test_multiple_extra_fields(self, monkeypatch):
        monkeypatch.setenv("TOPIC_WATCH_LOG_FORMAT", "json")

        stream = io.StringIO()
        setup_logging()
        logging.root.handlers[0].stream = stream

        logger = logging.getLogger("test.multi_extra")
        logger.info("msg", extra={"topic": "AI News", "feed_url": "https://example.com/rss"})

        parsed = json.loads(stream.getvalue().strip())
        assert parsed["extra"]["topic"] == "AI News"
        assert parsed["extra"]["feed_url"] == "https://example.com/rss"

    def test_no_extra_key_when_no_extras(self, monkeypatch):
        monkeypatch.setenv("TOPIC_WATCH_LOG_FORMAT", "json")

        stream = io.StringIO()
        setup_logging()
        logging.root.handlers[0].stream = stream

        logger = logging.getLogger("test.no_extra")
        logger.info("plain message")

        parsed = json.loads(stream.getvalue().strip())
        assert "extra" not in parsed
