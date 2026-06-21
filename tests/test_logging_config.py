"""Tests for app/logging_config.py — JSON and plain text logging setup."""

import io
import json
import logging

import pytest

from app.check_context import check_id_var
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


@pytest.fixture
def set_check_id():
    """Set check_id_var to a sentinel for the test, resetting it afterwards.

    Restores the prior contextvar token so the sentinel never leaks into other
    tests sharing the process-wide context.
    """
    token = check_id_var.set("abcd1234")
    try:
        yield "abcd1234"
    finally:
        check_id_var.reset(token)


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

    def test_text_output_renders_check_id_and_message(self, monkeypatch, set_check_id):
        """OVH-090: the default (text) branch must attach the CheckIdFilter and
        render the correlation id alongside the message. The two structural
        tests above never make a real log call, so a deleted filter loop would
        slip through; this redirects the handler to a StringIO and asserts the
        rendered line carries both the check_id and the message."""
        monkeypatch.setenv("TOPIC_WATCH_LOG_FORMAT", "text")
        # logging.basicConfig is a no-op if root already has a handler; under
        # pytest the logging plugin installs its own capture handler around the
        # call, so clear it first to guarantee setup_logging installs the text
        # handler we then redirect (mirrors the JSON branch's handler reset).
        logging.root.handlers.clear()
        setup_logging()

        stream = io.StringIO()
        root = logging.root
        text_handler = root.handlers[0]
        text_handler.stream = stream
        root.setLevel(logging.INFO)

        logging.getLogger("test.text_check_id").info("hello text mode")
        text_handler.flush()

        rendered = stream.getvalue()
        assert "abcd1234" in rendered, rendered
        assert "hello text mode" in rendered, rendered

    def test_text_output_renders_dash_when_no_check_id(self, monkeypatch):
        """At default (no check_id set) the text line carries the '-' placeholder."""
        monkeypatch.setenv("TOPIC_WATCH_LOG_FORMAT", "text")
        logging.root.handlers.clear()
        setup_logging()

        stream = io.StringIO()
        root = logging.root
        text_handler = root.handlers[0]
        text_handler.stream = stream
        root.setLevel(logging.INFO)

        logging.getLogger("test.text_no_check_id").info("no correlation id")
        text_handler.flush()

        rendered = stream.getvalue()
        assert "[-]" in rendered, rendered
        assert "no correlation id" in rendered, rendered


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

    def test_json_output_emits_check_id(self, monkeypatch, set_check_id):
        """OVH-089: when a check_id is set, it is carried into the JSON output —
        the whole point of the correlation-id feature. The existing JSON test
        runs with no var set, so a removed filter wiring would silently emit '-'
        and stay green; this asserts the id round-trips."""
        monkeypatch.setenv("TOPIC_WATCH_LOG_FORMAT", "json")

        stream = io.StringIO()
        setup_logging()
        logging.root.handlers[0].stream = stream

        logging.getLogger("test.json_check_id").info("with correlation id")

        parsed = json.loads(stream.getvalue().strip())
        assert parsed["check_id"] == set_check_id

    def test_json_output_check_id_dash_at_default(self, monkeypatch):
        """With no check_id set, the JSON output carries the '-' placeholder."""
        monkeypatch.setenv("TOPIC_WATCH_LOG_FORMAT", "json")

        stream = io.StringIO()
        setup_logging()
        logging.root.handlers[0].stream = stream

        logging.getLogger("test.json_no_check_id").info("no correlation id")

        parsed = json.loads(stream.getvalue().strip())
        assert parsed["check_id"] == "-"

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


class TestUvicornLoggersJSON:
    """OVH-041: in JSON mode, uvicorn loggers must flow through the root JSON handler."""

    @pytest.fixture(autouse=True)
    def reset_uvicorn_loggers(self):
        """Restore uvicorn loggers to a clean state around each test."""
        names = ("uvicorn", "uvicorn.error", "uvicorn.access")
        saved = {n: (logging.getLogger(n).handlers[:], logging.getLogger(n).propagate) for n in names}
        yield
        for n, (handlers, propagate) in saved.items():
            lg = logging.getLogger(n)
            lg.handlers = handlers
            lg.propagate = propagate

    def test_json_mode_retargets_uvicorn_loggers(self, monkeypatch):
        monkeypatch.setenv("TOPIC_WATCH_LOG_FORMAT", "json")
        # Simulate uvicorn having installed its own text handlers.
        for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
            lg = logging.getLogger(name)
            lg.handlers = [logging.StreamHandler()]
            lg.propagate = False

        setup_logging()

        for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
            lg = logging.getLogger(name)
            assert lg.handlers == [], f"{name} should have no own handlers in JSON mode"
            assert lg.propagate is True, f"{name} should propagate to root in JSON mode"

    def test_uvicorn_access_output_is_valid_json(self, monkeypatch):
        monkeypatch.setenv("TOPIC_WATCH_LOG_FORMAT", "json")
        setup_logging()

        stream = io.StringIO()
        logging.root.handlers[0].stream = stream

        access_logger = logging.getLogger("uvicorn.access")
        access_logger.setLevel(logging.INFO)
        access_logger.info('127.0.0.1 - "GET / HTTP/1.1" 200')

        output = stream.getvalue().strip()
        assert output, "Expected uvicorn.access output to reach the root JSON handler"
        parsed = json.loads(output)
        assert parsed["logger"] == "uvicorn.access"
        assert parsed["level"] == "INFO"
        assert "check_id" in parsed

    def test_plain_text_mode_leaves_uvicorn_loggers_untouched(self, monkeypatch):
        monkeypatch.setenv("TOPIC_WATCH_LOG_FORMAT", "text")
        own_handler = logging.StreamHandler()
        lg = logging.getLogger("uvicorn.access")
        lg.handlers = [own_handler]
        lg.propagate = False

        setup_logging()

        assert lg.handlers == [own_handler]
        assert lg.propagate is False


class TestNonRootLoggerHandling:
    """OVH-172: setup_logging configures only the root logger, so non-root
    application loggers (e.g. ``app.checker``) must inherit it via propagation —
    flowing through the same handler, formatter, and CheckIdFilter. These tests
    pin that scope (which the suite otherwise only assumed) for both modes.
    """

    def test_non_root_logger_propagates_to_json_handler_with_check_id(self, monkeypatch, set_check_id):
        """A child logger's record reaches the root JSON handler and carries the id."""
        monkeypatch.setenv("TOPIC_WATCH_LOG_FORMAT", "json")
        setup_logging()

        stream = io.StringIO()
        logging.root.handlers[0].stream = stream

        # A non-root logger with no handlers of its own (the production norm).
        child = logging.getLogger("app.checker")
        assert not child.handlers
        assert child.propagate is True
        child.info("child mode")

        parsed = json.loads(stream.getvalue().strip())
        assert parsed["logger"] == "app.checker"
        assert parsed["message"] == "child mode"
        # The CheckIdFilter on the root handler also applies to propagated records.
        assert parsed["check_id"] == set_check_id

    def test_non_root_logger_renders_through_text_handler_with_check_id(self, monkeypatch, set_check_id):
        """In text mode the child logger's line carries the check_id placeholder."""
        monkeypatch.setenv("TOPIC_WATCH_LOG_FORMAT", "text")
        logging.root.handlers.clear()
        setup_logging()

        stream = io.StringIO()
        text_handler = logging.root.handlers[0]
        text_handler.stream = stream
        logging.root.setLevel(logging.INFO)

        child = logging.getLogger("app.scraping.rss")
        child.info("scraper line")
        text_handler.flush()

        rendered = stream.getvalue()
        assert "scraper line" in rendered, rendered
        assert "abcd1234" in rendered, rendered
        assert "app.scraping.rss" in rendered, rendered


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
