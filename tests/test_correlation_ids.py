"""Tests for correlation ID (check_id) context management and logging filter."""

import contextvars
import io
import logging

import pytest

from app.check_context import CheckIdFilter, check_id_var, generate_check_id


def test_generate_check_id_returns_8_char_hex():
    """generate_check_id() returns an 8-character hex string."""
    cid = generate_check_id()
    assert len(cid) == 8
    # Verify it's valid hex
    int(cid, 16)


def test_generate_check_id_is_unique():
    """Each call to generate_check_id() returns a different value."""
    ids = {generate_check_id() for _ in range(20)}
    assert len(ids) == 20


def test_check_id_filter_adds_known_value():
    """CheckIdFilter adds check_id from context var to log records."""
    ctx = contextvars.copy_context()

    def run():
        check_id_var.set("abc12345")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test message",
            args=(),
            exc_info=None,
        )
        f = CheckIdFilter()
        result = f.filter(record)
        assert result is True
        assert record.check_id == "abc12345"

    ctx.run(run)


def test_check_id_filter_uses_dash_when_not_set():
    """CheckIdFilter uses '-' as placeholder when no check_id is in context."""
    ctx = contextvars.copy_context()

    def run():
        # Ensure no check_id is set
        check_id_var.set(None)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test message",
            args=(),
            exc_info=None,
        )
        f = CheckIdFilter()
        f.filter(record)
        assert record.check_id == "-"

    ctx.run(run)


def test_check_id_filter_uses_dash_for_default():
    """CheckIdFilter uses '-' when context var holds its default (None)."""
    ctx = contextvars.copy_context()

    def run():
        # Don't set the var at all — it defaults to None
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test message",
            args=(),
            exc_info=None,
        )
        f = CheckIdFilter()
        f.filter(record)
        assert record.check_id == "-"

    ctx.run(run)


async def _inner_reader() -> str | None:
    """Async helper that reads check_id_var from its context."""
    return check_id_var.get()


@pytest.mark.asyncio
async def test_check_id_propagates_through_async_context():
    """check_id_var propagates correctly through awaited async functions."""
    check_id_var.set("deadbeef")
    value = await _inner_reader()
    assert value == "deadbeef"
    # Cleanup
    check_id_var.set(None)


def test_check_id_appears_in_formatted_log_output():
    """check_id appears in formatted log output when filter is registered."""
    ctx = contextvars.copy_context()

    def run():
        check_id_var.set("feed1234")

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(name)s [%(check_id)s]: %(message)s"))
        handler.addFilter(CheckIdFilter())

        test_logger = logging.getLogger("test.correlation")
        test_logger.addHandler(handler)
        test_logger.setLevel(logging.DEBUG)
        # Prevent propagation to avoid interfering with root logger
        test_logger.propagate = False

        test_logger.info("hello world")

        output = stream.getvalue()
        assert "feed1234" in output
        assert "hello world" in output

        # Cleanup handler
        test_logger.removeHandler(handler)
        test_logger.propagate = True

    ctx.run(run)
