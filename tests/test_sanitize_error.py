"""Tests for the _sanitize_error Jinja2 filter."""

from app.web.routes import _sanitize_error


def test_none_input_returns_default_message():
    result = _sanitize_error(None)
    assert "<p>An unknown error occurred.</p>" in result


def test_empty_string_returns_default_message():
    result = _sanitize_error("")
    assert "<p>An unknown error occurred.</p>" in result


def test_short_error_displayed_as_is():
    msg = "Connection refused"
    result = _sanitize_error(msg)
    assert isinstance(result, str)
    assert f"<p>{msg}</p>" in result
    assert "<details>" not in result


def test_short_error_near_boundary():
    # 199 chars — still short
    msg = "x" * 199
    result = _sanitize_error(msg)
    assert "<details>" not in result
    assert f"<p>{msg}</p>" in result


def test_long_traceback_summary_is_last_line():
    traceback = (
        "Traceback (most recent call last):\n"
        '  File "/app/app/checker.py", line 45, in check_topic\n'
        "    articles = await fetch_new_articles_for_topic(...)\n"
        '  File "/app/app/scraping/__init__.py", line 28, in fetch_new_articles_for_topic\n'
        "    entries = await fetch_feeds_for_topic(topic)\n"
        "httpx.ConnectError: [Errno -2] Name or service not known"
    )
    result = _sanitize_error(traceback)
    assert isinstance(result, str)
    assert "<p>httpx.ConnectError: [Errno -2] Name or service not known</p>" in result
    assert "<details>" in result
    assert "<pre><code>" in result
    assert "Traceback (most recent call last):" in result


def test_long_traceback_full_error_in_details():
    traceback = "A" * 200 + "\nSomeError: something went wrong"
    result = _sanitize_error(traceback)
    assert "<details>" in result
    assert "<pre><code>" in result
    assert "SomeError: something went wrong" in result


def test_html_escaping_in_short_message():
    msg = "<script>alert('xss')</script>"
    result = _sanitize_error(msg)
    assert isinstance(result, str)
    assert "<script>" not in result
    assert "&lt;script&gt;" in result


def test_html_escaping_in_long_traceback():
    evil_line = "<script>alert('xss')</script>"
    # Build a long traceback that is definitely >= 200 chars
    padding = "  some long traceback line here\n" * 10
    traceback = "Traceback (most recent call last):\n" + padding + evil_line
    assert len(traceback) >= 200
    result = _sanitize_error(traceback)
    assert isinstance(result, str)
    assert "<script>" not in result
    assert "&lt;script&gt;" in result


def test_result_is_str_instance():
    result = _sanitize_error("short error")
    assert isinstance(result, str)

    long_error = "error\n" * 50
    result_long = _sanitize_error(long_error)
    assert isinstance(result_long, str)
