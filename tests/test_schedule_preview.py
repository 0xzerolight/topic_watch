"""Tests for the settings schedule-preview helper.

``_interval_preview`` guards ``parse_interval`` (which RAISES on invalid input)
and returns ``None`` so the template omits the preview rather than 500-ing.
"""

from app.web.routers.settings import _interval_preview


def test_valid_interval_renders_human_readable_preview():
    result = _interval_preview("6h")
    assert isinstance(result, str) and result  # non-empty human-readable string


def test_blank_returns_none():
    assert _interval_preview("") is None
    assert _interval_preview("   ") is None


def test_invalid_interval_returns_none():
    # parse_interval raises ValueError; the helper must swallow it, not propagate.
    assert _interval_preview("definitely not an interval") is None
