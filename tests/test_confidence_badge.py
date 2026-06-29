"""Tests for the _confidence_badge Jinja2 filter.

The filter emits class-based badges (``.badge--conf-{high,mid,low}``) — colors
live in components.css so every theme inherits them; no inline hex.
"""

import json

from app.web.routers.templates import _confidence_badge


def test_high_confidence_badge():
    data = {"confidence": 0.9, "has_new_info": True}
    result = _confidence_badge(json.dumps(data))
    assert isinstance(result, str)
    assert "badge--conf-high" in result
    assert "0.90" in result


def test_medium_confidence_badge():
    data = {"confidence": 0.65, "has_new_info": True}
    result = _confidence_badge(json.dumps(data))
    assert isinstance(result, str)
    assert "badge--conf-mid" in result
    assert "0.65" in result


def test_low_confidence_badge():
    data = {"confidence": 0.3, "has_new_info": False}
    result = _confidence_badge(json.dumps(data))
    assert isinstance(result, str)
    assert "badge--conf-low" in result
    assert "0.30" in result


def test_confidence_at_08_boundary_is_high():
    data = {"confidence": 0.8}
    result = _confidence_badge(json.dumps(data))
    assert "badge--conf-high" in result


def test_confidence_at_05_boundary_is_mid():
    data = {"confidence": 0.5}
    result = _confidence_badge(json.dumps(data))
    assert "badge--conf-mid" in result


def test_none_input_returns_dash():
    result = _confidence_badge(None)
    assert result == "-"


def test_empty_string_returns_dash():
    result = _confidence_badge("")
    assert result == "-"


def test_invalid_json_returns_dash():
    result = _confidence_badge("not valid json {{{")
    assert result == "-"


def test_json_missing_confidence_key_returns_dash():
    data = {"has_new_info": True, "summary": "some summary"}
    result = _confidence_badge(json.dumps(data))
    assert result == "-"


def test_html_structure_span_tag():
    data = {"confidence": 0.75}
    result = _confidence_badge(json.dumps(data))
    assert isinstance(result, str)
    assert "<span" in result
    assert "</span>" in result


def test_html_structure_title_attribute():
    data = {"confidence": 0.75}
    result = _confidence_badge(json.dumps(data))
    assert 'title="Confidence: 0.75"' in result


def test_html_structure_uses_badge_class_not_inline_hex():
    data = {"confidence": 0.9}
    result = _confidence_badge(json.dumps(data))
    assert 'class="badge badge--conf-high"' in result
    # No inline style / raw hex leaks through.
    assert "style=" not in result
    assert "#" not in result


def test_confidence_exactly_zero_is_low():
    data = {"confidence": 0.0}
    result = _confidence_badge(json.dumps(data))
    assert "badge--conf-low" in result


def test_confidence_exactly_one_is_high():
    data = {"confidence": 1.0}
    result = _confidence_badge(json.dumps(data))
    assert "badge--conf-high" in result
