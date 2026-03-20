"""Tests for the _confidence_badge Jinja2 filter."""

import json

from app.web.routes import _confidence_badge


def test_high_confidence_green_badge():
    data = {"confidence": 0.9, "has_new_info": True}
    result = _confidence_badge(json.dumps(data))
    assert isinstance(result, str)
    assert "#2ecc40" in result
    assert "0.90" in result


def test_medium_confidence_yellow_badge():
    data = {"confidence": 0.65, "has_new_info": True}
    result = _confidence_badge(json.dumps(data))
    assert isinstance(result, str)
    assert "#ffdc00" in result
    assert "0.65" in result


def test_low_confidence_red_badge():
    data = {"confidence": 0.3, "has_new_info": False}
    result = _confidence_badge(json.dumps(data))
    assert isinstance(result, str)
    assert "#ff4136" in result
    assert "0.30" in result


def test_confidence_at_08_boundary_is_green():
    data = {"confidence": 0.8}
    result = _confidence_badge(json.dumps(data))
    assert "#2ecc40" in result


def test_confidence_at_05_boundary_is_yellow():
    data = {"confidence": 0.5}
    result = _confidence_badge(json.dumps(data))
    assert "#ffdc00" in result


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


def test_html_structure_rounded_corners():
    data = {"confidence": 0.9}
    result = _confidence_badge(json.dumps(data))
    assert "border-radius" in result


def test_confidence_exactly_zero_is_red():
    data = {"confidence": 0.0}
    result = _confidence_badge(json.dumps(data))
    assert "#ff4136" in result


def test_confidence_exactly_one_is_green():
    data = {"confidence": 1.0}
    result = _confidence_badge(json.dumps(data))
    assert "#2ecc40" in result
