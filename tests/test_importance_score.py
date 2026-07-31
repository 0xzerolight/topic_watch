"""Tests for the _importance_score Jinja2 filter.

The filter reads ``importance`` out of a stored ``llm_response`` blob. It must
never raise on a malformed or pre-m023 blob — the check-history table renders one
per row, and a check recorded before importance scoring existed simply has no
score to show.
"""

import json

from app.web.routers.templates import _importance_score


def test_extracts_importance():
    assert _importance_score(json.dumps({"has_new_info": True, "importance": 4})) == 4


def test_extracts_boundary_values():
    assert _importance_score(json.dumps({"importance": 1})) == 1
    assert _importance_score(json.dumps({"importance": 5})) == 5


def test_missing_importance_returns_none():
    """Blobs written before m023 have no importance key."""
    assert _importance_score(json.dumps({"has_new_info": True, "confidence": 0.9})) is None


def test_null_importance_returns_none():
    assert _importance_score(json.dumps({"importance": None})) is None


def test_non_numeric_importance_returns_none():
    assert _importance_score(json.dumps({"importance": "high"})) is None


def test_float_importance_is_truncated_to_int():
    """A provider that emits 4.0 still renders a usable score."""
    assert _importance_score(json.dumps({"importance": 4.0})) == 4


def test_none_blob_returns_none():
    assert _importance_score(None) is None


def test_empty_blob_returns_none():
    assert _importance_score("") is None


def test_invalid_json_returns_none():
    assert _importance_score("not valid json {{{") is None


def test_non_object_json_returns_none():
    """A bare JSON scalar/array has no keys to read."""
    assert _importance_score("[1, 2, 3]") is None
    assert _importance_score("42") is None
