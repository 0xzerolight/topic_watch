"""Tests for the interval parsing and formatting module."""

import pytest

from app.interval import format_interval, parse_interval


class TestParseInterval:
    """Test parse_interval with valid and invalid inputs."""

    def test_minutes(self) -> None:
        assert parse_interval("30m") == 30

    def test_hours(self) -> None:
        assert parse_interval("6h") == 360

    def test_days(self) -> None:
        assert parse_interval("1d") == 1440

    def test_weeks(self) -> None:
        assert parse_interval("2w") == 20160

    def test_months(self) -> None:
        assert parse_interval("1M") == 43200

    def test_combined_units(self) -> None:
        assert parse_interval("1w 3d") == 10080 + 4320

    def test_combined_hours_minutes(self) -> None:
        assert parse_interval("2h 30m") == 150

    def test_combined_all_units(self) -> None:
        assert parse_interval("1M 1w 1d 1h 1m") == 43200 + 10080 + 1440 + 60 + 1

    def test_no_spaces(self) -> None:
        assert parse_interval("1w3d") == 10080 + 4320

    def test_whitespace_stripped(self) -> None:
        assert parse_interval("  6h  ") == 360

    def test_minimum_valid(self) -> None:
        assert parse_interval("10m") == 10

    def test_maximum_valid(self) -> None:
        assert parse_interval("6M") == 259200

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            parse_interval("")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            parse_interval("   ")

    def test_no_unit_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid interval"):
            parse_interval("30")

    def test_invalid_unit_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid interval"):
            parse_interval("6x")

    def test_text_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid interval"):
            parse_interval("six hours")

    def test_duplicate_unit_raises(self) -> None:
        with pytest.raises(ValueError, match="Duplicate unit"):
            parse_interval("2h 3h")

    def test_below_minimum_raises(self) -> None:
        with pytest.raises(ValueError, match="too short"):
            parse_interval("5m")

    def test_above_maximum_raises(self) -> None:
        with pytest.raises(ValueError, match="too long"):
            parse_interval("7M")

    def test_zero_value_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            parse_interval("0h")


class TestFormatInterval:
    """Test format_interval decomposition."""

    def test_exact_hours(self) -> None:
        assert format_interval(360) == "6h"

    def test_exact_days(self) -> None:
        assert format_interval(1440) == "1d"

    def test_exact_weeks(self) -> None:
        assert format_interval(10080) == "1w"

    def test_exact_months(self) -> None:
        assert format_interval(43200) == "1M"

    def test_hours_and_minutes(self) -> None:
        assert format_interval(150) == "2h 30m"

    def test_weeks_and_days(self) -> None:
        assert format_interval(14400) == "1w 3d"

    def test_just_minutes(self) -> None:
        assert format_interval(30) == "30m"

    def test_complex_decomposition(self) -> None:
        total = 43200 + 10080 + 1440 + 60 + 1
        assert format_interval(total) == "1M 1w 1d 1h 1m"

    def test_zero_returns_zero(self) -> None:
        assert format_interval(0) == "0m"

    def test_negative_returns_zero(self) -> None:
        assert format_interval(-5) == "0m"


class TestRoundTrip:
    """Verify parse → format → parse produces consistent results."""

    @pytest.mark.parametrize(
        "input_str",
        ["10m", "30m", "1h", "6h", "1d", "2d 12h", "1w", "1w 3d", "2w", "1M", "3M", "6M"],
    )
    def test_round_trip(self, input_str: str) -> None:
        minutes = parse_interval(input_str)
        formatted = format_interval(minutes)
        assert parse_interval(formatted) == minutes
