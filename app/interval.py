"""Human-readable interval parsing and formatting.

Supports units: m (minutes), h (hours), d (days), w (weeks), M (months).
Combined syntax: "1w 3d 2h" means 1 week + 3 days + 2 hours.
"""

import re

UNIT_MINUTES: dict[str, int] = {
    "m": 1,
    "h": 60,
    "d": 1440,
    "w": 10080,
    "M": 43200,  # 30 days
}

# Matches one or more "number + unit" tokens, e.g. "1w", "3d", "2h 30m"
_TOKEN_RE = re.compile(r"(\d+)\s*([mhdwM])")

MIN_INTERVAL_MINUTES = 10
MAX_INTERVAL_MINUTES = 6 * 43200  # 6 months = 259200 minutes


def parse_interval(s: str) -> int:
    """Parse a human-readable interval string into total minutes.

    Examples:
        "6h"     → 360
        "1w 3d"  → 14400
        "30m"    → 30
        "2M"     → 86400

    Args:
        s: Interval string using units m/h/d/w/M.

    Returns:
        Total minutes as an integer.

    Raises:
        ValueError: If the string is empty, has invalid format, duplicate units,
                    or the result is outside the allowed range.
    """
    s = s.strip()
    if not s:
        raise ValueError("Interval string is empty")

    tokens = _TOKEN_RE.findall(s)
    if not tokens:
        raise ValueError(
            f"Invalid interval format: '{s}'. "
            f"Use units: m (minutes), h (hours), d (days), w (weeks), M (months). "
            f"Example: '6h', '1w 3d', '2h 30m'"
        )

    # Verify the entire string is consumed by valid tokens (no trailing junk)
    reconstructed = re.sub(r"\s+", "", "".join(f"{n}{u}" for n, u in tokens))
    cleaned = re.sub(r"\s+", "", s)
    if reconstructed != cleaned:
        raise ValueError(
            f"Invalid interval format: '{s}'. "
            f"Use units: m (minutes), h (hours), d (days), w (weeks), M (months). "
            f"Example: '6h', '1w 3d', '2h 30m'"
        )

    seen_units: set[str] = set()
    total = 0
    for value_str, unit in tokens:
        if unit in seen_units:
            raise ValueError(f"Duplicate unit '{unit}' in interval '{s}'")
        seen_units.add(unit)
        value = int(value_str)
        if value <= 0:
            raise ValueError(f"Interval values must be positive, got {value}{unit}")
        total += value * UNIT_MINUTES[unit]

    if total < MIN_INTERVAL_MINUTES:
        raise ValueError(f"Interval too short: {total} minutes (minimum {MIN_INTERVAL_MINUTES} minutes)")
    if total > MAX_INTERVAL_MINUTES:
        raise ValueError(f"Interval too long: {total} minutes (maximum 6 months)")

    return total


def format_interval(minutes: int) -> str:
    """Format a minute count into a human-readable interval string.

    Decomposes into the largest fitting units, e.g.:
        360    → "6h"
        14400  → "1w 3d"
        90     → "1h 30m"

    Args:
        minutes: Total minutes (must be positive).

    Returns:
        Human-readable interval string.
    """
    if minutes <= 0:
        return "0m"

    parts: list[str] = []
    remaining = minutes

    for unit, unit_min in [("M", 43200), ("w", 10080), ("d", 1440), ("h", 60), ("m", 1)]:
        if remaining >= unit_min:
            count = remaining // unit_min
            remaining %= unit_min
            parts.append(f"{count}{unit}")

    return " ".join(parts)
