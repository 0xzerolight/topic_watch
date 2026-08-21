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

# One "number + unit" token, anchored at a cursor, e.g. "1w", " 3d", "2h".
# Matched with ``.match(s, pos)`` and never ``findall``: the unanchored form
# re-tried a greedy digit run from every offset, so a digit-only value cost
# quadratic time on the shared event loop (AUG-244).
_TOKEN_RE = re.compile(r"\s*(\d+)\s*([mhdwM])")

MIN_INTERVAL_MINUTES = 10
MAX_INTERVAL_MINUTES = 6 * 43200  # 6 months = 259200 minutes

# Longest accepted interval string. The longest meaningful value ("5M 3w 6d 23h
# 59m") is under 20 characters; anything past this is not a typo, and rejecting
# it before parsing keeps a crafted form field off the event loop entirely.
MAX_INTERVAL_CHARS = 64

_FORMAT_HELP = "Use units: m (minutes), h (hours), d (days), w (weeks), M (months). Example: '6h', '1w 3d', '2h 30m'"


def _invalid(value: str) -> ValueError:
    """The shared format error, echoing at most a short prefix of the input.

    The message used to interpolate the whole value, which turned a crafted
    field into an equally large error string, log line and re-rendered form.
    """
    shown = value[:32] + ("..." if len(value) > 32 else "")
    return ValueError(f"Invalid interval format: '{shown}'. {_FORMAT_HELP}")


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
    if len(s) > MAX_INTERVAL_CHARS:
        raise ValueError(f"Interval string is too long (maximum {MAX_INTERVAL_CHARS} characters)")

    # Single left-to-right pass. Each token is matched at the cursor and the
    # cursor advances past it, so the whole string is consumed exactly once and
    # trailing junk is a match failure rather than a separate reconstruction pass.
    tokens: list[tuple[str, str]] = []
    pos = 0
    while pos < len(s):
        match = _TOKEN_RE.match(s, pos)
        if match is None:
            raise _invalid(s)
        tokens.append((match.group(1), match.group(2)))
        pos = match.end()

    if not tokens:
        raise _invalid(s)

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
