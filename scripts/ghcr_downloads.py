#!/usr/bin/env python3
"""Emit a shields.io endpoint badge for the GHCR total-download count.

Reads the HTML of https://github.com/OWNER/REPO/pkgs/container/PKG on stdin and writes the badge
JSON to stdout. The page markup is the only source: GraphQL package statistics are deprecated for
the container registry and the REST package endpoints carry no download count.
"""

import json
import re
import sys

TOTAL_DOWNLOADS_RE = re.compile(r"Total downloads[\s\S]{0,500}?<h3\b([^>]*)>([^<]+)</h3>", re.IGNORECASE)
TITLE_ATTR_RE = re.compile(r'\btitle="([\d,]+)"', re.IGNORECASE)


def parse_count(html: str) -> int | None:
    """Return the total download count, preferring the exact value in the title attribute."""
    match = TOTAL_DOWNLOADS_RE.search(html)
    if not match:
        return None
    title = TITLE_ATTR_RE.search(match.group(1))
    digits = re.sub(r"\D", "", title.group(1) if title else match.group(2))
    return int(digits) if digits else None


def format_count(count: int) -> str:
    if count < 1000:
        return str(count)
    return f"{count / 1000:.1f}".removesuffix(".0") + "k"


def build_badge(count: int) -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "label": "docker pulls",
        "message": format_count(count),
        "color": "blue",
        "namedLogo": "docker",
    }


def main() -> int:
    count = parse_count(sys.stdin.read())
    if not count:
        print("no 'Total downloads' count found in package page HTML", file=sys.stderr)
        return 1
    print(json.dumps(build_badge(count)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
