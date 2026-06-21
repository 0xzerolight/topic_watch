"""Tests for OPML import/export functionality."""

from collections.abc import Iterator
from unittest.mock import patch

import pytest

from app.opml import MAX_IMPORT_TOPICS, export_opml, parse_opml


@pytest.fixture(autouse=True)
def _stub_feed_url_validation() -> Iterator[None]:
    """OVH-083: stub the SSRF resolver so parse tests never make live DNS calls.

    ``validate_feed_url`` resolves each host (fail-closed SSRF check), so any
    unmocked ``parse_opml`` call here would hit the network — flaky on CI runners
    without outbound DNS and green-for-the-wrong-reason behind a captive resolver.
    Every parse/round-trip test treats all URLs as valid; the one dedicated SSRF
    test (``test_ssrf_private_url_skipped``) overrides this with its own explicit,
    DNS-free mock.
    """
    with patch("app.opml.validate_feed_url", return_value=None):
        yield


VALID_OPML = """<?xml version="1.0" encoding="UTF-8"?>
<opml version="2.0">
    <head><title>Test Feeds</title></head>
    <body>
        <outline text="Hacker News" xmlUrl="https://news.ycombinator.com/rss" />
        <outline text="Lobsters" xmlUrl="https://lobste.rs/rss" />
    </body>
</opml>"""

NESTED_OPML = """<?xml version="1.0" encoding="UTF-8"?>
<opml version="2.0">
    <head><title>Test</title></head>
    <body>
        <outline text="Tech">
            <outline text="Hacker News" xmlUrl="https://news.ycombinator.com/rss" />
            <outline text="Lobsters" xmlUrl="https://lobste.rs/rss" />
        </outline>
        <outline text="Science">
            <outline text="ArXiv" xmlUrl="https://arxiv.org/rss/cs.AI" />
        </outline>
    </body>
</opml>"""


class TestParseOPML:
    def test_valid_flat_opml(self):
        result = parse_opml(VALID_OPML, set())
        assert len(result.topics) == 2
        assert result.topics[0]["name"] == "Hacker News"
        assert result.topics[0]["feed_urls"] == ["https://news.ycombinator.com/rss"]
        assert result.topics[1]["name"] == "Lobsters"
        assert result.skipped_dupes == 0
        assert result.skipped_invalid == 0

    def test_nested_opml_extracts_tags(self):
        result = parse_opml(NESTED_OPML, set())
        assert len(result.topics) == 3
        hn = next(t for t in result.topics if t["name"] == "Hacker News")
        assert hn["tags"] == ["Tech"]
        arxiv = next(t for t in result.topics if t["name"] == "ArXiv")
        assert arxiv["tags"] == ["Science"]

    def test_dedup_skips_existing_urls(self):
        existing = {"https://news.ycombinator.com/rss"}
        result = parse_opml(VALID_OPML, existing)
        assert len(result.topics) == 1
        assert result.topics[0]["name"] == "Lobsters"
        assert result.skipped_dupes == 1

    def test_dedup_within_same_import(self):
        opml = """<?xml version="1.0"?>
        <opml version="2.0"><body>
            <outline text="Feed A" xmlUrl="https://example.com/feed" />
            <outline text="Feed B" xmlUrl="https://example.com/feed" />
        </body></opml>"""
        result = parse_opml(opml, set())
        assert len(result.topics) == 1
        assert result.skipped_dupes == 1

    def test_malformed_xml(self):
        result = parse_opml("<not valid xml!!!>", set())
        assert len(result.topics) == 0
        assert len(result.warnings) == 1
        assert "Invalid XML" in result.warnings[0]

    def test_empty_opml(self):
        opml = '<?xml version="1.0"?><opml version="2.0"><head/><body/></opml>'
        result = parse_opml(opml, set())
        assert len(result.topics) == 0
        assert any("No feeds found" in w for w in result.warnings)

    def test_no_body_element(self):
        opml = '<?xml version="1.0"?><opml version="2.0"><head/></opml>'
        result = parse_opml(opml, set())
        assert len(result.topics) == 0
        assert any("No <body>" in w for w in result.warnings)

    def test_missing_title_uses_domain(self):
        opml = """<?xml version="1.0"?>
        <opml version="2.0"><body>
            <outline xmlUrl="https://example.com/feed.xml" />
        </body></opml>"""
        result = parse_opml(opml, set())
        assert len(result.topics) == 1
        assert result.topics[0]["name"] == "example.com"

    def test_ssrf_private_url_skipped(self):
        """OVH-083: the one dedicated SSRF test, mocked explicitly (no live DNS).

        ``validate_feed_url`` is mocked to reject ONLY the private URL, proving
        ``parse_opml`` routes a validation error into ``skipped_invalid`` and drops
        the offending feed — without resolving any host on the network.
        """
        opml = """<?xml version="1.0"?>
        <opml version="2.0"><body>
            <outline text="Private" xmlUrl="http://localhost:8080/feed" />
            <outline text="Public" xmlUrl="https://example.com/feed" />
        </body></opml>"""

        def fake_validate(url: str) -> str | None:
            if "localhost" in url:
                return f"Feed URL points to a private/reserved address: {url}"
            return None

        with patch("app.opml.validate_feed_url", side_effect=fake_validate):
            result = parse_opml(opml, set())
        assert len(result.topics) == 1
        assert result.topics[0]["name"] == "Public"
        assert result.skipped_invalid == 1

    def test_truncation_at_max_topics(self):
        outlines = "\n".join(
            f'<outline text="Feed {i:04d}" xmlUrl="https://example{i}.com/feed" />'
            for i in range(MAX_IMPORT_TOPICS + 50)
        )
        opml = f'<?xml version="1.0"?><opml version="2.0"><body>{outlines}</body></opml>'
        # Mock URL validation to avoid 550 real DNS lookups (is_private_url resolves
        # each host); this test exercises truncation logic, not SSRF validation.
        with patch("app.opml.validate_feed_url", return_value=None):
            result = parse_opml(opml, set())
        assert len(result.topics) == MAX_IMPORT_TOPICS
        assert any("Imported first" in w for w in result.warnings)

    def test_depth_limit_prevents_deep_nesting(self):
        # Build deeply nested outline (15 levels)
        inner = '<outline text="Deep" xmlUrl="https://deep.example.com/feed" />'
        for i in range(15):
            inner = f'<outline text="Level {i}">{inner}</outline>'
        opml = f'<?xml version="1.0"?><opml version="2.0"><body>{inner}</body></opml>'
        result = parse_opml(opml, set())
        # Feed is at depth 16, cap is 10, so it should not be found
        assert len(result.topics) == 0

    def test_empty_xmlurl_ignored(self):
        opml = """<?xml version="1.0"?>
        <opml version="2.0"><body>
            <outline text="Empty" xmlUrl="" />
            <outline text="Valid" xmlUrl="https://example.com/feed" />
        </body></opml>"""
        result = parse_opml(opml, set())
        assert len(result.topics) == 1


class TestParseOPMLStructuralWalk:
    """OVH-071: structural parsing must be unit-testable without sockets/DNS.

    The recursive walk extracts structure; SSRF validation is a separate pass.
    Mocking ``validate_feed_url`` proves no DNS happens during structural parsing.
    """

    def test_parse_without_dns_when_validation_mocked(self):
        with patch("app.opml.validate_feed_url", return_value=None) as mock_validate:
            result = parse_opml(VALID_OPML, set())
        assert len(result.topics) == 2
        # Validation runs once per surviving (deduped) candidate URL.
        assert mock_validate.call_count == 2

    def test_url_dedup_runs_before_validation(self):
        """Duplicate URLs are dropped structurally, so validation never sees them."""
        existing = {"https://news.ycombinator.com/rss"}
        with patch("app.opml.validate_feed_url", return_value=None) as mock_validate:
            result = parse_opml(VALID_OPML, existing)
        assert result.skipped_dupes == 1
        assert len(result.topics) == 1
        # Only the surviving (non-dupe) URL is validated.
        assert mock_validate.call_count == 1
        assert mock_validate.call_args.args == ("https://lobste.rs/rss",)

    def test_nested_structure_walked_without_dns(self):
        with patch("app.opml.validate_feed_url", return_value=None):
            result = parse_opml(NESTED_OPML, set())
        assert len(result.topics) == 3
        hn = next(t for t in result.topics if t["name"] == "Hacker News")
        assert hn["tags"] == ["Tech"]


class TestParseOPMLNameCollision:
    """OVH-072: name collisions with existing DB topics live in OPMLResult."""

    def test_existing_topic_name_skipped_and_counted(self):
        with patch("app.opml.validate_feed_url", return_value=None):
            result = parse_opml(VALID_OPML, set(), existing_topic_names={"Hacker News"})
        assert result.skipped_name_dupes == 1
        names = {t["name"] for t in result.topics}
        assert "Hacker News" not in names
        assert "Lobsters" in names

    def test_no_collision_when_name_absent(self):
        with patch("app.opml.validate_feed_url", return_value=None):
            result = parse_opml(VALID_OPML, set(), existing_topic_names={"Unrelated"})
        assert result.skipped_name_dupes == 0
        assert len(result.topics) == 2

    def test_default_no_existing_names_keeps_all(self):
        with patch("app.opml.validate_feed_url", return_value=None):
            result = parse_opml(VALID_OPML, set())
        assert result.skipped_name_dupes == 0
        assert len(result.topics) == 2

    def test_multi_feed_collision_counted_once(self):
        """A multi-feed topic colliding with a DB name counts as one name-dupe."""
        opml = """<?xml version="1.0"?>
        <opml version="2.0"><body>
            <outline text="Multi" xmlUrl="https://a.example.com/feed" />
            <outline text="Multi" xmlUrl="https://b.example.com/feed" />
        </body></opml>"""
        with patch("app.opml.validate_feed_url", return_value=None):
            result = parse_opml(opml, set(), existing_topic_names={"Multi"})
        assert result.skipped_name_dupes == 1
        assert result.topics == []


class TestExportOPML:
    def test_export_basic(self):
        topics = [
            {"name": "Hacker News", "feed_urls": ["https://news.ycombinator.com/rss"], "tags": []},
            {"name": "Lobsters", "feed_urls": ["https://lobste.rs/rss"], "tags": []},
        ]
        xml = export_opml(topics)
        assert "Hacker News" in xml
        assert "https://news.ycombinator.com/rss" in xml
        assert "Lobsters" in xml

    def test_export_with_tags_creates_folders(self):
        topics = [
            {"name": "HN", "feed_urls": ["https://hn.com/rss"], "tags": ["Tech"]},
            {"name": "ArXiv", "feed_urls": ["https://arxiv.org/rss"], "tags": ["Science"]},
        ]
        xml = export_opml(topics)
        assert 'text="Tech"' in xml
        assert 'text="Science"' in xml

    def test_export_empty_topics(self):
        xml = export_opml([])
        assert "<body" in xml
        assert "Topic Watch Export" in xml

    def test_round_trip(self):
        """Export then import should recover the same feeds."""
        original_topics = [
            {"name": "Feed A", "feed_urls": ["https://a.example.com/feed"], "tags": []},
            {"name": "Feed B", "feed_urls": ["https://b.example.com/feed"], "tags": ["Tech"]},
        ]
        xml = export_opml(original_topics)
        result = parse_opml(xml, set())
        assert len(result.topics) == 2
        names = {t["name"] for t in result.topics}
        assert "Feed A" in names
        assert "Feed B" in names

    def test_round_trip_multi_feed_topic(self):
        """A topic with multiple feeds must round-trip as ONE topic with both feeds."""
        original_topics = [
            {
                "name": "Multi",
                "feed_urls": ["https://a.example.com/feed", "https://b.example.com/feed"],
                "tags": [],
            },
        ]
        xml = export_opml(original_topics)
        result = parse_opml(xml, set())
        assert len(result.topics) == 1
        assert result.topics[0]["name"] == "Multi"
        assert set(result.topics[0]["feed_urls"]) == {
            "https://a.example.com/feed",
            "https://b.example.com/feed",
        }

    def test_round_trip_multi_feed_topic_in_folder(self):
        """Multi-feed topic inside a tag folder also merges into one topic."""
        original_topics = [
            {
                "name": "Multi",
                "feed_urls": ["https://a.example.com/feed", "https://b.example.com/feed"],
                "tags": ["Tech"],
            },
        ]
        xml = export_opml(original_topics)
        result = parse_opml(xml, set())
        assert len(result.topics) == 1
        assert result.topics[0]["tags"] == ["Tech"]
        assert set(result.topics[0]["feed_urls"]) == {
            "https://a.example.com/feed",
            "https://b.example.com/feed",
        }


class TestParseOPMLResolverTimeout:
    """OVH-148: OPML import DNS validation is bounded by a resolver timeout."""

    def test_slow_host_does_not_block_import(self, monkeypatch):
        """A crafted slow-resolving host can't occupy a worker for minutes.

        Runs the REAL SSRF validation path (overriding the autouse mock) so the
        bounded getaddrinfo in url_validation is exercised end-to-end: a host
        whose resolution hangs is given up on after the resolver timeout and
        skipped as invalid, rather than serializing into a multi-minute import.
        """
        import socket
        import time

        from app import url_validation

        monkeypatch.setattr(url_validation, "_RESOLVE_TIMEOUT", 0.1)

        def _slow(*_args, **_kwargs):
            time.sleep(5)  # would stall the whole import if the resolver weren't bounded
            return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))]

        monkeypatch.setattr(socket, "getaddrinfo", _slow)
        # Undo the autouse validate_feed_url stub so real DNS validation runs.
        monkeypatch.setattr("app.opml.validate_feed_url", url_validation.validate_feed_url)

        opml = """<?xml version="1.0"?>
        <opml version="2.0"><body>
            <outline text="Slow" xmlUrl="https://slow.example.com/feed" />
        </body></opml>"""

        start = time.monotonic()
        result = parse_opml(opml, set())
        elapsed = time.monotonic() - start

        assert elapsed < 2.0  # bounded — did not wait for the 5s resolver
        # Fail-closed: unverifiable host is skipped, never imported.
        assert result.topics == []
        assert result.skipped_invalid == 1
