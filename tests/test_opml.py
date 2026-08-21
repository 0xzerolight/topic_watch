"""Tests for OPML import/export functionality."""

from collections.abc import Iterator
from unittest.mock import patch

import pytest

from app.opml import MAX_IMPORT_TOPICS, export_opml, parse_opml
from app.url_validation import validate_feed_url as _real_validate_feed_url


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


class TestParseOPMLGroupIdentity:
    """AUG-204: display text is not identity — only our own group marker is."""

    def test_third_party_same_name_feeds_stay_separate(self):
        opml = """<?xml version="1.0"?>
        <opml version="2.0"><body>
            <outline text="Politics"><outline text="News" xmlUrl="https://a.example.com/feed" /></outline>
            <outline text="Sport"><outline text="News" xmlUrl="https://b.example.com/feed" /></outline>
        </body></opml>"""
        with patch("app.opml.validate_feed_url", return_value=None):
            result = parse_opml(opml, set())

        assert len(result.topics) == 2
        # The second keeps its own folder tag instead of being dropped into the first.
        by_tag = {t["tags"][0]: t for t in result.topics}
        assert set(by_tag) == {"Politics", "Sport"}
        assert by_tag["Sport"]["feed_urls"] == ["https://b.example.com/feed"]
        # The duplicate name is disambiguated, not silently merged or lost.
        assert len({t["name"] for t in result.topics}) == 2
        assert any("shared a name" in w for w in result.warnings)

    def test_marked_outlines_still_merge(self):
        from app.opml import TOPIC_ATTR

        opml = f"""<?xml version="1.0"?>
        <opml version="2.0"><body>
            <outline text="Multi" {TOPIC_ATTR}="Multi" xmlUrl="https://a.example.com/feed" />
            <outline text="Multi" {TOPIC_ATTR}="Multi" xmlUrl="https://b.example.com/feed" />
        </body></opml>"""
        with patch("app.opml.validate_feed_url", return_value=None):
            result = parse_opml(opml, set())

        assert len(result.topics) == 1
        assert len(result.topics[0]["feed_urls"]) == 2

    def test_per_topic_feed_cap(self):
        from app.opml import MAX_FEEDS_PER_TOPIC, TOPIC_ATTR

        feeds = "".join(
            f'<outline text="Multi" {TOPIC_ATTR}="Multi" xmlUrl="https://e{i}.example.com/feed" />'
            for i in range(MAX_FEEDS_PER_TOPIC + 3)
        )
        opml = f'<?xml version="1.0"?><opml version="2.0"><body>{feeds}</body></opml>'
        with patch("app.opml.validate_feed_url", return_value=None):
            result = parse_opml(opml, set())

        assert len(result.topics) == 1
        assert len(result.topics[0]["feed_urls"]) == MAX_FEEDS_PER_TOPIC

    def test_imported_name_is_length_capped(self):
        from app.opml import MAX_TOPIC_NAME_CHARS

        long_name = "N" * (MAX_TOPIC_NAME_CHARS + 200)
        opml = (
            f'<?xml version="1.0"?><opml version="2.0"><body>'
            f'<outline text="{long_name}" xmlUrl="https://a.example.com/feed" />'
            f"</body></opml>"
        )
        with patch("app.opml.validate_feed_url", return_value=None):
            result = parse_opml(opml, set())

        assert len(result.topics[0]["name"]) == MAX_TOPIC_NAME_CHARS


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

    def test_round_trip_preserves_every_tag(self):
        """TW-AUD-026: folders carry one tag; the export carries them all."""
        original_topics = [
            {"name": "Multi Tag", "feed_urls": ["https://a.example.com/feed"], "tags": ["Policy, Europe", "Energy"]},
        ]
        xml = export_opml(original_topics)
        result = parse_opml(xml, set())
        assert result.topics[0]["tags"] == ["Policy, Europe", "Energy"]

    def test_omitted_topics_are_disclosed_in_the_file(self):
        """TW-AUD-026: an OPML download must not pass for a complete backup."""
        xml = export_opml([{"name": "Kept", "feed_urls": ["https://a.example.com/feed"], "tags": []}], omitted_count=3)
        assert "3 topic(s) omitted" in xml
        assert "JSON export" in xml

    def test_nothing_omitted_adds_no_note(self):
        xml = export_opml([{"name": "Kept", "feed_urls": ["https://a.example.com/feed"], "tags": []}])
        assert "omitted" not in xml

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


class TestImportUrlBudget:
    """DNS work is capped BEFORE per-topic processing, not after (AUG-012)."""

    def _opml(self, n: int) -> str:
        outlines = "".join(f'<outline text="Feed {i}" xmlUrl="https://e{i}.example.com/feed" />' for i in range(n))
        return f'<?xml version="1.0"?><opml version="2.0"><body>{outlines}</body></opml>'

    def test_validation_stops_at_the_url_budget(self):
        from app.opml import MAX_IMPORT_FEED_URLS

        seen: list[str] = []

        def _fake(url: str) -> None:
            seen.append(url)
            return None

        with patch("app.opml.validate_feed_url", side_effect=_fake):
            result = parse_opml(self._opml(MAX_IMPORT_FEED_URLS + 25), set())

        assert len(seen) == MAX_IMPORT_FEED_URLS
        assert any(str(MAX_IMPORT_FEED_URLS) in w for w in result.warnings)


class TestEntityExpansion:
    """A DTD/entity declaration is refused before the tree is built (AUG-015)."""

    def test_internal_entity_document_is_rejected(self):
        opml = (
            '<?xml version="1.0"?>'
            '<!DOCTYPE opml [<!ENTITY a "AAAAAAAAAA">'
            '<!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">'
            '<!ENTITY c "&b;&b;&b;&b;&b;&b;&b;&b;&b;&b;">]>'
            '<opml version="2.0"><body>'
            '<outline text="&c;" xmlUrl="https://a.example.com/feed" />'
            "</body></opml>"
        )
        result = parse_opml(opml, set())

        assert result.topics == []
        assert any("entit" in w.lower() or "doctype" in w.lower() for w in result.warnings)

    def test_plain_opml_still_parses(self):
        result = parse_opml(VALID_OPML, set())
        assert len(result.topics) == 2


class TestMalformedUrlIsolation:
    """One malformed outline skips that entry, not the whole import (AUG-205)."""

    def test_malformed_url_skips_only_that_entry(self):
        opml = (
            '<?xml version="1.0"?><opml version="2.0"><body>'
            '<outline text="Broken" xmlUrl="http://[::1" />'
            '<outline text="Good" xmlUrl="https://good.example.com/feed" />'
            "</body></opml>"
        )
        # Real validation (no stub): the malformed URL must not propagate.
        with patch("app.opml.validate_feed_url", new=_real_validate_feed_url):
            result = parse_opml(opml, set())

        assert [t["name"] for t in result.topics] == ["Good"]
        assert result.skipped_invalid == 1
