"""Tests for OPML import/export functionality."""

from app.opml import MAX_IMPORT_TOPICS, export_opml, parse_opml

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
        opml = """<?xml version="1.0"?>
        <opml version="2.0"><body>
            <outline text="Private" xmlUrl="http://localhost:8080/feed" />
            <outline text="Public" xmlUrl="https://example.com/feed" />
        </body></opml>"""
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
