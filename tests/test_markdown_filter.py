"""Tests for the ``markdown`` Jinja2 filter that renders LLM knowledge summaries.

The knowledge prompt instructs the model to emit markdown (``**Current Status:**``
headers plus ``-`` bullet lists). The summary is rendered at
``topic_status.html`` and must become real HTML, not literal asterisks — while
staying safe against stored XSS from article-derived text. ``_markdown`` uses
markdown-it-py configured ``html=False`` (raw HTML escaped, unsafe link schemes
rejected) plus ``_normalize_markdown`` to restore list structure.
"""

from markupsafe import Markup

from app.web.routers.templates import _markdown, _normalize_markdown, templates


class TestNormalizeMarkdown:
    def test_blank_line_inserted_before_label_and_list(self) -> None:
        # Real prompt format: labels and bullets with no blank lines between them.
        text = "**Current Status:** ok\n**Confirmed Facts:**\n- a\n- b"
        normalized = _normalize_markdown(text)
        # A blank line separates the two labels and precedes the list run...
        assert "**Current Status:** ok\n\n**Confirmed Facts:**\n\n- a" in normalized
        # ...but consecutive bullets stay tight (no blank line between them).
        assert "- a\n- b" in normalized

    def test_idempotent_on_already_spaced_input(self) -> None:
        text = "**Current Status:** ok\n**Confirmed Facts:**\n- a\n- b"
        once = _normalize_markdown(text)
        assert _normalize_markdown(once) == once

    def test_plain_text_unchanged(self) -> None:
        assert _normalize_markdown("just a sentence") == "just a sentence"


class TestMarkdownFilter:
    def test_bold_label_renders_strong(self) -> None:
        out = _markdown("**Current Status:** still banned")
        assert "<strong>Current Status:</strong>" in out
        assert "**" not in out

    def test_bullets_render_as_list(self) -> None:
        out = _markdown("**Confirmed Facts:**\n- fact a\n- fact b")
        assert "<ul>" in out
        assert "<li>fact a</li>" in out
        assert "<li>fact b</li>" in out

    def test_labels_render_as_separate_blocks(self) -> None:
        out = _markdown("**Current Status:** ok\n**Confirmed Facts:**\n- a")
        # Each label is its own paragraph, not merged into one.
        assert "<strong>Current Status:</strong>" in out
        assert "<strong>Confirmed Facts:</strong>" in out
        assert "</p>\n<p>" in out

    def test_plain_text_passthrough(self) -> None:
        out = _markdown("just a sentence")
        assert "just a sentence" in out
        assert "**" not in out

    def test_none_returns_empty_markup(self) -> None:
        result = _markdown(None)
        assert result == Markup("")
        assert isinstance(result, Markup)

    def test_empty_string_returns_empty_markup(self) -> None:
        result = _markdown("")
        assert result == Markup("")
        assert isinstance(result, Markup)

    def test_filter_registered_on_environment(self) -> None:
        assert templates.env.filters.get("markdown") is _markdown

    def test_template_render_produces_html_not_literal_asterisks(self) -> None:
        # End-to-end: autoescape must NOT double-escape the Markup result.
        rendered = templates.env.from_string("{{ x | markdown }}").render(x="**Bold:** v")
        assert "<strong>Bold:</strong>" in rendered
        assert "**Bold:**" not in rendered


class TestMarkdownXssPolicy:
    def test_raw_script_tag_escaped(self) -> None:
        out = _markdown("<script>alert(1)</script>")
        assert "<script>" not in out
        assert "&lt;script&gt;" in out

    def test_inline_event_handler_tag_escaped(self) -> None:
        out = _markdown("text <img src=x onerror=alert(1)> more")
        assert "<img" not in out
        assert "&lt;img" in out

    def test_javascript_link_yields_no_href(self) -> None:
        out = _markdown("[x](javascript:alert(1))")
        assert 'href="javascript:' not in out
        assert "<a " not in out

    def test_http_link_is_inert_text_not_an_anchor(self) -> None:
        # AUG-016: the summary is model-derived and a malicious feed can steer it,
        # so a link inside it must not become a clickable anchor on a page the user
        # trusts. The old assertion here pinned exactly that anchor.
        out = _markdown("[ok](http://example.com)")
        assert "<a " not in out
        assert "href=" not in out
        # The destination stays visible as text — hiding it is the phishing shape.
        assert "ok" in out
        assert "http://example.com" in out

    def test_disguised_link_text_cannot_hide_the_destination(self) -> None:
        out = _markdown("Verify at [your account portal](https://attacker.example/login).")
        assert "<a " not in out
        assert "attacker.example" in out

    def test_autolink_is_inert_too(self) -> None:
        out = _markdown("<https://attacker.example/x>")
        assert "<a " not in out
        assert "attacker.example" in out

    def test_reference_link_is_inert_too(self) -> None:
        out = _markdown("See [the docs][1].\n\n[1]: https://attacker.example/x")
        assert "<a " not in out
        assert "attacker.example" in out

    def test_nested_links_do_not_leak_a_destination(self) -> None:
        # Two links in one paragraph must each keep their own URL.
        out = _markdown("[a](https://one.example) and [b](https://two.example)")
        assert "<a " not in out
        assert "one.example" in out
        assert "two.example" in out

    def test_inert_link_url_is_html_escaped(self) -> None:
        out = _markdown('[x](https://e.example/"><script>alert(1)</script>)')
        assert "<script>" not in out

    def test_data_uri_image_is_inert(self) -> None:
        out = _markdown("![a](data:text/html,<b>)")
        assert "<img" not in out

    def test_nul_byte_scheme_does_not_yield_javascript_href(self) -> None:
        # A NUL/control char between the scheme and ':' must not produce an
        # executable javascript: href (markdown-it normalization guarantee).
        out = _markdown("[x](javascript\x00:alert(1))")
        assert 'href="javascript:' not in out

    def test_code_fence_does_not_leak_live_tag(self) -> None:
        out = _markdown("```\n<script>alert(1)</script>\n```")
        assert "<code>" in out
        assert "<script>" not in out
