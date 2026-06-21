"""Reusable test infrastructure for live-exercising integration tests.

These helpers stub ONLY the outermost boundaries (HTTP and the LLM API),
so tests can exercise the real production pipeline between those edges
instead of mocking every internal seam. Heavy internal mocking is what
hides integration bugs; these helpers avoid it.

Modules:
    rss_fixtures      -- canned RSS/Atom feeds served over httpx.MockTransport.
    redirect_transport -- httpx.MockTransport returning 3xx redirects (SSRF building block).
    stub_llm          -- context managers stubbing the LLM boundary (no live calls).
"""

from tests.helpers.redirect_transport import build_redirect_transport
from tests.helpers.rss_fixtures import (
    RssEntry,
    build_atom_xml,
    build_rss_transport,
    build_rss_xml,
)
from tests.helpers.stub_llm import (
    make_compressed_knowledge,
    make_knowledge_update,
    make_novelty_result,
    stub_llm_boundary,
)

__all__ = [
    "RssEntry",
    "build_atom_xml",
    "build_redirect_transport",
    "build_rss_transport",
    "build_rss_xml",
    "make_compressed_knowledge",
    "make_knowledge_update",
    "make_novelty_result",
    "stub_llm_boundary",
]
