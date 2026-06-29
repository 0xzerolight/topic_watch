"""Consolidated data-label / column-count contract for the ``.data-table`` views.

The ``.data-table`` mobile collapse (components.css, <768px) turns every data
``<td>`` into a labelled card row via ``content: attr(data-label)``. A cell that
forgets its ``data-label`` collapses with no header on mobile. These tests render
the real templates and assert the contract holds across the dashboard, the feeds
page, and the topic-detail page — and that the dashboard's row template, table
header, and empty-state ``colspan`` all agree on the column count.
"""

import re
import sqlite3
from collections.abc import AsyncGenerator
from html.parser import HTMLParser
from pathlib import Path

import httpx
import pytest

from app.config import LLMSettings, NotificationSettings, Settings
from app.crud import (
    create_check_result,
    create_topic,
    upsert_feed_health_failure,
)
from app.main import app
from app.models import CheckResult, FeedMode, Topic, TopicStatus
from app.web.dependencies import get_db_conn, get_settings

_TEMPLATES = Path(__file__).resolve().parent.parent / "app" / "templates"


def _make_settings() -> Settings:
    return Settings(
        llm=LLMSettings(model="openai/gpt-4o-mini", api_key="test-key-12345678"),
        notifications=NotificationSettings(urls=["json://localhost"]),
    )


@pytest.fixture
async def client(db_conn: sqlite3.Connection) -> AsyncGenerator[httpx.AsyncClient, None]:
    settings = _make_settings()

    def override_db():
        yield db_conn

    app.dependency_overrides[get_db_conn] = override_db
    app.dependency_overrides[get_settings] = lambda: settings
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac
    app.dependency_overrides.clear()


class _DataTableCellChecker(HTMLParser):
    """Collect the data ``<td>``s inside every ``table.data-table`` body.

    Records, for each body cell, whether it declares a ``data-label`` attribute
    (the empty string counts — that is the opt-out for checkbox/actions cells).
    """

    def __init__(self) -> None:
        super().__init__()
        self._in_data_table = 0  # nesting depth of data-table <table>s
        self._in_thead = False
        self.cells: list[bool] = []  # one bool per data <td>: has a data-label attr?

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        adict = dict(attrs)
        if tag == "table":
            classes = (adict.get("class") or "").split()
            self._in_data_table += 1 if "data-table" in classes else 0
        elif tag == "thead" and self._in_data_table:
            self._in_thead = True
        elif tag == "td" and self._in_data_table and not self._in_thead:
            self.cells.append("data-label" in adict)

    def handle_endtag(self, tag: str) -> None:
        if tag == "thead":
            self._in_thead = False
        elif tag == "table" and self._in_data_table:
            self._in_data_table -= 1


def _assert_every_cell_labelled(html: str, *, where: str) -> None:
    parser = _DataTableCellChecker()
    parser.feed(html)
    assert parser.cells, f"expected at least one .data-table data <td> on {where}"
    missing = [i for i, ok in enumerate(parser.cells) if not ok]
    assert not missing, f"{where}: {len(missing)} .data-table <td>(s) missing a data-label attribute"


async def test_dashboard_data_table_cells_all_labelled(client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
    """Every dashboard topic-row <td> declares a data-label (checkbox/actions == "")."""
    topic = create_topic(
        db_conn,
        Topic(
            name="Labelled Topic",
            description="A description that is long enough to be truncated in the cell body.",
            feed_urls=["https://example.com/feed.xml"],
            feed_mode=FeedMode.MANUAL,
            status=TopicStatus.ERROR,
            error_message="Something went wrong during the check.",
            tags=["alpha", "beta"],
        ),
    )
    create_check_result(
        db_conn,
        CheckResult(topic_id=topic.id, articles_found=4, has_new_info=True, llm_response='{"confidence": 0.9}'),
    )
    db_conn.commit()

    response = await client.get("/")
    assert response.status_code == 200
    _assert_every_cell_labelled(response.text, where="dashboard")


async def test_feeds_data_table_cells_all_labelled(client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
    """Every feed-health row <td> declares a data-label."""
    upsert_feed_health_failure(db_conn, "https://example.com/feed.xml", "boom")
    db_conn.commit()

    response = await client.get("/feeds")
    assert response.status_code == 200
    _assert_every_cell_labelled(response.text, where="feeds")


async def test_topic_detail_data_table_cells_all_labelled(
    client: httpx.AsyncClient, db_conn: sqlite3.Connection
) -> None:
    """Every topic-detail table <td> (articles + check history) declares a data-label."""
    topic = create_topic(
        db_conn,
        Topic(
            name="Detail Topic",
            description="d",
            feed_urls=["https://example.com/feed.xml"],
            feed_mode=FeedMode.MANUAL,
            status=TopicStatus.READY,
        ),
    )
    create_check_result(
        db_conn,
        CheckResult(topic_id=topic.id, articles_found=2, has_new_info=True, llm_response='{"confidence": 0.7}'),
    )
    db_conn.commit()

    response = await client.get(f"/topics/{topic.id}")
    assert response.status_code == 200
    _assert_every_cell_labelled(response.text, where="topic-detail")


def _count_top_level_tds(template_body: str) -> int:
    """Count the row's direct-child <td> openers (nested tables aren't used here)."""
    return len(re.findall(r"<td\b", template_body))


def test_dashboard_column_counts_agree() -> None:
    """_topic_row <td> count == dashboard <thead> <th> count == _topic_list colspan."""
    row_html = (_TEMPLATES / "_topic_row.html").read_text()
    dashboard_html = (_TEMPLATES / "dashboard.html").read_text()
    list_html = (_TEMPLATES / "_topic_list.html").read_text()

    td_count = _count_top_level_tds(row_html)

    # The dashboard topic table's <thead> column count.
    thead = re.search(r'<table class="data-table">.*?<thead>(.*?)</thead>', dashboard_html, re.S)
    assert thead, "dashboard topic .data-table <thead> not found"
    th_count = len(re.findall(r"<th\b", thead.group(1)))

    # The empty-state colspan must match the live column count.
    colspan = re.search(r'colspan="(\d+)"', list_html)
    assert colspan, "_topic_list.html empty-state colspan not found"
    colspan_n = int(colspan.group(1))

    assert td_count == th_count == colspan_n, (
        f"column count mismatch: _topic_row <td>={td_count}, dashboard <th>={th_count}, _topic_list colspan={colspan_n}"
    )
