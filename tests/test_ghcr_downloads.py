"""Tests for the GHCR download-count parser used by the README badge workflow."""

import importlib.util
import io
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ghcr_downloads.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ghcr_downloads", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script() -> ModuleType:
    return _load_script()


def _page(total: str, displayed: str) -> str:
    """Package-page markup as GitHub renders it (verified against the live page on 2026-08-08)."""
    return f"""
    <div class="container-lg tmp-my-3 d-flex clearfix">
      <div class="lh-condensed d-flex flex-column flex-items-baseline tmp-pr-1">
        <span class="d-block color-fg-muted text-small tmp-mb-1">Total downloads</span>
        <h3 title="{total}">{displayed}</h3>
      </div>
      <div aria-label="Downloads for the last 30 days"><svg><rect data-merge-count="0"/></svg></div>
    </div>
    """


def test_prefers_exact_count_from_title_attribute(script: ModuleType) -> None:
    assert script.parse_count(_page("1,234", "1.2k")) == 1234


def test_falls_back_to_element_text_without_title(script: ModuleType) -> None:
    html = "<span>Total downloads</span><h3>428</h3>"
    assert script.parse_count(html) == 428


def test_zero_downloads_parses_as_zero_not_missing(script: ModuleType) -> None:
    assert script.parse_count(_page("0", "0")) == 0


def test_rejects_compact_fallback_text_instead_of_misparsing_it(script: ModuleType) -> None:
    """No `title` attribute and a compact displayed count (e.g. "1.2k") must not be
    stripped down to "12" — that silently reports a wildly wrong total (AUG-067)."""
    html = "<span>Total downloads</span><h3>1.2k</h3>"
    assert script.parse_count(html) is None


def test_badge_labels_the_count_as_docker_pulls(script: ModuleType) -> None:
    assert script.build_badge(428) == {
        "schemaVersion": 1,
        "label": "docker pulls",
        "message": "428",
        "color": "blue",
        "namedLogo": "docker",
    }


@pytest.mark.parametrize(
    ("count", "expected"),
    [(0, "0"), (999, "999"), (1000, "1k"), (1234, "1.2k"), (12345, "12.3k")],
)
def test_format_count_switches_to_compact_notation(script: ModuleType, count: int, expected: str) -> None:
    assert script.format_count(count) == expected


def test_main_fails_loudly_when_the_count_is_missing(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("<html><body>no packages here</body></html>"))
    assert script.main() == 1
    assert "Total downloads" in capsys.readouterr().err


def test_main_writes_badge_json_to_stdout(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(_page("428", "428")))
    assert script.main() == 0
    assert '"message": "428"' in capsys.readouterr().out


def test_main_succeeds_on_a_legitimate_zero_count(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A brand-new package with zero downloads must not be treated as a parse failure."""
    monkeypatch.setattr("sys.stdin", io.StringIO(_page("0", "0")))
    assert script.main() == 0
    assert '"message": "0"' in capsys.readouterr().out
