"""Tests for the Host-header allowlist (AUG-002 — browser DNS rebinding)."""

import httpx
import pytest
from fastapi import FastAPI

from app.web.host_allowlist import (
    ALLOWED_HOSTS_ENV,
    DEFAULT_ALLOWED_HOSTS,
    HostAllowlistMiddleware,
    host_is_allowed,
    parse_allowed_hosts,
    split_host,
)


@pytest.fixture(autouse=True)
def _clear_allowed_hosts_env(monkeypatch: pytest.MonkeyPatch):
    """conftest sets the var to "*" for the module-global app; these tests want the real default."""
    monkeypatch.delenv(ALLOWED_HOSTS_ENV, raising=False)


def _make_app(allowed_hosts=None) -> FastAPI:
    test_app = FastAPI()
    test_app.add_middleware(HostAllowlistMiddleware, allowed_hosts=allowed_hosts)

    @test_app.get("/probe")
    async def probe():
        return {"ok": True}

    return test_app


async def _get(app: FastAPI, host: str) -> httpx.Response:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url=f"http://{host}",
    ) as ac:
        return await ac.get("/probe")


# --- split_host ---


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("localhost", "localhost"),
        ("localhost:8000", "localhost"),
        ("LocalHost:8000", "localhost"),
        ("192.168.1.5:8000", "192.168.1.5"),
        ("[::1]:8000", "[::1]"),
        ("[::1]", "[::1]"),
        ("::1", "::1"),
        ("  example.com:443  ", "example.com"),
        ("example.com:notaport", "example.com:notaport"),
    ],
)
def test_split_host(raw: str, expected: str) -> None:
    assert split_host(raw) == expected


# --- parse_allowed_hosts ---


def test_parse_allowed_hosts_empty_returns_defaults_only() -> None:
    assert parse_allowed_hosts(None) == ()
    assert parse_allowed_hosts("") == ()
    assert parse_allowed_hosts("   ,  , ") == ()


def test_parse_allowed_hosts_splits_and_normalizes() -> None:
    assert parse_allowed_hosts(" News.Example.COM , *.internal ") == ("news.example.com", "*.internal")


# --- Loopback and IP-literal defaults ---


async def test_localhost_is_allowed_by_default() -> None:
    response = await _get(_make_app(), "localhost:8000")
    assert response.status_code == 200


async def test_loopback_ip_is_allowed_by_default() -> None:
    response = await _get(_make_app(), "127.0.0.1:8000")
    assert response.status_code == 200


async def test_lan_ip_literal_is_allowed_by_default() -> None:
    """TOPIC_WATCH_BIND_ADDR=0.0.0.0 LAN access keeps working.

    An IP literal carries no DNS lookup, so it cannot be rebound — accepting it
    costs nothing and keeps the documented LAN setup working.
    """
    response = await _get(_make_app(), "192.168.1.50:8000")
    assert response.status_code == 200


async def test_ipv6_literal_is_allowed_by_default() -> None:
    response = await _get(_make_app(), "[::1]:8000")
    assert response.status_code == 200


async def test_mdns_local_name_is_allowed_by_default() -> None:
    response = await _get(_make_app(), "my-nas.local:8000")
    assert response.status_code == 200


# --- The rebinding case ---


async def test_attacker_domain_is_rejected() -> None:
    """AUG-002: a rebound attacker domain no longer reaches the console."""
    response = await _get(_make_app(), "rebind.evil.example")
    assert response.status_code == 400
    assert "host header" in response.text.lower()


async def test_rejection_does_not_reflect_the_submitted_host() -> None:
    response = await _get(_make_app(), "rebind.evil.example")
    assert "rebind.evil.example" not in response.text


async def test_rejection_names_the_env_var() -> None:
    response = await _get(_make_app(), "topics.example.com")
    assert ALLOWED_HOSTS_ENV in response.text


async def test_missing_host_header_is_rejected() -> None:
    app = _make_app()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost") as ac:
        response = await ac.get("/probe", headers={"host": ""})
    assert response.status_code == 400


# --- Reverse-proxy compatibility ---


async def test_reverse_proxy_hostname_allowed_via_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """AUG-002 compatibility: a reverse-proxy deployment keeps working.

    Nginx/Caddy forward the user-facing hostname (SECURITY.md's example sets
    ``proxy_set_header Host $host``); listing it in the env var accepts it.
    """
    monkeypatch.setenv(ALLOWED_HOSTS_ENV, "topics.example.com")
    app = _make_app()  # reads the env var at construction, as app.main does
    response = await _get(app, "topics.example.com")
    assert response.status_code == 200


async def test_reverse_proxy_wildcard_covers_subdomains_and_apex(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ALLOWED_HOSTS_ENV, "*.example.com")
    app = _make_app()
    assert (await _get(app, "topics.example.com")).status_code == 200
    assert (await _get(app, "example.com")).status_code == 200
    assert (await _get(app, "example.com.evil.test")).status_code == 400


async def test_star_disables_the_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ALLOWED_HOSTS_ENV, "*")
    app = _make_app()
    assert (await _get(app, "anything.example")).status_code == 200


async def test_env_var_extends_rather_than_replaces_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ALLOWED_HOSTS_ENV, "topics.example.com")
    app = _make_app()
    assert (await _get(app, "localhost:8000")).status_code == 200


# --- host_is_allowed unit coverage ---


def test_host_is_allowed_rejects_suffix_lookalike() -> None:
    assert not host_is_allowed("notlocalhost", DEFAULT_ALLOWED_HOSTS)
    assert not host_is_allowed("localhost.evil.test", DEFAULT_ALLOWED_HOSTS)


def test_host_is_allowed_accepts_localhost_subdomain() -> None:
    assert host_is_allowed("app.localhost", DEFAULT_ALLOWED_HOSTS)
