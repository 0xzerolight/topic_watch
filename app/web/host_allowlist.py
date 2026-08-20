"""Host-header allowlist — blocks browser DNS rebinding into the console (AUG-002).

Topic Watch has no login screen, so any origin that can talk to it same-origin
owns it. A hostile site can serve script, then re-point its own DNS name at
127.0.0.1 or the LAN address; without a Host check the app answers under the
attacker's name and its script keeps same-origin access to the page and the
CSRF token.

Rejecting unknown *names* closes that. Names are the whole attack: a rebind
needs a DNS record, so an IP-literal Host is accepted unconditionally and the
documented ``TOPIC_WATCH_BIND_ADDR=0.0.0.0`` LAN setup keeps working.

Reverse-proxy deployments forward a real hostname, which is neither loopback
nor an IP literal — list it in ``TOPIC_WATCH_ALLOWED_HOSTS`` (see .env.example
and SECURITY.md).

Starlette ships ``TrustedHostMiddleware``, but its pattern language cannot
express "any IP literal" and it splits the Host on the first colon, which
mangles bracketed IPv6 authorities (``[::1]:8000`` becomes ``[``). This is the
same idea with those two gaps closed.
"""

import ipaddress
import logging
import os
from collections.abc import Sequence

from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)

ALLOWED_HOSTS_ENV = "TOPIC_WATCH_ALLOWED_HOSTS"

#: Names that always mean "this machine". Everything else is either an IP
#: literal (accepted, see module docstring) or must be configured.
DEFAULT_ALLOWED_HOSTS: tuple[str, ...] = ("localhost", "*.localhost", "*.local")

_REJECTION_BODY = (
    "Invalid host header.\n\n"
    "Topic Watch only answers to localhost, an IP address, or a hostname you "
    f"list in {ALLOWED_HOSTS_ENV} (comma-separated). Set it to the hostname "
    "your reverse proxy forwards — see SECURITY.md.\n"
)


def split_host(value: str) -> str:
    """Return the host part of a Host header value, lowercased, port removed.

    Handles the three authority spellings: ``name:port``, ``[v6]:port`` and a
    bare unbracketed IPv6 literal (which must not be split on its colons).
    """
    host = value.strip().lower()
    if host.startswith("["):
        end = host.find("]")
        return host if end == -1 else host[: end + 1]
    head, sep, tail = host.rpartition(":")
    if sep and tail.isdigit() and ":" not in head:
        return head
    return host


def parse_allowed_hosts(raw: str | None) -> tuple[str, ...]:
    """Parse the comma-separated env var into normalized patterns."""
    if not raw:
        return ()
    return tuple(part.strip().lower() for part in raw.split(",") if part.strip())


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return False
    return True


def host_is_allowed(host: str, patterns: Sequence[str]) -> bool:
    """True if ``host`` (already split/lowercased) matches the allowlist."""
    if not host:
        return False
    if "*" in patterns:
        return True
    if _is_ip_literal(host):
        return True
    for pattern in patterns:
        if host == pattern:
            return True
        # "*.example.com" covers every subdomain and the apex, so a user who
        # configures the wildcard does not also have to list the bare name.
        if pattern.startswith("*.") and (host.endswith(pattern[1:]) or host == pattern[2:]):
            return True
    return False


class HostAllowlistMiddleware:
    """Rejects requests whose Host header is not on the allowlist."""

    def __init__(self, app: ASGIApp, allowed_hosts: Sequence[str] | None = None) -> None:
        self.app = app
        if allowed_hosts is None:
            configured = parse_allowed_hosts(os.environ.get(ALLOWED_HOSTS_ENV))
        else:
            configured = tuple(allowed_hosts)
        self.allowed_hosts: tuple[str, ...] = DEFAULT_ALLOWED_HOSTS + configured

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        raw_host = ""
        for key, value in scope.get("headers", []):
            if key == b"host":
                raw_host = value.decode("latin-1")
                break

        if host_is_allowed(split_host(raw_host), self.allowed_hosts):
            await self.app(scope, receive, send)
            return

        # Log the rejected name so the operator can copy it into the env var;
        # never reflect it into the response body.
        logger.warning(
            "Rejected request with untrusted Host header %r — add it to %s to allow it",
            raw_host[:200],
            ALLOWED_HOSTS_ENV,
        )
        response = PlainTextResponse(_REJECTION_BODY, status_code=400)
        await response(scope, receive, send)
