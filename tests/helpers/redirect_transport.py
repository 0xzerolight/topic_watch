"""httpx.MockTransport factory that returns 3xx redirects.

A building block for later SSRF tests: a public-looking feed/article URL can be
made to redirect to an arbitrary target (e.g. a private/loopback address) so
tests can assert the scraping layer's SSRF defenses re-validate the redirect
target rather than only the original URL.

This module only provides the transport; it intentionally does NOT contain an
SSRF test.
"""

from __future__ import annotations

import httpx


def build_redirect_transport(
    target: str,
    *,
    status: int = 302,
    match: str | None = None,
    final_body: str = "redirected ok",
    final_status: int = 200,
) -> httpx.MockTransport:
    """Build a MockTransport that 3xx-redirects matching requests to ``target``.

    Args:
        target: absolute URL to redirect to (the ``Location`` header).
        status: redirect status code (301/302/303/307/308). Defaults to 302.
        match: optional substring; only requests whose URL contains it are
            redirected. When ``None``, the first request to any non-``target``
            URL is redirected. Requests to ``target`` itself return
            ``final_status``/``final_body``.
        final_body: body returned when ``target`` is finally requested.
        final_status: status returned when ``target`` is finally requested.

    Returns:
        An ``httpx.MockTransport``. Pair it with ``follow_redirects=True`` to
        exercise the full redirect chain.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == target or target in url:
            return httpx.Response(final_status, text=final_body)
        if match is None or match in url:
            return httpx.Response(status, headers={"location": target})
        return httpx.Response(404, text="Not found")

    return httpx.MockTransport(handler)
