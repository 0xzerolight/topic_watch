"""Executable behavior tests for app/static/notifications.js (TW-AUD-035).

``tests/test_web.py::TestHtmxErrorSurfacing`` and ``::TestBrowserNotificationRobustness``
grep the shipped script's source for event names, try/catch blocks, and
string literals — a cheap syntax check that never actually runs the code. A
regression that keeps every one of those substrings in the file (e.g. an
inverted condition, or a retry that fires unconditionally) would stay green
there.

These tests run the real script for real against a hand-rolled DOM/window
stub (``tests/helpers/dom_stub.js``, Node's built-in ``vm`` module — no
jsdom, no browser, no npm install) and assert on the resulting behavior:
HTMX failure/retry dispatch and the notification permission/persistence
flow. This does not replace the lexical tests, which stay as a fast
sanity check independent of a Node toolchain being present.
"""

from __future__ import annotations

import pytest

from tests.helpers.run_js import node_available, run_scenario

pytestmark = pytest.mark.skipif(not node_available(), reason="node executable not found on PATH")


# --- HTMX failure/retry (OVH-011, AUG-218) ---


def test_safe_verb_error_retries_via_htmx_ajax_and_dismisses_the_toast() -> None:
    result = run_scenario(
        """
        harness.loadScript(NOTIFICATIONS_JS);
        const elt = harness.document.createElement("button");
        harness.document.dispatchEvent({
            type: "htmx:responseError",
            detail: {
                xhr: { status: 500 },
                requestConfig: { verb: "get" },
                elt: elt,
                pathInfo: { requestPath: "/topics/1/status" },
            },
        });

        const toastCountBeforeClick = harness.document.body.findByClass("tw-toast").length;
        const retryButtons = harness.document.body.findByClass("tw-toast-retry");
        retryButtons[0].dispatchEvent({ type: "click" });

        result = {
            toastCountBeforeClick: toastCountBeforeClick,
            ajaxCalls: harness.ajaxCalls.map((c) => ({ verb: c.verb, path: c.path })),
            reloadCalls: harness.reloadCalls,
            toastCountAfterClick: harness.document.body.findByClass("tw-toast").length,
        };
        """
    )

    assert result["toastCountBeforeClick"] == 1
    assert result["ajaxCalls"] == [{"verb": "get", "path": "/topics/1/status"}]
    assert result["reloadCalls"] == 0
    assert result["toastCountAfterClick"] == 0  # clicking Retry dismisses the toast


def test_unsafe_verb_error_never_replays_and_offers_reload_instead() -> None:
    """AUG-218: a failed POST must never be re-issued — its outcome is unknown."""
    result = run_scenario(
        """
        harness.loadScript(NOTIFICATIONS_JS);
        const elt = harness.document.createElement("button");
        harness.document.dispatchEvent({
            type: "htmx:responseError",
            detail: {
                xhr: { status: 500 },
                requestConfig: { verb: "post" },
                elt: elt,
                pathInfo: { requestPath: "/topics/1/toggle-active" },
            },
        });

        const toastMessage = harness.document.body.findByClass("tw-toast-message")[0].textContent;
        const retryButtons = harness.document.body.findByClass("tw-toast-retry");
        const buttonLabel = retryButtons[0].textContent;
        retryButtons[0].dispatchEvent({ type: "click" });

        result = {
            toastMessage: toastMessage,
            buttonLabel: buttonLabel,
            ajaxCalls: harness.ajaxCalls.length,
            reloadCalls: harness.reloadCalls,
        };
        """
    )

    assert "may or may not have been applied" in result["toastMessage"]
    assert result["buttonLabel"] == "Reload"
    assert result["ajaxCalls"] == 0  # the POST is never replayed
    assert result["reloadCalls"] == 1


def test_send_error_surfaces_a_network_message_and_retries_on_click() -> None:
    result = run_scenario(
        """
        harness.loadScript(NOTIFICATIONS_JS);
        const elt = harness.document.createElement("a");
        harness.document.dispatchEvent({
            type: "htmx:sendError",
            detail: {
                requestConfig: { verb: "get" },
                elt: elt,
                pathInfo: { requestPath: "/topics/1/check" },
            },
        });

        const toastMessage = harness.document.body.findByClass("tw-toast-message")[0].textContent;
        harness.document.body.findByClass("tw-toast-retry")[0].dispatchEvent({ type: "click" });

        result = {
            toastMessage: toastMessage,
            ajaxCalls: harness.ajaxCalls.map((c) => ({ verb: c.verb, path: c.path })),
        };
        """
    )

    assert "could not reach the server" in result["toastMessage"]
    assert result["ajaxCalls"] == [{"verb": "get", "path": "/topics/1/check"}]


def test_retry_that_throws_falls_back_to_reload() -> None:
    result = run_scenario(
        """
        harness.loadScript(NOTIFICATIONS_JS);
        const elt = harness.document.createElement("button");
        harness.document.dispatchEvent({
            type: "htmx:responseError",
            detail: {
                xhr: { status: 502 },
                requestConfig: { verb: "get" },
                elt: elt,
                pathInfo: { requestPath: "/topics/1/status" },
            },
        });
        harness.document.body.findByClass("tw-toast-retry")[0].dispatchEvent({ type: "click" });

        result = { ajaxAttempted: harness.ajaxCalls.length, reloadCalls: harness.reloadCalls };
        """,
        opts={"ajaxThrows": True},
    )

    assert result["ajaxAttempted"] == 1  # htmx.ajax was tried
    assert result["reloadCalls"] == 1  # ...and the throw fell through to reload


# --- Browser notification permission / persistence (OVH-117/118/119, AUG-128, AUG-219) ---


def test_show_constructs_and_auto_closes_when_granted_and_enabled() -> None:
    result = run_scenario(
        """
        harness.loadScript(NOTIFICATIONS_JS);
        const api = harness.window.TopicWatchNotifications;
        api.setEnabled(true);
        const ok = api.show("Title", "Body text", { url: "http://target", tag: "topic-1" });
        harness.flushTimers();
        const n = harness.constructedNotifications[0];

        result = {
            ok: ok,
            constructedCount: harness.constructedNotifications.length,
            tag: n.tag,
            body: n.body,
            closedAfterFlush: n.closed,
        };
        """,
        opts={"permission": "granted"},
    )

    assert result["ok"] is True
    assert result["constructedCount"] == 1
    assert result["tag"] == "topic-1"
    assert result["body"] == "Body text"
    assert result["closedAfterFlush"] is True  # the 10s auto-close timer fired


def test_show_onclick_navigates_and_closes() -> None:
    result = run_scenario(
        """
        harness.loadScript(NOTIFICATIONS_JS);
        const api = harness.window.TopicWatchNotifications;
        api.setEnabled(true);
        api.show("Title", "Body", { url: "http://target/topic/9" });
        const n = harness.constructedNotifications[0];
        n.onclick();

        result = { href: harness.locationHref, closed: n.closed };
        """,
        opts={"permission": "granted"},
    )

    assert result["href"] == "http://target/topic/9"
    assert result["closed"] is True


def test_show_returns_false_when_permission_denied() -> None:
    result = run_scenario(
        """
        harness.loadScript(NOTIFICATIONS_JS);
        const api = harness.window.TopicWatchNotifications;
        api.setEnabled(true);
        const ok = api.show("Title", "Body", {});
        result = { ok: ok, constructedCount: harness.constructedNotifications.length };
        """,
        opts={"permission": "denied"},
    )

    assert result["ok"] is False
    assert result["constructedCount"] == 0


def test_show_returns_false_when_disabled_even_if_permission_granted() -> None:
    result = run_scenario(
        """
        harness.loadScript(NOTIFICATIONS_JS);
        const api = harness.window.TopicWatchNotifications;
        // setEnabled(true) is deliberately never called.
        const ok = api.show("Title", "Body", {});
        result = { ok: ok, constructedCount: harness.constructedNotifications.length };
        """,
        opts={"permission": "granted"},
    )

    assert result["ok"] is False
    assert result["constructedCount"] == 0


def test_show_catches_illegal_constructor_and_returns_false() -> None:
    """AUG-128: some platforms (Android Chrome) report permission "granted" but
    ``new Notification()`` throws — the guard must swallow it, not propagate."""
    result = run_scenario(
        """
        harness.loadScript(NOTIFICATIONS_JS);
        const api = harness.window.TopicWatchNotifications;
        api.setEnabled(true);
        const ok = api.show("Title", "Body", {});
        result = { ok: ok, constructedCount: harness.constructedNotifications.length };
        """,
        opts={"permission": "granted", "constructorThrows": True},
    )

    assert result["ok"] is False
    assert result["constructedCount"] == 0


def test_notification_tags_are_distinct_not_a_shared_literal() -> None:
    """AUG-219: a fixed tag would collapse same-origin notifications from
    different topics into one visible alert."""
    result = run_scenario(
        """
        harness.loadScript(NOTIFICATIONS_JS);
        const api = harness.window.TopicWatchNotifications;
        api.setEnabled(true);
        api.show("T1", "B1", { tag: "topic-1" });
        api.show("T2", "B2", { tag: "topic-2" });
        api.show("T3", "B3", {});

        result = {
            tags: harness.constructedNotifications.map((n) => n.tag || null),
            count: harness.constructedNotifications.length,
        };
        """,
        opts={"permission": "granted"},
    )

    assert result["count"] == 3
    assert result["tags"] == ["topic-1", "topic-2", None]
    assert "topic-watch" not in result["tags"]


def test_enabled_state_round_trips_through_local_storage() -> None:
    result = run_scenario(
        """
        harness.loadScript(NOTIFICATIONS_JS);
        const api = harness.window.TopicWatchNotifications;
        const before = api.isEnabled();
        api.setEnabled(true);
        const afterTrue = api.isEnabled();
        const stored = harness.localStorage.getItem("topic-watch-browser-notifications");
        api.setEnabled(false);
        const afterFalse = api.isEnabled();

        result = { before: before, afterTrue: afterTrue, stored: stored, afterFalse: afterFalse };
        """
    )

    assert result["before"] is False
    assert result["afterTrue"] is True
    assert result["stored"] == "true"
    assert result["afterFalse"] is False
