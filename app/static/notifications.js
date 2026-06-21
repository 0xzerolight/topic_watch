(function () {
    "use strict";

    var STORAGE_KEY = "topic-watch-browser-notifications";

    function isSupported() {
        return "Notification" in window;
    }

    function isEnabled() {
        try {
            return localStorage.getItem(STORAGE_KEY) === "true";
        } catch (e) {
            return false;
        }
    }

    function setEnabled(val) {
        try {
            localStorage.setItem(STORAGE_KEY, val ? "true" : "false");
        } catch (e) {}
    }

    function requestPermissionIfNeeded() {
        if (!isSupported()) return Promise.resolve("denied");
        if (Notification.permission === "granted") return Promise.resolve("granted");
        if (Notification.permission === "denied") return Promise.resolve("denied");
        return Notification.requestPermission();
    }

    /**
     * Show a browser notification if enabled and permitted.
     * @param {string} title
     * @param {string} body
     * @param {object} options - optional: { url: string }
     */
    function show(title, body, options) {
        if (!isSupported() || !isEnabled()) return;
        if (Notification.permission !== "granted") return;

        var opts = { body: body, tag: "topic-watch" };

        // On some platforms (e.g. Android Chrome) the Notification constructor is
        // present and permission can be "granted", yet `new Notification()` throws
        // "Illegal constructor" — only ServiceWorkerRegistration.showNotification is
        // allowed (Chromium issue 481856). Guard the construction so the throw does
        // not propagate out of show() and abort callers (e.g. the afterSwap handler)
        // or leave the UI in an inconsistent "enabled but nothing shown" state.
        var n;
        try {
            n = new Notification(title, opts);
        } catch (e) {
            return;
        }

        if (options && options.url) {
            n.onclick = function () {
                window.focus();
                window.location.href = options.url;
                n.close();
            };
        }

        // Auto-close after 10 seconds
        setTimeout(function () { n.close(); }, 10000);
    }

    // --- Non-blocking on-page toast (always visible, no permission needed) ---

    var TOAST_CONTAINER_ID = "tw-toast-container";

    function toastContainer() {
        var el = document.getElementById(TOAST_CONTAINER_ID);
        if (!el) {
            el = document.createElement("div");
            el.id = TOAST_CONTAINER_ID;
            el.className = "tw-toast-container";
            el.setAttribute("aria-live", "polite");
            el.setAttribute("aria-atomic", "false");
            document.body.appendChild(el);
        }
        return el;
    }

    /**
     * Show a non-blocking error toast at the corner of the page.
     * @param {string} message
     * @param {function} [onRetry] - optional callback wired to a "Retry" button
     */
    function toast(message, onRetry) {
        var container = toastContainer();
        var el = document.createElement("div");
        el.className = "tw-toast";
        el.setAttribute("role", "alert");

        var msg = document.createElement("span");
        msg.className = "tw-toast-message";
        msg.textContent = message;
        el.appendChild(msg);

        var timer = setTimeout(dismiss, 8000);

        function dismiss() {
            clearTimeout(timer);
            if (el.parentNode) el.parentNode.removeChild(el);
        }

        if (typeof onRetry === "function") {
            var retry = document.createElement("button");
            retry.type = "button";
            retry.className = "tw-toast-retry";
            retry.textContent = "Retry";
            retry.addEventListener("click", function () {
                dismiss();
                onRetry();
            });
            el.appendChild(retry);
        }

        var close = document.createElement("button");
        close.type = "button";
        close.className = "tw-toast-close";
        close.setAttribute("aria-label", "Dismiss");
        close.textContent = "×";
        close.addEventListener("click", dismiss);
        el.appendChild(close);

        container.appendChild(el);
        return el;
    }

    // --- Global HTMX error surfacing (OVH-011) ---
    // HTMX does not swap 4xx/5xx responses and network failures fire no swap at
    // all; without a listener they fail silently. One global handler covers every
    // partial action: surface a non-blocking toast with the status and a retry.

    function htmxRetry(evt) {
        var elt = evt && evt.detail ? evt.detail.elt : null;
        if (elt && window.htmx && typeof window.htmx.trigger === "function") {
            try {
                window.htmx.ajax(evt.detail.requestConfig.verb, evt.detail.pathInfo.requestPath, { source: elt });
                return;
            } catch (e) {
                /* fall through to reload */
            }
        }
        window.location.reload();
    }

    function wireHtmxErrorListeners() {
        // HTMX dispatches these events on the triggering element; they bubble to
        // document, so one document-level listener covers every partial action.
        document.addEventListener("htmx:responseError", function (evt) {
            var status = evt.detail && evt.detail.xhr ? evt.detail.xhr.status : "?";
            toast("Request failed (HTTP " + status + "). Please try again.", function () {
                htmxRetry(evt);
            });
        });

        document.addEventListener("htmx:sendError", function (evt) {
            toast("Network error — could not reach the server.", function () {
                htmxRetry(evt);
            });
        });
    }

    // This script loads in <head> without defer, so document.body is not yet
    // available; wait for the DOM before wiring listeners.
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", wireHtmxErrorListeners);
    } else {
        wireHtmxErrorListeners();
    }

    // Expose globally
    window.TopicWatchNotifications = {
        isSupported: isSupported,
        isEnabled: isEnabled,
        setEnabled: setEnabled,
        requestPermission: requestPermissionIfNeeded,
        show: show,
        toast: toast
    };
})();
