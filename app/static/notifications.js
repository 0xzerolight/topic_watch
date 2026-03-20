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

        var opts = { body: body, icon: "/static/icon.png", tag: "topic-watch" };
        var n = new Notification(title, opts);

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

    // Expose globally
    window.TopicWatchNotifications = {
        isSupported: isSupported,
        isEnabled: isEnabled,
        setEnabled: setEnabled,
        requestPermission: requestPermissionIfNeeded,
        show: show
    };
})();
