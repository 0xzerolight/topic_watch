"use strict";

// Minimal hand-rolled DOM/window/localStorage/Notification stand-ins, just
// enough surface for app/static/notifications.js to run for real (TW-AUD-035).
// No jsdom, no browser, no npm install — Node's built-in `vm` module runs the
// shipped script unmodified against these fakes.

const fs = require("fs");
const vm = require("vm");

class FakeEventTarget {
    constructor() {
        this._listeners = new Map();
    }
    addEventListener(type, fn) {
        if (!this._listeners.has(type)) this._listeners.set(type, []);
        this._listeners.get(type).push(fn);
    }
    removeEventListener(type, fn) {
        const fns = this._listeners.get(type);
        if (!fns) return;
        const i = fns.indexOf(fn);
        if (i >= 0) fns.splice(i, 1);
    }
    dispatchEvent(evt) {
        const fns = this._listeners.get(evt.type) || [];
        fns.slice().forEach((fn) => fn(evt));
    }
}

class FakeElement extends FakeEventTarget {
    constructor(tagName) {
        super();
        this.tagName = String(tagName || "").toUpperCase();
        this.children = [];
        this.parentNode = null;
        this.attributes = {};
        this.className = "";
        this.id = "";
        this._text = "";
    }
    set textContent(v) {
        this._text = String(v);
    }
    get textContent() {
        return this._text;
    }
    setAttribute(name, value) {
        this.attributes[name] = String(value);
    }
    getAttribute(name) {
        return Object.prototype.hasOwnProperty.call(this.attributes, name) ? this.attributes[name] : null;
    }
    appendChild(child) {
        child.parentNode = this;
        this.children.push(child);
        return child;
    }
    removeChild(child) {
        const i = this.children.indexOf(child);
        if (i >= 0) this.children.splice(i, 1);
        child.parentNode = null;
        return child;
    }
    // Not real DOM API — a small test-only helper for finding rendered nodes.
    findByClass(cls) {
        const out = [];
        const walk = (el) => {
            if (el.className && el.className.split(" ").includes(cls)) out.push(el);
            el.children.forEach(walk);
        };
        walk(this);
        return out;
    }
}

class FakeDocument extends FakeEventTarget {
    constructor() {
        super();
        this.readyState = "complete";
        this.body = new FakeElement("body");
    }
    createElement(tag) {
        return new FakeElement(tag);
    }
    getElementById(id) {
        let found = null;
        const walk = (el) => {
            if (found) return;
            if (el.id === id) {
                found = el;
                return;
            }
            el.children.forEach(walk);
        };
        walk(this.body);
        return found;
    }
}

/**
 * Build one harness instance: a fake window/document/localStorage/Notification
 * plus a `loadScript` that runs a real source file against them.
 *
 * @param {object} opts
 * @param {string} [opts.permission] - initial Notification.permission
 * @param {boolean} [opts.constructorThrows] - simulate the Android "Illegal
 *   constructor" case (AUG-128)
 * @param {boolean} [opts.ajaxThrows] - make window.htmx.ajax() throw
 * @param {boolean} [opts.brokenStorage] - make localStorage throw (private mode)
 */
function createHarness(opts) {
    opts = opts || {};
    const document = new FakeDocument();

    const storageBacking = {};
    const localStorage = {
        getItem(key) {
            return Object.prototype.hasOwnProperty.call(storageBacking, key) ? storageBacking[key] : null;
        },
        setItem(key, val) {
            storageBacking[key] = String(val);
        },
        removeItem(key) {
            delete storageBacking[key];
        },
    };
    if (opts.brokenStorage) {
        localStorage.getItem = () => {
            throw new Error("storage disabled");
        };
        localStorage.setItem = () => {
            throw new Error("storage disabled");
        };
    }

    const timers = [];
    function fakeSetTimeout(fn, delay) {
        const id = timers.length;
        timers.push({ id, fn, delay, fired: false });
        return id;
    }
    function fakeClearTimeout(id) {
        const t = timers.find((t) => t.id === id);
        if (t) t.fired = true;
    }
    function flushTimers() {
        timers.filter((t) => !t.fired).forEach((t) => {
            t.fired = true;
            t.fn();
        });
    }

    const ajaxCalls = [];
    const htmx = {
        ajax(verb, path, config) {
            ajaxCalls.push({ verb, path, config });
            if (opts.ajaxThrows) throw new Error("ajax failed");
        },
    };

    const locationState = { href: "http://test/dashboard" };
    let reloadCalls = 0;
    const window = {
        document,
        localStorage,
        htmx,
        location: {
            get href() {
                return locationState.href;
            },
            set href(v) {
                locationState.href = v;
            },
            reload() {
                reloadCalls += 1;
            },
        },
        focus() {},
    };

    let permission = opts.permission || "default";
    const constructedNotifications = [];
    class Notification {
        static get permission() {
            return permission;
        }
        static set permission(v) {
            permission = v;
        }
        static requestPermission() {
            return Promise.resolve(opts.requestPermissionResult || permission);
        }
        constructor(title, options) {
            if (opts.constructorThrows) {
                throw new Error("Illegal constructor");
            }
            this.title = title;
            this.body = options && options.body;
            this.tag = options && options.tag;
            this.onclick = null;
            this.closed = false;
            constructedNotifications.push(this);
        }
        close() {
            this.closed = true;
        }
    }

    // "Notification" in window is how notifications.js feature-detects support —
    // in a real browser `window` IS the global object, so window.Notification
    // and the bare global are the same binding. Mirror that here.
    window.Notification = Notification;

    const context = vm.createContext({
        window,
        document,
        localStorage,
        Notification,
        setTimeout: fakeSetTimeout,
        clearTimeout: fakeClearTimeout,
        console,
    });

    return {
        context,
        window,
        document,
        localStorage,
        ajaxCalls,
        get reloadCalls() {
            return reloadCalls;
        },
        get locationHref() {
            return locationState.href;
        },
        constructedNotifications,
        flushTimers,
        getPermission: () => permission,
        setPermission: (v) => {
            permission = v;
        },
        loadScript(path) {
            const src = fs.readFileSync(path, "utf8");
            vm.runInContext(src, context, { filename: path });
        },
    };
}

module.exports = { createHarness, FakeElement, FakeDocument };
