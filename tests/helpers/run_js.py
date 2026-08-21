"""Run a Node.js scenario against the ``dom_stub.js`` harness.

TW-AUD-035: the browser-behavior tests used to only grep the shipped script's
source for event names and try/catch blocks — they never executed it. This
runs the real ``app/static/notifications.js`` inside a hand-rolled DOM/window
stub via Node's built-in ``vm`` module (no jsdom, no browser, no npm install),
so a regression in the actual retry/toast/notification behavior fails a test
instead of only a text search.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

_HELPERS_DIR = Path(__file__).resolve().parent
DOM_STUB_JS = _HELPERS_DIR / "dom_stub.js"
NOTIFICATIONS_JS = _HELPERS_DIR.parent.parent / "app" / "static" / "notifications.js"


def node_available() -> bool:
    """Whether a ``node`` executable is on PATH (skip reason for CI images without it)."""
    return shutil.which("node") is not None


def run_scenario(body: str, *, opts: dict | None = None, timeout: float = 10.0) -> dict:
    """Run a scenario body against a fresh harness and return its JSON result.

    ``body`` is a JS statement list with a harness (``createHarness(OPTS)``)
    and ``NOTIFICATIONS_JS`` (the absolute path to the real script) already in
    scope. It must set ``result`` to a JSON-serializable value; this prints it
    as the script's only stdout line and returns the parsed value.
    """
    script = f"""
const {{ createHarness }} = require({json.dumps(str(DOM_STUB_JS))});
const NOTIFICATIONS_JS = {json.dumps(str(NOTIFICATIONS_JS))};
const OPTS = {json.dumps(opts or {})};

(function () {{
    const harness = createHarness(OPTS);
    let result;
{body}
    process.stdout.write(JSON.stringify(result));
}})();
"""
    node = shutil.which("node")
    if node is None:  # pragma: no cover - callers gate on node_available() first
        raise RuntimeError("node executable not found on PATH")
    proc = subprocess.run(  # noqa: S603 -- fixed args, no shell, absolute resolved executable
        [node],
        input=script,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise AssertionError(f"node scenario failed (exit {proc.returncode}):\n{proc.stderr}\n---\n{script}")
    if not proc.stdout.strip():
        raise AssertionError(f"node scenario produced no output.\nSTDERR:\n{proc.stderr}\n---\n{script}")
    return json.loads(proc.stdout)
