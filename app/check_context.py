"""Check correlation ID context management.

Provides context variables for tracking correlation IDs across async call chains
(`check_id_var` for the scheduler/checker pipeline, `request_id_var` for inbound
web requests), and a logging filter that surfaces whichever is set on log records.
"""

import contextvars
import logging
import uuid

# Context variable for the current check's correlation ID (scheduler/checker pipeline).
check_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("check_id", default=None)

# Context variable for the current inbound web request's correlation ID.
# Kept separate from check_id_var so pipeline vs request semantics are not conflated,
# but the same logging filter surfaces either so all log lines stay correlatable.
request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)

# Context variable for the current scheduler check-all cycle. Set once per
# cycle (check_all_topics) and left in place while its notification-retry,
# webhook-retry and per-topic child pipelines each set their OWN check_id_var
# — so every record any of them logs also carries the cycle that launched it,
# and a noisy or failed tick can be reconstructed as one unit (AUG-275). A
# scheduler tick has no inbound web request, so this is a separate var from
# request_id_var rather than overloading it.
cycle_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("cycle_id", default=None)


def generate_check_id() -> str:
    """Generate a short correlation ID (first 8 chars of UUID4)."""
    return uuid.uuid4().hex[:8]


class CheckIdFilter(logging.Filter):
    """Logging filter that adds correlation ids from contextvars to log records.

    ``check_id`` prefers the pipeline id, falling back to the web request id,
    so every record still carries a single "best" correlation id for the
    existing text format. ``request_id`` is set independently, straight from
    ``request_id_var`` — when a check runs synchronously inside the web
    request that triggered it, ``check_id_var`` is set *on top of* the still-
    live ``request_id_var``, so every record the child pipeline logs carries
    both fields at once. Without this, a record showed only whichever id
    ``check_id`` preferred, and no record linked the two — the parent
    request could not be found from the child's logs, or vice versa
    (AUG-273). Both default to '-' when unset.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.check_id = check_id_var.get() or request_id_var.get() or "-"  # type: ignore[attr-defined]
        record.request_id = request_id_var.get() or "-"  # type: ignore[attr-defined]
        record.cycle_id = cycle_id_var.get() or "-"  # type: ignore[attr-defined]
        return True
