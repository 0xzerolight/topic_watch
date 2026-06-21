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


def generate_check_id() -> str:
    """Generate a short correlation ID (first 8 chars of UUID4)."""
    return uuid.uuid4().hex[:8]


class CheckIdFilter(logging.Filter):
    """Logging filter that adds a correlation id from contextvars to log records.

    Prefers the pipeline ``check_id``; falls back to the web ``request_id`` so
    request-tier logs are correlatable too. Uses '-' as placeholder when neither
    is set.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.check_id = check_id_var.get() or request_id_var.get() or "-"  # type: ignore[attr-defined]
        return True
