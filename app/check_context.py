"""Check correlation ID context management.

Provides a context variable for tracking check IDs across async call chains,
and a logging filter that automatically adds the check_id to log records.
"""

import contextvars
import logging
import uuid

# Context variable for the current check's correlation ID
check_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("check_id", default=None)


def generate_check_id() -> str:
    """Generate a short correlation ID (first 8 chars of UUID4)."""
    return uuid.uuid4().hex[:8]


class CheckIdFilter(logging.Filter):
    """Logging filter that adds check_id from contextvars to log records.

    If no check_id is set in the current context, uses '-' as placeholder.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.check_id = check_id_var.get() or "-"  # type: ignore[attr-defined]
        return True
