"""Logging configuration for Topic Watch.

Supports plain text (default) and JSON structured logging modes.
Set TOPIC_WATCH_LOG_FORMAT=json to enable JSON output.
"""

import json
import logging
import os
from datetime import UTC, datetime
from typing import Any

from app.check_context import CheckIdFilter


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "check_id": getattr(record, "check_id", "-"),
        }
        # Include extra fields if present
        # Standard LogRecord attributes to exclude from extras
        standard_attrs = {
            "name",
            "msg",
            "args",
            "created",
            "relativeCreated",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "filename",
            "module",
            "pathname",
            "thread",
            "threadName",
            "process",
            "processName",
            "levelname",
            "levelno",
            "message",
            "msecs",
            "taskName",
            "check_id",
        }
        extras = {k: v for k, v in record.__dict__.items() if k not in standard_attrs and not k.startswith("_")}
        if extras:
            log_entry["extra"] = extras

        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


def setup_logging() -> None:
    """Configure logging based on TOPIC_WATCH_LOG_FORMAT env var.

    If TOPIC_WATCH_LOG_FORMAT=json, uses JSON formatter.
    Otherwise, uses plain text format (default).
    """
    log_format = os.environ.get("TOPIC_WATCH_LOG_FORMAT", "text").lower()
    check_id_filter = CheckIdFilter()

    if log_format == "json":
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        handler.addFilter(check_id_filter)
        logging.root.handlers.clear()
        logging.root.addHandler(handler)
        logging.root.setLevel(logging.INFO)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s [%(check_id)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        for h in logging.root.handlers:
            h.addFilter(check_id_filter)
