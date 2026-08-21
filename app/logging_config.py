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
            # The originating web request, kept separate from check_id (AUG-273):
            # a request-triggered check logs both at once, so the two can be
            # joined even after the check outlives the request that started it.
            "request_id": getattr(record, "request_id", "-"),
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
            "request_id",
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
    log_level = os.environ.get("TOPIC_WATCH_LOG_LEVEL", "INFO").upper()
    check_id_filter = CheckIdFilter()

    if log_format == "json":
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        handler.addFilter(check_id_filter)
        logging.root.handlers.clear()
        logging.root.addHandler(handler)
        logging.root.setLevel(getattr(logging, log_level, logging.INFO))
        # Retarget uvicorn's loggers so their startup/error/access lines flow through
        # the root JSON handler+filter instead of uvicorn's own text formatter. This
        # makes stdout a single all-JSON stream and gives access logs a check_id field.
        # LiteLLM installs its own "LiteLLM" logger with a plain StreamHandler at
        # import time and leaves propagate=True, so without retargeting it too, its
        # warnings render twice: once through its own ANSI handler, once JSON via
        # propagation to root (AUG-249).
        for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "LiteLLM"):
            target_logger = logging.getLogger(name)
            target_logger.handlers = []
            target_logger.propagate = True
    else:
        logging.basicConfig(
            level=getattr(logging, log_level, logging.INFO),
            format="%(asctime)s [%(levelname)s] %(name)s [%(check_id)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        for h in logging.root.handlers:
            h.addFilter(check_id_filter)
