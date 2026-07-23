"""Structured logging with structlog.

Why structured logs? Because logs are data: a `correlation_id` field that a
log aggregator can filter on beats regex-parsing prose at 3 a.m.

The middleware (`middleware.py`) binds a correlation id to structlog's
context variables at the start of every request, so every log line emitted
during that request — in any layer — carries it automatically.
"""

import logging
import sys

import structlog


def configure_logging(*, debug: bool = False) -> None:
    """Configure stdlib logging + structlog processors. Call once at startup."""
    level = logging.DEBUG if debug else logging.INFO

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            # JSON in production; pretty console output when debugging locally.
            structlog.dev.ConsoleRenderer() if debug else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )
