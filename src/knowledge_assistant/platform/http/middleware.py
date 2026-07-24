"""Correlation-ID middleware — and the last line of error defense.

Every request gets a correlation id: taken from the `X-Correlation-ID` header
if the caller provided a safe value (trace propagation), generated otherwise.
The id is

1. bound to structlog's contextvars → every log line in this request has it;
2. echoed back in the `X-Correlation-ID` response header → clients can quote
   it when reporting bugs.

This middleware ALSO converts unhandled exceptions into the unified 500
error envelope. Why here and not in a Starlette `Exception` handler?
Starlette routes a handler registered for `Exception` to its OUTERMOST
ServerErrorMiddleware — outside this middleware — so the response would
escape without the `X-Correlation-ID` header, breaking the "every error is
quotable" promise. Catching here keeps envelope AND header together. The
exception is deliberately NOT re-raised and its message/traceback are not
logged by default because either may contain user-controlled content.
"""

import re
import uuid
from collections.abc import Awaitable, Callable

import structlog
from opentelemetry import trace
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

CORRELATION_ID_HEADER = "X-Correlation-ID"
_CORRELATION_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")

logger = structlog.get_logger()


def resolve_correlation_id(header_value: str | None) -> str:
    """Return a safe correlation id for logs, traces, and response headers."""
    if header_value is not None and _CORRELATION_ID_PATTERN.fullmatch(header_value):
        return header_value
    return str(uuid.uuid4())


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        correlation_id = resolve_correlation_id(request.headers.get(CORRELATION_ID_HEADER))
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(correlation_id=correlation_id)
        trace.get_current_span().set_attribute("correlation.id", correlation_id)

        try:
            response = await call_next(request)
        except Exception as exc:
            # An exception that survived every handler and domain mapping is
            # a BUG or an untranslated outage — either way the client gets
            # the same honest shape as every other error: the unified
            # envelope with the correlation id, and nothing from the
            # traceback (which would leak internals).
            logger.error(
                "unhandled_exception",
                exception_type=type(exc).__qualname__,
                path=request.url.path,
                correlation_id=correlation_id,
            )
            response = JSONResponse(
                status_code=500,
                content={
                    "detail": "Internal Server Error",
                    "error": "Internal Server Error",
                    "correlation_id": correlation_id,
                },
            )
        response.headers[CORRELATION_ID_HEADER] = correlation_id
        return response
