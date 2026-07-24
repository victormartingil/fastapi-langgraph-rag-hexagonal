"""Mapping of domain errors to HTTP responses — and THE one error envelope.

This module is THE boundary where the domain meets HTTP: domain code raises
plain Python exceptions; here — and only here — they become status codes.
Adding a new domain error means adding one entry to the mapping below, with
zero changes in routers or use cases.

Every error response of the API — domain errors, HTTPException (401, 413),
and request-validation 422 — shares ONE envelope:

    {"detail": str, "error": str, "correlation_id": str}

(validation errors add an "errors" list with the field-level details).
`error` is the domain-error class name, or the HTTP status phrase for plain
HTTPExceptions. `correlation_id` matches the X-Correlation-ID response
header, so a client can quote a failing request end to end.
"""

from http import HTTPStatus

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from knowledge_assistant.assistant.domain.exceptions import (
    EmptyQuestionError,
    GenerationUnavailableError,
    InvalidModelOutputError,
    InvalidQuestionError,
    RetrievalUnavailableError,
)
from knowledge_assistant.knowledge_base.application.exceptions import (
    KnowledgeBaseUnavailableError,
)
from knowledge_assistant.knowledge_base.domain.exceptions import (
    ConcurrentIngestionError,
    DocumentNotFoundError,
    EmbeddingProviderUnavailableError,
    EmptyDocumentError,
    InvalidDocumentMetadataError,
    TextExtractionError,
    UnsupportedFileTypeError,
)
from knowledge_assistant.platform.http.middleware import safe_route_path
from knowledge_assistant.shared_kernel.exceptions import DomainError

logger = structlog.get_logger()

# Domain error type -> HTTP status. Explicit beats clever.
_STATUS_BY_ERROR: dict[type[DomainError], int] = {
    DocumentNotFoundError: 404,
    UnsupportedFileTypeError: 415,
    ConcurrentIngestionError: 409,
    EmptyDocumentError: 422,
    TextExtractionError: 422,
    EmptyQuestionError: 422,
    InvalidQuestionError: 422,
    InvalidDocumentMetadataError: 422,
    InvalidModelOutputError: 502,
    RetrievalUnavailableError: 503,
    GenerationUnavailableError: 503,
    EmbeddingProviderUnavailableError: 503,
    KnowledgeBaseUnavailableError: 503,
}


def _status_for(error: DomainError) -> int:
    for error_type, status in _STATUS_BY_ERROR.items():
        if isinstance(error, error_type):
            return status
    # An unmapped DomainError is a server-side OMISSION (someone added a
    # domain error and forgot to map it here), not a client fault — 500.
    return 500


def _correlation_id() -> str:
    """The id bound by CorrelationIdMiddleware (handlers run inside the
    middleware stack, so the contextvar is already set)."""
    return str(structlog.contextvars.get_contextvars().get("correlation_id", "unknown"))


def _envelope(detail: str, error: str, **extra: object) -> dict[str, object]:
    return {
        "detail": detail,
        "error": error,
        "correlation_id": _correlation_id(),
        **extra,
    }


def register_error_handlers(app: FastAPI) -> None:
    """Attach the error handlers to the app. Called in `create_app`."""

    @app.exception_handler(DomainError)
    async def domain_error_handler(request: Request, exc: DomainError) -> JSONResponse:
        status = _status_for(exc)
        logger.info(
            "domain_error",
            error=type(exc).__name__,
            status=status,
            path=safe_route_path(request),
            correlation_id=_correlation_id(),
        )
        return JSONResponse(
            status_code=status,
            content=_envelope(str(exc), type(exc).__name__),
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        # Same envelope as domain errors. `error` is the status phrase
        # ("Unauthorized", ...) — HTTPException's class name would say
        # nothing. Headers are preserved (WWW-Authenticate on 401 is an RFC
        # 9110 obligation, not a decoration).
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope(str(exc.detail), HTTPStatus(exc.status_code).phrase),
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Same envelope, plus the field-level details FastAPI collects.
        return JSONResponse(
            status_code=422,
            content=_envelope(
                "Request validation failed",
                "RequestValidationError",
                errors=jsonable_encoder(exc.errors()),
            ),
        )
