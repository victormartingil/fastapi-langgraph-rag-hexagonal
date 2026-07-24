"""HTTP edge privacy tests: correlation ids and content-safe logging."""

from http import HTTPStatus
from uuid import UUID

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from structlog.testing import capture_logs

from knowledge_assistant.assistant.domain.exceptions import InvalidQuestionError
from knowledge_assistant.platform.http.error_handlers import register_error_handlers
from knowledge_assistant.platform.http.middleware import (
    CORRELATION_ID_HEADER,
    CorrelationIdMiddleware,
    resolve_correlation_id,
)


def _is_uuid(value: str) -> bool:
    UUID(value)
    return True


class TestCorrelationIds:
    def test_safe_header_value_is_kept(self) -> None:
        assert resolve_correlation_id("request_123.TEST-ok") == "request_123.TEST-ok"

    @pytest.mark.parametrize(
        "header_value",
        [
            None,
            "",
            "line\nbreak",
            "cliente-ñ",
            "x" * 129,
        ],
    )
    def test_invalid_header_value_is_replaced(self, header_value: str | None) -> None:
        resolved = resolve_correlation_id(header_value)

        assert resolved != header_value
        assert _is_uuid(resolved)


class TestHttpLoggingPrivacy:
    async def test_domain_error_log_excludes_user_content_but_keeps_correlation_id(
        self,
    ) -> None:
        app = FastAPI()
        app.add_middleware(CorrelationIdMiddleware)
        register_error_handlers(app)
        secret_question = "private question: client account 998877"

        @app.get("/domain-error")
        async def domain_error() -> None:
            raise InvalidQuestionError(secret_question)

        with capture_logs() as logs:
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://testserver",
            ) as client:
                response = await client.get(
                    "/domain-error",
                    headers={CORRELATION_ID_HEADER: "safe-request-1"},
                )

        assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
        assert response.json()["detail"] == secret_question
        assert response.headers[CORRELATION_ID_HEADER] == "safe-request-1"
        assert secret_question not in str(logs)
        assert {
            "event": "domain_error",
            "error": "InvalidQuestionError",
            "status": 422,
            "path": "/domain-error",
            "correlation_id": "safe-request-1",
            "log_level": "info",
        } in logs

    async def test_unhandled_error_log_excludes_exception_message(self) -> None:
        app = FastAPI()
        app.add_middleware(CorrelationIdMiddleware)
        secret_payload = "uploaded file title Contract-A-Private"

        @app.get("/boom")
        async def boom() -> None:
            raise RuntimeError(secret_payload)

        with capture_logs() as logs:
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://testserver",
            ) as client:
                response = await client.get(
                    "/boom",
                    headers={CORRELATION_ID_HEADER: "safe-request-2"},
                )

        assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
        assert response.json()["correlation_id"] == "safe-request-2"
        assert response.headers[CORRELATION_ID_HEADER] == "safe-request-2"
        assert secret_payload not in str(logs)
        assert {
            "event": "unhandled_exception",
            "exception_type": "RuntimeError",
            "path": "/boom",
            "correlation_id": "safe-request-2",
            "log_level": "error",
        } in logs
