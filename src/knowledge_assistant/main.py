"""Application entry point: `create_app()`.

Everything process-wide happens here: logging, the composition root,
middleware, error handlers, routers. The lifespan owns startup/shutdown so
resources (engine pool, HTTP clients) are released cleanly.

Run it with:  uv run uvicorn knowledge_assistant.main:create_app --factory
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

import structlog
from anyio import to_thread
from fastapi import Depends, FastAPI, Response, status
from sqlalchemy import text

from knowledge_assistant import bootstrap
from knowledge_assistant.assistant.adapters.inbound.http.router import router as chat_router
from knowledge_assistant.bootstrap import Container, build_container
from knowledge_assistant.config import Settings, get_settings
from knowledge_assistant.knowledge_base.adapters.inbound.http.router import (
    router as documents_router,
)
from knowledge_assistant.platform.database.schema_meta import (
    assert_fts_language_parity,
)
from knowledge_assistant.platform.http.error_handlers import register_error_handlers
from knowledge_assistant.platform.http.middleware import CorrelationIdMiddleware
from knowledge_assistant.platform.observability.logging import configure_logging
from knowledge_assistant.platform.observability.telemetry import configure_telemetry

logger = structlog.get_logger()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory: builds a fully wired FastAPI app.

    Accepting `settings` as a parameter (instead of reading globals) keeps the
    factory testable: e2e tests inject settings pointing at a throwaway
    testcontainers database.
    """
    settings = settings or get_settings()
    configure_logging(debug=settings.debug)
    telemetry = configure_telemetry(
        enabled=settings.otel_enabled,
        service_name=settings.otel_service_name,
        otlp_endpoint=settings.otel_exporter_otlp_endpoint,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        container: Container | None = None
        try:
            container = build_container(settings)
            telemetry.instrument_engine(container.engine)
            # Fail fast on schema/config drift (ADR-0004): the FTS language is
            # baked into the migrated schema, so booting with a different one
            # would silently degrade full-text search.
            await assert_fts_language_parity(
                container.session_factory, expected=settings.fts_language
            )
            app.state.container = container
            logger.info(
                "app_started",
                app=settings.app_name,
                embedding_provider=settings.embedding_provider,
                llm_provider=settings.llm_provider,
            )
            yield
        finally:
            try:
                if container is not None:
                    await container.aclose()
            finally:
                await to_thread.run_sync(telemetry.shutdown)
                logger.info("app_stopped")

    app = FastAPI(
        title=settings.app_name,
        lifespan=lifespan,
        # When auth is on, the interactive docs are an information leak: they
        # enumerate every endpoint and schema to anyone who asks. The same
        # KA_API_KEY switch that protects /api/v1/* closes them. /health is
        # a route, not a docs surface, and always stays open for probes.
        docs_url="/docs" if settings.api_key is None else None,
        redoc_url="/redoc" if settings.api_key is None else None,
        openapi_url="/openapi.json" if settings.api_key is None else None,
    )

    app.add_middleware(CorrelationIdMiddleware)
    register_error_handlers(app)
    app.include_router(documents_router)
    app.include_router(chat_router)

    @app.get("/livez", tags=["health"])
    async def livez() -> dict[str, str]:
        """LIVENESS probe: is the process up and serving requests?

        Deliberately dependency-free — no database, no container access. If
        this endpoint fails, the process itself is dead (or unreachable) and
        the orchestrator should RESTART it. Checking the database here would
        conflate "app dead" with "dependency down" and turn a transient DB
        hiccup into a pointless restart loop. Docker's healthcheck and a
        Kubernetes livenessProbe belong here.
        """
        return {"status": "ok"}

    @app.get("/health", tags=["health"])
    @app.get("/healthz", tags=["health"])
    async def health(
        response: Response,
        container: Annotated[Container, Depends(bootstrap.get_container)],
    ) -> dict[str, str]:
        """READINESS probe: can the app actually serve traffic right now?

        Verifies the database connection; a 503 means "don't route traffic
        here yet" (Kubernetes readinessProbe semantics) — the process stays
        up. Exposed under both names: `/health` is the original endpoint
        (kept for backward compatibility), `/healthz` the conventional one.
        """
        try:
            async with container.session_factory() as session:
                await session.execute(text("SELECT 1"))
        except Exception as exc:
            logger.error(
                "health_check_failed",
                exception_type=type(exc).__qualname__,
                path="/health",
            )
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return {"status": "unhealthy"}
        return {"status": "ok"}

    telemetry.instrument_app(app)
    return app
