"""Optional OpenTelemetry setup and content-safe operation instrumentation."""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib.metadata import version

from fastapi import FastAPI
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk.metrics import MeterProvider as SdkMeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode, Tracer
from sqlalchemy.ext.asyncio import AsyncEngine

INSTRUMENTATION_NAME = "knowledge_assistant"
_meter = metrics.get_meter(INSTRUMENTATION_NAME)
_operation_duration = _meter.create_histogram(
    "rag.operation.duration",
    unit="ms",
    description="Duration of content-safe RAG pipeline operations",
)
_operation_count = _meter.create_counter(
    "rag.operation.count",
    description="Completed RAG pipeline operations by outcome",
)
_retry_count = _meter.create_counter(
    "rag.retry.count",
    description="Retries issued at provider boundaries",
)
_abstention_count = _meter.create_counter(
    "rag.abstention.count",
    description="Answers refused because no grounded evidence survived",
)
_evidence_count = _meter.create_histogram(
    "rag.evidence.count",
    unit="{item}",
    description="Evidence chunks available to generation",
)


def _tracer() -> Tracer:
    return trace.get_tracer(INSTRUMENTATION_NAME)


@contextmanager
def observe_operation(
    operation: str,
    attributes: Mapping[str, str | int | float | bool] | None = None,
) -> Iterator[None]:
    """Create a span and metrics without accepting prompts or document text."""
    started = time.perf_counter()
    outcome = "ok"
    with _tracer().start_as_current_span(
        f"rag.{operation}", attributes=dict(attributes or {})
    ) as span:
        try:
            yield
        except BaseException as exc:
            outcome = "error"
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
            raise
        finally:
            metric_attributes = {"rag.operation.name": operation, "outcome": outcome}
            _operation_count.add(1, metric_attributes)
            _operation_duration.record(
                (time.perf_counter() - started) * 1_000,
                metric_attributes,
            )


def record_retry(operation: str) -> None:
    _retry_count.add(1, {"rag.operation.name": operation})


def record_abstention() -> None:
    _abstention_count.add(1)


def record_evidence(count: int) -> None:
    _evidence_count.record(count)


@dataclass(slots=True)
class TelemetryRuntime:
    """Owns process-wide OTel providers and instrumentor lifecycle."""

    enabled: bool
    tracer_provider: TracerProvider | None = None
    meter_provider: SdkMeterProvider | None = None
    _fastapi_apps: list[FastAPI] = field(default_factory=list)
    _httpx_instrumented: bool = False
    _sqlalchemy_instrumented: bool = False

    def instrument_app(self, app: FastAPI) -> None:
        if not self.enabled:
            return
        FastAPIInstrumentor.instrument_app(app, tracer_provider=self.tracer_provider)
        self._fastapi_apps.append(app)

    def instrument_engine(self, engine: AsyncEngine) -> None:
        if not self.enabled:
            return
        SQLAlchemyInstrumentor().instrument(
            engine=engine.sync_engine,
            tracer_provider=self.tracer_provider,
        )
        self._sqlalchemy_instrumented = True

    def shutdown(self) -> None:
        if not self.enabled:
            return
        for app in self._fastapi_apps:
            FastAPIInstrumentor.uninstrument_app(app)
        if self._httpx_instrumented:
            HTTPXClientInstrumentor().uninstrument()
        if self._sqlalchemy_instrumented:
            SQLAlchemyInstrumentor().uninstrument()
        if self.meter_provider is not None:
            self.meter_provider.shutdown()
        if self.tracer_provider is not None:
            self.tracer_provider.shutdown()


def configure_telemetry(
    *,
    enabled: bool,
    service_name: str,
    otlp_endpoint: str,
) -> TelemetryRuntime:
    """Configure OTLP/HTTP exporters only when explicitly enabled."""
    if not enabled:
        return TelemetryRuntime(enabled=False)

    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": version("fastapi-langgraph-rag-hexagonal"),
        }
    )
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{otlp_endpoint.rstrip('/')}/v1/traces"))
    )
    metric_reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint=f"{otlp_endpoint.rstrip('/')}/v1/metrics")
    )
    meter_provider = SdkMeterProvider(resource=resource, metric_readers=[metric_reader])
    trace.set_tracer_provider(tracer_provider)
    metrics.set_meter_provider(meter_provider)
    HTTPXClientInstrumentor().instrument(tracer_provider=tracer_provider)
    return TelemetryRuntime(
        enabled=True,
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        _httpx_instrumented=True,
    )
