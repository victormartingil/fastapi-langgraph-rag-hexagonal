# Observability without content leakage

OpenTelemetry is an optional platform capability in this project. It provides
portable traces and metrics while preserving the central privacy rule:
**prompts, questions, document titles, filenames, chunk text, and generated
answers are not telemetry attributes**.

## What is measured

The API, outbound HTTP calls, and SQLAlchemy engine receive standard
instrumentation. Explicit content-safe spans cover:

- extraction, with file type only;
- embeddings, with provider, model, and batch size;
- retrieval, with requested and returned evidence counts;
- grading, with surviving evidence count;
- generation, with provider operation metadata;
- retry, error, latency, and abstention counters.

Every request has an `X-Correlation-ID`. The same value is bound to structured
logs and attached to the active request span as `correlation.id`, so an
operator can move from an error response to logs and traces without searching
for user content.

The GenAI attributes follow the evolving OpenTelemetry semantic conventions
where they are stable enough to be useful. Application-specific RAG attributes
use the `rag.*` namespace.

## Local trace UI

Telemetry stays disabled unless explicitly requested:

```bash
KA_OTEL_ENABLED=true docker compose --profile observability up --build
```

Open Jaeger at <http://localhost:16686>. The API exports OTLP/HTTP to the
Collector; the Collector forwards traces to Jaeger and prints aggregate
metrics through its debug exporter.

For an external collector:

```bash
KA_OTEL_ENABLED=true
KA_OTEL_SERVICE_NAME=knowledge-assistant
KA_OTEL_EXPORTER_OTLP_ENDPOINT=https://collector.example.com
```

The endpoint must be the OTLP/HTTP base URL, without `/v1/traces` or
`/v1/metrics`; the application appends those signal paths.

## Ownership and failure behavior

`platform/observability` owns provider setup and instrumentation lifecycle.
Domain and application code cannot import OpenTelemetry, enforced by the
architecture contracts. Adapter-level spans are closed under success, error,
and cancellation. Exporters flush during graceful shutdown.

An unavailable collector does not change a RAG response into an application
error: OTLP exporting is asynchronous and reports export failures through the
SDK. For regulated deployments, also review collector-side processors,
retention, access control, and network policy; content-safe application
attributes do not make an unrestricted telemetry backend safe by themselves.
