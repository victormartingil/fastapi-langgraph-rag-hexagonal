# Changelog

All notable changes to this project are documented here. The project follows
[Semantic Versioning](https://semver.org/).

## [1.1.0] - 2026-07-24

### Fixed

- Released PostgreSQL connections before LLM grading/generation on chat
  requests, including cancellation and error paths.
- Hardened Ollama structured output, provider-specific model construction,
  citation validation, and typed generation failures.
- Separated document and chunk identities with dedicated domain value objects.
- Hardened shutdown cleanup, content-safe error logging, and correlation ID
  validation.

### Changed

- Refreshed the platform stack for July 2026: Python 3.14.6 reference runtime,
  Pydantic AI 2.17.0, OpenTelemetry 1.44.0 / 0.65b0, pgvector 0.8.5-pg16,
  Ollama 0.32.3, Collector 0.157.0, and Jaeger 2.20.0.
- Switched the reproducible local Ollama default to `qwen3.5:2b-q4_K_M`;
  larger Qwen variants remain documented opt-in choices.
- Renamed retrieval adapters and exposed explicit dense, lexical, and hybrid
  retrieval strategies while keeping hybrid as the assistant default.

### Added

- Live RAG evaluation modes for the real PostgreSQL/pgvector and Ollama
  pipeline, including case-level generation quality reports.
- A documented quality reference with live retrieval metrics, model
  compatibility, citation limitations, and failure-reading guidance.
- Non-blocking Trivy HIGH/CRITICAL SARIF upload, release wheel/sdist/SBOM,
  checksums, and provenance attestations.

### Verified

- HTTP contract remains backward compatible.
- Full local suite passed with 210 tests.
- Fresh Docker Compose smoke passed with health, ingest, cited grounded answer,
  and out-of-domain refusal.
- GitHub CI and security checks passed for every merged PR in this release.

## [1.0.0] - 2026-07-24

### Added

- Two bounded contexts with executable hexagonal dependency rules.
- FastAPI ingestion and grounded question-answering APIs.
- Hybrid PostgreSQL/pgvector retrieval with Reciprocal Rank Fusion.
- LangGraph orchestration isolated behind an application port.
- Structured answer validation, mandatory citations, abstention, and typed
  provider failures.
- A 30-case RAG evaluation corpus with Recall@k, MRR, abstention, citation,
  fact-coverage, and latency metrics.
- Optional content-safe OpenTelemetry traces and metrics.
- Python 3.12–3.14 CI, real PostgreSQL integration/E2E tests, build and
  fresh-install checks, dependency audit, CodeQL, secret scanning, container
  scanning, SBOM generation, and digest-pinned build inputs.
- Architecture, threat-model, testing, observability, and evolution guides.

### Security

- Untrusted questions, titles, and document chunks are explicitly delimited.
- Affirmative answers without valid evidence indices fail rather than return
  a misleading success response.
- Telemetry omits prompts, questions, document metadata and content, answers,
  exception messages, and stack traces by default.
- Corrupt PDF headers are rejected before the parser can log uploaded bytes.

[1.1.0]: https://github.com/victormartingil/fastapi-langgraph-rag-hexagonal/releases/tag/v1.1.0
[1.0.0]: https://github.com/victormartingil/fastapi-langgraph-rag-hexagonal/releases/tag/v1.0.0
