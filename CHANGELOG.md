# Changelog

All notable changes to this project are documented here. The project follows
[Semantic Versioning](https://semver.org/).

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

[1.0.0]: https://github.com/victormartingil/fastapi-langgraph-rag-hexagonal/releases/tag/v1.0.0
