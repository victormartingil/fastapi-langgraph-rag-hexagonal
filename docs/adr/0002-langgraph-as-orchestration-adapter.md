# ADR-0002: LangGraph as an orchestration adapter

- **Status**: Accepted
- **Date**: 2026-07-01

## Context

The assistant flow needs state, conditional routing, and an evolution path
toward checkpointing. LangGraph supplies that runtime, but its graph types,
compiled state, and execution API are vendor-specific.

## Decision

LangGraph lives under
`assistant/adapters/outbound/orchestration/langgraph/` and implements the
application-owned `RagWorkflow` port. `AskQuestion` depends only on that port.

Filtering, generate/refuse routing, and refusal construction remain pure
functions in `assistant/application/policies.py`. Graph nodes coordinate
those policies with `KnowledgeSearch` and `AnswerGenerator`.

## Rationale

- The application owns decisions; the adapter owns LangGraph state and
  execution mechanics.
- A plain-Python, HTTP, or durable workflow implementation can replace the
  adapter without changing `AskQuestion`.
- Pure policies and individual nodes remain independently testable, while
  adapter tests cover routing and partial graph execution.

## Consequences

- Domain and application contain no LangGraph imports; import-linter enforces
  that boundary.
- `bootstrap.py` selects `LangGraphRagWorkflow`.
- A checkpointer is intentionally absent until durable execution, memory, or
  human-in-the-loop creates a real requirement.

## Alternatives considered

- **Plain `if/else` in `AskQuestion`**: sufficient today, but it would couple
  the use case to one execution model as the workflow grows.
- **LangGraph directly in application**: rejected because compiled graph and
  state APIs are replaceable infrastructure concerns.
