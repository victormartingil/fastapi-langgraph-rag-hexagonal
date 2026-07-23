# ADR-0002: LangGraph in the application layer

- **Status**: Accepted
- **Date**: 2026-07-01

## Context

The read side (retrieve → grade → generate/refuse) needs orchestration with a
conditional branch. Two placement options:

- Treat LangGraph as infrastructure and hide the whole pipeline behind a
  single port (`QuestionAnsweringPipeline`).
- **Place the graph in the application layer**, where its nodes call output
  ports (`ChunkRetriever`, `AnswerGenerator`).

## Decision

The LangGraph graph lives in `assistant/application/graph/`. Nodes are factory
functions closing over Protocol ports. The domain never imports LangGraph
(enforced by import-linter). The compiled graph is injected into the
`AskQuestion` use case by the composition root.

## Rationale

- **"Retrieve, grade, then answer or refuse" is application policy**, not an
  infrastructure detail. It would exist with any orchestrator — LangGraph is
  merely the library that executes it, the same way SQLAlchemy is the library
  that executes persistence inside an adapter.
- Hiding the graph behind one coarse port would push the *policy* (node
  wiring, routing, thresholds) into infrastructure, where it could not be
  unit-tested without infrastructure. As placed, every node, the router, and
  the full compiled pipeline are unit-tested with hand-written fakes — no
  Docker, no LLM.
- The roadmap (Postgres checkpointer, router node for ad-hoc documents) grows
  the graph; keeping it in the application layer keeps those changes
  policy-level.

## Consequences

- The application layer has one extra dependency (`langgraph`). Acceptable:
  application layers may depend on libraries; they may not depend on
  adapters/vendors. The domain remains pure.
- **Pydantic AI** is treated differently: it *is* a vendor SDK, so it is
  quarantined in the `PydanticAiAnswerGenerator` infrastructure adapter and
  imported nowhere else.

## Alternatives considered

- **Plain `if/else` in the service**: honestly sufficient for three nodes;
  rejected because the project is a reference for growing RAG pipelines, and
  explicit graphs teach better.
- **Graph as infrastructure behind one port**: rejected (see Rationale).
