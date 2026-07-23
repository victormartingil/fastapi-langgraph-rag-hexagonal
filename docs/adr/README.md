# Architecture decision records

ADRs capture decisions that change a boundary, invariant, persistence model,
or deployment trade-off. They describe context, decision, consequences, and
rejected alternatives; they are not a second implementation guide.

| ADR | Decision | Status |
| --- | --- | --- |
| [0001](0001-pgvector-as-vector-store.md) | PostgreSQL + pgvector as the vector store | Accepted |
| [0002](0002-langgraph-as-orchestration-adapter.md) | LangGraph behind an application-owned workflow port | Accepted |
| [0003](0003-hybrid-retrieval.md) | Dense + lexical retrieval fused with RRF | Accepted |
| [0004](0004-schema-bound-config-parity-guard.md) | record schema-bound configuration and fail fast on drift | Accepted |
| [0005](0005-short-transaction-ingest.md) | short transactions around, never across, slow ingest work | Accepted |

Add an ADR when a change:

- moves ownership between bounded contexts;
- introduces or removes a deployable component;
- changes a public application API or port;
- changes storage, consistency, grounding, security, or failure semantics;
- accepts a material operational trade-off.

Do not add an ADR for a local refactor whose decision is obvious from the
code and existing rules.
