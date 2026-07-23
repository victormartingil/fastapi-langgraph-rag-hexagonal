# Advanced exercises

These exercises extend a real seam without putting completed solutions in the
main learning path. Each one should finish with an ADR or short design note,
tests at the appropriate level, and unchanged architecture contracts.

## 1. Add a DOCX extractor

Implement the existing `TextExtractor` port for `.docx`.

Acceptance questions:

- How are malformed, encrypted, and oversized archives rejected?
- Which parsing work must leave the event loop?
- How do you prove no partial document is persisted?
- Does the adapter name expose its technology?

## 2. Add another embedding or LLM provider

Select a provider with a genuinely different API rather than another
OpenAI-compatible URL.

Acceptance questions:

- Can the existing port express batching and ordering honestly?
- Which failures are transient?
- How is model dimension validated against the schema?
- Are prompts and documents absent from logs and telemetry?

## 3. Replace the in-process knowledge bridge with HTTP

Keep `AskQuestion`, assistant policies, and the `KnowledgeSearch` port
unchanged.

Acceptance questions:

- What is the versioned wire schema?
- How are deadlines, retries, cancellation, and correlation propagation
  handled?
- Which remote statuses become `RetrievalUnavailableError`?
- Can contract tests run against both in-process and HTTP adapters?

## 4. Add a LangGraph checkpointer

Introduce durable execution for one explicit requirement such as
human-in-the-loop approval.

Acceptance questions:

- What identifies a thread and who may resume it?
- What user data is persisted and for how long?
- How are graph/state schema changes migrated?
- Which partial-execution tests prove recovery?

## 5. Move ingestion to a worker

Create a worker inbound adapter and a durable queue boundary.

Acceptance questions:

- Where do uploaded bytes live before the worker runs?
- What is the idempotency key?
- Who owns retry and dead-letter policy?
- How does the API expose job state?
- Can the same ingestion policies still be tested without a broker?

## 6. Build a tenant-isolation threat test

Add tenant-aware ingestion and retrieval with adversarial tests attempting
cross-tenant access.

Acceptance questions:

- Is tenant scope mandatory in every port?
- Is isolation defended in both application and database?
- Do dense and lexical retrieval apply identical filters?
- Can logs, traces, caches, or eval reports leak one tenant to another?

## 7. Replace deterministic grading with a reranker

Add a reranker or LLM grader without removing the deterministic safety net
until evaluation demonstrates the replacement is superior.

Acceptance questions:

- What labeled cases measure the benefit?
- What are the latency and cost budgets?
- How does failure degrade?
- Does Recall@k improve without reducing abstention precision or citation
  validity?

Run the architecture contracts and the deterministic evaluation suite after
every exercise:

```bash
uv run --locked pytest tests/architecture tests/evals
```
