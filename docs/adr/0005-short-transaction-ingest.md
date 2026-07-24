# ADR-0005: Slow RAG work uses short database transactions

- **Status**: Accepted
- **Date**: 2026-07-23

## Context

Ingestion and chat retrieval both mix fast database work with slower AI work.

Ingestion has two very different speeds:

1. fast, cheap database work — the dedup lookup, the final `INSERT`;
2. slow, fallible work — text extraction (hundreds of ms of CPU) and the
   embedding HTTP calls (seconds, against a provider that may be busy).

Originally `IngestDocument` received a repository bound to the
request-scoped session (`provide_session`): one session — and, after the
first query, one pooled database CONNECTION — per request, held from the
dedup check until commit. That meant the connection stayed checked out for
the entire embedding phase: seconds per upload. PostgreSQL's default pool
(five connections + ten overflow) would exhaust under a handful of
concurrent uploads, turning a slow provider into a database outage for
every OTHER endpoint — including `/health` and `/chat`.

Holding an open transaction across slow, fallible work is also a
correctness smell: a crash mid-embedding leaves an idle-in-transaction
session behind (until the connection is recycled), and long transactions
delay vacuum and hold locks they do not need.

Chat has the same failure mode in reverse. Hybrid retrieval is fast SQL, but
grading and answer generation can wait on a local or remote LLM for seconds.
If `/chat` keeps a request-scoped session until the HTTP response is fully
rendered, one pooled connection can remain checked out while the application
is no longer using PostgreSQL at all.

## Decision

`IngestDocument` depends on a new port, `OpenRepository`
(`knowledge_base/application/ports.py`): a zero-argument factory returning an
async context manager around a `DocumentRepository`. Each `async with` is
one SHORT unit of work (session + transaction via `session_scope`), opened
and committed in milliseconds:

```
extract (thread) → hash
  → scope 1: dedup check            (short tx, closed immediately)
chunk → embed in batches            (NO session, NO transaction, NO tx to
                                     roll back if the provider dies)
  → scope 2: save                   (short tx; DuplicateDocumentError
                                     escapes and rolls back on exit)
  → scope 3 (only on race): re-read the winner → 200, or 409 if winnerless
```

The composition root wires the port with `repository_scope_factory`
(bootstrap.py), which also re-applies the outage doctrine at COMMIT time
(`is_db_outage_error` → `KnowledgeBaseUnavailableError` → 503), because
commit lives in `session_scope`, outside the repository's own translation.

The dedup check deliberately stays OUTSIDE the save transaction. Keeping
them in one long transaction would not close the TOCTOU window anyway —
two concurrent identical uploads can both pass the check under READ
COMMITTED no matter how the scopes are drawn. The unique index on
`content_hash` remains the actual arbiter; the race recovery (scope 3) is
the documented safety net.

`SearchKnowledge` follows the same rule through `OpenKnowledgeRetriever`: each
question opens one short retrieval scope, materializes the `KnowledgeHit`
list, commits/closes the session, and only then returns evidence to the
assistant context. LangGraph grading and `AnswerGenerator.generate()` run
after PostgreSQL has been released.

The composition root wires this with `retriever_scope_factory`. `GET` and
`LIST` endpoints still use the request-scoped session because they perform
only short SQL reads and do not wait on extraction, embeddings, grading, or
generation.

Observability: unit tests prove embedding calls execute with zero repository
scopes open and generation starts after the retrieval scope has closed. An
integration test runs chat with a database pool of one connection and a
blocked generator; a second SQL query must still complete while the first
chat request waits on the LLM.

## Alternatives considered

- **Keep one request-scoped session, but embed BEFORE first use.** The
  session only acquires a connection lazily, so embedding before the dedup
  check would avoid pinning a connection — but would embed documents that
  turn out to be duplicates (wasted provider calls on every re-upload) and
  still holds the transaction open across chunk assembly and save. Rejected:
  it optimizes the accidental detail (lazy checkout) instead of stating the
  intended transaction shape.
- **A single long transaction with `SELECT ... FOR UPDATE`-style locking**
  to serialize identical uploads: heavier, blocks the loser for the whole
  embedding duration, and buys nothing the unique index + race recovery
  already provide.
- **Raise pool size / add PgBouncer**: treats the symptom. The pool stays
  sized for short transactions; no upload may hold a connection while
  waiting on a third-party HTTP call.
- **Keep chat retrieval on the request-scoped session because the query is
  fast**: rejected because the session lifetime is the HTTP request, not the
  SQL query. A fast query can still leave the connection checked out while
  answer generation waits.

## Consequences

- A slow or down embedding provider can no longer exhaust the database
  pool; ingest concurrency is bounded by HTTP timeouts, not connections.
- A slow or down answer generator can no longer exhaust the database pool
  after retrieval has already completed.
- Each scope is independently committed/rolled back: a failed save leaves
  no partial document (chunks cascade in the same transaction), and a
  failed race recovery leaves the (committed or absent) winner untouched.
- `provide_ingest_document` and `provide_ask_question` no longer depend on
  `provide_session`. `GetDocument` and `ListDocuments` keep the request-scoped
  session because they perform one fast read each.
- The use case now opens 2–3 tiny transactions per ingest instead of one
  long one: a negligible overhead (microseconds of pool checkout) against
  seconds of embedding time.
- Chat opens one tiny transaction per retrieval instead of holding a session
  until the response finishes.
