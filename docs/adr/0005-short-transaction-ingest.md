# ADR-0005: Ingest runs short-lived transactions around, not across, slow work

- **Status**: Accepted
- **Date**: 2026-07-23

## Context

Ingestion is a pipeline with two very different speeds:

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

Observability: a unit test (`RepositoryScopeRecorder` in tests/unit/fakes.py)
proves the embedding calls execute with ZERO repository scopes open, so the
transaction shape cannot silently regress.

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

## Consequences

- A slow or down embedding provider can no longer exhaust the database
  pool; ingest concurrency is bounded by HTTP timeouts, not connections.
- Each scope is independently committed/rolled back: a failed save leaves
  no partial document (chunks cascade in the same transaction), and a
  failed race recovery leaves the (committed or absent) winner untouched.
- `provide_ingest_document` no longer depends on `provide_session`;
  `GetDocument`/`ListDocuments`/`AskQuestion` keep the request-scoped
  session — they perform one fast read each, so a long-lived scope would
  buy nothing there.
- The use case now opens 2–3 tiny transactions per ingest instead of one
  long one: a negligible overhead (microseconds of pool checkout) against
  seconds of embedding time.
