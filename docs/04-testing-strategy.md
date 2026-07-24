# 04 — Testing Strategy

> The pyramid, the fakes, the architecture tests, and exactly what each suite
> proves — including what it cannot prove.

## The suites

```
                 ┌──────────┐
                 │   e2e    │  31 tests — full HTTP flow, real DB, faked AI,
                 │          │  DB-down/probe-split/unhandled-exception paths
             ┌───┴──────────┴───┐
             │   integration    │  18 tests — repository + hybrid SQL vs real pgvector,
             │                  │  FTS parity guard, Spanish FTS in its own container
         ┌───┴──────────────────┴───┐
         │   architecture           │  3 tests — import contracts + naming rules
     ┌───┴──────────────────────────┴───┐
     │   unit                           │  132 tests — domain, use cases, graph nodes,
     │                                  │  mappers, AI + extraction adapters, outage
     │                                  │  translation, container wiring
     └──────────────────────────────────┘
```

| Suite | Command | Count | Needs Docker? |
| --- | --- | ---: | --- |
| unit              | `uv run --locked pytest tests/unit`         | 132 | No |
| architecture      | `uv run --locked pytest tests/architecture` | 3 | No |
| integration       | `uv run --locked pytest tests/integration`  | 18 | Yes |
| e2e               | `uv run --locked pytest tests/e2e`          | 31 | Yes |
| eval              | `uv run --locked pytest tests/evals`        | 8 | No |

Markers are registered in `pyproject.toml`, so you can also select by mark:
`uv run --locked pytest -m "not integration and not e2e"`.

## Tests and evaluations answer different questions

The deterministic suites prove software contracts: invariants, mappings,
transactions, SQL, HTTP behavior, architecture rules, and metric
calculations. A passing application suite must not depend on a live model.

The versioned `evals/` corpus measures RAG behavior:

| Dimension | Metric or evidence |
| --- | --- |
| retrieval coverage | Recall@5 |
| ranking quality | MRR |
| refusal behavior | abstention accuracy |
| provenance shape | citation validity |
| answer coverage | expected fact-phrase coverage |
| operational cost | latency p50/p95 |

The committed lexical baseline is reproducible and offline. Live retrieval is
also versioned, but it is explicitly model/infrastructure-bound:
Testcontainers starts pgvector, Alembic builds the schema, the real chunker
seeds deterministic document/chunk IDs, Ollama creates embeddings, and the
same PostgreSQL adapter runs `dense`, `lexical`, and `hybrid` SQL. Full
generation additionally executes the LangGraph workflow and Ollama LLM.

Regression thresholds—5 percentage points for Recall@5 and 0.05 for MRR—catch
movement; they do not certify a deployment. Each real corpus needs
representative and adversarial cases of its own.

This separation prevents flaky model behavior from weakening CI while also
preventing high code coverage from being mistaken for RAG quality.
It follows the distinction in Google's
[Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml)
between reliable infrastructure and measured model behavior. The corpus also
contains indirect-injection and competing-source cases; deployment-specific
red teaming should extend it following
[adversarial testing guidance](https://developers.google.com/machine-learning/guides/adv-testing).

## Unit tests — fakes, not mocks

`tests/unit/fakes.py` contains tiny **working** implementations of the ports:
a repository backed by a dict, an embedding provider returning deterministic
vectors, a retriever with canned chunks. Tests then read as executable
specifications:

```python
async def test_no_relevant_evidence_produces_an_honest_refusal(self) -> None:
    retriever = FakeChunkRetriever([make_retrieved_chunk(score=0.001)])
    generator = FakeAnswerGenerator(make_answer())

    answer = await self.build_use_case(retriever, generator).execute("...")

    assert answer.text == REFUSAL_MESSAGE
    assert generator.calls == []  # the LLM was never even called
```

Why not `unittest.mock`? A mock asserts *that* a method was called with
arguments; a fake lets you assert *what the system did*. Fakes are also
checked by mypy against the real Protocols (mypy runs over `src` **and**
`tests` — see CI and `.pre-commit-config.yaml`), so a port change breaks the
fake — exactly the coupling you want.

**What unit tests prove**: chunking boundaries (including the overlap tail on
hard splits and the trailing-sliver merge), value-object invariants, the
use-case pipeline (extract → chunk → embed → save), ingestion edge cases
(content-hash dedup, the lost-race recovery against a `RaceLosingRepository`
fake, and the rarer winnerless race → `ConcurrentIngestionError` → HTTP 409,
embedding batching, the first-batch dimension assertion, paginated
listing), the composition root's guards (dimension fail-fast, provider
defaults, missing API keys) and knob wiring (LLM timeout, default top_k),
every graph node, the routing decision, the refusal path, the
retriever-outage signal propagating through the compiled graph (never
swallowed into a refusal), the retriever adapter's outage translation
(transient provider/DB failures → `RetrievalUnavailableError`; permanent
failures untouched), the FTS-language parity comparison (mismatch → fail
fast naming both languages) and its missing-table pgcode classification
(42P01 vs permission errors), the fts_language config validator, the Unicode-aware FTS question tokenizer
(accents, umlauts, CJK, operator injection, pathological mega-token cap),
the extraction adapters' error quarantine (corrupt PDF →
`TextExtractionError`), and all mappers.

**What they cannot prove**: that SQL works, that the ORM mapping matches the
real schema, that the HTTP layer wires dependencies correctly.

## Unit tests for the AI adapters — respx, not fakes

The three HTTP-facing adapters (`OllamaEmbeddingProvider`,
`OpenAiEmbeddingProvider`, `PydanticAiAnswerGenerator`) are tested in
`tests/unit/test_ai_adapters.py` with the **real adapter code and real httpx**,
mocking only the HTTP boundary via [respx](https://lundberg.github.io/respx/)
— no Docker, no network, so they belong in the unit tier. Covered: response
parsing, OpenAI's index-ordered batching, the Authorization header, tenacity
retry-then-success on transient failures, the transient taxonomy itself
(5xx/429/timeouts/dropped connections retried; a permanent 401 is NOT —
one HTTP call, then failure), and the error doctrine on exhausted retries:
embedding adapters raise `EmbeddingProviderUnavailableError`, the LLM
adapter raises `GenerationUnavailableError` (connection-down included),
permanent LLM errors propagate loud — plus citation resolution and rejection
of missing or out-of-range citations.

Not covered, honestly documented: pydantic-ai's *internal* agent loop (tool
execution and validation retries). That machinery belongs to the vendor SDK;
we mock its HTTP boundary and trust its own test suite above it.

## Architecture tests — the rules are executable

Two layers of protection:

1. **import-linter contracts** (`pyproject.toml`, run by
   `tests/architecture/test_import_contracts.py`): layered contracts per
   context (adapters → application → domain), forbidden cross-context imports,
   and a "domain stays pure" contract banning framework imports in
   `*/domain/`.
2. **Naming/structure conventions** (`test_naming_conventions.py`): adapter
   classes must carry a technology prefix; port modules may contain only
   Protocol classes.

These are the tests that keep the architecture intact six months and fifty
PRs from now — including PRs written by AI agents (see `.ai-guidelines.md`).

## Integration tests — real PostgreSQL, thrown away every run

`tests/integration` uses [testcontainers] to start
the same digest-pinned `pgvector/pgvector:0.8.5-pg16` image as Compose,
then **runs the Alembic migrations** before yielding a session. That means
the migration itself is under test: if the schema cannot be built from
scratch, everything fails loudly at fixture setup.

- `test_repository.py`: ORM mapping vs the real schema, UUID and vector
  round-trips through asyncpg, cascade persistence, the `content_hash`
  unique index (a deliberate violation must surface as
  `DuplicateDocumentError`, the domain signal the use case recovers from),
  the dedup-race recovery path replayed across two real transactions, and
  the summary projection — paginated chunk counts computed by SQL **without a
  single `ChunkModel` entering the identity map** (asserted explicitly).
- `test_hybrid_retriever.py`: the hybrid SQL — dense leg, tsvector leg, RRF
  fusion — verified with hand-crafted vectors and wording, so the expected
  ranking is known exactly. Also home of the **query-plan guard**: an EXPLAIN
  assertion that the dense leg plans as an Index Scan over
  `ix_chunks_embedding_hnsw` (with `enable_seqscan = off`, because three rows
  would talk the planner out of any index — the test pins the query SHAPE,
  not the planner's cost math). A plan regression here is a silent
  performance cliff no functional test would catch.
- `test_multilingual_fts.py`: the configurable FTS language, proven end to
  end in a **second container** migrated with `KA_FTS_LANGUAGE=spanish` (the
  language is fixed at migration time, so it cannot share the English
  session container). A Spanish question must hit BOTH legs; a stopword-only
  question must degrade to pure dense retrieval; the stemming/stop-word
  contract itself is pinned with direct SQL assertions, so a PostgreSQL or
  image upgrade that changes Spanish text search fails loudly; and
  `schema_meta` must record `spanish` — introspected from the column, never
  copied from the environment.
- `test_schema_meta.py`: the startup parity guard (ADR-0004) against the
  real migrated database: parity passes, a tampered `schema_meta` row fails
  fast, and a missing row says "run the migrations". Tampering is always
  restored — the suite shares one container per session.

**Coverage honesty**: the 80% coverage gate (`fail_under` in
`pyproject.toml`) is measured on the unit suite over domain + application
(currently 100%). Infrastructure adapters are deliberately **excluded** from
that metric — the AI adapters are exercised by the respx-based unit tests
above, and the database adapters here, against real infrastructure. A
percentage over vendor-boundary code would mean little; executing it means
everything.

## E2E tests — everything real except the AI

`tests/e2e` boots the real FastAPI app (routers, middleware, error handlers),
the real Alembic-built database, the real repository, the real hybrid SQL and
the real compiled LangGraph graph. Only AI behavior is controlled, by
installing two fakes at the same composition seams used to swap vendors:

- `EmbeddingProvider` → deterministic vectors (offline, reproducible);
- `AnswerGenerator` → a scripted generator echoing its evidence as sources.

The tests walk the whole user journey: ingest `samples/return-policy.md` →
list → get → ask "Can I return a product after two months?" → answer with
cited sources → ask an unrelated question → honest refusal. Plus error
mapping (404 domain error / 413 over-limit / 415 unsupported type /
422 corrupt PDF, malformed UUID or empty extraction / 503 retrieval backend
down at query time / 503 embedding provider down during ingest / 503 LLM
down at answer time), the unified error envelope (401 with
WWW-Authenticate, domain errors, 422 validation with field details), the correlation-ID middleware, and the operational
hardening paths:

- **Idempotent re-upload** — the same bytes return 200 with the existing
  document id, and the listing still totals 1.
- **Upload limit** — a fixture builds the app with a ~104-byte cap, so the
  893-byte sample is rejected with 413.
- **Pagination** — `limit`/`offset` are echoed back, pages don't overlap, and
  `total` reflects the whole corpus.
- **API-key auth** — a fixture boots the app with `KA_API_KEY=test-secret`:
  `/api/v1/*` returns 401 without the header, 200 with it, `/health` stays
  open, and the interactive docs (`/docs`, `/redoc`, `/openapi.json`) return
  404 — open by default, closed when the API needs a key. The default-off
  behavior is covered by every other e2e test, which never sends the header.
- **Real container wiring** — one fixture installs NO dependency overrides
  and only swaps the two AI ports on the container, so the real
  per-request assembly runs (chunking, batch size, fetch_limit, rrf_k,
  min_relevance_score). It sets `retrieval_top_k=1`: omitting `top_k` in the
  request must yield one source even with two matching documents — if the
  knob were not wired, the hardcoded default would give two.
- **Provider-down wiring** — sibling fixtures keep the real wiring but
  install an embedding provider (or an LLM) that always fails:
  `POST /api/v1/documents` and `POST /api/v1/chat` must answer 503 with a
  truthful "temporarily unavailable" (retrieval, ingest, AND generation
  sides), proving each adapter's outage translation reaches the HTTP
  boundary.

All fixtures share one client-factory body; they differ only in the
`Settings` overrides — the same seam an operator uses via environment
variables.

## Writing a new test — decision guide

- Pure logic (domain, mappers, node behavior)? → **unit**, with a fake.
- HTTP adapter behavior (parsing, retries, outage translation)? → **unit**, with respx.
- "Does this SQL/ORM actually work?" → **integration**.
- "Does the whole request path work?" → **e2e**.
- "Could someone break the layering?" → **architecture**.

[testcontainers]: https://testcontainers-python.readthedocs.io/
