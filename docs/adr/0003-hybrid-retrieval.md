# ADR-0003: Hybrid retrieval with Reciprocal Rank Fusion

- **Status**: Accepted
- **Date**: 2026-07-01

## Context

Pure dense (vector) retrieval misses exact terms; pure full-text retrieval
misses paraphrases. The grading step (and ultimately the refusal behavior)
needs a ranked candidate list it can trust.

## Decision

Implement **hybrid retrieval** in `PgVectorHybridRetriever` as one SQL
statement with three parts:

1. **Dense leg** — cosine distance over the `vector(768)` column (HNSW).
   Written as an inner `ORDER BY embedding <=> :q LIMIT fetch_limit` subquery
   (the shape the HNSW index serves structurally), with `ROW_NUMBER()` over
   the survivors only — see the module docstring for the measured plans.
2. **Full-text leg** — `ts_rank_cd` over a generated `tsvector` column (GIN),
   queried with an **OR-ed `to_tsquery`** of the question words
   (recall-first; stop words dropped by PostgreSQL), pre-limited to the
   top-`fetch_limit` matches before ranking so the sort stays bounded
   regardless of match-set size. The text-search configuration is a bound
   parameter (`KA_FTS_LANGUAGE`, default `english`), not a hard-coded literal.
3. **RRF fusion** — score `Σ 1/(k + rank_leg)` with `k = 60`, full outer join
   on chunk id.

Grading then applies `min_relevance_score = 0.028`. Since the best single-leg
RRF score is `1/61 ≈ 0.0164`, the threshold effectively requires **consensus
between the two legs** — a deterministic, zero-cost grader.

## Rationale

- **RRF needs no score calibration.** Dense distances and `ts_rank` values
  live on unrelated scales; rank-based fusion sidesteps normalization
  entirely. This is why RRF (Cormack et al., 2009) is the default fusion
  method in production search systems.
- **One round-trip.** Both legs and the fusion run inside PostgreSQL; the
  alternative (two queries + fusion in Python) doubles latency and moves
  ranking logic out of the engine built for it.
- **OR semantics for full-text**: natural questions carry many non-content
  words; AND-ing them (`plainto_tsquery`) would silently empty the leg. The
  dense leg and the consensus threshold provide the precision the OR gives up.

## Consequences

- The threshold's meaning is stricter than "consensus": it is consensus at
  **rank ≲ 11 on both legs**. Two legs at rank *r* score `2/(60 + r)`, which
  clears 0.028 only while `60 + r ≤ 71.4`; a chunk ranked 12th on both legs
  (`2/72 ≈ 0.0278`) is *below* the bar. The bound is asymmetric: a top rank on
  one leg compensates a weak rank on the other (1st + 26th scores
  `1/61 + 1/86 ≈ 0.028`, which passes). Tuners should think in "required
  rank", not just "both legs".
- The threshold's meaning ("consensus of both legs") is coupled to `rrf_k`;
  the comment in `config.py` documents the arithmetic. Changing `rrf_k`
  without revisiting `min_relevance_score` silently changes grading.
- **Consensus grading is precision-first — it trades recall for trust.**
  Paraphrased questions with no shared terms can miss the full-text leg and
  trigger unwarranted refusals, and RRF ranks (hence the threshold's meaning)
  shift as the corpus grows. Accepted deliberately: a wrong refusal is
  recoverable, a confident hallucination is not.
- An LLM-as-judge grader remains a roadmap item; it would *add* a node, not
  replace this deterministic safety net.
- **The FTS language is schema-bound.** The generated `tsv` column bakes in
  the text-search configuration chosen via `KA_FTS_LANGUAGE` at migration
  time (migration 0003). Changing languages therefore means rebuilding the
  schema on a fresh database —
  `KA_FTS_LANGUAGE=spanish uv run --locked alembic upgrade head` — not
  flipping a runtime flag. This mirrors the embedding
  dimension (ADR-0001): retrieval correctness depends on the stored column
  matching the query-time configuration. Drift between the two is guarded
  twice (ADR-0004): Alembic reads the same `.env` as the app, and a startup
  parity check compares the configured language against the one recorded in
  `schema_meta`, refusing to boot on mismatch. For mixed-language corpora,
  `simple` (no stemming, no stop words) is the supported fallback; CJK
  corpora get no segmentation from tsvector and rely on the dense leg.

## Alternatives considered

- **Dense-only**: rejected — exact-term queries (product names, error codes)
  are the classic failure mode.
- **Weighted score fusion** (`α·dense + β·fts`): rejected — requires
  per-model score calibration and retuning per embedding model.
- **Reranker model (cross-encoder)**: deferred — adds a model dependency and
  latency; RRF is the right Phase-1 trade-off.
