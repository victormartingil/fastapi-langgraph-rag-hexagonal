# 02 — RAG Explained (with this codebase as the example)

> Retrieval-Augmented Generation, from chunking to cited answers. Every
> section points at the file that implements it.

## The problem RAG solves

An LLM only knows what was in its training data, and it will happily invent
plausible-sounding facts ("hallucinate") when it doesn't know. RAG fixes the
*knowledge* part: before asking the model, we **retrieve** relevant pieces of
our own documents and paste them into the prompt as evidence. The model then
answers *grounded* in that evidence — and, in this project, must cite it.

## The pipeline at a glance

```
WRITE (once per document)                    READ (once per question)
─────────────────────────                    ─────────────────────────
file ──► extract text                        question ──► embed
  │                                              │
  ▼                                              ▼
chunk ──► embed each chunk ──► PostgreSQL    hybrid search (dense + full-text)
                               + pgvector        │  fused with RRF
                                                 ▼
                                           grade: relevant enough?
                                                 │
                                      ┌──────────┴──────────┐
                                      ▼                     ▼
                                 LLM answer            honest refusal
                                 + cited sources       (no hallucination)
```

## Step 1 — Extraction

`documents/infrastructure/extraction/`. Uploads arrive as bytes; an adapter
per format (`PlainTextExtractor`, `PdfTextExtractor`) turns them into text.
The use case picks the first extractor whose `supports(file_name)` matches —
adding DOCX later is one new adapter.

## Step 2 — Chunking

`documents/domain/chunking.py`. Embeddings degrade when text is too long
(everything averages into mush) and retrieval degrades when chunks are too
short (no context). So we split into overlapping pieces of ~800 characters:

- split on paragraph boundaries (semantic units),
- greedily merge until the size budget is spent,
- carry a 120-character tail into the next chunk so a sentence cut at a
  boundary still exists whole *somewhere* — including when a single overlong
  paragraph has to be hard-split mid-text,
- merge a trailing "sliver" chunk (less than a quarter of the budget) into its
  predecessor — a slightly oversized chunk beats a context-free fragment.

This is a **pure domain service**: chunking policy is a business decision,
not an infrastructure concern.

## Step 3 — Embeddings

An embedding model maps text to a vector such that *similar meanings land
close together*. We embed each chunk (default: `nomic-embed-text`, 768
dimensions, running locally in Ollama — no API keys) and store the vector
next to the chunk in a `vector(768)` pgvector column.

## Step 4 — Hybrid retrieval

`chat/infrastructure/retrieval/pgvector_hybrid.py`. Neither search style is
enough alone:

|                     | Dense (vector) search           | Full-text (tsvector) search      |
| ------------------- | ------------------------------- | -------------------------------- |
| Finds               | *meaning* — "money back" ≈ "refund" | *exact words* — SKU, names, jargon |
| Misses              | rare exact terms                | paraphrases, synonyms            |

So the retriever runs **both** in one SQL statement:

1. **Dense leg**: rank by cosine distance (`embedding <=> query_vector`).
   Written as an inner `ORDER BY ... LIMIT fetch_limit` kNN subquery — the
   one shape the HNSW index serves structurally — with `ROW_NUMBER()` applied
   afterwards over just those survivors. (A window function sharing one
   SELECT with the ORDER BY + LIMIT leaves the bound to the planner's mercy:
   it happens to short-circuit on pgvector 0.8 / PG16, but the two-step shape
   makes it version-proof.)
2. **Full-text leg**: rank by `ts_rank_cd(tsv, to_tsquery(...))` against a
   `tsv` column that PostgreSQL keeps generated from `content` (GIN-indexed).
   Terms are OR-ed — this leg optimizes for recall. Pre-limiting to the
   top-`fetch_limit` matches before ranking keeps the sort bounded (top-N
   heapsort) even when thousands of rows match; RRF ranks are identical.
3. **RRF fusion** (Reciprocal Rank Fusion): each chunk scores
   `1 / (k + rank)` per leg, summed. RRF needs no normalization between the
   two score scales and rewards chunks found by *both* legs.

Two boundary semantics are worth knowing before tuning:

- **Per-leg pre-limit.** Each leg is bounded to its top `fetch_limit`
  candidates *before* fusion. A chunk ranked 21st on the dense leg but 1st on
  full-text therefore contributes only its full-text rank — its dense
  contribution never happened. This is inherent to fetching bounded candidate
  sets; it is the price of an index-served kNN.
- **`fetch_limit` vs `top_k`.** The API caps `top_k` at 20 and the retriever's
  `fetch_limit` defaults to 20, so even a maximum-size request draws from both
  legs' full candidate windows. Raising `top_k` beyond `fetch_limit` would
  silently return dense-leg filler below the fused candidates.

### Multilingual retrieval

The dense leg is language-agnostic by construction — the embedding model maps
Spanish, German, or Chinese text into the same semantic space. The full-text
leg needs language configuration at two levels:

- **Tokenizing the question** (`_WORDS` in `pgvector_hybrid.py`) uses the
  Unicode-aware pattern `[^\W_]+`, so accented characters (`cómo`), umlauts
  (`für`), and CJK characters survive as terms instead of being shredded by
  an ASCII-only `\w+`.
- **The FTS language** (`KA_FTS_LANGUAGE`, default `english`) picks the
  PostgreSQL text-search configuration used for stemming and stop words —
  `to_tsquery(CAST(:tsconfig AS regconfig), ...)` at query time and the
  generated `tsv` column at rest. Like the embedding dimension, this choice is
  **schema-bound**: migration 0003 builds the `tsv` column for the language
  named in the environment *at migration time*, so switching languages means
  rebuilding the schema on a fresh database
  (`KA_FTS_LANGUAGE=spanish uv run alembic upgrade head`). For genuinely
  mixed-language corpora, `simple` (no stemming, no stop words) is the honest
  fallback.

Because the language is baked into the schema, configuration and reality can
drift — migrate with one value, run with another, and full-text search
silently degrades. Two defenses prevent that (ADR-0004):

1. **Single source of truth.** Alembic loads the same `.env` the app reads,
   so `KA_FTS_LANGUAGE` in `.env` reaches migration 0003 exactly as it
   reaches `Settings` — same file, same precedence (real environment
   variables win in both).
2. **Startup parity guard.** Migration 0004 records the language the column
   was *actually* built with — introspected from the column expression, not
   copied from the environment — in a `schema_meta` table. At startup the
   app compares it against `KA_FTS_LANGUAGE` and **refuses to boot** on
   mismatch, with an error naming both languages and the fix.

What happens when the language is *wrong* for the corpus? The full-text leg
quietly degrades (foreign words are not stop words, but stemming mismatches
and position-heavy ranking make hits weak), the consensus grader then refuses
more often — degradation by design, not corruption. The dense leg still
carries semantically close results into the candidate list.

One caveat: PostgreSQL's tsvector does **not segment CJK text** (Chinese,
Japanese, Korean have no spaces between words), so those corpora rely almost
entirely on the dense leg. That is a known limitation, documented here rather
than fixed, because proper CJK tokenization (e.g. zhparser, sudachi) is an
extension-level commitment.

## Step 5 — Grading

`chat/application/graph/nodes.py::make_grade_node`. Retrieval always returns
*something*, even for nonsense questions — the top result of a bad search is
still a result. Grading filters chunks below `min_relevance_score`.

With `rrf_k = 60`, the best a **single-leg** match can score is
`1/61 ≈ 0.0164`. Our default threshold `0.028` is unreachable for single-leg
matches, so a chunk effectively must be found by **both** legs to count as
evidence — a cheap, deterministic "consensus" grader. (An LLM-as-judge
grader is a documented roadmap item.)

More precisely, consensus alone is not enough — it must be consensus at
**rank ≲ 11 on both legs**: two legs at rank *r* score `2/(60 + r)`, which
clears 0.028 only while `60 + r ≤ 71.4`. A chunk ranked 12th on both legs
(`2/72 ≈ 0.0278`) is *below* the bar. The bound is asymmetric, though: rank 1
on one leg compensates a weak rank on the other — 1st + 26th scores
`1/61 + 1/86 ≈ 0.028`, which also passes. Keep that mental model when tuning:
`min_relevance_score` and `rrf_k` move together.

**The honest trade-off.** Consensus grading is precision-first: it happily
sacrifices recall to keep wrong evidence out of the prompt. Two known failure
modes:

- **Paraphrase defeat.** A question that restates the document in different
  words ("When does the money-back window close?" vs. a policy that says
  "refunds within 30 days") scores well on the dense leg but may miss the
  full-text leg entirely — no shared terms, no consensus, unwarranted refusal.
- **Corpus-size dependence.** RRF scores depend on rank, and ranks depend on
  how many chunks compete. A threshold tuned on a handful of documents can be
  too strict (or too lax) on a thousand.

We accept both because a wrong refusal is recoverable (rephrase, or lower
`KA_MIN_RELEVANCE_SCORE`) while a confident hallucination is not. The real
fix is an LLM-as-judge grader that scores each chunk semantically —
[docs/03](docs/03-langgraph-orchestration.md#extending-the-graph-guided-exercise)
blueprints it as a guided exercise.

## Step 6 — Generation (or honest refusal)

If nothing survives grading, the graph routes to the **refuse** node and
answers a fixed, truthful "I could not find relevant information" — *without
calling the LLM at all*. A RAG system's most important feature is knowing
when to say no.

Otherwise `PydanticAiAnswerGenerator` sends question + surviving chunks to
the chat model with strict instructions ("answer ONLY from the context"),
and — critically — demands **structured output**: JSON validated against a
Pydantic schema `{answer, source_indices}`. Chunks are numbered `[1]`, `[2]`,
… in the prompt and the model cites those **numbers**, not database IDs —
small integers are far easier for an LLM to copy faithfully than UUIDs.
Citations pointing at chunks we never provided are dropped, so a hallucinated
citation cannot leak through.

**Degradation is honest on every path — one doctrine everywhere.** A
*transient* failure that outlives its retries becomes a domain signal the
HTTP layer answers with **503 "temporarily unavailable"** (the client learns
"try again later"): `RetrievalUnavailableError` when the embedding provider
or database is down at query time, `GenerationUnavailableError` when the
LLM is down at answer time, `EmbeddingProviderUnavailableError` when the
provider is down during ingestion. A *permanent* failure (a dead API key, a
SQL bug, malformed-output validation) is deliberately NOT translated —
misconfiguration reported as "temporary" would send clients into endless
retry loops, and a 200-with-fallback-message would be indistinguishable
from a real answer. Those stay loud: 500-class, in the logs, in your face.
Every error response — domain errors, HTTP errors, validation failures —
shares one envelope: `{detail, error, correlation_id}`.

## Try it yourself

```bash
curl -F "file=@samples/return-policy.md" -F "title=Return Policy" \
     http://localhost:8000/api/v1/documents

curl -X POST http://localhost:8000/api/v1/chat \
     -H 'Content-Type: application/json' \
     -d '{"question": "Can I return a product after two months?"}'
```

The policy says refunds stop at 30 days (store credit *maybe* until day 90),
so a grounded answer to "two months" is "no refund, possibly store credit" —
with the exact chunk cited. Ask "How do I bake sourdough?" and you get the
refusal instead.
