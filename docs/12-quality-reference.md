# 12 — Verified RAG Quality Reference

> What has been verified, what the numbers mean, and which model choices are
> intended for local learning versus higher-quality experiments.

## Verification snapshot

Last verified: **2026-07-24**.

Environment:

- Python 3.14.6 container runtime, package syntax compatible with Python 3.12.
- PostgreSQL 16 + pgvector `0.8.5-pg16`, digest-pinned in Compose and
  Testcontainers.
- Ollama `0.32.3`, digest-pinned in Compose.
- Embeddings: `nomic-embed-text`, 768 dimensions.
- Default local chat model: `qwen3.5:2b-q4_K_M`.
- Workflow: real chunking, Ollama embeddings, PostgreSQL dense/FTS/RRF
  retrieval, deterministic relevance grading, LangGraph orchestration, and
  provider-native structured output for Ollama.

## Local model compatibility

The public default optimizes for a reliable **fresh clone + Docker Compose**
experience. Larger models are useful, but should be explicit choices because
they can fail on small Docker VMs.

| Model | Role | Output mode | Verified result | Practical requirement |
| --- | --- | --- | --- | --- |
| `nomic-embed-text` | Embeddings | Ollama `/api/embed` | Passed live retrieval baseline | Fits modest local Docker VMs |
| `qwen3.5:2b-q4_K_M` | Default chat | Native JSON Schema via Pydantic AI | Passed fresh Compose smoke: health, ingest, cited answer, refusal | Good default for reproducible demos |
| `qwen3.5:4b-q4_K_M` | Optional chat | Native JSON Schema via Pydantic AI | Not part of the default gate | More memory; validate locally before using as a baseline |
| `qwen3.5:9b-q4_K_M` | Optional chat | Native JSON Schema via Pydantic AI | Not part of the default gate | Higher memory and slower startup |

Empirical note: unqualified `qwen3.5:4b` and `qwen3.5:9b` failed to load in
the local Docker VM used for the v1.1 platform refresh because the Ollama
model process was killed by the host. The repository therefore avoids making
those variants the default, even if they may work on a larger machine.

## Retrieval baseline

The committed live retrieval baseline runs against real PostgreSQL/pgvector
SQL and real Ollama embeddings:

| Strategy | Recall@5 | MRR |
| --- | ---: | ---: |
| dense | 1.000 | 1.000 |
| lexical | 0.864 | 0.856 |
| hybrid | 1.000 | 0.955 |

Hybrid retrieval is the assistant default because it gives dense retrieval
coverage while retaining lexical exact-match behavior. The baseline is a
regression guard for this educational corpus, not a universal threshold for a
customer corpus.

## Generation quality checks

The live generation path validates:

- the model returns structured output matching the schema;
- at least one source index is present for non-refusal answers;
- every source index points to a provided evidence chunk;
- invalid structure or invalid citations are retried and then returned as a
  typed 502 instead of a successful answer;
- out-of-domain questions refuse without calling the generator when retrieval
  leaves no relevant evidence.

A structurally valid citation is **not** proof of entailment. It proves that
the answer points to a known evidence chunk. It does not prove every sentence
is logically supported by that chunk. The evaluation harness therefore also
tracks expected fact-phrase coverage, and production deployments should add a
domain-specific entailment or review step if the cost of a wrong grounded
answer is high.

## How to read failures

- Retrieval failure: inspect the case-level `case_id`, expected document ids,
  retrieved document ids, and strategy metrics.
- Citation failure: inspect whether the model omitted sources, cited an
  out-of-range index, or produced a refusal when evidence existed.
- Fact coverage failure: inspect whether the model missed required facts or
  introduced unsupported claims.
- Abstention failure: split false positives from false negatives. They require
  different fixes: stricter grading for false positives, retrieval/corpus
  improvement for false negatives.

Reports intentionally avoid storing prompts, chunks, questions, or generated
answers by default. Case ids and aggregate metrics are enough for regression
tracking without leaking document content.

## Reproduce the evidence

```bash
# Deterministic metric tests
uv run --locked pytest tests/evals

# Live retrieval against real pgvector + Ollama embeddings
TESTCONTAINERS_RYUK_DISABLED=true \
uv run --locked python -m knowledge_assistant.evaluation.runner \
  --mode live-retrieval \
  --baseline evals/live-baseline-qwen3.5-2b-q4_K_M.json

# Optional full generation run
TESTCONTAINERS_RYUK_DISABLED=true \
uv run --locked python -m knowledge_assistant.evaluation.runner \
  --mode live-full \
  --llm-model qwen3.5:2b-q4_K_M
```

For a new domain, treat this as the harness to copy, not as a finished
quality guarantee. Replace the corpus with representative documents,
questions, adversarial cases, and acceptance thresholds before trusting the
system in production.
