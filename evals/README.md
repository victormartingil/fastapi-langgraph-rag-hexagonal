# RAG evaluation

Evaluation is separate from the deterministic application test suite. The
versioned corpus contains 30 answerable, out-of-domain, paraphrased,
multilingual, competing-source, and indirect prompt-injection cases.

Run the deterministic lexical baseline:

```bash
uv run --locked python -m knowledge_assistant.evaluation.runner
```

Run the live retrieval pipeline. This starts an ephemeral pgvector database
with Testcontainers, applies Alembic, seeds the corpus with deterministic
document/chunk IDs, uses the real chunker and Ollama embeddings, then compares
the real SQL strategies (`dense`, `lexical`, `hybrid`):

```bash
TESTCONTAINERS_RYUK_DISABLED=true \
uv run --locked python -m knowledge_assistant.evaluation.runner \
  --mode live-retrieval \
  --baseline evals/live-baseline-qwen3.5-9b.json
```

Run live generation through LangGraph + Ollama:

```bash
TESTCONTAINERS_RYUK_DISABLED=true \
uv run --locked python -m knowledge_assistant.evaluation.runner \
  --mode live-full \
  --llm-model qwen3.5:9b
```

The runner writes both JSON and Markdown reports. It exits non-zero when
Recall@5 falls more than 5 percentage points or MRR more than 0.05 below a
mode present in the selected baseline.

The live baseline records that a zero-false-positive relevance threshold on
this small generic corpus has **0.0 answerable recall**. That is an explicit
quality finding, not a hidden failure: for real deployments, calibrate the
threshold against a representative corpus instead of copying this value.

A metric pass is a regression guard, not proof of production quality; inspect
failed cases and retune the corpus for each real deployment.
