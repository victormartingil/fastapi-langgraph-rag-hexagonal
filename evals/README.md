# RAG evaluation

Evaluation is separate from the deterministic application test suite. The
versioned corpus contains 30 answerable, out-of-domain, paraphrased,
multilingual, competing-source, and indirect prompt-injection cases.

Run the deterministic lexical baseline:

```bash
uv run --locked python -m knowledge_assistant.evaluation.runner
```

Compare lexical, dense, and reciprocal-rank-fused hybrid retrieval using a
live Ollama embedding model:

```bash
uv run --locked python -m knowledge_assistant.evaluation.runner \
  --ollama-url http://localhost:11434 \
  --embedding-model nomic-embed-text
```

With the Docker Compose API running, add end-to-end abstention, citation,
fact-phrase coverage, and latency p50/p95:

```bash
uv run --locked python -m knowledge_assistant.evaluation.runner \
  --ollama-url http://localhost:11434 \
  --api-url http://localhost:8000
```

The runner writes both `evals/report.json` and `evals/report.md`. It exits
non-zero when Recall@5 falls more than 5 percentage points or MRR more than
0.05 below a mode present in `baseline.json`. A metric pass is a regression
guard, not proof of production quality; inspect failed cases and retune the
corpus for each real deployment.
