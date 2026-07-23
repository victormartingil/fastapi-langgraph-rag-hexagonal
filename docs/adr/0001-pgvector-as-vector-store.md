# ADR-0001: pgvector as the vector store

- **Status**: Accepted
- **Date**: 2026-07-01

## Context

The RAG system needs vector similarity search over chunk embeddings. Options:

- A dedicated vector database (Qdrant, Weaviate, Milvus, Pinecone).
- **pgvector**, an extension to the PostgreSQL we already run.

## Decision

Use **pgvector** inside the existing PostgreSQL.

## Rationale

- **One less moving part.** Documents, chunks, vectors and the full-text
  index live in one database, one backup story, one transaction boundary.
  The chunks table owns both the `vector(768)` column (HNSW-indexed) and the
  generated `tsvector` column (GIN-indexed) — the hybrid retrieval of
  ADR-0003 becomes a single SQL statement.
- **Consistency.** Vector store and metadata can never drift apart; deletes
  cascade; there is no dual-write problem.
- **Scale honesty.** Our target (thousands–low millions of chunks) is well
  within pgvector+HNSW territory. A dedicated vector DB would be resume
  driven design, not engineering.

## Consequences

- **The embedding dimension is part of the schema.** The migration creates
  `vector(768)` for the default model (`nomic-embed-text`). Switching to a
  provider with a different dimension (e.g. OpenAI's 1536) requires
  regenerating the migration and re-embedding the corpus. The dimension is
  defended at two layers, because config and reality can disagree twice:
  1. **Startup (config vs schema)** — the composition root compares the
     resolved dimension against `SCHEMA_EMBEDDING_DIMENSION` (the constant
     that mirrors the migration, in `container.py`) and **refuses to boot**
     with a `ValueError` naming this ADR on mismatch.
  2. **First ingest (reality vs config)** — `IngestDocument` checks the
     provider's first actual vector against the configured dimension and
     raises `EmbeddingDimensionMismatchError` naming the model and both
     dimensions, so a model/setting disagreement (e.g. a 3072-dim model
     override under a 1536 setting) fails loudly instead of corrupting the
     corpus at insert time.
- If the corpus ever outgrows pgvector, the `ChunkRetriever` port is the
  single seam to re-implement — the architecture was chosen to keep that
  option cheap.

## Alternatives considered

- **Qdrant/Weaviate**: better vector-specific features (payload filtering
  DSL, quantization), rejected as an unnecessary second datastore for Phase 1.
- **In-memory/FAISS**: rejected — the knowledge base must be permanent.
