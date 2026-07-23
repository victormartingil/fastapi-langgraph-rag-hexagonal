"""PgVectorHybridRetriever: hybrid search adapter (dense + full-text + RRF).

HYBRID RETRIEVAL, explained by the code below:

1. DENSE leg — embed the question, rank chunks by cosine distance (`<=>`)
   against their stored vectors. Good at *meaning*: "give money back" matches
   "refund policy".
2. FULL-TEXT leg — rank chunks by `ts_rank_cd` of the generated `tsv` column
   against an OR-ed `to_tsquery` of the question words. Good at *exact words*:
   model numbers, rare terms, names that embeddings blur. We OR the terms
   (rather than `plainto_tsquery`, which ANDs them) because this leg optimizes
   for RECALL: precision comes from the dense leg and from RRF + grading.
   Stop words are dropped by PostgreSQL; an all-stop-word query yields an
   empty tsquery, which simply matches nothing.
3. RRF FUSION — Reciprocal Rank Fusion scores each chunk by
   `1 / (k + rank)` in each leg and sums the results. It needs no score
   normalization (dense distances and ts_rank live on different scales) and
   consistently beats either leg alone.

The whole thing is ONE SQL statement: the database is the right place to rank
rows. The SQL is written out in full (not hidden behind an ORM) because this
file is also the documentation — see docs/02-rag-explained.md.

QUERY SHAPE MATTERS: each leg is written as an inner `ORDER BY ... LIMIT`
subquery whose result is THEN ranked by `ROW_NUMBER()`, so the window ranks
only the `fetch_limit` survivors. The one-SELECT alternative (window +
ORDER BY + LIMIT together) leaves the bound to the planner's mercy:

- dense leg: on pgvector 0.8 / PG16 the planner happens to short-circuit it
  (incremental ROW_NUMBER + LIMIT over the index-ordered scan), but the
  WindowAgg still sits over a full-table-costed partition — one planner
  regression or version change away from a whole-table sort. The two-step
  shape puts the `LIMIT` node DIRECTLY on the HNSW index scan: the kNN bound
  is structural, not planner-dependent.
- full-text leg: measured difference today. With a large match set the old
  shape sorts EVERY match (EXPLAIN ANALYZE at 20k matching rows: quicksort,
  1706 kB); the pre-limited shape bounds the sort to the top-N (top-N
  heapsort, 26 kB). RRF ranks are identical — the window sees the same top-N
  in the same order.

An integration test asserts the dense leg plans as an Index Scan over
`ix_chunks_embedding_hnsw` (`test_hybrid_retriever.py`) — regressions here
are silent performance cliffs, not functional failures.

asyncpg note: the question vector is sent as a string literal and CAST to
`vector` inside SQL; this avoids driver-level codec concerns for parameters.
"""

import re

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from knowledge_assistant.knowledge_base.application.exceptions import (
    KnowledgeBaseUnavailableError,
)
from knowledge_assistant.knowledge_base.application.ports import EmbeddingProvider
from knowledge_assistant.knowledge_base.application.read_models import KnowledgeHit
from knowledge_assistant.knowledge_base.domain.exceptions import (
    EmbeddingProviderUnavailableError,
)
from knowledge_assistant.platform.database.session import is_db_outage_error
from knowledge_assistant.platform.http.resilience import (
    is_transient_http_error,
)

HYBRID_SEARCH_SQL = """
WITH dense_knn AS (
    -- Index-friendly kNN first: a plain ORDER BY ... LIMIT over the vector
    -- column is the one shape PostgreSQL can serve with the HNSW index.
    -- Deliberately NO id tiebreaker here: a composite sort key can no
    -- longer be served by the index ordering, the LIMIT loses its index
    -- scan, and the kNN bound silently degrades to a full sort (the EXPLAIN
    -- guard test catches exactly this). Determinism is restored one level
    -- up, over the already-bounded survivor set.
    SELECT c.id,
           c.embedding <=> CAST(:query_embedding AS vector) AS distance
    FROM chunks c
    ORDER BY c.embedding <=> CAST(:query_embedding AS vector)
    LIMIT :fetch_limit
),
dense AS (
    -- The window now ranks only the fetch_limit survivors, not the table.
    -- The id tiebreaker makes equal-distance ranks deterministic — cheap
    -- here because it sorts ~fetch_limit rows, never the table.
    SELECT id, ROW_NUMBER() OVER (ORDER BY distance, id) AS rank
    FROM dense_knn
),
full_text_top AS (
    -- The GIN-indexed @@ filter bounds the match set; pre-limiting to the
    -- top-N before ranking keeps the window small. RRF ranks are identical
    -- to ranking the full match set (same ordering, same top-N). The id
    -- tiebreaker makes equal-rank pages deterministic.
    SELECT c.id,
           ts_rank_cd(c.tsv, query) AS rank_score
    FROM chunks c, to_tsquery(CAST(:tsconfig AS regconfig), :or_query) AS query
    WHERE query <> ''::tsquery AND c.tsv @@ query
    ORDER BY rank_score DESC, c.id
    LIMIT :fetch_limit
),
full_text AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY rank_score DESC, id) AS rank
    FROM full_text_top
),
fused AS (
    SELECT COALESCE(d.id, f.id) AS id,
           COALESCE(1.0 / (:rrf_k + d.rank), 0.0)
             + COALESCE(1.0 / (:rrf_k + f.rank), 0.0) AS rrf_score
    FROM dense d
    FULL OUTER JOIN full_text f ON f.id = d.id
)
SELECT ch.id          AS chunk_id,
       ch.document_id AS document_id,
       doc.title      AS document_title,
       ch.content     AS content,
       fused.rrf_score AS score
FROM fused
JOIN chunks ch ON ch.id = fused.id
JOIN documents doc ON doc.id = ch.document_id
ORDER BY fused.rrf_score DESC, ch.id
LIMIT :limit
"""

_WORDS = re.compile(r"[^\W_]+")

# PostgreSQL raises (and the API would 500) when a tsquery lexeme exceeds
# 2047 bytes — reachable with a single pathological "word" (a pasted hash, a
# base64 dump). Such tokens carry no lexical signal, so they are DROPPED, not
# truncated: the remaining words still query, and an all-mega-token question
# degrades to the dense leg exactly like an all-stop-word one. 2000 bytes
# keeps a safe margin under the engine's limit.
_MAX_TOKEN_BYTES = 2000


def _to_or_query(question: str) -> str:
    """Turn free text into an OR tsquery: 'can I return it?' -> 'can | i | return | it'.

    The tokenizer is Unicode-aware (`[^\\W_]+` = word characters, any
    language, minus underscore): 'cómo' stays whole (ASCII-only `[a-z0-9]+`
    would mangle it into 'c | mo'), and CJK characters pass through instead
    of vanishing. Only word characters are kept, so tsquery operators
    (`&`, `|`, `!`, `:*`, parentheses) can never be smuggled in — the
    injection-safety property is structural, not a blacklist.

    PostgreSQL drops stop words of the configured language; if nothing
    remains, the empty tsquery matches nothing and the dense leg carries
    retrieval alone.
    """
    tokens = (
        token
        for token in _WORDS.findall(question.lower())
        if len(token.encode()) <= _MAX_TOKEN_BYTES
    )
    return " | ".join(tokens)


class PgVectorHybridRetriever:
    """Implements the knowledge-base retrieval port with PostgreSQL + pgvector."""

    def __init__(
        self,
        session: AsyncSession,
        embedding_provider: EmbeddingProvider,
        *,
        fetch_limit: int = 20,
        rrf_k: int = 60,
        tsconfig: str = "english",
    ) -> None:
        self._session = session
        self._embedding_provider = embedding_provider
        self._fetch_limit = fetch_limit
        self._rrf_k = rrf_k
        self._tsconfig = tsconfig

    async def retrieve(self, question: str, limit: int) -> list[KnowledgeHit]:
        try:
            [query_embedding] = await self._embedding_provider.embed([question])
        except EmbeddingProviderUnavailableError as exc:
            # The provider adapter translated the outage per the port
            raise KnowledgeBaseUnavailableError(
                "embedding provider unreachable after retries"
            ) from exc
        except httpx.HTTPError as exc:
            # Defensive: a provider adapter that does not honor the contract
            # (raises raw httpx). Transient outages -> 503; permanent errors
            # (a 401 is a misconfiguration, not an outage) keep propagating.
            if not is_transient_http_error(exc):
                raise
            raise KnowledgeBaseUnavailableError(
                "embedding provider unreachable after retries"
            ) from exc

        # pgvector accepts the canonical text form "[0.1, 0.2, ...]".
        vector_literal = "[" + ",".join(repr(v) for v in query_embedding.values) + "]"

        try:
            result = await self._session.execute(
                text(HYBRID_SEARCH_SQL),
                {
                    "query_embedding": vector_literal,
                    "or_query": _to_or_query(question),
                    # Bound parameter, cast to regconfig IN the database: the FTS
                    # configuration name can never become SQL injection.
                    "tsconfig": self._tsconfig,
                    "fetch_limit": self._fetch_limit,
                    "rrf_k": self._rrf_k,
                    "limit": limit,
                },
            )
        except Exception as exc:
            # Connection-level database failures only: an outage is a 503.
            # ProgrammingError and friends are bugs and must stay 500-visible.
            # (is_db_outage_error also covers asyncpg's raw-OSError connect
            # failures, which SQLAlchemy does not wrap.)
            if not is_db_outage_error(exc):
                raise
            raise KnowledgeBaseUnavailableError() from exc
        return [
            KnowledgeHit(
                chunk_id=str(row.chunk_id),
                document_id=str(row.document_id),
                document_title=row.document_title,
                content=row.content,
                score=float(row.score),
            )
            for row in result
        ]
