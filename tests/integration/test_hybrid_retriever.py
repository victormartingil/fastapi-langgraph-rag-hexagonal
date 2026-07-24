"""Integration tests: PgVectorRetriever against real pgvector.

This is the heart of the read side, tested where it can only be tested:
against a real database. We insert three chunks with hand-crafted vectors and
wording, then verify that the hybrid SQL (dense leg + tsvector leg + RRF
fusion) ranks exactly as designed:

- "refund" chunk: dense-close AND full-text hit -> must rank first (both legs)
- "cafeteria" chunk: dense-far, no full-text hit -> must rank last
"""

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from knowledge_assistant.knowledge_base.adapters.outbound.persistence.repository import (
    SqlAlchemyDocumentRepository,
)
from knowledge_assistant.knowledge_base.adapters.outbound.retrieval.pgvector import (
    HYBRID_SEARCH_SQL,
    PgVectorRetriever,
)
from knowledge_assistant.knowledge_base.application.retrieval import RetrievalStrategy
from knowledge_assistant.knowledge_base.domain.models import Chunk, Document
from knowledge_assistant.knowledge_base.domain.value_objects import (
    ChunkId,
    ChunkText,
    DocumentId,
    EmbeddingVector,
)
from tests.unit.fakes import FakeEmbeddingProvider

pytestmark = pytest.mark.integration

DIM = 768


def vector(fill: float) -> EmbeddingVector:
    return EmbeddingVector(tuple([fill] * DIM))


def make_chunk(text: str, position: int, embedding: EmbeddingVector) -> Chunk:
    return Chunk(id=ChunkId(), text=ChunkText(text), position=position, embedding=embedding)


@pytest.fixture
async def seeded_session(session: AsyncSession) -> AsyncSession:
    """Two documents: one semantically+lexically about refunds, one not."""
    policy = Document(
        id=DocumentId(),
        title="Return Policy",
        file_name="return-policy.md",
        raw_text="Refunds within 30 days.",
        chunks=(
            make_chunk(
                "You may return any product within 30 days of purchase for a full refund.",
                0,
                vector(0.5),  # dense-close to the query (same direction)
            ),
            make_chunk(
                "Refunds are issued to the original payment method.",
                1,
                vector(0.5),  # also dense-close, and mentions "refund"
            ),
        ),
    )
    cafeteria = Document(
        id=DocumentId(),
        title="Cafeteria Guide",
        file_name="cafeteria.md",
        raw_text="The cafeteria opens at noon.",
        chunks=(
            make_chunk(
                "The cafeteria opens at noon on weekdays.",
                0,
                vector(-0.5),  # opposite direction: dense-far
            ),
        ),
    )
    repository = SqlAlchemyDocumentRepository(session)
    await repository.save(policy)
    await repository.save(cafeteria)
    return session


class TestPgVectorRetriever:
    async def test_hybrid_ranking_puts_the_double_match_first(
        self, seeded_session: AsyncSession
    ) -> None:
        # The question vector points in the same direction as the policy chunks.
        embedding_provider = FakeEmbeddingProvider(dimension=DIM, fill=0.5)
        retriever = PgVectorRetriever(seeded_session, embedding_provider)

        results = await retriever.retrieve("Can I get a refund for a returned product?", limit=5)

        assert len(results) == 3
        # Both full-text hits rank above the pure-dense cafeteria chunk.
        assert results[0].document_title == "Return Policy"
        assert results[1].document_title == "Return Policy"
        assert results[2].document_title == "Cafeteria Guide"
        # Scores are RRF scores, strictly ordered.
        assert results[0].score >= results[1].score > results[2].score
        # A chunk matched by BOTH legs scores higher than a single-leg match.
        assert results[0].score > 1.0 / 61  # > best possible single-leg score

    async def test_limit_is_respected(self, seeded_session: AsyncSession) -> None:
        embedding_provider = FakeEmbeddingProvider(dimension=DIM, fill=0.5)
        retriever = PgVectorRetriever(seeded_session, embedding_provider)

        results = await retriever.retrieve("refund", limit=1)

        assert len(results) == 1
        assert results[0].document_title == "Return Policy"

    async def test_explicit_sql_strategies_are_available(
        self, seeded_session: AsyncSession
    ) -> None:
        embedding_provider = FakeEmbeddingProvider(dimension=DIM, fill=0.5)
        retriever = PgVectorRetriever(seeded_session, embedding_provider)

        dense = await retriever.retrieve(
            "Can I get a refund for a returned product?",
            limit=5,
            strategy=RetrievalStrategy.DENSE,
        )
        lexical = await retriever.retrieve(
            "Can I get a refund for a returned product?",
            limit=5,
            strategy=RetrievalStrategy.LEXICAL,
        )
        hybrid = await retriever.retrieve(
            "Can I get a refund for a returned product?",
            limit=5,
            strategy=RetrievalStrategy.HYBRID,
        )

        assert [hit.document_title for hit in dense] == [
            "Return Policy",
            "Return Policy",
            "Cafeteria Guide",
        ]
        assert [hit.document_title for hit in lexical] == ["Return Policy", "Return Policy"]
        assert [hit.document_title for hit in hybrid] == [
            "Return Policy",
            "Return Policy",
            "Cafeteria Guide",
        ]

    async def test_no_rows_means_no_results(self, session: AsyncSession) -> None:
        embedding_provider = FakeEmbeddingProvider(dimension=DIM, fill=0.5)
        retriever = PgVectorRetriever(session, embedding_provider)

        assert await retriever.retrieve("anything", limit=5) == []

    async def test_dense_leg_plan_uses_the_hnsw_index(self, seeded_session: AsyncSession) -> None:
        """Regression guard for the W1 fix: the dense leg must be an
        index-served kNN, bounded BEFORE the window function runs.

        The two-step query shape (inner ORDER BY ... LIMIT, then ROW_NUMBER
        over the survivors) puts the LIMIT directly on the HNSW index scan —
        the kNN bound is structural. The one-SELECT shape it replaced left
        the bound to the planner (an incremental ROW_NUMBER short-circuited
        by LIMIT — or not, depending on version and plan), and its full-text
        leg sorted the entire match set. Silent performance cliffs, invisible
        to functional tests — so this test EXPLAINs the real query and
        asserts the index appears.

        `enable_seqscan` is switched off because the seeded table holds three
        rows: at that size the planner would (correctly) judge a seq scan
        cheaper and hide the plan shape under test. We are asserting that
        this query shape CAN use the index, not second-guessing the
        planner's cost arithmetic on a handful of rows.
        """
        query_vector = "[" + ",".join(["0.5"] * DIM) + "]"
        # EXPLAIN cannot take bind parameters through this driver stack, so
        # the literals are substituted directly — fine inside a throwaway
        # test database.
        explained = (
            HYBRID_SEARCH_SQL.replace(":query_embedding", f"'{query_vector}'")
            .replace(":or_query", "'refund'")
            .replace(":tsconfig", "'english'")
            .replace(":fetch_limit", "20")
            .replace(":rrf_k", "60")
            .replace(":limit", "5")
        )

        await seeded_session.execute(text("SET LOCAL enable_seqscan = off"))
        result = await seeded_session.execute(text(f"EXPLAIN {explained}"))
        plan = "\n".join(row[0] for row in result.all())

        assert "Index Scan using ix_chunks_embedding_hnsw" in plan
