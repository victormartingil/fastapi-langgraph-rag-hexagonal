"""Integration tests: the FTS language is a schema-bound, configurable choice.

Migration 0003 rebuilds the `tsv` generated column for the language named by
KA_FTS_LANGUAGE at migration time. These tests prove the whole chain against
a real database migrated with `KA_FTS_LANGUAGE=spanish`:

- a Spanish question matches a Spanish document through BOTH legs (dense +
  Spanish-stemmed full text),
- a question made only of Spanish stopwords produces an EMPTY tsquery, so the
  FTS leg silently drops out and the dense leg carries the answer alone,
- the expected stemming/stopword behavior itself, asserted directly in SQL so
  the test documents what PostgreSQL actually does (verified empirically on
  pgvector/pgvector:0.8.1-pg16).

The suite runs its own container because the language is fixed at migration
time: it cannot share the English session container from the root conftest.
"""

from collections.abc import AsyncIterator, Iterator

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from knowledge_assistant.knowledge_base.adapters.outbound.persistence.repository import (
    SqlAlchemyDocumentRepository,
)
from knowledge_assistant.knowledge_base.adapters.outbound.retrieval.pgvector import (
    PgVectorRetriever,
)
from knowledge_assistant.knowledge_base.domain.models import Chunk, Document
from knowledge_assistant.knowledge_base.domain.value_objects import (
    ChunkId,
    ChunkText,
    DocumentId,
    EmbeddingVector,
)
from knowledge_assistant.platform.database.schema_meta import (
    assert_fts_language_parity,
)
from knowledge_assistant.platform.database.session import (
    create_engine,
    create_session_factory,
    session_scope,
)
from tests.conftest import _run_migrations
from tests.unit.fakes import FakeEmbeddingProvider

pytestmark = pytest.mark.integration

DIM = 768


@pytest.fixture(scope="module")
def spanish_database_url() -> Iterator[str]:
    """A second container, migrated with the Spanish FTS configuration."""
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer("pgvector/pgvector:0.8.1-pg16") as postgres:
        host = postgres.get_container_host_ip()
        port = postgres.get_exposed_port(5432)
        url = (
            f"postgresql+asyncpg://{postgres.username}:{postgres.password}"
            f"@{host}:{port}/{postgres.dbname}"
        )
        _run_migrations(url, extra_env={"KA_FTS_LANGUAGE": "spanish"})
        yield url


@pytest.fixture
async def spanish_session(spanish_database_url: str) -> AsyncIterator[AsyncSession]:
    """A committed-on-success session against a freshly emptied Spanish database."""
    engine = create_engine(spanish_database_url)
    session_factory = create_session_factory(engine)

    async with session_factory() as cleanup:
        await cleanup.execute(text("DELETE FROM chunks"))
        await cleanup.execute(text("DELETE FROM documents"))
        await cleanup.commit()

    async with session_scope(session_factory) as session:
        yield session

    await engine.dispose()


def vector(fill: float) -> EmbeddingVector:
    return EmbeddingVector(tuple([fill] * DIM))


def make_chunk(content: str, position: int, embedding: EmbeddingVector) -> Chunk:
    return Chunk(id=ChunkId(), text=ChunkText(content), position=position, embedding=embedding)


@pytest.fixture
async def seeded_spanish_session(spanish_session: AsyncSession) -> AsyncSession:
    """A Spanish return policy (dense-close) and a Spanish cafeteria note (dense-far)."""
    policy = Document(
        id=DocumentId(),
        title="Política de Devoluciones",
        file_name="politica-devoluciones.md",
        raw_text="Puede devolver cualquier producto dentro de los 30 días.",
        chunks=(
            make_chunk(
                "Puede devolver cualquier producto dentro de los 30 días posteriores a la compra.",
                0,
                vector(0.5),
            ),
        ),
    )
    cafeteria = Document(
        id=DocumentId(),
        title="Guía de la Cafetería",
        file_name="cafeteria.md",
        raw_text="La cafetería abre al mediodía.",
        chunks=(make_chunk("La cafetería abre al mediodía entre semana.", 0, vector(-0.5)),),
    )
    repository = SqlAlchemyDocumentRepository(spanish_session)
    await repository.save(policy)
    await repository.save(cafeteria)
    return spanish_session


class TestSpanishFullTextSearch:
    async def test_spanish_question_matches_through_both_legs(
        self, seeded_spanish_session: AsyncSession
    ) -> None:
        """'¿Cómo puedo devolver un producto?' stems to 'com | pued | devolv |
        product' — all present in the policy chunk's Spanish tsvector. With the
        dense leg also close, the chunk must fuse from BOTH legs."""
        embedding_provider = FakeEmbeddingProvider(dimension=DIM, fill=0.5)
        retriever = PgVectorRetriever(
            seeded_spanish_session, embedding_provider, tsconfig="spanish"
        )

        results = await retriever.retrieve("¿Cómo puedo devolver un producto?", limit=5)

        assert len(results) == 2
        assert results[0].document_title == "Política de Devoluciones"
        # Both legs contributed: strictly above the best single-leg RRF score.
        assert results[0].score > 1.0 / 61
        # The cafeteria chunk matches no query term: dense leg only.
        assert results[1].score == pytest.approx(1.0 / 62)

    async def test_stopword_only_question_falls_back_to_the_dense_leg(
        self, seeded_spanish_session: AsyncSession
    ) -> None:
        """'el la un de' are all Spanish stopwords: the tsquery is EMPTY, the
        FTS leg returns nothing, and retrieval is pure dense — by design."""
        embedding_provider = FakeEmbeddingProvider(dimension=DIM, fill=0.5)
        retriever = PgVectorRetriever(
            seeded_spanish_session, embedding_provider, tsconfig="spanish"
        )

        results = await retriever.retrieve("el la un de", limit=5)

        assert len(results) == 2  # the dense leg still returns both chunks
        assert results[0].document_title == "Política de Devoluciones"
        assert results[0].score == pytest.approx(1.0 / 61)  # dense rank 1 alone
        assert results[1].score == pytest.approx(1.0 / 62)  # dense rank 2 alone

    async def test_schema_meta_records_the_introspected_language(
        self, spanish_database_url: str
    ) -> None:
        """Migration 0004 must record what the column ACTUALLY is ('spanish'
        here), never a copy of the environment — so the startup parity guard
        passes on this container with 'spanish' and would fail with
        'english'."""
        engine = create_engine(spanish_database_url)
        session_factory = create_session_factory(engine)
        try:
            await assert_fts_language_parity(session_factory, expected="spanish")
            with pytest.raises(ValueError, match="'spanish'"):
                await assert_fts_language_parity(session_factory, expected="english")
        finally:
            await engine.dispose()

    async def test_spanish_stemming_and_stopwords_as_postgres_defines_them(
        self, spanish_session: AsyncSession
    ) -> None:
        """Pin the linguistic contract directly: accents fold into stems and
        stopwords vanish. If a future PostgreSQL/image changes these rules,
        this test fails loudly instead of silently degrading retrieval."""
        result = await spanish_session.execute(
            text(
                "SELECT to_tsquery(CAST('spanish' AS regconfig),"
                " 'cómo | puedo | devolver | un | producto')::text"
            )
        )
        assert result.scalar_one() == "'com' | 'pued' | 'devolv' | 'product'"

        result = await spanish_session.execute(
            text("SELECT to_tsquery(CAST('spanish' AS regconfig), 'el | la | un | de')::text")
        )
        assert result.scalar_one() == ""  # every token was a stopword
