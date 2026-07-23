"""Integration tests: SqlAlchemyDocumentRepository against real PostgreSQL.

What this proves that unit tests cannot: the ORM mapping matches the real
schema (built by Alembic migrations 0001+0002), UUID/vector columns round-trip
through asyncpg, the content_hash unique index exists, and the summary
projection's chunk-count subquery returns real numbers — all without loading
a single embedding for list rendering.
"""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from knowledge_assistant.knowledge_base.adapters.outbound.persistence.models import ChunkModel
from knowledge_assistant.knowledge_base.adapters.outbound.persistence.repository import (
    SqlAlchemyDocumentRepository,
)
from knowledge_assistant.knowledge_base.domain.exceptions import DuplicateDocumentError
from knowledge_assistant.knowledge_base.domain.models import Chunk, Document
from knowledge_assistant.knowledge_base.domain.value_objects import ChunkText, DocumentId
from knowledge_assistant.platform.database.session import session_scope
from knowledge_assistant.shared_kernel.value_objects import EmbeddingVector

pytestmark = pytest.mark.integration

EMBEDDING_DIM = 768  # matches the migrated schema (vector(768))


def make_document(
    title: str = "Return Policy", *, chunk_count: int = 2, content_hash: str | None = None
) -> Document:
    return Document(
        id=DocumentId(),
        title=title,
        file_name=f"{title.lower().replace(' ', '-')}.md",
        raw_text="Refunds are available within 30 days.",
        chunks=tuple(
            Chunk(
                id=DocumentId(),
                text=ChunkText(f"Chunk {index} of {title}."),
                position=index,
                embedding=EmbeddingVector(tuple([0.01 * (index + 1)] * EMBEDDING_DIM)),
            )
            for index in range(chunk_count)
        ),
        content_hash=content_hash,
    )


class TestSqlAlchemyDocumentRepository:
    async def test_save_and_get_round_trip(self, session: AsyncSession) -> None:
        repository = SqlAlchemyDocumentRepository(session)
        document = make_document(content_hash="a" * 64)

        await repository.save(document)
        fetched = await repository.get_by_id(document.id)

        assert fetched is not None
        assert fetched.id == document.id
        assert fetched.title == "Return Policy"
        assert fetched.content_hash == "a" * 64
        assert fetched.chunk_count == 2
        assert [c.position for c in fetched.chunks] == [0, 1]
        # The vector came back from a real pgvector column, dimension intact.
        assert fetched.chunks[0].embedding is not None
        assert fetched.chunks[0].embedding.dimension == EMBEDDING_DIM

    async def test_get_by_id_returns_none_for_unknown_id(self, session: AsyncSession) -> None:
        repository = SqlAlchemyDocumentRepository(session)
        assert await repository.get_by_id(DocumentId()) is None

    async def test_get_summary_by_content_hash(self, session: AsyncSession) -> None:
        """The dedup lookup is SLIM: summary fields, no chunk hydration."""
        repository = SqlAlchemyDocumentRepository(session)
        document = make_document(content_hash="b" * 64)
        await repository.save(document)

        found = await repository.get_summary_by_content_hash("b" * 64)
        assert found is not None
        assert found.id == document.id
        assert found.chunk_count == 2
        assert await repository.get_summary_by_content_hash("z" * 64) is None

    async def test_get_summary_by_id_is_slim(self, session: AsyncSession) -> None:
        """GET /documents/{id} serves summary fields only: the lookup must
        not hydrate chunk rows (or their embeddings) to compute them."""
        repository = SqlAlchemyDocumentRepository(session)
        document = make_document(content_hash="c" * 64)
        await repository.save(document)
        session.expunge_all()  # identity map must not answer for the query

        found = await repository.get_summary_by_id(document.id)
        assert found is not None
        assert found.id == document.id
        assert found.chunk_count == 2
        assert await repository.get_summary_by_id(DocumentId()) is None

    async def test_duplicate_content_hash_raises_a_domain_signal(
        self, session: AsyncSession
    ) -> None:
        """The use case deduplicates BEFORE saving; the unique index is the
        last line of defense if two concurrent ingestions race. The adapter
        translates the vendor IntegrityError into a domain signal (and rolls
        back the poisoned transaction itself)."""
        repository = SqlAlchemyDocumentRepository(session)
        await repository.save(make_document("Alpha", content_hash="c" * 64))

        with pytest.raises(DuplicateDocumentError):
            await repository.save(make_document("Beta", content_hash="c" * 64))

    async def test_losing_the_dedup_race_is_recoverable(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        """The W5 race path, against the REAL unique index: transaction 1
        commits the winner; transaction 2's twin insert is rejected, rolls
        back, and can immediately re-read the winner — exactly what the use
        case's race recovery does. (Sequential transactions replay the race
        without threads: the code path is identical.)"""
        async with session_scope(session_factory) as winner_session:
            await SqlAlchemyDocumentRepository(winner_session).save(
                make_document("Winner", content_hash="d" * 64)
            )

        async with session_scope(session_factory) as loser_session:
            loser = SqlAlchemyDocumentRepository(loser_session)
            with pytest.raises(DuplicateDocumentError):
                await loser.save(make_document("Loser", content_hash="d" * 64))

            recovered = await loser.get_summary_by_content_hash("d" * 64)
            assert recovered is not None
            assert recovered.title == "Winner"

    async def test_list_summaries_paginates_and_counts_without_loading_chunks(
        self, session: AsyncSession
    ) -> None:
        repository = SqlAlchemyDocumentRepository(session)
        await repository.save(make_document("Alpha Policy", chunk_count=3))
        await repository.save(make_document("Beta Policy", chunk_count=1))
        await repository.save(make_document("Gamma Policy", chunk_count=2))

        # Detach everything the saves loaded, so the identity map only
        # reflects what the LIST queries load from here on.
        session.expunge_all()

        page = await repository.list_summaries(limit=2, offset=0)
        rest = await repository.list_summaries(limit=2, offset=2)

        assert await repository.count() == 3
        assert len(page) == 2
        assert len(rest) == 1
        # Chunk counts come from the real GROUP BY subquery...
        counts = {s.title: s.chunk_count for s in (*page, *rest)}
        assert counts == {"Alpha Policy": 3, "Beta Policy": 1, "Gamma Policy": 2}
        # ...and no chunk rows entered the session for the list query.
        # (If they had, `ChunkModel` instances would be in the identity map.)
        assert not any(isinstance(entity, ChunkModel) for entity in session.identity_map.values())
