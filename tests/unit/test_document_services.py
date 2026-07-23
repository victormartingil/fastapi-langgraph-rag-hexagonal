"""Unit tests for the documents use cases, driven entirely by fakes.

These tests are the executable specification of ingestion:
"upload → extract → deduplicate → chunk → embed → persist", with no Docker
and no network.
"""

import pytest

from knowledge_assistant.knowledge_base.application.ingest import IngestDocument
from knowledge_assistant.knowledge_base.application.queries import (
    GetDocument,
    ListDocuments,
)
from knowledge_assistant.knowledge_base.application.read_models import DocumentSummary
from knowledge_assistant.knowledge_base.domain.exceptions import (
    ConcurrentIngestionError,
    DocumentNotFoundError,
    EmbeddingDimensionMismatchError,
    EmptyDocumentError,
    InvalidDocumentMetadataError,
    UnsupportedFileTypeError,
)
from knowledge_assistant.knowledge_base.domain.models import Document
from knowledge_assistant.knowledge_base.domain.value_objects import DocumentId, EmbeddingVector
from tests.unit.fakes import (
    FakeDocumentRepository,
    FakeEmbeddingProvider,
    FakeTextExtractor,
    RaceLosingRepository,
    RepositoryScopeRecorder,
    fake_repository_scope,
)

LONG_TEXT = "\n\n".join(f"Paragraph {index} of the policy." for index in range(10))


def make_ingest(
    repository: FakeDocumentRepository | None = None,
    text: str = "First paragraph.\n\nSecond paragraph.",
    *,
    embedding_batch_size: int = 32,
    chunk_max_chars: int = 800,
) -> tuple[IngestDocument, FakeDocumentRepository]:
    repo = repository or FakeDocumentRepository()
    use_case = IngestDocument(
        open_repository=fake_repository_scope(repo),
        embedding_provider=FakeEmbeddingProvider(dimension=8),
        text_extractors=[FakeTextExtractor(text=text)],
        chunk_max_chars=chunk_max_chars,
        embedding_batch_size=embedding_batch_size,
    )
    return use_case, repo


class TestIngestDocument:
    async def test_ingests_a_document_with_embedded_chunks(self) -> None:
        use_case, repo = make_ingest()

        result = await use_case.execute("policy.md", b"ignored by the fake extractor")

        assert result.created is True
        document = result.document
        assert repo.documents[str(document.id)] is document
        assert document.title == "policy.md"  # defaults to the file name
        # Both paragraphs fit in a single 800-char chunk.
        assert document.chunk_count == 1
        assert document.content_hash is not None
        assert len(document.content_hash) == 64  # SHA-256 hex
        # Every persisted chunk carries an embedding of the right dimension.
        for chunk in document.chunks:
            assert chunk.embedding is not None
            assert chunk.embedding.dimension == 8

    async def test_title_can_be_overridden(self) -> None:
        use_case, _ = make_ingest()
        result = await use_case.execute("policy.md", b"x", title="Return Policy")
        assert result.document.title == "Return Policy"

    @pytest.mark.parametrize(
        ("file_name", "title"),
        [
            ("x" * 256, None),
            ("policy.md", "x" * 201),
            ("policy.md", "   "),
        ],
    )
    async def test_invalid_metadata_is_rejected_before_extraction(
        self, file_name: str, title: str | None
    ) -> None:
        use_case, _ = make_ingest()

        with pytest.raises(InvalidDocumentMetadataError):
            await use_case.execute(file_name, b"x", title=title)

    async def test_reingesting_identical_content_returns_the_original(self) -> None:
        """Idempotent ingestion: same content twice -> no duplicate document."""
        use_case, repo = make_ingest()
        first = await use_case.execute("policy.md", b"x")
        second = await use_case.execute("policy-copy.md", b"x")

        assert second.created is False
        assert second.document.id == first.document.id
        assert len(repo.documents) == 1  # no twin was stored

    async def test_same_file_name_with_different_content_is_not_a_duplicate(self) -> None:
        repo = FakeDocumentRepository()
        use_case, _ = make_ingest(repository=repo, text="Version one.")
        await use_case.execute("policy.md", b"x")

        other_use_case, _ = make_ingest(repository=repo, text="Version two.")
        second = await other_use_case.execute("policy.md", b"x")

        assert second.created is True
        assert len(repo.documents) == 2

    async def test_losing_the_dedup_race_returns_the_winner(self) -> None:
        """TOCTOU: two concurrent identical uploads both pass the hash check;
        the loser trips the unique index — and must still get the idempotent
        'already exists' outcome, not a 500."""
        text = "Content both racers carry."
        winner_result = await make_ingest(text=text)[0].execute("original.md", b"x")
        winner_document = winner_result.document
        assert isinstance(winner_document, Document)
        winner_summary = DocumentSummary(
            id=winner_document.id,
            title=winner_document.title,
            file_name=winner_document.file_name,
            created_at=winner_document.created_at,
            chunk_count=winner_document.chunk_count,
        )

        use_case = IngestDocument(
            open_repository=fake_repository_scope(RaceLosingRepository(winner_summary)),
            embedding_provider=FakeEmbeddingProvider(dimension=8),
            text_extractors=[FakeTextExtractor(text=text)],
        )

        result = await use_case.execute("racing-twin.md", b"x")

        assert result.created is False
        # The recovered winner is the SLIM summary — no chunk hydration.
        assert result.document == winner_summary

    async def test_losing_the_race_to_a_rolled_back_winner_is_a_409(self) -> None:
        """The rarest branch: we lost the race, but the winner's transaction
        ALSO rolled back — there is no document to return idempotently, so
        the honest answer is a 409-mapped error, not a fake 200."""
        use_case = IngestDocument(
            open_repository=fake_repository_scope(RaceLosingRepository(winner=None)),
            embedding_provider=FakeEmbeddingProvider(dimension=8),
            text_extractors=[FakeTextExtractor(text="Vanished winner content.")],
        )

        with pytest.raises(ConcurrentIngestionError):
            await use_case.execute("raced.md", b"x")

    async def test_wrong_provider_dimension_fails_loudly_on_first_batch(self) -> None:
        """The startup guard validates configuration against the schema; this
        validates REALITY against configuration (ADR-0001)."""
        use_case = IngestDocument(
            open_repository=fake_repository_scope(FakeDocumentRepository()),
            embedding_provider=FakeEmbeddingProvider(dimension=8),
            text_extractors=[FakeTextExtractor(text="Any text.")],
            expected_embedding_dimension=768,
            embedding_model_name="nomic-embed-text",
        )

        with pytest.raises(EmbeddingDimensionMismatchError, match="8-dimensional vectors"):
            await use_case.execute("a.md", b"x")

    async def test_embeddings_are_batched(self) -> None:
        """Many chunks must not become one giant provider call."""
        embedding_provider = FakeEmbeddingProvider(dimension=8)
        use_case = IngestDocument(
            open_repository=fake_repository_scope(FakeDocumentRepository()),
            embedding_provider=embedding_provider,
            text_extractors=[FakeTextExtractor(text=LONG_TEXT)],
            chunk_max_chars=30,  # each 25-char paragraph becomes its own chunk
            chunk_overlap_chars=0,
            embedding_batch_size=3,
        )

        result = await use_case.execute("policy.md", b"x")

        # 10 chunks / batch size 3 -> 4 provider calls, order preserved.
        assert len(embedding_provider.received_texts) == 4
        flattened = [t for batch in embedding_provider.received_texts for t in batch]
        assert flattened[0].startswith("Paragraph 0")
        assert result.document.chunk_count == len(flattened)

    async def test_unsupported_file_type_raises_domain_error(self) -> None:
        use_case, _ = make_ingest()
        with pytest.raises(UnsupportedFileTypeError):
            await use_case.execute("photo.png", b"\x89PNG")

    async def test_embedding_runs_with_no_database_scope_open(self) -> None:
        """ADR-0005, made observable: the slow embedding HTTP calls must run
        BETWEEN database scopes — holding a pooled connection across them
        would exhaust the pool under modest upload concurrency.

        The recorder counts how many repository scopes are open RIGHT NOW;
        the embedding spy snapshots that count the moment it is called."""
        recorder = RepositoryScopeRecorder(FakeDocumentRepository())
        open_scopes_at_embed_time: list[int] = []

        class SpyEmbeddingProvider:
            async def embed(self, texts: list[str]) -> list[EmbeddingVector]:
                open_scopes_at_embed_time.append(recorder.currently_open)
                return [EmbeddingVector((0.5,) * 8) for _ in texts]

        use_case = IngestDocument(
            open_repository=recorder,
            embedding_provider=SpyEmbeddingProvider(),
            text_extractors=[FakeTextExtractor(text="Some policy text.")],
        )

        result = await use_case.execute("policy.md", b"x")

        assert result.created is True
        # Embedding happened with ZERO database scopes open...
        assert open_scopes_at_embed_time == [0]
        # ...and exactly two short scopes existed in total: the dedup check
        # and the save (no race recovery needed on the happy path).
        assert recorder.total_opened == 2
        assert recorder.currently_open == 0

    async def test_empty_extraction_raises_domain_error(self) -> None:
        use_case, _ = make_ingest(text="   \n  ")
        with pytest.raises(EmptyDocumentError):
            await use_case.execute("empty.md", b"x")


class TestGetDocument:
    async def test_returns_the_document_summary(self) -> None:
        """The use case serves the read model (slim projection), matching
        what DocumentResponse actually exposes — not the full aggregate."""
        use_case, repo = make_ingest()
        result = await use_case.execute("policy.md", b"x")

        fetched = await GetDocument(repo).execute(result.document.id)
        assert fetched.id == result.document.id
        assert fetched.title == result.document.title
        assert fetched.chunk_count == result.document.chunk_count

    async def test_missing_document_raises_domain_error(self) -> None:
        with pytest.raises(DocumentNotFoundError):
            await GetDocument(FakeDocumentRepository()).execute(DocumentId())


class TestListDocuments:
    async def test_returns_a_page_with_totals(self) -> None:
        use_case, repo = make_ingest()
        await use_case.execute("a.md", b"x")
        other, _ = make_ingest(repository=repo, text="Different content.")
        await other.execute("b.md", b"x")

        page = await ListDocuments(repo).execute(limit=10, offset=0)

        assert page.total == 2
        assert page.limit == 10
        assert page.offset == 0
        assert {s.file_name for s in page.items} == {"a.md", "b.md"}
        # Summaries expose chunk_count without carrying chunks.
        assert all(s.chunk_count == 1 for s in page.items)

    async def test_pagination_slices_the_results(self) -> None:
        repo = FakeDocumentRepository()
        for index in range(3):
            use_case, _ = make_ingest(repository=repo, text=f"Content number {index}.")
            await use_case.execute(f"doc-{index}.md", b"x")

        first_page = await ListDocuments(repo).execute(limit=2, offset=0)
        second_page = await ListDocuments(repo).execute(limit=2, offset=2)

        assert len(first_page.items) == 2
        assert len(second_page.items) == 1
        assert first_page.total == second_page.total == 3
        all_ids = [str(s.id) for s in (*first_page.items, *second_page.items)]
        assert len(set(all_ids)) == 3  # no overlap between pages
