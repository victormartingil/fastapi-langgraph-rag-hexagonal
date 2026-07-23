"""Use cases of the knowledge-base context.

A use case is one application-level action, named with a verb. It receives
PORTS (Protocols) via constructor injection — never concrete adapters — so the
same class runs against PostgreSQL in production and in-memory fakes in tests.

This is the entire "service layer": orchestration of domain logic + ports.
No SQL, no HTTP, no vendor SDKs.
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

from anyio import to_thread

from knowledge_assistant.knowledge_base.application.ports import (
    DocumentRepository,
    EmbeddingProvider,
    KnowledgeRetriever,
    OpenRepository,
    TextExtractor,
)
from knowledge_assistant.knowledge_base.application.read_models import (
    DocumentPage,
    DocumentSummary,
    KnowledgeHit,
)
from knowledge_assistant.knowledge_base.domain.chunking import chunk_text
from knowledge_assistant.knowledge_base.domain.exceptions import (
    ConcurrentIngestionError,
    DocumentNotFoundError,
    DuplicateDocumentError,
    EmbeddingDimensionMismatchError,
    EmptyDocumentError,
    UnsupportedFileTypeError,
)
from knowledge_assistant.knowledge_base.domain.models import Chunk, Document
from knowledge_assistant.knowledge_base.domain.value_objects import DocumentId
from knowledge_assistant.shared.domain.value_objects import EmbeddingVector


@dataclass(frozen=True)
class IngestResult:
    """Outcome of ingestion: the document plus whether it was newly created.

    `created=False` means an identical document already existed and is being
    returned — ingestion is idempotent by content. Duplicates come back as a
    SLIM `DocumentSummary`: there is no point hydrating chunks and their
    768-float embeddings just to say "already exists".
    """

    document: Document | DocumentSummary
    created: bool


class IngestDocument:
    """Upload → extract → deduplicate → chunk → embed → persist.

    The pipeline is deliberately written as a flat sequence of steps: this is
    the code you show someone who asks "what does ingestion do?".

    Transaction shape (ADR-0005): the use case receives a repository-SCOPE
    factory, not a repository. Each `async with self._open_repository()` is a
    short unit of work — dedup check, save, race recovery — so the SLOW steps
    (extraction, the embedding HTTP calls) run with NO database connection
    held. Holding the request's session across embedding would pin a pooled
    connection for seconds per upload and exhaust the pool under modest
    concurrency.
    """

    def __init__(
        self,
        open_repository: OpenRepository,
        embedding_provider: EmbeddingProvider,
        text_extractors: Sequence[TextExtractor],
        *,
        chunk_max_chars: int = 800,
        chunk_overlap_chars: int = 120,
        embedding_batch_size: int = 32,
        expected_embedding_dimension: int | None = None,
        embedding_model_name: str = "the configured model",
    ) -> None:
        self._open_repository = open_repository
        self._embedding_provider = embedding_provider
        self._text_extractors = text_extractors
        self._chunk_max_chars = chunk_max_chars
        self._chunk_overlap_chars = chunk_overlap_chars
        self._embedding_batch_size = embedding_batch_size
        self._expected_embedding_dimension = expected_embedding_dimension
        self._embedding_model_name = embedding_model_name

    async def execute(self, file_name: str, data: bytes, title: str | None = None) -> IngestResult:
        extractor = self._pick_extractor(file_name)
        # Extraction can block for hundreds of milliseconds (pypdf parses the
        # whole file synchronously). A blocking call on the event loop stalls
        # EVERY concurrent request, so it runs in a worker thread instead.
        # The port stays sync: "extract text" is not inherently async — the
        # threading is an execution detail of the use case.
        raw_text = await to_thread.run_sync(extractor.extract, file_name, data)
        if not raw_text.strip():
            raise EmptyDocumentError(f"No extractable text in {file_name!r}")

        # Deduplication: identical content ingested twice returns the
        # original document (idempotent ingestion). The check is SLIM — a
        # summary query, no chunk hydration — and runs in its OWN short
        # transaction (scope 1): the scope closes before the slow work starts.
        content_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        async with self._open_repository() as repository:
            existing = await repository.get_summary_by_content_hash(content_hash)
        if existing is not None:
            return IngestResult(document=existing, created=False)

        chunk_texts = chunk_text(
            raw_text,
            max_chars=self._chunk_max_chars,
            overlap_chars=self._chunk_overlap_chars,
        )
        # The slow, fallible step — seconds of HTTP against the embedding
        # provider — deliberately happens BETWEEN scopes: no connection, no
        # transaction, nothing to roll back or pin (ADR-0005).
        embeddings = await self._embed_in_batches([str(t) for t in chunk_texts])

        document_id = DocumentId()
        chunks = tuple(
            Chunk(
                id=DocumentId(),
                text=text,
                position=index,
                embedding=embeddings[index],
            )
            for index, text in enumerate(chunk_texts)
        )
        document = Document(
            id=document_id,
            title=title or file_name,
            file_name=file_name,
            raw_text=raw_text,
            chunks=chunks,
            content_hash=content_hash,
        )
        return await self._save_or_recover_from_race(document, content_hash)

    async def _save_or_recover_from_race(
        self, document: Document, content_hash: str
    ) -> IngestResult:
        """Persist (scope 2), recovering from the dedup TOCTOU race.

        Two concurrent identical uploads can both pass the dedup check above;
        the loser trips the unique index at save time. That is not a failure —
        it is the same idempotent outcome one step later: in a FRESH scope (3)
        fetch the winner's (now committed) row and report "already exists".
        The failed scope 2 rolled itself back on the way out.
        """
        try:
            async with self._open_repository() as repository:
                await repository.save(document)
        except DuplicateDocumentError:
            async with self._open_repository() as repository:
                winner = await repository.get_summary_by_content_hash(content_hash)
            if winner is not None:
                return IngestResult(document=winner, created=False)
            # The winner's transaction rolled back too: nothing exists to
            # return idempotently, so the honest answer is 409 — the client
            # can retry the upload. (`from None`: the caught duplicate signal
            # would be a misleading cause — the real story is the vanished
            # winner.)
            raise ConcurrentIngestionError(content_hash) from None
        return IngestResult(document=document, created=True)

    async def _embed_in_batches(self, texts: list[str]) -> list[EmbeddingVector]:
        """Embed in fixed-size batches: one giant call is both a timeout risk
        and a single point of failure for the whole ingestion."""
        embeddings: list[EmbeddingVector] = []
        for start in range(0, len(texts), self._embedding_batch_size):
            batch = texts[start : start + self._embedding_batch_size]
            vectors = await self._embedding_provider.embed(batch)
            if start == 0:
                self._assert_dimension(vectors[0])
            embeddings.extend(vectors)
        return embeddings

    def _assert_dimension(self, vector: EmbeddingVector) -> None:
        """The startup guard checks CONFIGURATION against the schema; this
        checks REALITY against configuration — the provider's first actual
        vector must have the configured dimension, or persisting it would
        corrupt every future similarity search (ADR-0001)."""
        expected = self._expected_embedding_dimension
        if expected is not None and vector.dimension != expected:
            raise EmbeddingDimensionMismatchError(
                self._embedding_model_name, expected, vector.dimension
            )

    def _pick_extractor(self, file_name: str) -> TextExtractor:
        for extractor in self._text_extractors:
            if extractor.supports(file_name):
                return extractor
        raise UnsupportedFileTypeError(file_name)


class GetDocument:
    """Fetch one document by id, or raise a domain error.

    Uses the SLIM projection, not the full aggregate: DocumentResponse only
    exposes summary fields, so hydrating raw_text + embeddings just for the
    mapper to drop them would be pure waste.
    """

    def __init__(self, repository: DocumentRepository) -> None:
        self._repository = repository

    async def execute(self, document_id: DocumentId) -> DocumentSummary:
        summary = await self._repository.get_summary_by_id(document_id)
        if summary is None:
            raise DocumentNotFoundError(str(document_id))
        return summary


class ListDocuments:
    """List documents page by page, with totals for pagination."""

    def __init__(self, repository: DocumentRepository) -> None:
        self._repository = repository

    async def execute(self, limit: int, offset: int) -> DocumentPage:
        summaries = await self._repository.list_summaries(limit, offset)
        total = await self._repository.count()
        return DocumentPage(items=tuple(summaries), total=total, limit=limit, offset=offset)


class SearchKnowledge:
    """Public application API for ranked knowledge-base search."""

    def __init__(self, retriever: KnowledgeRetriever) -> None:
        self._retriever = retriever

    async def execute(self, question: str, limit: int) -> list[KnowledgeHit]:
        return await self._retriever.retrieve(question, limit)
