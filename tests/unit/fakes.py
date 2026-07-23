"""Hand-written fakes: in-memory implementations of the output ports.

Why fakes and not mocks? A fake is a tiny WORKING implementation (this
repository actually stores documents; this embedder actually returns
vectors), so tests read like executable specifications of the use case:
"when I ingest a file, a document with embedded chunks is saved". No mocking
framework DSL, no asserting on calls — just arrange/act/assert.

Each fake satisfies its port Protocol STRUCTURALLY; if a port signature
changes, mypy (run over tests/ too) flags the fake.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from knowledge_assistant.assistant.domain.models import Answer, RetrievedChunk
from knowledge_assistant.knowledge_base.application.ports import OpenRepository
from knowledge_assistant.knowledge_base.application.read_models import DocumentSummary
from knowledge_assistant.knowledge_base.domain.exceptions import DuplicateDocumentError
from knowledge_assistant.knowledge_base.domain.models import Document
from knowledge_assistant.knowledge_base.domain.value_objects import DocumentId
from knowledge_assistant.shared.domain.value_objects import EmbeddingVector


class FakeDocumentRepository:
    """Implements DocumentRepository with a dict."""

    def __init__(self) -> None:
        self.documents: dict[str, Document] = {}

    async def save(self, document: Document) -> None:
        self.documents[str(document.id)] = document

    async def get_by_id(self, document_id: DocumentId) -> Document | None:
        return self.documents.get(str(document_id))

    async def get_summary_by_content_hash(self, content_hash: str) -> DocumentSummary | None:
        match = next(
            (d for d in self.documents.values() if d.content_hash == content_hash),
            None,
        )
        if match is None:
            return None
        return DocumentSummary(
            id=match.id,
            title=match.title,
            file_name=match.file_name,
            created_at=match.created_at,
            chunk_count=match.chunk_count,
        )

    async def get_summary_by_id(self, document_id: DocumentId) -> DocumentSummary | None:
        match = self.documents.get(str(document_id))
        if match is None:
            return None
        return DocumentSummary(
            id=match.id,
            title=match.title,
            file_name=match.file_name,
            created_at=match.created_at,
            chunk_count=match.chunk_count,
        )

    async def list_summaries(self, limit: int, offset: int) -> list[DocumentSummary]:
        ordered = sorted(self.documents.values(), key=lambda d: d.created_at, reverse=True)
        return [
            DocumentSummary(
                id=d.id,
                title=d.title,
                file_name=d.file_name,
                created_at=d.created_at,
                chunk_count=d.chunk_count,
            )
            for d in ordered[offset : offset + limit]
        ]

    async def count(self) -> int:
        return len(self.documents)


class RaceLosingRepository(FakeDocumentRepository):
    """Simulates LOSING the dedup TOCTOU race against a concurrent upload.

    Sequence of a real lost race: our dedup check misses (the winner has not
    committed yet), our insert trips the unique index, and by the time we
    recover, the winner's row is visible. This fake replays that exact
    sequence against the use case, no threads required.
    """

    def __init__(self, winner: DocumentSummary | None) -> None:
        super().__init__()
        self._winner = winner
        self._first_check_done = False

    async def get_summary_by_content_hash(self, content_hash: str) -> DocumentSummary | None:
        if not self._first_check_done:
            self._first_check_done = True
            return None  # the winner had not committed when we looked
        return self._winner  # None if the winner rolled back instead

    async def save(self, document: Document) -> None:
        # The unique index does its job: our twin insert is rejected.
        raise DuplicateDocumentError(document.content_hash or "")


def fake_repository_scope(repository: FakeDocumentRepository) -> OpenRepository:
    """The `OpenRepository` port over a fake: every scope yields the SAME
    in-memory repository (the fake's dict is the 'database'). The scope is
    a real context manager, so the use case's enter/exit discipline is
    exercised, not stubbed away."""

    @asynccontextmanager
    async def _scope() -> AsyncIterator[FakeDocumentRepository]:
        yield repository

    return _scope


class RepositoryScopeRecorder:
    """An `OpenRepository` that COUNTS: how many scopes are open right now
    and how many were opened in total. Exists so a test can prove the slow
    embedding step runs with ZERO database scopes open (ADR-0005)."""

    def __init__(self, repository: FakeDocumentRepository) -> None:
        self.repository = repository
        self.currently_open = 0
        self.total_opened = 0

    @asynccontextmanager
    async def __call__(self) -> AsyncIterator[FakeDocumentRepository]:
        self.currently_open += 1
        self.total_opened += 1
        try:
            yield self.repository
        finally:
            self.currently_open -= 1


class FakeEmbeddingProvider:
    """Deterministic embeddings: constant vectors of the configured dimension.

    Constant vectors make every cosine distance equal — perfect for tests that
    care about plumbing, not about semantic ranking.
    """

    def __init__(self, dimension: int = 8, fill: float = 0.5) -> None:
        self.dimension = dimension
        self.fill = fill
        self.received_texts: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[EmbeddingVector]:
        self.received_texts.append(list(texts))
        return [EmbeddingVector(tuple([self.fill] * self.dimension)) for _ in texts]


class FakeTextExtractor:
    """Implements TextExtractor for a fixed set of suffixes."""

    def __init__(
        self, suffixes: tuple[str, ...] = (".md", ".txt"), text: str | None = None
    ) -> None:
        self.suffixes = suffixes
        self.text = text

    def supports(self, file_name: str) -> bool:
        return file_name.lower().endswith(self.suffixes)

    def extract(self, file_name: str, data: bytes) -> str:
        return self.text if self.text is not None else data.decode("utf-8")


class FakeKnowledgeSearch:
    """Implements KnowledgeSearch with canned results (records the calls)."""

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.calls: list[tuple[str, int]] = []

    async def search(self, question: str, limit: int) -> list[RetrievedChunk]:
        self.calls.append((question, limit))
        return self.chunks[:limit]


class FailingKnowledgeSearch:
    """Implements KnowledgeSearch by raising — for outage-path tests."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    async def search(self, question: str, limit: int) -> list[RetrievedChunk]:
        raise self.error


class FakeAnswerGenerator:
    """Implements AnswerGenerator with a fixed answer (records the calls)."""

    def __init__(self, answer: Answer) -> None:
        self.answer = answer
        self.calls: list[tuple[str, list[RetrievedChunk]]] = []

    async def generate(self, question: str, chunks: list[RetrievedChunk]) -> Answer:
        self.calls.append((question, chunks))
        return self.answer


def make_retrieved_chunk(
    chunk_id: str = "chunk-1",
    document_title: str = "Return Policy",
    score: float = 0.05,
    content: str = "You may return any product within 30 days of purchase.",
) -> RetrievedChunk:
    """Small factory so tests state their data needs explicitly."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        document_title=document_title,
        content=content,
        score=score,
    )
