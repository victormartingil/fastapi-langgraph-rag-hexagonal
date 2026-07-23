"""Unit tests for the persistence mappers: domain <-> ORM conversion.

Mappers are pure functions over in-memory objects, so they belong in the unit
suite even though they live in the infrastructure layer — no database needed.
"""

import pytest

from knowledge_assistant.knowledge_base.domain.models import Chunk, Document
from knowledge_assistant.knowledge_base.domain.value_objects import ChunkText, DocumentId
from knowledge_assistant.knowledge_base.infrastructure.persistence.mappers import (
    chunk_to_model,
    document_to_domain,
    document_to_model,
)
from knowledge_assistant.shared.domain.value_objects import EmbeddingVector


def make_chunk(with_embedding: bool = True) -> Chunk:
    return Chunk(
        id=DocumentId(),
        text=ChunkText("Refunds within 30 days."),
        position=0,
        embedding=EmbeddingVector((0.1, 0.2, 0.3)) if with_embedding else None,
    )


class TestChunkMapping:
    def test_domain_to_model(self) -> None:
        chunk = make_chunk()
        document_id = DocumentId()

        model = chunk_to_model(chunk, document_id)

        assert model.id == chunk.id.value
        assert model.document_id == document_id.value
        assert model.content == "Refunds within 30 days."
        assert model.embedding == [0.1, 0.2, 0.3]

    def test_chunk_without_embedding_cannot_be_persisted(self) -> None:
        with pytest.raises(ValueError, match="embedding"):
            chunk_to_model(make_chunk(with_embedding=False), DocumentId())


class TestDocumentMapping:
    def test_round_trip_preserves_everything_that_matters(self) -> None:
        document = Document(
            id=DocumentId(),
            title="Return Policy",
            file_name="return-policy.md",
            raw_text="Refunds within 30 days.",
            chunks=(make_chunk(),),
        )

        restored = document_to_domain(document_to_model(document))

        assert restored.id == document.id
        assert restored.title == document.title
        assert restored.file_name == document.file_name
        assert restored.raw_text == document.raw_text
        assert len(restored.chunks) == 1
        assert restored.chunks[0].text == document.chunks[0].text
        assert restored.chunks[0].embedding == document.chunks[0].embedding
