"""Unit tests for value objects: invalid states must be unrepresentable."""

import uuid

import pytest

from knowledge_assistant.knowledge_base.domain.value_objects import (
    ChunkText,
    DocumentId,
    EmbeddingVector,
)


class TestChunkText:
    @pytest.mark.parametrize("raw", ["", "   ", "\n\t"])
    def test_rejects_empty_or_whitespace(self, raw: str) -> None:
        with pytest.raises(ValueError, match="empty"):
            ChunkText(raw)

    def test_valid_text_round_trips(self) -> None:
        assert str(ChunkText("hello")) == "hello"
        assert len(ChunkText("hello")) == 5


class TestEmbeddingVector:
    def test_rejects_empty_vector(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            EmbeddingVector(())

    def test_dimension_property(self) -> None:
        assert EmbeddingVector((0.1, 0.2, 0.3)).dimension == 3


class TestDocumentId:
    def test_generates_unique_ids_by_default(self) -> None:
        assert DocumentId() != DocumentId()

    def test_from_string_round_trip(self) -> None:
        raw = str(uuid.uuid4())
        assert str(DocumentId.from_string(raw)) == raw

    def test_equality_by_value(self) -> None:
        raw = str(uuid.uuid4())
        assert DocumentId.from_string(raw) == DocumentId.from_string(raw)
