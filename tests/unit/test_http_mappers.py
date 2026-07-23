"""Unit tests for the HTTP mapper of the documents context."""

from knowledge_assistant.documents.domain.models import Document
from knowledge_assistant.documents.domain.value_objects import DocumentId
from knowledge_assistant.documents.infrastructure.http.mappers import document_to_response


def test_document_to_response_maps_the_public_contract() -> None:
    document = Document(
        id=DocumentId(),
        title="Return Policy",
        file_name="return-policy.md",
        raw_text="...",
    )

    response = document_to_response(document)

    assert response.id == str(document.id)
    assert response.title == "Return Policy"
    assert response.file_name == "return-policy.md"
    assert response.chunk_count == 0
