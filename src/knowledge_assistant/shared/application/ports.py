"""Ports of the shared kernel: capabilities both contexts consume.

Embedding is the canonical example in this system:

- the write side (documents) embeds chunks during ingestion;
- the read side (chat) embeds the question before retrieval.

If the port lived in either context, the other would have to import it — a
forbidden cross-context dependency. So the contract lives here, in the shared
kernel, and BOTH contexts depend on it (dependencies still point inward).
"""

from typing import Protocol

from knowledge_assistant.shared.domain.value_objects import EmbeddingVector


class EmbeddingProvider(Protocol):
    """Port toward whatever model turns text into dense vectors."""

    async def embed(self, texts: list[str]) -> list[EmbeddingVector]:
        """Embed a batch of texts. Output order must match input order."""
        ...
