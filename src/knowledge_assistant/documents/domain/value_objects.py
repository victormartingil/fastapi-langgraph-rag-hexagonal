"""Value objects of the documents context.

A value object is defined ONLY by its attributes: two ``DocumentId`` instances
with the same UUID are the same id. They are immutable (frozen dataclasses)
and validate themselves at construction time, so invalid states are
unrepresentable — you simply cannot build a ``ChunkText("")``.

Why not plain strings/uuids? Because ``ChunkText`` and ``DocumentId`` are
*different concepts* that happen to share a representation. Wrapping them
makes type signatures self-documenting and lets mypy catch mistakes such as
passing a document title where a chunk text is expected.
"""

import uuid
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DocumentId:
    """Strongly typed identity of a Document."""

    value: uuid.UUID = field(default_factory=uuid.uuid4)

    @classmethod
    def from_string(cls, raw: str) -> "DocumentId":
        """Parse a DocumentId from its canonical string form."""
        return cls(uuid.UUID(raw))

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class ChunkText:
    """A piece of text that fits in one embedding.

    The rule lives here, not scattered across the codebase: a chunk is never
    empty or whitespace-only.
    """

    value: str

    def __post_init__(self) -> None:
        if not self.value.strip():
            msg = "ChunkText cannot be empty or whitespace-only"
            raise ValueError(msg)

    def __len__(self) -> int:
        return len(self.value)

    def __str__(self) -> str:
        return self.value


# NOTE: `EmbeddingVector` is NOT defined here. Both bounded contexts need it
# (write side embeds chunks, read side embeds questions), so it lives in the
# shared kernel: knowledge_assistant.shared.domain.value_objects.
