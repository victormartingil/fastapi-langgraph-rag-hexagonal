"""Value objects of the knowledge-base context.

A value object is defined ONLY by its attributes: two ``DocumentId`` instances
with the same UUID are the same id. They are immutable (frozen dataclasses)
and validate themselves at construction time, so invalid states are
unrepresentable — you simply cannot build a ``ChunkText("")``.

Why not plain strings/uuids? Because ``DocumentId``, ``ChunkId`` and
``ChunkText`` are *different concepts* that happen to share simple
representations. Wrapping them makes type signatures self-documenting and
lets mypy catch mistakes such as passing a document id where a chunk id is
expected.
"""

import uuid
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class DocumentId:
    """Strongly typed identity of a Document."""

    value: uuid.UUID = field(default_factory=uuid.uuid4)

    @classmethod
    def from_string(cls, raw: str) -> "DocumentId":
        """Parse a DocumentId from its canonical string form."""
        return cls(uuid.UUID(raw))

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True, slots=True)
class ChunkId:
    """Strongly typed identity of a Chunk."""

    value: uuid.UUID = field(default_factory=uuid.uuid4)

    @classmethod
    def from_string(cls, raw: str) -> "ChunkId":
        """Parse a ChunkId from its canonical string form."""
        return cls(uuid.UUID(raw))

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
class EmbeddingVector:
    """A dense vector produced by the knowledge-base embedding provider.

    Only non-emptiness is intrinsic to the value. The configured model
    dimension is enforced by `IngestDocument` and by the fixed-size pgvector
    column, where that deployment-specific invariant belongs.
    """

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.values:
            msg = "EmbeddingVector cannot be empty"
            raise ValueError(msg)

    @property
    def dimension(self) -> int:
        return len(self.values)
