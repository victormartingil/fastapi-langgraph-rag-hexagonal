"""Value objects of the shared kernel.

An embedding vector is needed by BOTH bounded contexts: the write side embeds
chunks, the read side embeds the question. Rather than letting one context
depend on the other, the concept lives in the shared kernel — the DDD pattern
for "a few concepts genuinely belong to everyone".
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingVector:
    """A dense vector produced by an embedding model.

    Only NON-EMPTINESS is validated here: the value object does not know
    which model produced it, so it cannot know the expected dimension.
    Dimensionality is enforced one layer out — `IngestDocument` checks the
    provider's first real vector against the configured dimension — and at
    the bottom by the fixed-size pgvector column itself.
    """

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.values) == 0:
            msg = "EmbeddingVector cannot be empty"
            raise ValueError(msg)

    @property
    def dimension(self) -> int:
        return len(self.values)
