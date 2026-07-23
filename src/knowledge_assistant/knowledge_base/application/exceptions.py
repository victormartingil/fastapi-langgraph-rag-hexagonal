"""Failures exposed by the knowledge-base application boundary."""

from knowledge_assistant.shared_kernel.exceptions import DomainError


class KnowledgeBaseUnavailableError(DomainError):
    """The knowledge-base application cannot currently serve a request.

    This is an application-boundary signal rather than a domain invariant:
    adapters translate connection-level failures into it, and callers can
    handle the outage without importing knowledge-base domain internals.
    """

    def __init__(self, reason: str = "database unreachable") -> None:
        super().__init__(
            f"The knowledge base is temporarily unavailable ({reason}). Please try again shortly."
        )


class EmbeddingBatchCardinalityError(RuntimeError):
    """An embedding provider violated the one-vector-per-input port contract."""

    def __init__(self, expected: int, actual: int) -> None:
        super().__init__(
            f"Embedding provider returned {actual} vectors for {expected} input texts; "
            "refusing to associate embeddings with the wrong chunks"
        )
        self.expected = expected
        self.actual = actual
