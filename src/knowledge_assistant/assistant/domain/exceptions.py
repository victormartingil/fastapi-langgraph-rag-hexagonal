"""Domain errors of the chat bounded context (read side)."""

from knowledge_assistant.shared_kernel.exceptions import DomainError


class EmptyQuestionError(DomainError):
    """Raised when the user submits an empty or whitespace-only question."""


class InvalidQuestionError(DomainError):
    """Raised when a question or retrieval limit violates domain bounds."""


class InvalidModelOutputError(DomainError):
    """The LLM exhausted structured-output retries without valid grounding."""


class RetrievalUnavailableError(DomainError):
    """The knowledge base cannot be searched right now (transient outage).

    Raised by infrastructure adapters at the port boundary — the same pattern
    as `DuplicateDocumentError` on the write side — when the embedding
    provider or the database fails transiently (timeouts, connection errors,
    5xx, 429). Permanent failures (a 401, a SQL bug) are NOT translated: they
    are server errors and must stay visible as such. Mapped to HTTP 503 so
    the client learns "try again later".
    """


class GenerationUnavailableError(DomainError):
    """The answer-generation service cannot respond right now (transient outage).

    Raised by the LLM adapter when its retries are EXHAUSTED on transient
    errors — symmetric with RetrievalUnavailableError on the retrieval side.
    Permanent failures (a dead API key, a validation bug) are NOT translated:
    a 200-with-fallback-message would report misconfiguration as "temporary"
    and be indistinguishable from a real answer, so those stay loud (500).
    Mapped to HTTP 503.
    """
