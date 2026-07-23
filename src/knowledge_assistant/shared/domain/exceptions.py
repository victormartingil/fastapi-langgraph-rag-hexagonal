"""Shared domain exceptions.

Every domain error in the system inherits from ``DomainError``. This gives the
HTTP layer a single, stable seam: the error handler maps subclasses of
``DomainError`` to HTTP responses without knowing anything else about the
domain. Domain code, in turn, never imports HTTP concepts.
"""


class DomainError(Exception):
    """Base class for all business-rule violations in the system."""


class EmbeddingProviderUnavailableError(DomainError):
    """The embedding service cannot be reached right now (transient outage).

    Raised by the embedding provider ADAPTERS at the port boundary, after
    their retries are exhausted — part of the `EmbeddingProvider` port's
    contract, so it lives in the shared kernel next to the port itself
    (shared/application/ports.py). Only TRANSIENT failures are translated
    (timeouts, connection errors, 5xx); a 401 or 404 is a configuration
    error and propagates raw. Mapped to HTTP 503 — on BOTH the ingest path
    and, re-wrapped as RetrievalUnavailableError, the chat path.
    """
