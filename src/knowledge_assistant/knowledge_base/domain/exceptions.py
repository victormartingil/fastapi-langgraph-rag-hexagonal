"""Domain errors of the documents bounded context (write side)."""

from knowledge_assistant.shared_kernel.exceptions import DomainError


class EmbeddingProviderUnavailableError(DomainError):
    """The embedding provider exhausted retries for a transient outage."""


class InvalidDocumentMetadataError(DomainError):
    """A title or filename violates the public domain limits."""

    def __init__(self, field: str, reason: str) -> None:
        super().__init__(f"Invalid {field}: {reason}")
        self.field = field


class UnsupportedFileTypeError(DomainError):
    """Raised when an uploaded file has an extension we cannot extract text from."""

    def __init__(self, file_name: str) -> None:
        super().__init__(f"Unsupported file type: {file_name!r}. Allowed: .md, .txt, .pdf")
        self.file_name = file_name


class TextExtractionError(DomainError):
    """Raised when extraction fails on a SUPPORTED file type.

    Distinct from UnsupportedFileTypeError: the extension was fine, but the
    content is unreadable — a corrupt or encrypted PDF, for example. Vendor
    parsers (pypdf) raise a zoo of exception types; the adapters quarantine
    that zoo and raise this single domain signal instead, so a broken upload
    surfaces as a 422 the caller can act on, never as an opaque 500.
    """

    def __init__(self, file_name: str) -> None:
        super().__init__(
            f"Could not extract text from {file_name!r}: "
            "the file appears to be corrupt, encrypted, or otherwise unreadable"
        )
        self.file_name = file_name


class DocumentNotFoundError(DomainError):
    """Raised when a document id does not exist in the knowledge base."""

    def __init__(self, document_id: object) -> None:
        super().__init__(f"Document not found: {document_id}")
        self.document_id = document_id


class EmptyDocumentError(DomainError):
    """Raised when extraction yields no usable text."""


class DuplicateDocumentError(DomainError):
    """Raised by the repository when the content_hash unique index rejects an
    insert — the loser's signal in the dedup race.

    Two concurrent identical uploads can both pass the dedup check before
    either commits (TOCTOU); the unique index is the last line of defense.
    This signal is INTERNAL to the adapter → use-case boundary: the use case
    converts it into an idempotent "already exists" result — or, in the rare
    case the winner also rolled back, into a ConcurrentIngestionError.
    """

    def __init__(self, content_hash: str) -> None:
        super().__init__(f"A document with content hash {content_hash[:12]}... already exists")
        self.content_hash = content_hash


class ConcurrentIngestionError(DomainError):
    """Raised when a dedup race ends with NO winner: our insert was rejected
    by the unique index, yet no committed row exists to return idempotently
    (the winning transaction rolled back). Nothing truthful can be returned,
    so the honest answer is 409 Conflict — the client can retry the upload.
    """

    def __init__(self, content_hash: str) -> None:
        super().__init__(
            "A concurrent upload of the same content raced this one and was "
            "rolled back; please retry the upload "
            f"(content hash {content_hash[:12]}...)"
        )
        self.content_hash = content_hash


class EmbeddingDimensionMismatchError(RuntimeError):
    """Raised when the embedding provider returns vectors of the wrong size.

    Deliberately NOT a DomainError: this is a server-side configuration bug
    (model and KA_EMBEDDING_DIMENSION disagree), not a client problem — it
    should surface as a loud 500 the operator fixes, not a 4xx the caller
    retries. The startup guard in bootstrap.py checks CONFIGURATION against
    the schema; this check validates REALITY against configuration, on the
    first embedded batch of each ingestion (ADR-0001).
    """

    def __init__(self, model_name: str, expected: int, actual: int) -> None:
        super().__init__(
            f"Embedding model {model_name!r} returned {actual}-dimensional vectors, "
            f"but {expected} were configured (KA_EMBEDDING_DIMENSION). Fix the model "
            "or the dimension setting — persisting these vectors would corrupt retrieval."
        )
        self.model_name = model_name
        self.expected = expected
        self.actual = actual
