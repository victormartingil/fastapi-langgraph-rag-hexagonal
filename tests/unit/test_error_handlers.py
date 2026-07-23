"""Unit tests for the domain-error -> HTTP-status mapping.

The mapping table is the single place where status codes are chosen; these
tests pin both the explicit entries and, critically, the DEFAULT: an
unmapped DomainError is a server-side omission (someone forgot to add the
entry), never a client fault — so the default is 500, not 400.
"""

from knowledge_assistant.assistant.domain.exceptions import (
    EmptyQuestionError,
    GenerationUnavailableError,
    RetrievalUnavailableError,
)
from knowledge_assistant.knowledge_base.domain.exceptions import (
    ConcurrentIngestionError,
    DocumentNotFoundError,
    EmbeddingProviderUnavailableError,
)
from knowledge_assistant.platform.http.error_handlers import _status_for
from knowledge_assistant.shared_kernel.exceptions import DomainError


class ForgottenDomainError(DomainError):
    """A domain error nobody mapped — stands in for the next one added."""


class TestStatusMapping:
    def test_explicit_mappings(self) -> None:
        assert _status_for(DocumentNotFoundError("x")) == 404
        assert _status_for(ConcurrentIngestionError("x")) == 409
        assert _status_for(EmptyQuestionError("x")) == 422
        assert _status_for(RetrievalUnavailableError("x")) == 503
        assert _status_for(GenerationUnavailableError("x")) == 503
        assert _status_for(EmbeddingProviderUnavailableError("x")) == 503

    def test_unmapped_domain_error_defaults_to_500(self) -> None:
        # A missing mapping is OUR bug, not the client's: 500 keeps it loud
        # instead of blaming the caller with a 4xx.
        assert _status_for(ForgottenDomainError("x")) == 500
