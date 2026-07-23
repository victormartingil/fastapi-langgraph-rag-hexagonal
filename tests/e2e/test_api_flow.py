"""E2E tests: the complete user journey over HTTP.

ingest sample doc → list → get → ask → cited answer
                                      → unrelated question → honest refusal
"""

import pytest
from httpx import AsyncClient

from tests.e2e.conftest import SAMPLE_DOC

pytestmark = pytest.mark.e2e


async def _ingest_sample(client: AsyncClient) -> dict[str, object]:
    response = await client.post(
        "/api/v1/documents",
        files={"file": (SAMPLE_DOC.name, SAMPLE_DOC.read_bytes(), "text/markdown")},
        data={"title": "Return Policy"},
    )
    assert response.status_code == 201, response.text
    body: dict[str, object] = response.json()
    return body


class TestHealthEndpoint:
    async def test_health_is_ok(self, client: AsyncClient) -> None:
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    async def test_correlation_id_is_echoed_back(self, client: AsyncClient) -> None:
        response = await client.get("/health", headers={"X-Correlation-ID": "test-123"})
        assert response.headers["X-Correlation-ID"] == "test-123"


class TestProbeSplit:
    """Liveness and readiness are DIFFERENT questions (F7): /livez answers
    "is the process up?", /health(z) answers "can it serve traffic?"."""

    async def test_livez_is_ok(self, client: AsyncClient) -> None:
        response = await client.get("/livez")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    async def test_healthz_is_the_readiness_alias(self, client: AsyncClient) -> None:
        response = await client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    async def test_livez_stays_up_when_the_database_is_down(
        self, db_down_client: AsyncClient
    ) -> None:
        """The whole point of the split: a dead dependency must NOT restart
        the process. Liveness has no dependencies, so it answers 200 even
        with the database refusing connections."""
        response = await db_down_client.get("/livez")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    async def test_readiness_reports_unhealthy_when_the_database_is_down(
        self, db_down_client: AsyncClient
    ) -> None:
        for path in ("/health", "/healthz"):
            response = await db_down_client.get(path)
            assert response.status_code == 503
            assert response.json() == {"status": "unhealthy"}


class TestDatabaseOutage:
    """The DATABASE is down (dead port). Symmetric with the provider-outage
    doctrine: a dead knowledge base is a 503 outage signal with the unified
    envelope — never a bare 500 with a SQLAlchemy traceback."""

    async def test_ingest_with_db_down_is_a_truthful_503(self, db_down_client: AsyncClient) -> None:
        response = await db_down_client.post(
            "/api/v1/documents",
            files={"file": (SAMPLE_DOC.name, SAMPLE_DOC.read_bytes(), "text/markdown")},
        )

        assert response.status_code == 503, response.text
        body = response.json()
        assert body["error"] == "KnowledgeBaseUnavailableError"
        assert "temporarily unavailable" in body["detail"]
        assert body["correlation_id"] == response.headers["X-Correlation-ID"]

    async def test_list_with_db_down_is_a_truthful_503(self, db_down_client: AsyncClient) -> None:
        response = await db_down_client.get("/api/v1/documents")

        assert response.status_code == 503, response.text
        assert response.json()["error"] == "KnowledgeBaseUnavailableError"


class TestUnhandledExceptions:
    async def test_unhandled_exception_gets_the_unified_500_envelope(
        self, broken_client: AsyncClient
    ) -> None:
        """A pipeline stage raises a plain RuntimeError (a bug, not a domain
        signal). The last line of defense must answer with the SAME envelope
        every other error uses — detail + error + correlation_id — plus the
        X-Correlation-ID header, and leak nothing from the traceback."""
        response = await broken_client.post(
            "/api/v1/chat",
            json={"question": "Can I return a product after two months?"},
        )

        assert response.status_code == 500, response.text
        body = response.json()
        assert body["error"] == "Internal Server Error"
        assert body["detail"] == "Internal Server Error"
        assert "boom" not in response.text  # no internals leak
        assert body["correlation_id"] == response.headers["X-Correlation-ID"]


class TestDocumentEndpoints:
    async def test_full_document_lifecycle(self, client: AsyncClient) -> None:
        created = await _ingest_sample(client)
        assert created["title"] == "Return Policy"
        chunk_count = created["chunk_count"]
        assert isinstance(chunk_count, int)
        assert chunk_count >= 1

        listing = await client.get("/api/v1/documents")
        assert listing.status_code == 200
        assert listing.json()["total"] == 1
        assert listing.json()["documents"][0]["id"] == created["id"]

        fetched = await client.get(f"/api/v1/documents/{created['id']}")
        assert fetched.status_code == 200
        assert fetched.json()["file_name"] == "return-policy.md"

    async def test_get_unknown_document_is_a_404_domain_error(self, client: AsyncClient) -> None:
        import uuid

        response = await client.get(f"/api/v1/documents/{uuid.uuid4()}")
        assert response.status_code == 404
        body = response.json()
        assert body["error"] == "DocumentNotFoundError"
        # The unified error envelope: detail + error + correlation_id.
        assert "detail" in body
        assert body["correlation_id"] == response.headers["X-Correlation-ID"]

    async def test_malformed_document_id_is_a_422(self, client: AsyncClient) -> None:
        # The path parameter is typed uuid.UUID: FastAPI rejects malformed
        # ids at the boundary, before any domain code runs.
        response = await client.get("/api/v1/documents/not-a-uuid")
        assert response.status_code == 422

    async def test_uploading_an_unsupported_type_is_a_415(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/documents",
            files={"file": ("photo.png", b"\x89PNG", "image/png")},
        )
        assert response.status_code == 415
        assert response.json()["error"] == "UnsupportedFileTypeError"

    async def test_uploading_a_corrupt_pdf_is_a_422(self, client: AsyncClient) -> None:
        """A supported extension with unreadable content: the pypdf failure is
        quarantined into a domain error, never an opaque 500."""
        response = await client.post(
            "/api/v1/documents",
            files={"file": ("corrupt.pdf", b"this is not a real PDF", "application/pdf")},
        )
        assert response.status_code == 422
        assert response.json()["error"] == "TextExtractionError"

    async def test_upload_over_the_configured_limit_is_a_413(
        self, limited_client: AsyncClient
    ) -> None:
        # The fixture caps uploads at ~104 bytes; the sample is 893 bytes.
        response = await limited_client.post(
            "/api/v1/documents",
            files={"file": (SAMPLE_DOC.name, SAMPLE_DOC.read_bytes(), "text/markdown")},
        )
        assert response.status_code == 413

    async def test_embedding_provider_outage_during_ingest_is_a_truthful_503(
        self, provider_down_client: AsyncClient
    ) -> None:
        """The embedding provider is DOWN during ingestion. Symmetric with
        the chat path, the transient outage must surface as 503 'temporarily
        unavailable' — never a bare 500 with a raw httpx traceback."""
        response = await provider_down_client.post(
            "/api/v1/documents",
            files={"file": (SAMPLE_DOC.name, SAMPLE_DOC.read_bytes(), "text/markdown")},
        )

        assert response.status_code == 503, response.text
        body = response.json()
        assert body["error"] == "EmbeddingProviderUnavailableError"
        assert "temporarily unavailable" in body["detail"]

    async def test_reuploading_the_same_content_is_idempotent(self, client: AsyncClient) -> None:
        created = await _ingest_sample(client)

        # Same bytes, different file name: the content hash still matches.
        duplicate = await client.post(
            "/api/v1/documents",
            files={"file": ("copy.md", SAMPLE_DOC.read_bytes(), "text/markdown")},
            data={"title": "Return Policy (copy)"},
        )

        assert duplicate.status_code == 200  # 200, not 201: nothing new was created
        assert duplicate.json()["id"] == created["id"]

        listing = await client.get("/api/v1/documents")
        assert listing.json()["total"] == 1  # no twin row

    async def test_listing_is_paginated(self, client: AsyncClient) -> None:
        await _ingest_sample(client)
        await client.post(
            "/api/v1/documents",
            files={"file": ("notes.md", b"Unrelated release notes.", "text/markdown")},
        )

        first_page = await client.get("/api/v1/documents", params={"limit": 1, "offset": 0})
        assert first_page.status_code == 200
        body = first_page.json()
        assert body["total"] == 2
        assert body["limit"] == 1
        assert body["offset"] == 0
        assert len(body["documents"]) == 1

        second_page = await client.get("/api/v1/documents", params={"limit": 1, "offset": 1})
        assert second_page.json()["offset"] == 1
        assert second_page.json()["documents"][0]["id"] != body["documents"][0]["id"]


class TestApiKeyAuth:
    async def test_request_without_the_key_is_a_401(self, authed_client: AsyncClient) -> None:
        response = await authed_client.get("/api/v1/documents")
        assert response.status_code == 401
        # RFC 9110 §11.6.1: a 401 MUST carry a WWW-Authenticate challenge.
        assert response.headers["WWW-Authenticate"] == 'ApiKey realm="api"'
        # Same unified envelope as every other error.
        body = response.json()
        assert body["error"] == "Unauthorized"
        assert "detail" in body
        assert "correlation_id" in body

    async def test_validation_error_uses_the_same_envelope(self, client: AsyncClient) -> None:
        """FastAPI's request-validation 422 joins the unified envelope, and
        keeps the field-level details under "errors"."""
        response = await client.post("/api/v1/chat", json={"question": 42})

        assert response.status_code == 422
        body = response.json()
        assert body["error"] == "RequestValidationError"
        assert body["detail"] == "Request validation failed"
        assert body["correlation_id"] == response.headers["X-Correlation-ID"]
        assert any(error["loc"] == ["body", "question"] for error in body["errors"])

    @pytest.mark.parametrize("question", ["   ", "x" * 4_001])
    async def test_question_limits_return_422(self, client: AsyncClient, question: str) -> None:
        response = await client.post("/api/v1/chat", json={"question": question})

        assert response.status_code == 422
        assert response.json()["error"] == "RequestValidationError"

    async def test_request_with_the_key_passes(self, authed_client: AsyncClient) -> None:
        response = await authed_client.get(
            "/api/v1/documents", headers={"X-API-Key": "test-secret"}
        )
        assert response.status_code == 200

    async def test_health_stays_open_when_auth_is_on(self, authed_client: AsyncClient) -> None:
        # Probes must not need credentials.
        response = await authed_client.get("/health")
        assert response.status_code == 200

    async def test_api_docs_are_closed_when_auth_is_on(self, authed_client: AsyncClient) -> None:
        """The interactive docs enumerate every endpoint and schema: when the
        API needs a key, the docs must not stay open to the keyless."""
        assert (await authed_client.get("/docs")).status_code == 404
        assert (await authed_client.get("/redoc")).status_code == 404
        assert (await authed_client.get("/openapi.json")).status_code == 404

    async def test_api_docs_are_open_by_default(self, client: AsyncClient) -> None:
        assert (await client.get("/openapi.json")).status_code == 200


class TestRealContainerWiring:
    """No dependency overrides: the REAL container providers assemble the use
    cases, with only the two AI ports faked on the container. This is where
    settings wiring (fetch_limit, rrf_k, min_relevance_score, default top_k)
    is proven end to end."""

    async def test_server_default_top_k_comes_from_settings(
        self, real_wiring_client: AsyncClient
    ) -> None:
        # The fixture sets retrieval_top_k=1: omitting top_k must yield one
        # source even though TWO documents match the question.
        await _ingest_sample(real_wiring_client)
        await real_wiring_client.post(
            "/api/v1/documents",
            files={
                "file": (
                    "returns-faq.md",
                    b"You can return a product after two months with a receipt.",
                    "text/markdown",
                )
            },
        )

        defaulted = await real_wiring_client.post(
            "/api/v1/chat",
            json={"question": "Can I return a product after two months?"},
        )
        assert defaulted.status_code == 200, defaulted.text
        assert len(defaulted.json()["sources"]) == 1

        # An explicit top_k always wins over the server default.
        explicit = await real_wiring_client.post(
            "/api/v1/chat",
            json={"question": "Can I return a product after two months?", "top_k": 5},
        )
        assert explicit.status_code == 200, explicit.text
        assert len(explicit.json()["sources"]) >= 2


class TestChatEndpoint:
    async def test_question_about_the_policy_gets_a_cited_answer(self, client: AsyncClient) -> None:
        await _ingest_sample(client)

        response = await client.post(
            "/api/v1/chat",
            json={"question": "Can I return a product after two months?"},
        )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["answer"].startswith("Grounded answer to:")
        # The answer cites the ingested document — provenance end-to-end.
        assert len(body["sources"]) >= 1
        assert body["sources"][0]["document_title"] == "Return Policy"
        assert "return" in body["sources"][0]["excerpt"].lower()

    async def test_unrelated_question_gets_an_honest_refusal(self, client: AsyncClient) -> None:
        await _ingest_sample(client)

        response = await client.post(
            "/api/v1/chat",
            json={"question": "How do I bake sourdough bread?"},
        )

        assert response.status_code == 200
        body = response.json()
        assert "could not find any relevant information" in body["answer"]
        assert body["sources"] == []

    async def test_embedding_provider_outage_is_a_truthful_503(
        self, provider_down_client: AsyncClient
    ) -> None:
        """The embedding provider is DOWN at query time. Symmetric with the
        generation side, retrieval must degrade honestly:
        503 'temporarily unavailable', never an opaque 500."""
        response = await provider_down_client.post(
            "/api/v1/chat",
            json={"question": "Can I return a product after two months?"},
        )

        assert response.status_code == 503, response.text
        body = response.json()
        assert body["error"] == "RetrievalUnavailableError"
        assert "temporarily unavailable" in body["detail"]

    async def test_llm_outage_is_a_truthful_503_not_a_degraded_200(
        self, llm_down_client: AsyncClient
    ) -> None:
        """The LLM is DOWN at answer time (after the adapter's retries).
        The error doctrine on the generation side: 503 'temporarily
        unavailable' — never a 200 fallback message indistinguishable from
        a real answer."""
        await _ingest_sample(llm_down_client)  # embeddings work; only the LLM is down

        response = await llm_down_client.post(
            "/api/v1/chat",
            json={"question": "Can I return a product after two months?"},
        )

        assert response.status_code == 503, response.text
        body = response.json()
        assert body["error"] == "GenerationUnavailableError"
        assert "temporarily unavailable" in body["detail"]
