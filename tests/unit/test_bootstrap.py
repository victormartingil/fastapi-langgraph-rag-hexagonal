"""Unit tests for composition-root guards and wiring.

`build_container` performs fail-fast validation that must happen BEFORE any
connection is opened — these tests need neither Docker nor network. They also
prove that configuration knobs actually reach the objects they claim to
control (a knob nobody reads is a lie in config.py).
"""

import httpx
import pytest
from pydantic import ValidationError

from knowledge_assistant.assistant.adapters.outbound.llm.pydantic_ai import (
    PydanticAiAnswerGenerator,
)
from knowledge_assistant.bootstrap import build_container
from knowledge_assistant.config import Settings


class TestFtsLanguageValidation:
    def test_language_is_normalized_to_lowercase(self) -> None:
        # PostgreSQL folds regconfig names; the parity guard compares exact
        # strings — so config canonicalizes instead of drifting.
        assert Settings(fts_language="English").fts_language == "english"

    def test_invalid_language_is_rejected_at_boot(self) -> None:
        with pytest.raises(ValidationError, match="fts_language"):
            Settings(fts_language="engl1sh!")

    def test_simple_and_underscored_names_are_accepted(self) -> None:
        assert Settings(fts_language="simple").fts_language == "simple"


class TestEmbeddingDimensionGuard:
    def test_mismatched_dimension_refuses_to_start(self) -> None:
        """A 1536-dim provider against the vector(768) schema is a startup
        error, not a runtime surprise (ADR-0001)."""
        with pytest.raises(ValueError, match=r"vector\(768\)"):
            build_container(Settings(embedding_dimension=1536))

    def test_openai_embeddings_resolve_to_1536_and_hit_the_guard(self) -> None:
        """The provider flag drives defaults symmetric with the LLM: OpenAI
        embeddings default to text-embedding-3-small (1536 dims), which the
        shipped vector(768) schema cannot hold — so the guard names the fix
        (regenerate the migration) instead of booting a broken system."""
        with pytest.raises(ValueError, match=r"vector\(768\)"):
            build_container(Settings(embedding_provider="openai", embedding_api_key="sk-test"))

    def test_openai_embeddings_require_an_api_key(self) -> None:
        with pytest.raises(ValueError, match="KA_EMBEDDING_API_KEY"):
            build_container(Settings(embedding_provider="openai"))

    async def test_default_settings_resolve_the_ollama_defaults(self) -> None:
        container = build_container(Settings())
        assert container.embedding_config.model == "nomic-embed-text"
        assert container.embedding_config.dimension == 768
        assert container.embedding_config.base_url == "http://localhost:11434"
        await container.aclose()


class TestConfigKnobsAreWired:
    async def test_llm_timeout_reaches_the_answer_generator(self) -> None:
        """KA_LLM_TIMEOUT_SECONDS must not be a dead knob: it becomes the HTTP
        timeout of the LLM adapter's client."""
        container = build_container(Settings(llm_timeout_seconds=7.0))
        generator = container.answer_generator
        assert isinstance(generator, PydanticAiAnswerGenerator)
        assert generator.timeout_seconds == 7.0
        await container.aclose()

    async def test_aclose_releases_both_http_clients(self) -> None:
        """Shutdown hygiene: the LLM client is container-owned (injected into
        the generator), so aclose must release it too — an adapter-owned
        client would leak open connections at process exit."""
        container = build_container(Settings())

        await container.aclose()

        assert container._embedding_http_client.is_closed
        assert container._llm_http_client.is_closed

    async def test_one_failing_close_does_not_skip_the_rest(self) -> None:
        """Each resource closes independently: a broken client must not
        prevent the LLM client and engine pool from closing. Failures are
        re-raised as a group so a broken shutdown stays visible."""

        class UnclosableClient(httpx.AsyncClient):
            async def aclose(self) -> None:
                raise RuntimeError("close failed")

        container = build_container(Settings())
        container._embedding_http_client = UnclosableClient()

        with pytest.raises(ExceptionGroup, match="container shutdown"):
            await container.aclose()

        assert container._llm_http_client.is_closed

    # KA_RETRIEVAL_TOP_K reaches AskQuestion as the default top_k; that path
    # needs a database session, so it is covered by the e2e "real wiring"
    # fixture instead (TestRealContainerWiring in tests/e2e).
