"""Unit tests for the three HTTP-level AI adapters, with respx mocks.

These adapters never ran in any test before: they were only *wired* in e2e
(with fakes replacing them). respx lets us test them honestly — real adapter
code, real httpx, mocked HTTP — with no Docker and no network, so they live
in the unit tier.

What is covered:
- response parsing (embeddings; OpenAI's index-ordered batching),
- the Authorization header on the OpenAI adapter,
- tenacity retry behavior (transient 5xx → retry → success; permanent 4xx →
  no retry; exhausted transient retries → EmbeddingProviderUnavailableError),
- the LLM adapter's error doctrine: exhausted transient retries →
  GenerationUnavailableError (503 signal), permanent errors propagate loud,
- citation resolution, including dropping out-of-range citation indices.

What is NOT covered, by design: pydantic-ai's internal agent loop (tool
execution, validation retries). That machinery belongs to the vendor SDK; we
mock its HTTP boundary and trust its own test suite above it.
"""

import json

import httpx
import pytest
import respx
from pydantic_ai.exceptions import ModelHTTPError

from knowledge_assistant.assistant.domain.exceptions import GenerationUnavailableError
from knowledge_assistant.assistant.domain.models import RetrievedChunk
from knowledge_assistant.assistant.infrastructure.llm.pydantic_ai import (
    PydanticAiAnswerGenerator,
)
from knowledge_assistant.knowledge_base.infrastructure.embeddings.ollama import (
    OllamaEmbeddingProvider,
)
from knowledge_assistant.knowledge_base.infrastructure.embeddings.openai import (
    OpenAiEmbeddingProvider,
)
from knowledge_assistant.shared.domain.exceptions import (
    EmbeddingProviderUnavailableError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_chunk(chunk_id: str = "c1") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="d1",
        document_title="Return Policy",
        content="You may return any product within 30 days.",
        score=0.033,
    )


def completion_response(answer: str, source_indices: list[int]) -> httpx.Response:
    """The OpenAI-compatible response pydantic-ai expects for structured
    output: a tool call to `final_result` with the payload as JSON arguments."""
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "final_result",
                                    "arguments": json.dumps(
                                        {
                                            "answer": answer,
                                            "source_indices": source_indices,
                                        }
                                    ),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )


SERVER_ERROR = httpx.Response(500, json={"error": "boom"})


# ---------------------------------------------------------------------------
# OllamaEmbeddingProvider
# ---------------------------------------------------------------------------


class TestOllamaEmbeddingProvider:
    @respx.mock
    async def test_parses_embeddings_response(self) -> None:
        route = respx.post("http://ollama.test/api/embed").mock(
            return_value=httpx.Response(
                200, json={"embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]}
            )
        )
        provider = OllamaEmbeddingProvider(
            httpx.AsyncClient(base_url="http://ollama.test"), model="nomic-embed-text"
        )

        vectors = await provider.embed(["hello", "world"])

        assert [v.values for v in vectors] == [(0.1, 0.2, 0.3), (0.4, 0.5, 0.6)]
        assert route.called
        sent = json.loads(route.calls[0].request.content)
        assert sent == {"model": "nomic-embed-text", "input": ["hello", "world"]}

    @respx.mock
    async def test_empty_input_short_circuits_without_http(self) -> None:
        route = respx.post("http://ollama.test/api/embed")
        provider = OllamaEmbeddingProvider(
            httpx.AsyncClient(base_url="http://ollama.test"), model="nomic-embed-text"
        )

        assert await provider.embed([]) == []
        assert not route.called

    @respx.mock
    async def test_retries_transient_errors_then_succeeds(self) -> None:
        route = respx.post("http://ollama.test/api/embed").mock(
            side_effect=[SERVER_ERROR, httpx.Response(200, json={"embeddings": [[1.0]]})]
        )
        provider = OllamaEmbeddingProvider(
            httpx.AsyncClient(base_url="http://ollama.test"),
            model="nomic-embed-text",
            max_retries=2,
        )

        vectors = await provider.embed(["x"])

        assert [v.values for v in vectors] == [(1.0,)]
        assert route.call_count == 2  # one failure + one success

    @respx.mock
    async def test_permanent_client_errors_are_not_retried(self) -> None:
        # A 401 means the credentials are wrong: retrying cannot help, it only
        # multiplies the failure. The retry predicate is transient-only.
        route = respx.post("http://ollama.test/api/embed").mock(
            return_value=httpx.Response(401, json={"error": "unauthorized"})
        )
        provider = OllamaEmbeddingProvider(
            httpx.AsyncClient(base_url="http://ollama.test"),
            model="nomic-embed-text",
            max_retries=5,
        )

        with pytest.raises(httpx.HTTPStatusError):
            await provider.embed(["hello"])

        assert route.call_count == 1

    @respx.mock
    async def test_exhausted_transient_retries_become_unavailable_error(self) -> None:
        """Port contract: a transient outage that survives every retry becomes
        EmbeddingProviderUnavailableError (-> HTTP 503 on BOTH call paths),
        not a raw httpx error (-> bare 500 on the ingest path)."""
        route = respx.post("http://ollama.test/api/embed").mock(
            side_effect=[SERVER_ERROR, SERVER_ERROR]
        )
        provider = OllamaEmbeddingProvider(
            httpx.AsyncClient(base_url="http://ollama.test"),
            model="nomic-embed-text",
            max_retries=2,
        )

        with pytest.raises(EmbeddingProviderUnavailableError, match="temporarily unavailable"):
            await provider.embed(["hello"])

        assert route.call_count == 2


# ---------------------------------------------------------------------------
# OpenAiEmbeddingProvider
# ---------------------------------------------------------------------------


class TestOpenAiEmbeddingProvider:
    @respx.mock
    async def test_orders_results_by_index_and_sends_auth_header(self) -> None:
        # The API returns items in arbitrary order; the `index` field is the
        # contract, and the adapter must restore input order.
        route = respx.post("http://openai.test/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": [
                        {"index": 1, "embedding": [0.9]},
                        {"index": 0, "embedding": [0.1]},
                        {"index": 2, "embedding": [0.5]},
                    ]
                },
            )
        )
        provider = OpenAiEmbeddingProvider(
            httpx.AsyncClient(base_url="http://openai.test"),
            model="text-embedding-3-small",
            api_key="sk-test",
        )

        vectors = await provider.embed(["a", "b", "c"])

        assert [v.values for v in vectors] == [(0.1,), (0.9,), (0.5,)]
        assert route.calls[0].request.headers["Authorization"] == "Bearer sk-test"

    @respx.mock
    async def test_retries_transient_errors_then_succeeds(self) -> None:
        route = respx.post("http://openai.test/v1/embeddings").mock(
            side_effect=[
                SERVER_ERROR,
                httpx.Response(200, json={"data": [{"index": 0, "embedding": [2.0]}]}),
            ]
        )
        provider = OpenAiEmbeddingProvider(
            httpx.AsyncClient(base_url="http://openai.test"),
            model="text-embedding-3-small",
            api_key="sk-test",
            max_retries=2,
        )

        vectors = await provider.embed(["x"])

        assert [v.values for v in vectors] == [(2.0,)]
        assert route.call_count == 2

    @respx.mock
    async def test_exhausted_transient_retries_become_unavailable_error(self) -> None:
        """Same port contract as the Ollama adapter: transient outage after
        retries -> EmbeddingProviderUnavailableError, never raw httpx."""
        route = respx.post("http://openai.test/v1/embeddings").mock(
            side_effect=[SERVER_ERROR, SERVER_ERROR]
        )
        provider = OpenAiEmbeddingProvider(
            httpx.AsyncClient(base_url="http://openai.test"),
            model="text-embedding-3-small",
            api_key="sk-test",
            max_retries=2,
        )

        with pytest.raises(EmbeddingProviderUnavailableError, match="temporarily unavailable"):
            await provider.embed(["hello"])

        assert route.call_count == 2


# ---------------------------------------------------------------------------
# PydanticAiAnswerGenerator
# ---------------------------------------------------------------------------


def make_generator(
    *,
    base_url: str = "http://llm.test/v1",
    max_retries: int = 1,
) -> PydanticAiAnswerGenerator:
    return PydanticAiAnswerGenerator(
        model_name="test-model",
        base_url=base_url,
        api_key="test-key",
        http_client=httpx.AsyncClient(),  # owned by the container in production
        max_retries=max_retries,
    )


class TestPydanticAiAnswerGenerator:
    @respx.mock
    async def test_parses_structured_output_and_resolves_citations(self) -> None:
        route = respx.post("http://llm.test/v1/chat/completions").mock(
            return_value=completion_response("30 days, unused, with receipt.", [1])
        )
        generator = make_generator()

        answer = await generator.generate("Return window?", [make_chunk("c1")])

        assert answer.text == "30 days, unused, with receipt."
        assert len(answer.sources) == 1
        assert answer.sources[0].chunk_id == "c1"
        assert answer.sources[0].document_title == "Return Policy"
        # The prompt carried the evidence as a NUMBERED source, plus content.
        sent = json.loads(route.calls[0].request.content)
        prompt = sent["messages"][-1]["content"]
        assert "[1]" in prompt
        assert "Return window?" in prompt
        assert "return any product within 30 days" in prompt

    @respx.mock
    async def test_drops_out_of_range_citation_indices(self) -> None:
        # The model cites a chunk that does not exist: an out-of-range index
        # is the index-based equivalent of a hallucinated citation — dropped.
        respx.post("http://llm.test/v1/chat/completions").mock(
            return_value=completion_response("Invented citation test.", [1, 99])
        )
        generator = make_generator()

        answer = await generator.generate("q?", [make_chunk("c1")])

        assert [s.chunk_id for s in answer.sources] == ["c1"]

    @respx.mock
    async def test_retries_transient_llm_errors_then_succeeds(self) -> None:
        route = respx.post("http://llm.test/v1/chat/completions").mock(
            side_effect=[SERVER_ERROR, completion_response("Recovered.", [1])]
        )
        generator = make_generator(max_retries=2)

        answer = await generator.generate("q?", [make_chunk("c1")])

        assert answer.text == "Recovered."
        assert route.call_count == 2

    @respx.mock
    async def test_exhausted_transient_retries_raise_generation_unavailable(self) -> None:
        """The error doctrine, on the generation side: a transient outage that
        outlives every retry is a 503-class domain signal — never a degraded
        200 with a fallback message (that shape is indistinguishable from a
        real answer and reports the outage as success).

        The call count IS the retry contract: tenacity is the single retry
        authority (the SDK client is built with max_retries=0), so
        max_retries=2 means exactly two HTTP calls — not 2 x 3."""
        route = respx.post("http://llm.test/v1/chat/completions").mock(
            side_effect=[SERVER_ERROR, SERVER_ERROR]
        )
        generator = make_generator(max_retries=2)

        with pytest.raises(GenerationUnavailableError, match="temporarily unavailable"):
            await generator.generate("q?", [make_chunk("c1")])

        assert route.call_count == 2

    @respx.mock
    async def test_unreachable_llm_raises_generation_unavailable(self) -> None:
        """Server DOWN (connection refused): the SDK surfaces it as a
        connection error, which is transient by definition — retried, then
        the honest 503 signal. Again: one adapter attempt == one HTTP call,
        because the SDK's internal retries are disabled."""
        route = respx.post("http://llm.test/v1/chat/completions").mock(
            side_effect=httpx.ConnectError("connection refused")
        )
        generator = make_generator(max_retries=2)

        with pytest.raises(GenerationUnavailableError, match="temporarily unavailable"):
            await generator.generate("q?", [make_chunk("c1")])

        assert route.call_count == 2

    @respx.mock
    async def test_permanent_llm_errors_are_not_retried_and_stay_loud(self) -> None:
        # A 401 (bad API key) is permanent: one HTTP call, then the error
        # propagates UNTRANSLATED — reporting misconfiguration as "temporarily
        # unavailable" would send clients into endless retry loops.
        route = respx.post("http://llm.test/v1/chat/completions").mock(
            return_value=httpx.Response(401, json={"error": {"message": "bad key"}})
        )
        generator = make_generator(max_retries=5)

        with pytest.raises(ModelHTTPError):
            await generator.generate("q?", [make_chunk("c1")])

        assert route.call_count == 1
