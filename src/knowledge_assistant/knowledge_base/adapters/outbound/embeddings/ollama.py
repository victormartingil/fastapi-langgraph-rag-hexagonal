"""OllamaEmbeddingProvider: embeddings from a local Ollama server.

Default provider — no API keys, no cloud, fully reproducible demos. Talks to
Ollama's native `/api/embed` endpoint with an httpx client, and retries
transient failures with tenacity (exponential backoff): a local model server
being briefly busy is normal and should not fail an ingestion.

The two embedding adapters (this one and `openai.py`) are interchangeable
because both satisfy the `EmbeddingProvider` Protocol — the Strategy pattern
with structural typing.
"""

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

from knowledge_assistant.knowledge_base.domain.exceptions import (
    EmbeddingProviderUnavailableError,
)
from knowledge_assistant.platform.http.resilience import (
    is_transient_http_error,
)
from knowledge_assistant.platform.observability.telemetry import (
    observe_operation,
    record_retry,
)
from knowledge_assistant.shared_kernel.value_objects import EmbeddingVector

logger = structlog.get_logger()


class OllamaEmbeddingProvider:
    """Embeds texts with a local Ollama model (default: nomic-embed-text)."""

    def __init__(self, client: httpx.AsyncClient, model: str, *, max_retries: int = 3) -> None:
        self._client = client
        self._model = model
        # Only TRANSIENT failures are retried (timeouts, connection errors,
        # 5xx): a 4xx will never succeed on retry — see http_resilience.
        # Backoff carries jitter: without it, clients that failed together
        # retry in lockstep and re-flood the recovering server.
        self._embed_with_retry = retry(
            retry=retry_if_exception(is_transient_http_error),
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=0.5, max=8) + wait_random(0, 0.5),
            before_sleep=lambda _: record_retry("embeddings"),
            reraise=True,
        )(self._embed_once)

    async def embed(self, texts: list[str]) -> list[EmbeddingVector]:
        if not texts:
            return []
        logger.debug("embedding_batch", provider="ollama", model=self._model, size=len(texts))
        with observe_operation(
            "embeddings",
            {
                "gen_ai.operation.name": "embeddings",
                "gen_ai.provider.name": "ollama",
                "gen_ai.request.model": self._model,
                "gen_ai.request.input.count": len(texts),
            },
        ):
            try:
                return await self._embed_with_retry(texts)
            except httpx.HTTPError as exc:
                # Port contract: a transient outage that survived retries becomes
                # a domain signal (-> HTTP 503); permanent errors propagate raw.
                if not is_transient_http_error(exc):
                    raise
                msg = (
                    "The embedding service is temporarily unavailable "
                    "(provider 'ollama' unreachable after retries). "
                    "Please try again shortly."
                )
                raise EmbeddingProviderUnavailableError(msg) from exc

    async def _embed_once(self, texts: list[str]) -> list[EmbeddingVector]:
        response = await self._client.post(
            "/api/embed",
            json={"model": self._model, "input": texts},
        )
        response.raise_for_status()
        payload = response.json()
        embeddings = payload["embeddings"]
        return [EmbeddingVector(tuple(float(v) for v in vector)) for vector in embeddings]
