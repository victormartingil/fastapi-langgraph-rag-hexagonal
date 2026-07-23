"""OpenAiEmbeddingProvider: embeddings from the OpenAI API.

Alternative provider, selected via `KA_EMBEDDING_PROVIDER=openai` in `.env`.
Kept intentionally parallel to the Ollama adapter so the Strategy pattern is
visible at a glance: same Protocol, different vendor.

WARNING (documented in ADR-0001): OpenAI embedding models produce 1536-dim
vectors by default, while the schema ships with vector(768) for
nomic-embed-text. Switching providers requires adjusting
`KA_EMBEDDING_DIMENSION` and regenerating the migration for the new dimension.
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

from knowledge_assistant.shared.domain.exceptions import (
    EmbeddingProviderUnavailableError,
)
from knowledge_assistant.shared.domain.value_objects import EmbeddingVector
from knowledge_assistant.shared.infrastructure.http_resilience import (
    is_transient_http_error,
)

logger = structlog.get_logger()


class OpenAiEmbeddingProvider:
    """Embeds texts with the OpenAI embeddings API."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        model: str,
        api_key: str,
        *,
        max_retries: int = 3,
    ) -> None:
        self._client = client
        self._model = model
        self._api_key = api_key
        # Only TRANSIENT failures are retried (timeouts, connection errors,
        # 5xx): a 401 means a bad key — retrying it wastes time and budget.
        # Backoff carries jitter, like every adapter here (thundering herd).
        self._embed_with_retry = retry(
            retry=retry_if_exception(is_transient_http_error),
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=0.5, max=8) + wait_random(0, 0.5),
            reraise=True,
        )(self._embed_once)

    async def embed(self, texts: list[str]) -> list[EmbeddingVector]:
        if not texts:
            return []
        logger.debug("embedding_batch", provider="openai", model=self._model, size=len(texts))
        try:
            return await self._embed_with_retry(texts)
        except httpx.HTTPError as exc:
            # Port contract: a transient outage that survived retries becomes
            # a domain signal (-> HTTP 503); permanent errors propagate raw.
            if not is_transient_http_error(exc):
                raise
            msg = (
                "The embedding service is temporarily unavailable "
                "(provider 'openai' unreachable after retries). Please try again shortly."
            )
            raise EmbeddingProviderUnavailableError(msg) from exc

    async def _embed_once(self, texts: list[str]) -> list[EmbeddingVector]:
        response = await self._client.post(
            "/v1/embeddings",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"model": self._model, "input": texts},
        )
        response.raise_for_status()
        payload = response.json()
        ordered = sorted(payload["data"], key=lambda item: item["index"])
        return [EmbeddingVector(tuple(float(v) for v in item["embedding"])) for item in ordered]
