"""Application settings (12-factor: config comes from the environment).

`pydantic-settings` gives us typed, validated, documented configuration with
`.env` support for local development. Every knob has a sane default so the app
boots with zero configuration against the docker-compose stack.

All variables use the `KA_` prefix (Knowledge Assistant) to avoid colliding
with unrelated environment variables.
"""

import re
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# PostgreSQL text-search configuration names are lowercase identifiers
# ('english', 'spanish', 'simple', ...) — the SAME pattern migration 0003
# enforces before interpolating the name into DDL.
_FTS_LANGUAGE_PATTERN = re.compile(r"^[a-z_]+$")


class Settings(BaseSettings):
    """Root settings object; a single instance lives on `app.state`."""

    model_config = SettingsConfigDict(env_prefix="KA_", env_file=".env", extra="ignore")

    # --- HTTP -------------------------------------------------------------
    app_name: str = "knowledge-assistant"
    debug: bool = False

    # --- Database ---------------------------------------------------------
    # Async driver: postgresql+asyncpg. The default matches docker-compose.
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/knowledge"

    # --- Embeddings -------------------------------------------------------
    # Provider-driven defaults are resolved in bootstrap.py (symmetric with
    # the LLM); leave these unset (None) to inherit the provider's default
    # model/endpoint/dimension.
    embedding_provider: Literal["ollama", "openai"] = "ollama"
    embedding_model: str | None = None
    embedding_base_url: str | None = None
    embedding_api_key: str = ""  # only needed for the OpenAI provider
    # Must match the pgvector column (ADR-0001). The shipped schema is
    # vector(768): any other dimension fails fast at startup unless you
    # regenerate the migration (and SCHEMA_EMBEDDING_DIMENSION with it).
    embedding_dimension: int | None = None
    embedding_timeout_seconds: float = 60.0

    # --- LLM (answer generation) ------------------------------------------
    # Provider-driven defaults are resolved in bootstrap.py; leave these
    # unset (None) to inherit the provider's default model/endpoint/key.
    llm_provider: Literal["ollama", "openai"] = "ollama"
    llm_model: str | None = None
    llm_base_url: str | None = None  # OpenAI-compatible endpoint
    llm_api_key: str | None = None  # Ollama ignores it; OpenAI requires a real key
    llm_timeout_seconds: float = 120.0
    llm_max_retries: int = 3

    # --- RAG behavior ------------------------------------------------------
    chunk_max_chars: int = 800
    chunk_overlap_chars: int = 120
    retrieval_top_k: int = 5
    # How many candidates each retrieval leg fetches before RRF fusion.
    retrieval_fetch_limit: int = 20
    # RRF constant k (standard value from the original paper).
    rrf_k: int = 60
    # PostgreSQL text search configuration used by the full-text leg — any
    # regconfig name: 'english', 'spanish', 'simple', ... `simple` is
    # language-agnostic (no stemming, no stop words): the pragmatic choice
    # for mixed-language corpora. SCHEMA-BOUND like the embedding dimension:
    # migration 0003 builds the tsv column with the same variable, so change
    # it together and rebuild the schema on a fresh database (ADR-0003).
    fts_language: str = "english"
    # Minimum RRF score for a chunk to count as relevant evidence. With
    # rrf_k=60, the best a SINGLE-leg match can score is 1/(60+1) ~= 0.0164,
    # so 0.028 effectively requires consensus: the chunk must be found by
    # BOTH the dense and the full-text leg.
    #
    # KNOWN TRADE-OFF (see docs/02 and ADR-0003): this is precision-first.
    # Paraphrase questions that share no content words with the chunk
    # ("how do I get my money back?" vs "full refund") get no full-text hit,
    # score below the bar, and the system REFUSES — exactly the class of
    # question dense retrieval exists for. Absolute RRF thresholds are also
    # corpus-size dependent. We accept this deliberately: for a knowledge
    # base, an honest refusal beats a shaky answer. The real fix is an
    # LLM-as-judge grader node (blueprinted in docs/03, roadmap Phase 2).
    min_relevance_score: float = 0.028

    # --- Ingestion ---------------------------------------------------------
    # Uploads larger than this are rejected with HTTP 413 before buffering.
    max_upload_size_mb: float = 10.0
    # Chunks are embedded in batches of this size: one giant call is both a
    # timeout risk and a whole-ingestion single point of failure.
    embedding_batch_size: int = 32

    # --- Security ----------------------------------------------------------
    # Optional API key: when set, /api/v1/* requires the X-API-Key header.
    # When unset (default), auth is off — the zero-friction quick start keeps
    # working. /health is always open. See README "Security".
    api_key: str | None = None

    @field_validator("fts_language", mode="before")
    @classmethod
    def _normalize_fts_language(cls, value: object) -> object:
        """Lowercase and pattern-check the FTS language.

        PostgreSQL folds unquoted regconfig names, so `KA_FTS_LANGUAGE=English`
        would WORK at query time — but the startup parity guard compares the
        recorded schema language exactly, and migrations interpolate the raw
        string into DDL. Normalizing here keeps config, guard, and migration
        on the same canonical form by construction.
        """
        if isinstance(value, str):
            value = value.lower()
            if not _FTS_LANGUAGE_PATTERN.fullmatch(value):
                msg = (
                    f"Invalid fts_language {value!r}: a PostgreSQL text-search "
                    "configuration name (lowercase letters and underscores, "
                    "e.g. 'english', 'spanish', 'simple')"
                )
                raise ValueError(msg)
        return value


def get_settings() -> Settings:
    """Factory used by the composition root and overridable in tests."""
    return Settings()
