"""Recreate the `tsv` column for the configured FTS language (KA_FTS_LANGUAGE).

Migration 0001 hardcoded `to_tsvector('english', content)`. Full-text search
is language-sensitive — stemming and stop words differ per language — so the
text search configuration must be an operator decision. Like the embedding
dimension (ADR-0001), the FTS language is SCHEMA-BOUND: it is baked into the
generated column's expression, so it is chosen at migration time via the
`KA_FTS_LANGUAGE` environment variable (default 'english'), the same variable
the retriever reads at runtime. Changing it later means rebuilding the
schema on a fresh database (or writing a new migration like this one).

Workflow for a Spanish knowledge base:

    KA_FTS_LANGUAGE=spanish uv run alembic upgrade head   # fresh database

The language name is validated against a strict pattern before being
interpolated into DDL — migrations run with DDL privileges, and defense in
depth costs one regex.

Revision ID: 0003_configurable_fts_language
Revises: 0002_add_content_hash
Create Date: 2026-07-22
"""

import os
import re
from collections.abc import Sequence

from alembic import op

revision: str = "0003_configurable_fts_language"
down_revision: str = "0002_add_content_hash"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# PostgreSQL configuration names are lowercase identifiers ('english',
# 'spanish', 'simple', ...). Anything else is rejected before interpolation.
_LANGUAGE_PATTERN = re.compile(r"^[a-z_]+$")


def _fts_language() -> str:
    # Lowercased before validation, mirroring the Settings field validator:
    # PostgreSQL folds regconfig names anyway, and both sides must agree on
    # the canonical form the startup parity guard compares.
    language = os.environ.get("KA_FTS_LANGUAGE", "english").lower()
    if not _LANGUAGE_PATTERN.fullmatch(language):
        msg = f"Unsafe KA_FTS_LANGUAGE value: {language!r} (lowercase letters and underscores only)"
        raise ValueError(msg)
    return language


def _recreate_tsv(language: str) -> None:
    # Dropping the column also drops the GIN index built on it.
    op.execute("ALTER TABLE chunks DROP COLUMN tsv")
    op.execute(
        "ALTER TABLE chunks ADD COLUMN tsv tsvector "
        f"GENERATED ALWAYS AS (to_tsvector('{language}', content)) STORED"
    )
    op.execute("CREATE INDEX ix_chunks_tsv_gin ON chunks USING gin (tsv)")


def upgrade() -> None:
    _recreate_tsv(_fts_language())


def downgrade() -> None:
    _recreate_tsv("english")
