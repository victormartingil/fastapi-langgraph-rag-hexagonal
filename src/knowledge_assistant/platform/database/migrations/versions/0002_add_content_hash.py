"""Add content_hash to documents for upload deduplication.

`content_hash` is the SHA-256 of the extracted text, computed at ingestion
time. The unique index makes "same content uploaded twice" detectable — the
use case returns the existing document instead of storing a twin.

Nullable on purpose: rows ingested before this migration have no hash, and
PostgreSQL unique indexes treat NULLs as distinct, so they never collide.

Revision ID: 0002_add_content_hash
Revises: 0001_create_knowledge_base
Create Date: 2026-07-22
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_add_content_hash"
down_revision: str | None = "0001_create_knowledge_base"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("content_hash", sa.String(length=64), nullable=True))
    op.create_index("ix_documents_content_hash", "documents", ["content_hash"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_documents_content_hash", table_name="documents")
    op.drop_column("documents", "content_hash")
