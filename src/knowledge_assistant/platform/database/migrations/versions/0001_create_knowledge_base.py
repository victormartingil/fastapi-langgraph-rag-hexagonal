"""Create the knowledge base schema: documents + chunks + pgvector.

This migration does four things a plain `CREATE TABLE` cannot express in
SQLAlchemy models, so they are written as explicit SQL/DDL here:

1. Enables the `vector` extension (pgvector).
2. Creates `chunks.embedding` as vector(768) — the dimension of the default
   embedding model (nomic-embed-text). See ADR-0001 before changing it.
3. Adds `chunks.tsv`, a GENERATED tsvector column kept in sync by PostgreSQL.
4. Creates the two indexes that make retrieval fast: HNSW for the vector leg,
   GIN for the full-text leg.

Revision ID: 0001_create_knowledge_base
Revises:
Create Date: 2026-07-01
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0001_create_knowledge_base"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("file_name", sa.String(length=500), nullable=False),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(768), nullable=False),
        sa.Column(
            "tsv",
            postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('english', content)", persisted=True),
            nullable=False,
        ),
    )
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])
    op.create_index("ix_chunks_tsv_gin", "chunks", ["tsv"], postgresql_using="gin")
    # HNSW: approximate nearest-neighbor index for the dense leg.
    # vector_cosine_ops matches the `<=>` operator used by the retriever.
    op.execute(
        "CREATE INDEX ix_chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.drop_index("ix_chunks_embedding_hnsw", table_name="chunks")
    op.drop_index("ix_chunks_tsv_gin", table_name="chunks")
    op.drop_index("ix_chunks_document_id", table_name="chunks")
    op.drop_table("chunks")
    op.drop_table("documents")
    op.execute("DROP EXTENSION IF EXISTS vector")
