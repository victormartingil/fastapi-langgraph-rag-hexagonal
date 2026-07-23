"""SQLAlchemy models: the PERSISTENCE representation.

These classes are an infrastructure detail. They are shaped by database
concerns (column types, indexes, FK constraints), NOT by business rules, and
they never cross the infrastructure boundary — mappers convert them to/from
frozen domain dataclasses.

Note the `tsv` column: a stored, generated tsvector kept in sync by PostgreSQL
itself, indexed with GIN for full-text search. The vector column uses an HNSW
index for approximate nearest-neighbor search (created in the migration).
"""

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Computed, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Single declarative base for the whole schema."""


class DocumentModel(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    file_name: Mapped[str] = mapped_column(String(500), nullable=False)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    # SHA-256 of the extracted text; the deduplication key (see migration 0002).
    content_hash: Mapped[str | None] = mapped_column(
        String(64), unique=True, nullable=True, index=True
    )

    chunks: Mapped[list["ChunkModel"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="ChunkModel.position",
        lazy="selectin",
    )


class ChunkModel(Base):
    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # The dimension must match the configured embedding model (ADR-0001).
    embedding: Mapped[list[float]] = mapped_column(Vector(768), nullable=False)
    # Maintained by PostgreSQL, never written by the application. The DDL
    # lives in the Alembic migrations (0001 creates it, 0003 re-creates it
    # for the configured KA_FTS_LANGUAGE); the expression below mirrors the
    # default and is only metadata for autogenerate.
    tsv: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', content)", persisted=True),
        nullable=False,
    )

    document: Mapped[DocumentModel] = relationship(back_populates="chunks")

    __table_args__ = (
        Index("ix_chunks_tsv_gin", "tsv", postgresql_using="gin"),
        # HNSW index is created in the Alembic migration (needs an op class).
    )
