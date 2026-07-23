"""Create `schema_meta` and record the FTS language the schema was built with.

The FTS language is baked into the generated `tsv` column's expression
(ADR-0003), so runtime configuration and schema reality can drift: migrate
with one KA_FTS_LANGUAGE, run with another, and full-text search silently
degrades. This migration introduces the defense-in-depth half of the fix
(ADR-0004): a key/value table that records schema-bound configuration, which
the application verifies at startup.

The recorded value is INTROSPECTED from the actual column expression
(pg_get_expr), not copied from the environment: databases already migrated
with 0003 may have been built with any language, and the meta table must
describe reality, not repeat a claim. Parsing happens ONCE here, at
migration time, where a failure is loud and atomic — not on every boot.

KNOWN LIMITATION: because this migration queries the live schema, it cannot
run in Alembic's OFFLINE mode (`alembic upgrade --sql`) — data-introspecting
migrations need a real connection. This project always migrates online
(Dockerfile entrypoint, tests, README workflow), so the limitation is
documented, not engineered around.

Revision ID: 0004_schema_meta_fts_language
Revises: 0003_configurable_fts_language
Create Date: 2026-07-22
"""

import re
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004_schema_meta_fts_language"
down_revision: str = "0003_configurable_fts_language"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# The generation expression looks like: to_tsvector('spanish'::regconfig, content)
_EXPRESSION_PATTERN = re.compile(r"to_tsvector\('([a-z_]+)'::regconfig")


def _introspect_tsv_language() -> str:
    expression = (
        op.get_bind()
        .execute(
            sa.text(
                "SELECT pg_get_expr(d.adbin, d.adrelid) "
                "FROM pg_attrdef d "
                "JOIN pg_attribute a ON a.attrelid = d.adrelid AND a.attnum = d.adnum "
                "WHERE a.attrelid = 'chunks'::regclass AND a.attname = 'tsv'"
            )
        )
        .scalar_one()
    )
    match = _EXPRESSION_PATTERN.search(expression)
    if match is None:
        msg = (
            "Cannot parse the text-search configuration from the chunks.tsv "
            f"generation expression: {expression!r}. Record schema_meta."
            "fts_language manually to match the column."
        )
        raise RuntimeError(msg)
    return match.group(1)


def upgrade() -> None:
    op.create_table(
        "schema_meta",
        sa.Column("key", sa.Text(), primary_key=True),
        sa.Column("value", sa.Text(), nullable=False),
    )
    op.execute(
        sa.text(
            "INSERT INTO schema_meta (key, value) VALUES ('fts_language', :language)"
        ).bindparams(language=_introspect_tsv_language())
    )


def downgrade() -> None:
    op.drop_table("schema_meta")
