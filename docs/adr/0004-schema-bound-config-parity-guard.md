# ADR-0004: Record schema-bound configuration in `schema_meta` and verify it at startup

- **Status**: Accepted
- **Date**: 2026-07-22

## Context

Some configuration is baked INTO the schema at migration time. The FTS
language lives in the generated `tsv` column's expression (ADR-0003); the
embedding dimension lives in the `vector(768)` column type (ADR-0001). When
runtime configuration and schema reality disagree, the failure mode is
*silent degradation*, the worst kind: the app boots, answers flow, and
full-text search quietly underperforms.

The concrete drift window: `Settings` reads `.env`, but Alembic's `env.py`
read only `os.environ`. `KA_FTS_LANGUAGE=spanish` in `.env` + migrating
without it exported (or the reverse) builds the schema for one language and
queries with another.

The embedding dimension already has a two-layer defense (startup guard vs
`SCHEMA_EMBEDDING_DIMENSION` + first-ingest reality check). The FTS language
had none.

## Decision

Two complementary defenses:

1. **Single source of truth (root cause).** Alembic's `env.py` loads `.env`
   via `python-dotenv` (without overriding real environment variables — the
   same precedence pydantic-settings applies), so migrate-time and run-time
   resolve `KA_FTS_LANGUAGE` from the same file with the same rules.

2. **Startup parity guard (defense in depth).** Migration 0004 creates a
   `schema_meta` key/value table and records `fts_language` — **introspected
   from the actual column expression** (`pg_get_expr`), never copied from the
   environment, so the table describes reality even for databases migrated
   before this mechanism existed. At startup
   (`shared/infrastructure/schema_meta.py`, called from the lifespan) the app
   compares the recorded language against `KA_FTS_LANGUAGE`:
   - **mismatch** → refuse to boot, naming both languages and the fix;
   - **table/row missing** → refuse to boot: schema older than the app, run
     `alembic upgrade head`;
   - **database unreachable** → warn and boot anyway: availability is
     `/health`'s job, and a transient DB outage must not become a boot loop.

## Alternatives considered

- **Introspect the column expression at every startup** (`pg_get_expr` on
  `chunks.tsv`, parse out the regconfig): zero new schema and reads reality
  directly — but parsing DDL text on every boot is brittle across PostgreSQL
  versions (formatting and casts change). Rejected *for the hot path*; the
  same introspection is used **once**, at migration time, to populate
  `schema_meta`, where a parse failure is loud and atomic.
- **Record the embedding dimension in `schema_meta` too**: deliberately not
  done. The dimension guard runs *before* the engine exists (pure config vs
  constant comparison) and already has its first-ingest reality check;
  moving it behind a database read would weaken it for no unification gain.
  The table remains available if a future schema-bound knob needs it.
- **Do nothing beyond the dotenv fix**: rejected — the dotenv fix closes
  same-machine drift, but a database migrated in one environment and served
  in another (the realistic production topology) still drifts silently.

## Consequences

- Booting with a mismatched `KA_FTS_LANGUAGE` fails fast with an actionable
  error instead of degrading retrieval quality invisibly.
- Operators get one obvious contract: `schema_meta` is written by migrations
  and read at startup; never edit it by hand (the integration suite tampers
  with it precisely to prove the guard fires).
- The guard adds one cheap `SELECT` to startup. If the database is down at
  boot, startup behavior is unchanged (warn + boot, `/health` reports).
