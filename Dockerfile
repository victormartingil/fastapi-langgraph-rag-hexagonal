# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Multi-stage build:
#   1. `builder` — uses uv to resolve and install locked dependencies into a
#      virtualenv, with a cache mount so rebuilds are fast.
#   2. `runtime` — copies only the venv and the source; runs as a NON-ROOT
#      user; contains no build tools.
# ---------------------------------------------------------------------------

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

WORKDIR /app

# Install dependencies first (better layer caching: sources change often,
# the lockfile rarely does).
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev


FROM python:3.13-slim-bookworm AS runtime

# Non-root user: the app does not need any privilege.
RUN groupadd --system app && useradd --system --gid app app

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY --from=builder /app/.venv /app/.venv
COPY src/ src/
COPY alembic.ini ./

USER app

EXPOSE 8000

# Run DB migrations, then serve. Migrations are idempotent; in a real
# deployment you may prefer a separate migration job.
# `exec` matters: it REPLACES the shell with uvicorn, so uvicorn becomes
# PID 1 and receives SIGTERM/SIGINT directly (graceful shutdown). Without
# it, signals land on the wrapper shell and the server is killed uncleanly.
CMD ["/bin/sh", "-c", "alembic upgrade head && exec uvicorn knowledge_assistant.main:create_app --factory --host 0.0.0.0 --port 8000"]
