# Contributing

Contributions are welcome when they preserve the repository's purpose: a
generic, didactic, production-shaped RAG architecture reference.

## Before changing code

1. Read the [architecture overview](docs/00-architecture-overview.md) and
   [coding-agent guidelines](.ai-guidelines.md).
2. Locate the bounded context that owns the behavior.
3. Add a port only if the change crosses a volatile boundary.
4. Add an ADR when changing ownership, topology, persistence, security, or
   failure semantics.

## Setup and checks

Install [uv](https://docs.astral.sh/uv/) and sync exactly what the lockfile
resolves:

```bash
uv sync --locked
uv run --locked pre-commit install
```

Before opening a pull request:

```bash
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy --strict src tests
uv run --locked pytest tests/unit tests/architecture tests/evals
uv run --locked pytest tests/integration tests/e2e
```

The last command requires Docker. On Rancher Desktop, Testcontainers may need:

```bash
TESTCONTAINERS_DOCKER_SOCKET_OVERRIDE=/var/run/docker.sock \
  uv run --locked pytest tests/integration tests/e2e
```

## Pull requests

- Use a focused branch and
  [Conventional Commit](https://www.conventionalcommits.org/) messages.
- Explain the behavior and trade-off, not only the files changed.
- Add tests at the lowest meaningful level.
- Update the RAG baseline only when the metric movement is understood.
- Never commit credentials, private documents, prompt/response captures, or
  telemetry containing user content.
- Keep all code, comments, commits, and public documentation in English.

CI verifies Python 3.12–3.14, packaging, dependency audit, SBOM generation,
real PostgreSQL integration/E2E, CodeQL, secret scanning, and the container
image.
