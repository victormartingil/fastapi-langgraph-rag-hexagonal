## What changed

<!-- Describe behavior and the reason for the change. -->

## Architecture

<!-- Name the owning bounded context, affected ports/adapters, and ADR if needed. -->

- [ ] Dependencies still point inward.
- [ ] No new abstraction was added without a concrete variation point.
- [ ] Public contracts, failure behavior, and migrations are documented.

## Verification

- [ ] Ruff and formatting
- [ ] mypy strict
- [ ] unit and architecture tests
- [ ] integration/E2E when an adapter or HTTP path changed
- [ ] evaluation baseline reviewed when RAG behavior changed
- [ ] no secrets, private data, prompts, or document content in code/logs/telemetry
