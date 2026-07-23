# Threat model

This reference handles documents and model input as hostile data. The goal is
risk reduction and observable failure, not a claim that prompt injection has
been solved.

## Assets and trust boundaries

Assets include uploaded document contents, generated answers, provider
credentials, the vector index, and operational metadata. Trust boundaries are
the public HTTP API, file extractors, model-provider HTTP APIs, PostgreSQL, and
the boundary between retrieved evidence and the generation prompt.

The API accepts no tenant or customer concept. A production multi-tenant
deployment must add authorization-scoped storage and retrieval before storing
private documents.

## Principal threats and controls

| Threat | Existing controls | Residual risk |
| --- | --- | --- |
| Direct prompt injection in a question | question is serialized as untrusted JSON; system instructions explicitly reject embedded instructions; answers require valid evidence citations | a capable model can still be manipulated |
| Indirect prompt injection in documents or titles | retrieved fields are serialized, delimited, and labeled as data; no tools are available to the generator | malicious text may influence the model despite instructions |
| Hallucinated or missing citations | structured output requires at least one index; indices are range-checked; invalid output is retried and then returns typed HTTP 502 | a valid index does not prove every sentence is entailed |
| Unsupported claims with plausible citations | relevance grading and source provenance; deterministic refusal when no evidence survives | factual entailment is not yet independently verified |
| Resource exhaustion | bounded question, title, filename, top-k, upload, chunk, batch, timeout, and retry settings | PDF decompression/parser complexity still needs deployment-level limits |
| SQL injection | bound SQL parameters; the FTS configuration is pattern-validated and cast in PostgreSQL | application or dependency defects remain possible |
| Credential exposure | environment configuration; prompts, questions, and documents are not logged by default | operator misconfiguration can still expose secrets |
| Dependency or image compromise | dependency lock, automated audit/scanning, SBOM, pinned CI actions and images | upstream compromise and zero-days remain possible |

## Required production extensions

- tenant-aware authorization on ingestion, listing, retrieval, and deletion;
- malware scanning and file-type verification before extraction;
- request rate limits, quotas, and reverse-proxy body/time limits;
- encryption and retention policies appropriate to the data classification;
- red-team and regression datasets specific to the deployment language and
  document types;
- human review for high-impact answers.

See the
[OWASP Prompt Injection Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html)
for the defense-in-depth rationale.
