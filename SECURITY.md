# Security policy

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting for this repository
rather than opening a public issue. Include the affected version, impact,
reproduction steps, and any suggested mitigation. Do not include real
credentials, private documents, personal data, or production prompts.

If private reporting is unavailable, contact the maintainer through the
public profile linked in the README and request a private channel before
sharing details.

## Supported versions

Security fixes target the latest tagged release and `main`. This reference
project does not promise maintenance of older tags.

## Scope and deployment responsibility

The repository includes supply-chain scanning, bounded inputs, optional API
key authentication, grounded-output validation, and a documented
[threat model](docs/06-threat-model.md). It is not a turnkey security
boundary for private or multi-tenant data.

Before a production deployment, add tenant-aware authorization, malware/file
validation, rate limiting, network policy, secrets management, encryption,
retention/deletion controls, deployment-specific adversarial evaluation, and
human review where answers can cause material harm.
