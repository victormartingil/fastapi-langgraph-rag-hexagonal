"""knowledge_assistant — a didactic RAG API built with hexagonal architecture.

The package is organized into bounded contexts (`documents`, `chat`) plus a
`shared` kernel. Each context follows the same three-layer rule:

    domain  <-  application  <-  infrastructure

Dependencies may only point INWARD. The rule is not a convention; it is
enforced by import-linter contracts (see tests/architecture).
"""

__version__ = "0.1.0"
