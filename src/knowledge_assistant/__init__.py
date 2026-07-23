"""knowledge_assistant — a didactic RAG API built with hexagonal architecture.

The package is organized into bounded contexts (`knowledge_base`, `assistant`) plus a
`shared_kernel`. Each context follows the same three-layer rule:

    domain  <-  application  <-  adapters

Dependencies may only point INWARD. The rule is not a convention; it is
enforced by import-linter contracts (see tests/architecture).
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fastapi-langgraph-rag-hexagonal")
except PackageNotFoundError:  # source tree imported without installing the project
    __version__ = "0+unknown"
