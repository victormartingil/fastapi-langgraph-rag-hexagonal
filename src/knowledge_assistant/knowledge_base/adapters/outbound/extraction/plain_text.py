"""PlainTextExtractor: the trivial adapter for .md and .txt uploads.

Markdown is treated as plain text on purpose: for RAG chunking, markdown
structure (headings, lists) is useful signal for the LLM, so we keep it
instead of stripping it.
"""

from knowledge_assistant.platform.observability.telemetry import observe_operation

SUPPORTED_SUFFIXES = (".md", ".markdown", ".txt")


class PlainTextExtractor:
    """Extracts text from UTF-8 text files (supports .md/.markdown/.txt)."""

    def supports(self, file_name: str) -> bool:
        return file_name.lower().endswith(SUPPORTED_SUFFIXES)

    def extract(self, file_name: str, data: bytes) -> str:
        with observe_operation("extraction", {"file.type": "text"}):
            return data.decode("utf-8", errors="replace")
