"""PlainTextExtractor: the trivial adapter for .md and .txt uploads.

Markdown is treated as plain text on purpose: for RAG chunking, markdown
structure (headings, lists) is useful signal for the LLM, so we keep it
instead of stripping it.
"""

SUPPORTED_SUFFIXES = (".md", ".markdown", ".txt")


class PlainTextExtractor:
    """Extracts text from UTF-8 text files (supports .md/.markdown/.txt)."""

    def supports(self, file_name: str) -> bool:
        return file_name.lower().endswith(SUPPORTED_SUFFIXES)

    def extract(self, file_name: str, data: bytes) -> str:
        return data.decode("utf-8", errors="replace")
