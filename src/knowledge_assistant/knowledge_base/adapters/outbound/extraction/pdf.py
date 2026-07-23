"""PdfTextExtractor: text extraction from PDF uploads via pypdf.

PDF parsing is messy vendor territory — exactly the kind of concern that
belongs in an adapter, behind the `TextExtractor` port. If we ever swap pypdf
for a smarter parser (layout-aware, OCR), only this file changes.
"""

import io

from pypdf import PdfReader

from knowledge_assistant.knowledge_base.domain.exceptions import TextExtractionError
from knowledge_assistant.platform.observability.telemetry import observe_operation

SUPPORTED_SUFFIXES = (".pdf",)


class PdfTextExtractor:
    """Extracts concatenated page text from .pdf files."""

    def supports(self, file_name: str) -> bool:
        return file_name.lower().endswith(SUPPORTED_SUFFIXES)

    def extract(self, file_name: str, data: bytes) -> str:
        with observe_operation("extraction", {"file.type": "pdf"}):
            try:
                reader = PdfReader(io.BytesIO(data))
                pages = [(page.extract_text() or "") for page in reader.pages]
            except Exception as exc:
                # pypdf raises a zoo of exception types on corrupt or encrypted
                # files. The port speaks domain errors, so the zoo is quarantined
                # here — a broken upload becomes a 422, never an opaque 500.
                raise TextExtractionError(file_name) from exc
        return "\n\n".join(page for page in pages if page.strip())
