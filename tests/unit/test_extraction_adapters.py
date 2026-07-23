"""Unit tests for the extraction adapters.

Extraction adapters are where vendor parsers live; their failures must speak
domain errors, not vendor exception zoos (a corrupt PDF is a 422, not a 500).
"""

import pytest

from knowledge_assistant.documents.domain.exceptions import TextExtractionError
from knowledge_assistant.documents.infrastructure.extraction.pdf import PdfTextExtractor
from knowledge_assistant.documents.infrastructure.extraction.plain_text import (
    PlainTextExtractor,
)


class TestPdfTextExtractor:
    def test_corrupt_pdf_raises_a_domain_error(self) -> None:
        """pypdf raises its own exception zoo on unreadable files; the adapter
        quarantines it into one domain signal."""
        extractor = PdfTextExtractor()

        with pytest.raises(TextExtractionError, match=r"broken\.pdf"):
            extractor.extract("broken.pdf", b"this is not a PDF at all")

    def test_supports_only_pdf(self) -> None:
        extractor = PdfTextExtractor()
        assert extractor.supports("report.PDF")
        assert not extractor.supports("notes.md")


class TestPlainTextExtractor:
    def test_undecodable_bytes_are_replaced_not_raised(self) -> None:
        """Text files are decoded leniently: a stray invalid byte must not
        reject an otherwise readable upload."""
        text = PlainTextExtractor().extract("notes.md", b"valid \xff\xfe bytes")
        assert "valid" in text
