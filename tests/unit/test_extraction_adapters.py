"""Unit tests for the extraction adapters.

Extraction adapters are where vendor parsers live; their failures must speak
domain errors, not vendor exception zoos (a corrupt PDF is a 422, not a 500).
"""

import pytest

from knowledge_assistant.knowledge_base.adapters.outbound.extraction.pdf import PdfTextExtractor
from knowledge_assistant.knowledge_base.adapters.outbound.extraction.plain_text import (
    PlainTextExtractor,
)
from knowledge_assistant.knowledge_base.domain.exceptions import TextExtractionError


class TestPdfTextExtractor:
    def test_corrupt_pdf_raises_without_logging_uploaded_bytes(
        self,
        caplog: pytest.LogCaptureFixture,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """pypdf raises its own exception zoo on unreadable files; the adapter
        quarantines it into one domain signal."""
        extractor = PdfTextExtractor()
        private_bytes = b"private customer account 1234"

        with pytest.raises(TextExtractionError, match=r"broken\.pdf"):
            extractor.extract("broken.pdf", private_bytes)

        captured = capsys.readouterr()
        assert "private customer" not in caplog.text
        assert "private customer" not in captured.out
        assert "private customer" not in captured.err

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
