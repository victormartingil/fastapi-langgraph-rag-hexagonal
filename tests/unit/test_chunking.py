"""Unit tests for the pure chunking domain service.

No infrastructure, no async — the domain is the cheapest code to test and the
most valuable to keep correct, because every retrieval result downstream
depends on these boundaries.
"""

import pytest

from knowledge_assistant.knowledge_base.domain.chunking import chunk_text


class TestChunkText:
    def test_short_text_fits_in_one_chunk(self) -> None:
        chunks = chunk_text("A short policy statement.", max_chars=800)
        assert [str(c) for c in chunks] == ["A short policy statement."]

    def test_paragraphs_are_merged_up_to_max_chars(self) -> None:
        text = "aaa\n\nbbb\n\nccc"
        chunks = chunk_text(text, max_chars=8, overlap_chars=0)
        # "aaa\n\nbbb" is exactly 8 chars; adding "ccc" would overflow.
        assert [str(c) for c in chunks] == ["aaa\n\nbbb", "ccc"]

    def test_overlap_carries_tail_context_into_next_chunk(self) -> None:
        text = "aaaaaaaa\n\nbbbb"
        chunks = chunk_text(text, max_chars=12, overlap_chars=3)
        # The second chunk starts with the last 3 chars of the first one, so
        # boundary context is preserved.
        assert [str(c) for c in chunks] == ["aaaaaaaa", "aaa\n\nbbbb"]

    def test_a_single_long_paragraph_is_hard_split(self) -> None:
        text = "x" * 25
        chunks = chunk_text(text, max_chars=10, overlap_chars=0)
        assert [len(str(c)) for c in chunks] == [10, 10, 5]

    def test_hard_split_also_carries_the_overlap_tail(self) -> None:
        """The overlap is not a paragraph-path-only luxury: a hard-split must
        carry the tail too, or a sentence cut mid-word loses its context."""
        text = "ABCDEFGHIJKLMNOPQRSTUVWXY"  # 25 distinguishable characters
        chunks = [str(c) for c in chunk_text(text, max_chars=10, overlap_chars=3)]

        assert chunks == ["ABCDEFGHIJ", "HIJKLMNOPQ", "OPQRSTUVWX", "VWXY"]
        # Each chunk resumes with the last 3 characters of its predecessor.
        assert chunks[1][:3] == chunks[0][-3:]
        assert chunks[2][:3] == chunks[1][-3:]

    def test_trailing_sliver_is_merged_into_the_previous_chunk(self) -> None:
        """Two 700-char paragraphs at max 800/overlap 120 used to produce a
        tiny third chunk — an embedding call and a vector row for ~20 chars
        of retrieval-useless text. The sliver now merges into its predecessor."""
        text = "a" * 700 + "\n\n" + "b" * 700
        chunks = [str(c) for c in chunk_text(text, max_chars=800, overlap_chars=120)]

        assert len(chunks) == 2
        # The merged second chunk is a soft-ceiling exception: it contains the
        # whole overlap tail AND the remainder of the second paragraph.
        assert chunks[1].endswith("b" * 22)
        assert len(chunks[1]) == 944

    def test_empty_or_whitespace_text_yields_no_chunks(self) -> None:
        assert chunk_text("   \n\n  ") == []

    def test_never_returns_empty_chunks(self) -> None:
        chunks = chunk_text("a\n\nb\n\nc", max_chars=3, overlap_chars=1)
        assert all(str(c).strip() for c in chunks)

    @pytest.mark.parametrize(
        ("max_chars", "overlap"),
        [(0, 0), (-1, 0), (10, 10), (10, 20)],
        ids=["zero-max", "negative-max", "overlap-equals-max", "overlap-exceeds-max"],
    )
    def test_invalid_parameters_raise(self, max_chars: int, overlap: int) -> None:
        with pytest.raises(ValueError, match="chars"):
            chunk_text("text", max_chars=max_chars, overlap_chars=overlap)
