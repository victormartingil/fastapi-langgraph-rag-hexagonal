"""Unit tests for the full-text query builder: the Unicode-aware tokenizer.

The tokenizer is the multilingual gate of the FTS leg: an ASCII-only pattern
mangles accented words ('cómo' -> 'c | mo') and erases CJK input entirely,
which — combined with the consensus grader — guarantees refusal for any
non-English question. These tests pin the Unicode behavior and the
injection-safety property.
"""

from knowledge_assistant.knowledge_base.adapters.outbound.retrieval.pgvector_hybrid import (
    _to_or_query,
)


class TestToOrQuery:
    def test_spanish_question_keeps_accented_words_whole(self) -> None:
        query = _to_or_query("¿Cómo puedo devolver un producto?")
        # 'cómo' stays whole — an ASCII tokenizer would split it into 'c | mo'.
        assert query == "cómo | puedo | devolver | un | producto"

    def test_german_umlauts_and_eszett_stay_whole(self) -> None:
        query = _to_or_query("Müller äußert sich über Rückerstattung")
        assert "müller" in query
        assert "äußert" in query
        assert "über" in query
        assert "rückerstattung" in query

    def test_cjk_input_produces_tokens_instead_of_vanishing(self) -> None:
        # tsvector does not SEGMENT CJK (documented limitation — the dense leg
        # carries those queries), but the tokens must at least survive.
        query = _to_or_query("返品 ポリシー")
        assert query == "返品 | ポリシー"

    def test_tsquery_operators_cannot_be_smuggled_in(self) -> None:
        """Injection safety is structural: only word characters survive, so
        `&`, `|`, `!`, `:*` and parentheses never reach to_tsquery."""
        query = _to_or_query("foo & bar | (baz):* !qux")
        assert query == "foo | bar | baz | qux"

    def test_symbol_only_input_yields_an_empty_query(self) -> None:
        assert _to_or_query("🎉 !!!") == ""

    def test_pathological_mega_tokens_are_dropped(self) -> None:
        """A single >2KB "word" (pasted hash, base64 dump) makes to_tsquery
        raise at the database — a 500 for what is really bad input. Such
        tokens carry no lexical signal, so they are dropped at the boundary;
        the remaining words still query, and an all-mega-token question
        degrades to the dense leg like an all-stop-word one."""
        mega = "x" * 5000

        assert _to_or_query(f"refund {mega} policy") == "refund | policy"
        assert _to_or_query(mega) == ""

    def test_cjk_tokens_are_capped_by_bytes_not_characters(self) -> None:
        # The PostgreSQL lexeme limit is in BYTES: 700 CJK characters (~2100
        # bytes UTF-8) must be dropped even though 700 < 2000 characters.
        mega_cjk = "返" * 700
        assert _to_or_query(f"refund {mega_cjk}") == "refund"
