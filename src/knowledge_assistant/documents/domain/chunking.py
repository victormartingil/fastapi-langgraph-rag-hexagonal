"""Chunking: a PURE domain service.

Chunking is the first design decision of any RAG system, and it is pure
business logic: *how* you split a document determines what the retriever can
find later. It has nothing to do with databases or LLMs, so it lives in the
domain and is unit-tested without any infrastructure.

Strategy used here — deliberately simple and didactic:
split on paragraph boundaries, then greedily merge paragraphs into chunks of
at most `max_chars`, with `overlap_chars` of trailing context carried into the
next chunk so that sentences cut at a boundary still appear complete somewhere.
The overlap is carried on BOTH split paths — paragraph boundaries and the
hard-split of an over-long paragraph — and a trailing "sliver" chunk (less
than a quarter of `max_chars`) is merged into its predecessor: a 20-character
chunk costs an embedding call and a vector row but retrieves nothing useful.
`max_chars` is therefore a SOFT ceiling: the sliver merge may exceed it by up
to the floor.

Production systems swap this for token-aware or semantic chunking; the port of
this module (a single function) would not change.
"""

from knowledge_assistant.documents.domain.value_objects import ChunkText

# A trailing chunk smaller than this fraction of max_chars is a "sliver":
# merged into the previous chunk instead of standing alone.
_SLIVER_FRACTION = 4


def chunk_text(text: str, max_chars: int = 800, overlap_chars: int = 120) -> list[ChunkText]:
    """Split `text` into overlapping chunks of at most `max_chars` characters.

    The split prefers paragraph boundaries; a single paragraph longer than
    `max_chars` is hard-split on character count (last resort). Both paths
    carry `overlap_chars` of trailing context into the next chunk, and a
    trailing sliver (< max_chars // 4) is merged into its predecessor —
    see the module docstring.

    Empty (or whitespace-only) text is not an error: it returns an empty
    list — the caller (extraction pipeline) rejects empty documents earlier,
    with a domain error.

    Raises:
        ValueError: if parameters are inconsistent.
    """
    if max_chars <= 0:
        msg = "max_chars must be positive"
        raise ValueError(msg)
    if overlap_chars < 0 or overlap_chars >= max_chars:
        msg = "overlap_chars must be >= 0 and < max_chars"
        raise ValueError(msg)

    cleaned = text.strip()
    if not cleaned:
        return []

    paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
            # Carry a tail of the previous chunk into the next one so context
            # is not lost at the boundary.
            overlap = current[-overlap_chars:] if overlap_chars else ""
            current = f"{overlap}\n\n{paragraph}" if overlap else paragraph
        else:
            current = paragraph
        # A single paragraph that still does not fit is hard-split. The tail
        # is carried here too: resuming exactly at the cut would split a word
        # or sentence with no overlap to stitch it back together.
        while len(current) > max_chars:
            head = current[:max_chars]
            chunks.append(head)
            tail = head[-overlap_chars:] if overlap_chars else ""
            current = tail + current[max_chars:]
    if current:
        chunks.append(current)

    # Sliver guard: a tiny trailing chunk costs an embedding call and a vector
    # row but retrieves nothing useful — merge it into its predecessor.
    if len(chunks) >= 2 and len(chunks[-1]) < max_chars // _SLIVER_FRACTION:
        chunks[-2] = f"{chunks[-2]}\n\n{chunks[-1]}"
        chunks.pop()

    return [ChunkText(chunk) for chunk in chunks]
