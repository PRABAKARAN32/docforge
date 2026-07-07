"""Chunking: split a page's markdown into small pieces for embedding.

Embedding models have an input-size limit, and retrieval is more accurate on focused
pieces than on whole pages, so each page is split into overlapping chunks.

Every chunk carries its ``source_url`` (Decision 5.3): this is the stable link from a chunk
back to its page, so that when a page changes we can delete *exactly* that page's chunks
before re-inserting the new ones. Getting this right is what keeps the vector store free of
orphaned/duplicated chunks.

Strategy: split on blank-line (paragraph) boundaries, greedily pack paragraphs up to
``max_chars``, and carry a small ``overlap_chars`` tail into the next chunk so context isn't
cut mid-thought. Char-based for now; token-based sizing is a possible refinement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class Chunk:
    """One embeddable piece of a page.

    ``source_url`` links it back to its page; ``index`` is its 0-based position within the
    page (so chunks are ordered and identifiable). ``text`` is the chunk content.
    """

    source_url: str
    index: int
    text: str


def chunk_markdown(
    markdown: str,
    *,
    source_url: str,
    max_chars: int = 1200,
    overlap_chars: int = 150,
) -> list[Chunk]:
    """Split ``markdown`` into overlapping chunks, each tagged with ``source_url``.

    Paragraphs are packed together up to ``max_chars``; a paragraph longer than ``max_chars``
    on its own is hard-split. Each new chunk (after the first) begins with the last
    ``overlap_chars`` characters of the previous one, so meaning that spans a boundary is not
    lost. Returns ``[]`` for empty/whitespace-only input.
    """
    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT.split(markdown) if p.strip()]

    texts: list[str] = []
    buffer = ""

    def flush() -> None:
        nonlocal buffer
        if buffer.strip():
            texts.append(buffer.strip())
        buffer = ""

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            # A single oversized paragraph: flush what we have, then hard-split it.
            flush()
            for start in range(0, len(paragraph), max_chars):
                texts.append(paragraph[start : start + max_chars].strip())
            continue

        if buffer and len(buffer) + len(paragraph) + 2 > max_chars:
            previous = buffer
            flush()
            tail = previous[-overlap_chars:] if overlap_chars else ""
            buffer = f"{tail}\n\n{paragraph}" if tail else paragraph
        else:
            buffer = paragraph if not buffer else f"{buffer}\n\n{paragraph}"

    flush()

    return [Chunk(source_url=source_url, index=i, text=text) for i, text in enumerate(texts)]
