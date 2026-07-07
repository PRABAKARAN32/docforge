"""Unit tests for chunking. Pure function -> fast and deterministic."""

from docforge.chunking import Chunk, chunk_markdown

URL = "https://d/page"


def test_empty_input_yields_no_chunks() -> None:
    assert chunk_markdown("", source_url=URL) == []
    assert chunk_markdown("   \n\n  ", source_url=URL) == []


def test_short_page_is_one_chunk_tagged_with_source_url() -> None:
    chunks = chunk_markdown("# Title\n\nJust a little content.", source_url=URL)
    assert len(chunks) == 1
    assert chunks[0] == Chunk(source_url=URL, index=0, text="# Title\n\nJust a little content.")


def test_paragraphs_pack_together_under_the_limit() -> None:
    md = "Para one.\n\nPara two.\n\nPara three."
    chunks = chunk_markdown(md, source_url=URL, max_chars=1000)
    assert len(chunks) == 1  # all fit in one chunk
    assert "Para one." in chunks[0].text and "Para three." in chunks[0].text


def test_content_splits_into_multiple_indexed_chunks() -> None:
    # Each paragraph ~40 chars; small max forces several chunks.
    md = "\n\n".join(f"Paragraph number {i} with some filler text." for i in range(6))
    chunks = chunk_markdown(md, source_url=URL, max_chars=80, overlap_chars=0)

    assert len(chunks) > 1
    assert [c.index for c in chunks] == list(range(len(chunks)))  # 0,1,2,...
    assert all(c.source_url == URL for c in chunks)  # every chunk keeps its page link


def test_oversized_paragraph_is_hard_split() -> None:
    md = "x" * 250  # one paragraph, no blank lines, bigger than max
    chunks = chunk_markdown(md, source_url=URL, max_chars=100, overlap_chars=0)
    assert len(chunks) == 3  # 100 + 100 + 50
    assert "".join(c.text for c in chunks) == md


def test_overlap_carries_context_between_chunks() -> None:
    md = "AAAA.\n\nBBBB.\n\nCCCC.\n\nDDDD."
    chunks = chunk_markdown(md, source_url=URL, max_chars=14, overlap_chars=6)
    assert len(chunks) > 1
    # The 2nd chunk should begin with a tail of the 1st chunk's text (overlap).
    assert chunks[1].text[:4] in chunks[0].text
