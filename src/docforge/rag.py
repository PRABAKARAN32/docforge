"""RAG sync: turn a change-detection result into vector-store updates.

This is the M2 payoff -- it connects M1's diff to the vector store, touching only what
changed:

  * pages that CHANGED or were DELETED  -> delete their old chunks (by source_url)
  * pages that are NEW or CHANGED        -> chunk -> embed -> upsert their new chunks
  * pages that are UNCHANGED             -> nothing (the whole point)

Pages are processed one at a time (Notes/M2: stream, don't merge), so only a single page's
chunks are ever held in memory during embedding, and every chunk keeps its ``source_url``.
"""

from __future__ import annotations

from docforge.chunking import chunk_markdown
from docforge.detector import ChangeDetectionResult
from docforge.diff import deletions_to_apply
from docforge.embedder import Embedder
from docforge.vectorstore import VectorStore


def embed_changes(
    result: ChangeDetectionResult,
    embedder: Embedder,
    store: VectorStore,
) -> None:
    """Apply a detection result to the vector store: delete stale chunks, embed new ones.

    Deletions respect the crawl-success guard (Decision 5.5): a partial crawl never deletes
    a page's chunks just because it looked missing.
    """
    store.ensure_collection(embedder.dimension)

    report = result.report
    guarded_deletes = deletions_to_apply(report, crawl_succeeded=result.crawl_succeeded)

    # Remove old chunks for pages that changed or were (safely) deleted.
    for source_url in report.changed | guarded_deletes:
        store.delete_by_source_url(source_url)

    # Embed and upsert new content for new/changed pages, one page at a time.
    for source_url in report.new | report.changed:
        chunks = chunk_markdown(result.pages[source_url], source_url=source_url)
        if not chunks:
            continue
        vectors = embedder.embed([chunk.text for chunk in chunks])
        store.upsert_chunks(chunks, vectors)
