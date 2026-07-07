"""Tests for RAG sync (embed_changes), using a fake embedder + in-process Qdrant."""

from collections.abc import Sequence

from docforge.detector import ChangeDetectionResult
from docforge.diff import DiffReport
from docforge.rag import embed_changes
from docforge.vectorstore import QdrantVectorStore


class FakeEmbedder:
    @property
    def dimension(self) -> int:
        return 4

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(len(t)), 1.0, 0.0, 0.0] for t in texts]


def memory_store() -> QdrantVectorStore:
    from qdrant_client import QdrantClient

    return QdrantVectorStore(client=QdrantClient(location=":memory:"), collection="test")


def make_result(
    *, new=(), changed=(), deleted=(), pages=None, crawl_succeeded=True
) -> ChangeDetectionResult:
    report = DiffReport(
        new=frozenset(new), changed=frozenset(changed), deleted=frozenset(deleted), unchanged=frozenset()
    )
    return ChangeDetectionResult(
        report=report, current_hashes={}, crawl_succeeded=crawl_succeeded, pages=pages or {}
    )


def test_new_pages_are_embedded() -> None:
    store = memory_store()
    result = make_result(new=["https://d/a"], pages={"https://d/a": "# A\n\nAlpha content."})
    embed_changes(result, FakeEmbedder(), store)

    assert store.count() >= 1
    hits = store.search([1.0, 1.0, 0.0, 0.0], limit=5)
    assert any(h.source_url == "https://d/a" for h in hits)


def test_changed_page_replaces_old_chunks() -> None:
    store = memory_store()
    emb = FakeEmbedder()

    embed_changes(make_result(new=["https://d/a"], pages={"https://d/a": "original alpha"}), emb, store)
    embed_changes(
        make_result(changed=["https://d/a"], pages={"https://d/a": "brand new beta text"}), emb, store
    )

    texts = " ".join(h.text for h in store.search([1.0, 1.0, 0.0, 0.0], limit=50))
    assert "beta" in texts
    assert "original alpha" not in texts  # old chunks were deleted


def test_deleted_page_removed_when_crawl_succeeded() -> None:
    store = memory_store()
    emb = FakeEmbedder()

    embed_changes(
        make_result(new=["https://d/a", "https://d/b"], pages={"https://d/a": "aaa", "https://d/b": "bbb"}),
        emb,
        store,
    )
    embed_changes(make_result(deleted=["https://d/a"], crawl_succeeded=True), emb, store)

    hits = store.search([1.0, 1.0, 0.0, 0.0], limit=50)
    assert all(h.source_url != "https://d/a" for h in hits)


def test_deletion_guard_keeps_chunks_when_crawl_failed() -> None:
    store = memory_store()
    emb = FakeEmbedder()

    embed_changes(make_result(new=["https://d/a"], pages={"https://d/a": "aaa"}), emb, store)
    before = store.count()
    embed_changes(make_result(deleted=["https://d/a"], crawl_succeeded=False), emb, store)

    assert store.count() == before  # guard: not deleted after a failed crawl
