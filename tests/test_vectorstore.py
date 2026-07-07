"""Tests for the Qdrant vector store.

These run against a real in-process Qdrant (`location=":memory:"`) -- no Docker needed --
so they exercise the actual upsert/delete/search logic, fast and deterministically.
"""

import pytest

from docforge.chunking import Chunk
from docforge.vectorstore import QdrantVectorStore, VectorStore

DIM = 4


@pytest.fixture
def store() -> QdrantVectorStore:
    from qdrant_client import QdrantClient

    vs = QdrantVectorStore(client=QdrantClient(location=":memory:"), collection="test")
    vs.ensure_collection(DIM)
    return vs


def _chunks(source_url: str, n: int) -> list[Chunk]:
    return [Chunk(source_url=source_url, index=i, text=f"{source_url} chunk {i}") for i in range(n)]


def _vecs(n: int) -> list[list[float]]:
    # Distinct simple unit-ish vectors; exact values don't matter for these tests.
    return [[float(i + 1), 0.0, 0.0, 0.0] for i in range(n)]


def test_qdrant_store_satisfies_the_protocol(store: QdrantVectorStore) -> None:
    assert isinstance(store, VectorStore)


def test_upsert_then_count(store: QdrantVectorStore) -> None:
    chunks = _chunks("https://d/a", 3)
    store.upsert_chunks(chunks, _vecs(3))
    assert store.count() == 3


def test_upsert_is_idempotent(store: QdrantVectorStore) -> None:
    chunks = _chunks("https://d/a", 2)
    store.upsert_chunks(chunks, _vecs(2))
    store.upsert_chunks(chunks, _vecs(2))  # same chunks again -> same ids, no duplicates
    assert store.count() == 2


def test_delete_by_source_url_removes_only_that_page(store: QdrantVectorStore) -> None:
    store.upsert_chunks(_chunks("https://d/a", 3), _vecs(3))
    store.upsert_chunks(_chunks("https://d/b", 2), _vecs(2))
    assert store.count() == 5

    store.delete_by_source_url("https://d/a")

    assert store.count() == 2  # only page b's chunks remain
    hits = store.search([1.0, 0.0, 0.0, 0.0], limit=10)
    assert hits
    assert all(hit.source_url == "https://d/b" for hit in hits)


def test_search_returns_payload(store: QdrantVectorStore) -> None:
    store.upsert_chunks(_chunks("https://d/a", 1), _vecs(1))
    hits = store.search([1.0, 0.0, 0.0, 0.0], limit=1)
    assert len(hits) == 1
    assert hits[0].source_url == "https://d/a"
    assert "chunk 0" in hits[0].text


def test_embedded_path_mode_persists_to_disk(tmp_path) -> None:
    # No Docker, no server: Qdrant embedded on disk (like SQLite). Data must survive reopen.
    path = str(tmp_path / "vectors")

    first = QdrantVectorStore(path=path, collection="test")
    first.ensure_collection(DIM)
    first.upsert_chunks(_chunks("https://d/a", 2), _vecs(2))
    assert first.count() == 2
    first.close()  # release the folder lock

    reopened = QdrantVectorStore(path=path, collection="test")
    assert reopened.count() == 2  # persisted on disk across separate clients
    reopened.close()
