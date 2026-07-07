"""Vector store: persist chunk embeddings and delete them by page, behind an interface.

The vector store holds one point per chunk: its embedding vector plus a payload carrying
``source_url`` (the page it came from), ``chunk_index``, and ``text``.

The make-or-break operation is :meth:`delete_by_source_url` (Decision 5.3): because each point
is tagged with its page URL, removing a changed/deleted page's chunks is a single filtered
delete. Sync = "delete this page's old chunks, then upsert its new ones."

``VectorStore`` is a Protocol (a contract), so Qdrant can be swapped for another store without
touching the rest of DocForge (Decision 5.2). The default implementation is Qdrant.
"""

from __future__ import annotations

import uuid
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from docforge.chunking import Chunk

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

# Fixed namespace so a chunk's point id is a stable function of (source_url, index):
# re-embedding the same chunk maps to the same id (idempotent upserts).
_POINT_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")


@dataclass(frozen=True)
class SearchHit:
    """One retrieval result: which page, the chunk text, and the similarity score."""

    source_url: str
    text: str
    score: float


@runtime_checkable
class VectorStore(Protocol):
    """The contract every vector store must satisfy."""

    def ensure_collection(self, dimension: int) -> None: ...
    def upsert_chunks(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None: ...
    def delete_by_source_url(self, source_url: str) -> None: ...
    def count(self) -> int: ...
    def search(self, vector: Sequence[float], *, limit: int = 5) -> list[SearchHit]: ...


def _point_id(chunk: Chunk) -> str:
    return str(uuid.uuid5(_POINT_NAMESPACE, f"{chunk.source_url}#{chunk.index}"))


class QdrantVectorStore:
    """Qdrant-backed vector store.

    Pass ``client`` to inject a connection (tests use ``QdrantClient(location=":memory:")``
    for a real in-process Qdrant with no Docker); otherwise it connects to ``url``.
    """

    def __init__(
        self,
        *,
        client: QdrantClient | None = None,
        url: str = "http://localhost:6333",
        path: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
        collection: str = "docforge",
    ) -> None:
        # Three ways to connect, in priority order:
        #   client=...  -> injected (tests use QdrantClient(location=":memory:"))
        #   path=...    -> embedded on-disk Qdrant, in-process, NO server/Docker (like SQLite)
        #   url=...     -> a Qdrant server (Docker, native, or remote/Qdrant Cloud with api_key)
        # timeout (seconds) applies to server requests -- generous by default so confirmed
        # writes to a distant/remote cluster don't time out (the client default is ~5s).
        if client is None:
            from qdrant_client import QdrantClient as _QdrantClient

            if path is not None:
                client = _QdrantClient(path=path)
            else:
                client = _QdrantClient(url=url, api_key=api_key, timeout=timeout)
        self._client = client
        self._collection = collection

    def ensure_collection(self, dimension: int) -> None:
        from qdrant_client import models

        if not self._client.collection_exists(self._collection):
            self._client.create_collection(
                self._collection,
                vectors_config=models.VectorParams(
                    size=dimension, distance=models.Distance.COSINE
                ),
            )
            # Index source_url so delete/filter by page is fast. In embedded (local) mode this
            # is a harmless no-op that emits a warning -- suppress just that noise.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._client.create_payload_index(
                    self._collection,
                    field_name="source_url",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )

    def upsert_chunks(
        self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]
    ) -> None:
        from qdrant_client import models

        points = [
            models.PointStruct(
                id=_point_id(chunk),
                vector=list(vector),
                payload={
                    "source_url": chunk.source_url,
                    "chunk_index": chunk.index,
                    "text": chunk.text,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        if points:
            self._client.upsert(self._collection, points=points)

    def delete_by_source_url(self, source_url: str) -> None:
        from qdrant_client import models

        self._client.delete(
            self._collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source_url", match=models.MatchValue(value=source_url)
                        )
                    ]
                )
            ),
        )

    def count(self) -> int:
        return self._client.count(self._collection, exact=True).count

    def close(self) -> None:
        """Release the client. Matters in embedded (path) mode, which locks its folder."""
        self._client.close()

    def search(self, vector: Sequence[float], *, limit: int = 5) -> list[SearchHit]:
        response = self._client.query_points(
            self._collection, query=list(vector), limit=limit, with_payload=True
        )
        return [
            SearchHit(
                source_url=point.payload["source_url"],
                text=point.payload["text"],
                score=point.score,
            )
            for point in response.points
        ]
