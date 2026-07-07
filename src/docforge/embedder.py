"""Embedding: turn chunk text into vectors, behind a pluggable interface.

An *embedding* is a list of numbers representing a piece of text's meaning; similar texts
get nearby vectors. The vector store searches over these.

We define ``Embedder`` as a Protocol (a contract), not a concrete class, so the rest of
DocForge depends on the *interface* -- ``.dimension`` and ``.embed(texts)`` -- and any
implementation can be swapped in (Decision 5.2, pluggable interfaces). The default
implementation is fastembed: lightweight local ONNX models, no PyTorch.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

# A small, fast, good-quality English model (384-dim). Runs locally on CPU.
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


@runtime_checkable
class Embedder(Protocol):
    """The contract every embedder must satisfy.

    ``dimension`` is the length of each vector (the vector store needs it up front to size
    its collection). ``embed`` turns a batch of texts into one vector each, in order.
    """

    @property
    def dimension(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class FastEmbedEmbedder:
    """Local embeddings via fastembed (ONNX). Downloads the model once on first use."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        # Imported lazily: fastembed pulls in ONNX runtime, so importing this module stays
        # cheap and only paid when an embedder is actually created.
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)
        self._model_name = model_name
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        # Determine the vector length once by embedding a tiny probe -- works for any model
        # without hard-coding dimensions.
        if self._dimension is None:
            self._dimension = len(self.embed(["dimension probe"])[0])
        return self._dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        # fastembed yields numpy arrays; convert to plain lists so callers/tests don't need numpy.
        return [vector.tolist() for vector in self._model.embed(list(texts))]
