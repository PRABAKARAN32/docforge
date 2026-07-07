"""Tests for the embedder.

The contract is tested with a tiny fake (fast, no model). The real fastembed model is
opt-in via DOCFORGE_MODEL_TESTS because the first run downloads a ~90MB model.
"""

import os
from collections.abc import Sequence

import pytest

from docforge.embedder import Embedder, _want_cuda

_MODEL_TESTS = os.getenv("DOCFORGE_MODEL_TESTS")


class FakeEmbedder:
    """A trivial embedder satisfying the Embedder protocol, for fast tests."""

    def __init__(self, dim: int = 3) -> None:
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(len(t))] * self._dim for t in texts]


def test_fake_embedder_satisfies_the_protocol() -> None:
    assert isinstance(FakeEmbedder(), Embedder)  # structural (runtime_checkable) check


def test_embed_returns_one_vector_per_text_of_the_right_dimension() -> None:
    emb = FakeEmbedder(dim=4)
    vectors = emb.embed(["a", "bb", "ccc"])
    assert len(vectors) == 3
    assert all(len(v) == emb.dimension for v in vectors)


def test_want_cuda_device_logic() -> None:
    # cpu: never; cuda: always attempt (fallback handled at construction); auto: only if available
    assert _want_cuda("cpu", cuda_available=True) is False
    assert _want_cuda("cpu", cuda_available=False) is False
    assert _want_cuda("cuda", cuda_available=False) is True
    assert _want_cuda("cuda", cuda_available=True) is True
    assert _want_cuda("auto", cuda_available=True) is True
    assert _want_cuda("auto", cuda_available=False) is False


def test_want_cuda_rejects_unknown_device() -> None:
    with pytest.raises(ValueError, match="unknown device"):
        _want_cuda("tpu", cuda_available=True)


@pytest.mark.skipif(not _MODEL_TESTS, reason="set DOCFORGE_MODEL_TESTS=1 (downloads a model)")
def test_fastembed_device_falls_back_to_cpu_without_gpu() -> None:
    from docforge.embedder import FastEmbedEmbedder

    # This machine has no GPU: auto and forced-cpu both end up on CPU (no crash).
    assert FastEmbedEmbedder(device="cpu").device == "cpu"
    assert FastEmbedEmbedder(device="auto").device == "cpu"


@pytest.mark.skipif(not _MODEL_TESTS, reason="set DOCFORGE_MODEL_TESTS=1 (downloads a model)")
def test_fastembed_produces_real_vectors() -> None:
    from docforge.embedder import DEFAULT_MODEL, FastEmbedEmbedder

    emb = FastEmbedEmbedder(DEFAULT_MODEL)
    assert emb.dimension == 384  # bge-small-en-v1.5

    vectors = emb.embed(["hello world", "hello world"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 384
    assert vectors[0] == vectors[1]  # identical text -> identical vector (deterministic)
