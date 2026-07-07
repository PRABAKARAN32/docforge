"""Embedding: turn chunk text into vectors, behind a pluggable interface.

An *embedding* is a list of numbers representing a piece of text's meaning; similar texts
get nearby vectors. The vector store searches over these.

We define ``Embedder`` as a Protocol (a contract), not a concrete class, so the rest of
DocForge depends on the *interface* -- ``.dimension`` and ``.embed(texts)`` -- and any
implementation can be swapped in (Decision 5.2, pluggable interfaces). The default
implementation is fastembed: lightweight local ONNX models, no PyTorch.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

# A small, fast, good-quality English model (384-dim). Runs locally on CPU.
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_DEVICE = "auto"  # auto = use GPU if available, else CPU


@runtime_checkable
class Embedder(Protocol):
    """The contract every embedder must satisfy.

    ``dimension`` is the length of each vector (the vector store needs it up front to size
    its collection). ``embed`` turns a batch of texts into one vector each, in order.
    """

    @property
    def dimension(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def _cuda_available() -> bool:
    """True if the ONNX runtime reports a usable CUDA (GPU) execution provider."""
    try:
        import onnxruntime

        return "CUDAExecutionProvider" in onnxruntime.get_available_providers()
    except Exception:  # noqa: BLE001 -- any import/runtime issue just means "no GPU"
        return False


def _want_cuda(device: str, *, cuda_available: bool) -> bool:
    """Decide whether to *attempt* GPU, from the requested device. Pure + testable.

    ``cpu`` -> never; ``cuda`` -> always attempt (caller handles fallback if it fails);
    ``auto`` -> attempt only if a GPU looks available.
    """
    normalized = device.lower()
    if normalized == "cpu":
        return False
    if normalized == "cuda":
        return True
    if normalized == "auto":
        return cuda_available
    raise ValueError(f"unknown device {device!r} (use auto, cpu, or cuda)")


class FastEmbedEmbedder:
    """Local embeddings via fastembed (ONNX). Downloads the model once on first use.

    ``device`` selects the compute backend: ``auto`` (GPU if available, else CPU),
    ``cpu``, or ``cuda``. If GPU is requested/detected but can't actually be used, it
    falls back to CPU with a warning -- so it never crashes for lack of a GPU.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, *, device: str = DEFAULT_DEVICE) -> None:
        # Imported lazily: fastembed pulls in ONNX runtime, so importing this module stays
        # cheap and only paid when an embedder is actually created.
        from fastembed import TextEmbedding

        self._model_name = model_name
        self._dimension: int | None = None

        if _want_cuda(device, cuda_available=_cuda_available()):
            try:
                self._model = TextEmbedding(model_name=model_name, cuda=True)
                self._device = "cuda"
                return
            except Exception as exc:  # noqa: BLE001 -- GPU setup can fail many ways; fall back
                warnings.warn(
                    f"GPU embedding unavailable ({exc}); falling back to CPU.", stacklevel=2
                )

        self._model = TextEmbedding(model_name=model_name)
        self._device = "cpu"

    @property
    def device(self) -> str:
        """Which backend is actually in use: ``cpu`` or ``cuda``."""
        return self._device

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
