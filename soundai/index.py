"""FAISS-based nearest-neighbour index for audio embeddings.

Embeddings are assumed to be L2-normalised, so inner product == cosine
similarity and IndexFlatIP gives exact cosine nearest neighbours.
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss  # type: ignore
import numpy as np


class EmbeddingIndex:
    """Stores audio embeddings and metadata for fast similarity search.

    Parameters
    ----------
    dim:
        Embedding dimensionality (512 for CLAP).
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(dim)
        self._paths: list[str] = []

    # ------------------------------------------------------------------
    # Building the index
    # ------------------------------------------------------------------
    def add(self, embeddings: np.ndarray, paths: list[str]) -> None:
        """Add embeddings with corresponding file paths."""
        if embeddings.shape[0] != len(paths):
            raise ValueError("embeddings and paths must have the same length")
        self._index.add(embeddings.astype(np.float32))
        self._paths.extend(paths)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------
    def search(
        self, query: np.ndarray, k: int = 10
    ) -> list[dict[str, float | str]]:
        """Return top-k most similar samples.

        Parameters
        ----------
        query:
            (1, dim) or (dim,) query embedding (L2-normalised).
        k:
            Number of results to return.

        Returns
        -------
        List of dicts with keys ``path`` and ``score`` (cosine similarity).
        """
        q = np.atleast_2d(query).astype(np.float32)
        scores, indices = self._index.search(q, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({"path": self._paths[idx], "score": float(score)})
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, directory: Path | str) -> None:
        """Save index and metadata to *directory*."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(d / "index.faiss"))
        (d / "paths.json").write_text(json.dumps(self._paths, ensure_ascii=False))

    @classmethod
    def load(cls, directory: Path | str) -> "EmbeddingIndex":
        """Load a previously saved index."""
        d = Path(directory)
        obj = cls.__new__(cls)
        obj._index = faiss.read_index(str(d / "index.faiss"))
        obj.dim = obj._index.d
        obj._paths = json.loads((d / "paths.json").read_text())
        return obj

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def size(self) -> int:
        return self._index.ntotal
