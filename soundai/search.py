"""High-level search interface combining CLAPEmbedder and EmbeddingIndex."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from soundai.embed import CLAPEmbedder
from soundai.index import EmbeddingIndex

SearchResults = list[dict[str, float | str]]


class SampleSearcher:
    """One-stop interface for text and audio similarity search.

    Parameters
    ----------
    index_dir:
        Directory containing a saved EmbeddingIndex.  If ``None``, an
        empty index is created (useful for building from scratch).
    embedder:
        Optional pre-built CLAPEmbedder instance.  A new one is created
        if omitted.
    """

    def __init__(
        self,
        index_dir: Path | str | None = None,
        embedder: CLAPEmbedder | None = None,
    ) -> None:
        self.embedder = embedder or CLAPEmbedder()
        if index_dir is not None:
            self.index = EmbeddingIndex.load(index_dir)
        else:
            self.index = EmbeddingIndex()

    # ------------------------------------------------------------------
    # Search modes
    # ------------------------------------------------------------------
    def search_by_text(self, query: str, k: int = 10) -> SearchResults:
        """Find samples matching a natural-language description.

        Example
        -------
        >>> results = searcher.search_by_text("bright acoustic guitar strum")
        """
        emb = self.embedder.embed_texts([query])
        return self.index.search(emb[0], k=k)

    def search_by_file(self, path: Path | str, k: int = 10) -> SearchResults:
        """Find samples acoustically similar to *path*."""
        emb = self.embedder.embed_files([path])
        if emb.shape[0] == 0:
            raise ValueError(f"Could not embed file: {path}")
        return self.index.search(emb[0], k=k)

    def search_by_embedding(
        self, embedding: np.ndarray, k: int = 10
    ) -> SearchResults:
        """Find samples similar to a pre-computed embedding vector."""
        return self.index.search(embedding, k=k)

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------
    def save_index(self, directory: Path | str) -> None:
        """Persist the current index to disk."""
        self.index.save(directory)

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------
    @staticmethod
    def pretty_print(results: SearchResults, max_rows: int = 20) -> None:
        """Print results in a readable table."""
        print(f"{'Score':>6}  Path")
        print("-" * 60)
        for r in results[:max_rows]:
            print(f"{r['score']:>6.3f}  {Path(str(r['path'])).name}")
