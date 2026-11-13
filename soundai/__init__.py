"""soundai — audio sample search.  Heavy deps (CLAP, FAISS) load lazily."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from soundai.embed import CLAPEmbedder
    from soundai.index import EmbeddingIndex
    from soundai.search import SampleSearcher

__all__ = ["CLAPEmbedder", "EmbeddingIndex", "SampleSearcher"]


def __getattr__(name: str) -> Any:
    if name == "CLAPEmbedder":
        from soundai.embed import CLAPEmbedder

        return CLAPEmbedder
    if name == "EmbeddingIndex":
        from soundai.index import EmbeddingIndex

        return EmbeddingIndex
    if name == "SampleSearcher":
        from soundai.search import SampleSearcher

        return SampleSearcher
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
