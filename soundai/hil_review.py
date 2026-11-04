"""Human-in-the-loop relevance review for text search, powered by hil-eval.

Each text query becomes one hil-eval review item whose output is the top-k
result list; every result carries its audio file as a ``Path`` so hil-eval
stores it as a content-addressed audio asset that the review UI can play.
"""

from __future__ import annotations

import hashlib
import math
import struct
import wave
from pathlib import Path

import numpy as np

import hil_eval as hil

from soundai.search import SampleSearcher

PROJECT = "soundai-text-search"


def to_review_output(results: list[dict]) -> list[dict]:
    """Convert search results so hil-eval captures each file as an audio asset."""
    out = []
    for rank, r in enumerate(results, start=1):
        p = Path(str(r["path"]))
        out.append({"rank": rank, "name": p.name, "score": round(float(r["score"]), 4), "audio": p})
    return out


def score_judge(arguments: dict, output) -> hil.Judgment:
    """Confidence = top-1 cosine score (crude proxy; humans make the real call)."""
    if not output:
        return hil.Judgment(confidence=0.0, reason="no results")
    top = max(0.0, min(1.0, float(output[0]["score"])))
    return hil.Judgment(confidence=top, reason=f"top1 cosine={top:.3f}")


def make_reviewed_search(searcher: SampleSearcher, threshold: float = 1.0, audit_rate: float = 0.0):
    """Return ``search(query, k)`` wrapped with ``@hil.review``.

    threshold=1.0 routes everything (dogfood mode); lower it to review only
    low-confidence queries.
    """

    @hil.review(judge=score_judge, threshold=threshold, audit_rate=audit_rate, project=PROJECT,
                tags={"mode": "text"})
    def search_text(query: str, k: int = 5) -> list[dict]:
        return to_review_output(searcher.search_by_text(query, k=k))

    return search_text


# ----------------------------------------------------------------------
# Offline stub (no CLAP / no real samples needed)
# ----------------------------------------------------------------------
STUB_SOUNDS = {"kick": 55.0, "snare": 200.0, "hihat": 8000.0, "bass": 80.0, "pad": 440.0, "bell": 1200.0}


class KeywordStubEmbedder:
    """Deterministic CLAPEmbedder stand-in: embeds by keywords in filename/text."""

    DIM = 512

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.DIM, dtype=np.float32)
        for idx, word in enumerate(STUB_SOUNDS):
            if word in text.lower():
                v[idx] += 1.0
        seed = int(hashlib.sha1(text.encode()).hexdigest()[:8], 16)
        v += np.random.default_rng(seed).normal(0, 0.01, self.DIM).astype(np.float32)
        return v / (np.linalg.norm(v) or 1.0)

    def embed_texts(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_files(self, paths):
        return np.stack([self._vec(Path(p).stem) for p in paths]) if paths else np.empty((0, self.DIM), np.float32)


def write_synthetic_wavs(directory: Path | str, seconds: float = 0.5, sr: int = 16000) -> list[Path]:
    """Write tiny sine-tone WAVs (2 variants per sound name)."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, freq in STUB_SOUNDS.items():
        for variant in (1, 2):
            p = d / f"{name}_{variant:02d}.wav"
            n = int(seconds * sr)
            frames = b"".join(
                struct.pack("<h", int(12000 * math.sin(2 * math.pi * freq * variant * i / sr) * (1 - i / n)))
                for i in range(n)
            )
            with wave.open(str(p), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sr)
                w.writeframes(frames)
            paths.append(p)
    return paths


def build_stub_searcher(sample_dir: Path | str) -> SampleSearcher:
    embedder = KeywordStubEmbedder()
    searcher = SampleSearcher(embedder=embedder)  # type: ignore[arg-type]
    paths = write_synthetic_wavs(sample_dir)
    searcher.index.add(embedder.embed_files(paths), [str(p.resolve()) for p in paths])
    return searcher
