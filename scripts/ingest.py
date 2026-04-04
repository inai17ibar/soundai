"""Ingest audio files and build a searchable FAISS index.

Usage
-----
    uv run python scripts/ingest.py --samples data/samples --out data/embeddings

The script scans *samples* recursively, generates CLAP embeddings in
batches, and saves the index to *out*.  Re-running is incremental: files
already present in the existing index are skipped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from soundai.embed import CLAPEmbedder, SUPPORTED_EXTENSIONS
from soundai.index import EmbeddingIndex


def collect_audio_files(root: Path) -> list[Path]:
    files = [
        p for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return files


def load_existing_paths(out_dir: Path) -> set[str]:
    paths_file = out_dir / "paths.json"
    if paths_file.exists():
        return set(json.loads(paths_file.read_text()))
    return set()


def ingest(
    samples_dir: Path,
    out_dir: Path,
    batch_size: int = 32,
) -> None:
    print(f"Scanning {samples_dir} …")
    all_files = collect_audio_files(samples_dir)
    print(f"Found {len(all_files)} audio files")

    # Load existing index if present
    if (out_dir / "index.faiss").exists():
        print("Loading existing index …")
        index = EmbeddingIndex.load(out_dir)
        existing = set(index._paths)
    else:
        index = EmbeddingIndex()
        existing = set()

    new_files = [f for f in all_files if str(f) not in existing]
    if not new_files:
        print("Nothing new to ingest.")
        return
    print(f"New files to embed: {len(new_files)}")

    embedder = CLAPEmbedder()

    for i in tqdm(range(0, len(new_files), batch_size), desc="Embedding"):
        batch = new_files[i : i + batch_size]
        embeddings = embedder.embed_files(batch)
        if embeddings.shape[0] == 0:
            continue
        # Only keep files that were successfully embedded
        valid_paths = [
            str(p) for p in batch
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        ][: embeddings.shape[0]]
        index.add(embeddings, valid_paths)

    index.save(out_dir)
    print(f"Index saved to {out_dir}  ({index.size} total entries)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest audio files into FAISS index")
    parser.add_argument(
        "--samples",
        type=Path,
        default=Path("data/samples"),
        help="Root directory containing audio files",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/embeddings"),
        help="Directory to save the FAISS index",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of files per embedding batch",
    )
    args = parser.parse_args()
    ingest(args.samples, args.out, args.batch_size)


if __name__ == "__main__":
    main()
