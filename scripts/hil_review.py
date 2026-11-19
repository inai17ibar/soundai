"""Run example text queries through hil-eval review, then serve or export.

  uv run --extra hil python scripts/hil_review.py run   [--real --index-dir DIR]
  uv run --extra hil hil-eval --db .hil/hil.db serve      # review at http://127.0.0.1:8765
  uv run --extra hil python scripts/hil_review.py export -o reviews.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import hil_eval as hil

from soundai.hil_review import PROJECT, build_stub_searcher, make_reviewed_search

QUERIES = ["punchy kick drum", "crisp hihat", "deep sub bass", "warm ambient pad", "bright bell hit"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "export"])
    ap.add_argument("--db", default=".hil/hil.db")
    ap.add_argument("--real", action="store_true", help="use real CLAP + a saved index")
    ap.add_argument("--index-dir", default="index")
    ap.add_argument("--samples", default=".hil/samples", help="synthetic wav dir (stub mode)")
    ap.add_argument("-k", type=int, default=3)
    ap.add_argument("--format", default="jsonl", choices=["jsonl", "eval-set", "openai-ft"])
    ap.add_argument("-o", "--output", default="reviews.jsonl")
    a = ap.parse_args()

    Path(a.db).parent.mkdir(parents=True, exist_ok=True)
    hil.init(project=PROJECT, store=f"sqlite:///{a.db}")

    if a.cmd == "export":
        n = 0
        with open(a.output, "w") as f:
            for rec in hil.export(fmt=a.format, include_images=False):
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                n += 1
        print(f"exported {n} reviewed items -> {a.output}")
        return

    if a.real:
        from soundai.search import SampleSearcher

        searcher = SampleSearcher(index_dir=a.index_dir)
    else:
        searcher = build_stub_searcher(a.samples)
    search = make_reviewed_search(searcher)
    for q in QUERIES:
        res = search(q, k=a.k)
        print(f"{q!r}: " + ", ".join(f"{r['name']}({r['score']:.2f})" for r in res))
    print(f"queued: {len(hil.list_items())} items. Review with:\n"
          f"  uv run --extra hil hil-eval --db {a.db} serve")


if __name__ == "__main__":
    main()
