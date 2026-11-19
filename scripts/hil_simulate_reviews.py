"""Simulate human labels from ESC-50 metadata via the review HTTP API."""
import argparse
import csv
import json
from pathlib import Path
import hil_eval as hil
import httpx
from soundai.hil import PROJECT, HilClient, simulate_reviews


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default=".hil/esc50.db")
    p.add_argument("--project", default=PROJECT)
    p.add_argument("--url", default="http://127.0.0.1:8765")
    p.add_argument("--queries", type=Path, default=Path("data/hil_queries.json"))
    p.add_argument("--meta", type=Path, default=Path("data/samples/esc50/meta.csv"))
    p.add_argument("--limit", type=int, default=None,
                   help="review only the first N pending items (leave the rest for the UI)")
    a = p.parse_args()
    hil.init(project=a.project, store=f"sqlite:///{a.db}")
    with a.meta.open() as f:
        categories = {r["filename"]: r["category"] for r in csv.DictReader(f)}
    targets = {q["query"]: q["category"] for q in json.loads(a.queries.read_text())}
    with httpx.Client(base_url=a.url, timeout=30) as client:
        print("Reviewed:", simulate_reviews(hil.list_items()[: a.limit], categories, targets, HilClient(client), a.project))


if __name__ == "__main__":
    main()
