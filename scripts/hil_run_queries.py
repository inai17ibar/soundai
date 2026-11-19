"""Queue individual ESC-50 search results through the hil-eval decorator."""
import argparse
import json
from pathlib import Path
import hil_eval as hil
from soundai.hil import PROJECT, reviewed_search


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default=".hil/esc50.db")
    p.add_argument("--project", default=PROJECT)
    p.add_argument("--queries", type=Path, default=Path("data/hil_queries.json"))
    p.add_argument("--index-dir", default="data/embeddings/esc50")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--threshold", type=float, default=0.8)
    p.add_argument("--audit-rate", type=float, default=0.05)
    a = p.parse_args()
    from soundai.search import SampleSearcher
    Path(a.db).parent.mkdir(parents=True, exist_ok=True)
    hil.init(project=a.project, store=f"sqlite:///{a.db}")
    from hil_eval.store.sqlite import SQLiteStore
    # Persist the routing threshold so `hil-eval tune` weights samples consistently.
    store = SQLiteStore(a.db)
    store.ensure_project(a.project)
    store.update_threshold(a.project, a.threshold)
    searcher = SampleSearcher(index_dir=a.index_dir)
    before = {i.id for i in hil.list_items()}
    for q in json.loads(a.queries.read_text()):
        reviewed_search(searcher, q["query"], a.k, project=a.project,
                        tags={"query_set": "esc50-demo-v1"}, threshold=a.threshold, audit_rate=a.audit_rate)
    for item in hil.list_items():
        if item.id not in before:
            print(item.id, item.input["query"], item.input["rank"], item.confidence)


if __name__ == "__main__":
    main()
