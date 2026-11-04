"""Write paired judge/human precision@k and review coverage."""
import argparse
import json
from pathlib import Path
import hil_eval as hil
from soundai.hil import PROJECT, precision_metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default=".hil/esc50.db")
    p.add_argument("--project", default=PROJECT)
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--output", type=Path, default=Path(".hil/esc50-metrics.json"))
    a = p.parse_args()
    hil.init(project=a.project, store=f"sqlite:///{a.db}")
    report = precision_metrics(hil.list_items(), hil.export(fmt="eval-set", include_images=False), a.k)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{'Query':45} {'Judge':>7} {'Human':>7} {'Coverage':>9}")
    for row in report["queries"]:
        fmt = lambda v: f"{v:.3f}" if v is not None else "n/a"
        print(f"{row['query']:45} {fmt(row['judge_precision']):>7} {fmt(row['human_precision']):>7} {row['coverage']:9.1%}")
    print("Incomplete coverage means precision is on reviewed ranks only.")
    print(f"Saved {a.output}")


if __name__ == "__main__":
    main()
