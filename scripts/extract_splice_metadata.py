"""Extract rule-based metadata from a Splice path index (no audio is read).

Usage
-----
    uv run python scripts/extract_splice_metadata.py \\
        --input /workspace/soundai-data/splice_index.tsv \\
        --output /workspace/soundai-data/splice_metadata.parquet

The TSV has columns ``size_bytes<TAB>relative_path``.  Paths are relative to
``/Volumes/Extreme SSD/Splice/sounds/packs``.  Nothing is uploaded.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from soundai.metadata.extractor import extract_metadata, should_skip  # noqa: E402
from soundai.metadata.taxonomy import RULES_VERSION  # noqa: E402


def read_index(path: Path) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                size_s, rel = line.split("\t", 1)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: expected size<TAB>path") from exc
            rows.append((int(size_s), rel))
    return rows


def extract_rows(index: list[tuple[int, str]]) -> list[dict]:
    out: list[dict] = []
    for size, rel in index:
        if should_skip(rel):
            continue
        out.append(extract_metadata(rel, size_bytes=size).to_dict())
    return out


def write_output(rows: list[dict], dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    suffix = dest.suffix.lower()
    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            jsonl = dest.with_suffix(".jsonl")
            print(f"pandas/pyarrow not available ({exc}); writing {jsonl} instead", file=sys.stderr)
            return write_output(rows, jsonl)
        df = pd.DataFrame(rows)
        try:
            df.to_parquet(dest, index=False)
        except Exception as exc:  # noqa: BLE001 — fall back to jsonl
            jsonl = dest.with_suffix(".jsonl")
            print(f"parquet write failed ({exc}); writing {jsonl} instead", file=sys.stderr)
            return write_output(rows, jsonl)
        return dest
    if suffix in {".jsonl", ".json"}:
        with dest.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return dest
    raise ValueError(f"unsupported output suffix: {dest.suffix}")


def _pct(n: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{100.0 * n / total:.1f}%"


def coverage_report(rows: list[dict]) -> str:
    total = len(rows)
    lines = [
        f"rules_version={RULES_VERSION}",
        f"files={total}",
    ]
    leaf_known = sum(1 for r in rows if r["leaf"] != "unknown")
    family_known = sum(1 for r in rows if r["family"] != "unknown")
    lines.append(f"leaf instrument assigned: {leaf_known}/{total} ({_pct(leaf_known, total)})")
    lines.append(f"family assigned:         {family_known}/{total} ({_pct(family_known, total)})")

    conf = Counter(r["instrument_conf"] for r in rows)
    lines.append("instrument_conf:")
    for level in ("high", "medium", "low", "none"):
        n = conf.get(level, 0)
        lines.append(f"  {level:8s} {n:5d}  {_pct(n, total)}")

    typ = Counter(r["type"] for r in rows)
    lines.append("type:")
    for key in ("loop", "one_shot", "unknown"):
        n = typ.get(key, 0)
        lines.append(f"  {key:8s} {n:5d}  {_pct(n, total)}")

    bpm_n = sum(1 for r in rows if r["bpm"] is not None)
    key_n = sum(1 for r in rows if r["key_root"])
    lines.append(f"bpm extracted: {bpm_n}/{total} ({_pct(bpm_n, total)})")
    lines.append(f"key extracted: {key_n}/{total} ({_pct(key_n, total)})")

    conflict_n = sum(1 for r in rows if r["conflict"])
    lines.append(f"conflicts: {conflict_n}/{total} ({_pct(conflict_n, total)})")

    leaves = Counter((r["family"], r["leaf"]) for r in rows)
    lines.append("top 15 leaves (family/leaf):")
    for (family, leaf), n in leaves.most_common(15):
        lines.append(f"  {n:5d}  {family}/{leaf}")

    lines.append("10 example unknown/low-confidence paths:")
    examples = [
        r for r in rows
        if r["leaf"] == "unknown" or r["instrument_conf"] in {"low", "none"}
    ]
    # Prefer unknown first, then low
    examples.sort(key=lambda r: (0 if r["leaf"] == "unknown" else 1, r["rel_path"]))
    for r in examples[:10]:
        lines.append(
            f"  [{r['instrument_conf']}] {r['family']}/{r['leaf']}  {r['rel_path']}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="splice_index.tsv")
    parser.add_argument("--output", type=Path, required=True, help=".parquet or .jsonl")
    parser.add_argument("--report", type=Path, default=None, help="optional text report path")
    args = parser.parse_args(argv)

    index = read_index(args.input)
    rows = extract_rows(index)
    written = write_output(rows, args.output)
    report = coverage_report(rows)
    print(f"wrote {len(rows)} rows to {written}")
    print(report)
    if args.report is not None:
        args.report.write_text(report + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
