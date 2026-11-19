"""Build a hil-eval label-review subset from the Splice metadata catalog.

Usage
-----
    uv run python scripts/build_review_subset.py \\
        --input /workspace/soundai-data/splice_metadata.parquet \\
        --output /workspace/soundai-data/review_subset.jsonl

The JSONL is compatible with ``hil.submit`` (project ``soundai-labels``).
Each row is one file.  Audio lives only on the Mac mini, so items carry the
intended path but do not ingest audio (see docs/hil-eval-label-review-request.md).
Nothing is uploaded.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_INPUT = Path("/workspace/soundai-data/splice_metadata.parquet")
DEFAULT_OUTPUT = Path("/workspace/soundai-data/review_subset.jsonl")
DEFAULT_SEED = 20261114
PROJECT = "soundai-labels"
OPTIONS = ["correct", "wrong_instrument", "wrong_type", "wrong_bpm_key", "unsure"]
AUDIO_ROOT = "/Volumes/Extreme SSD/Splice/sounds/packs"
HIGH_PER_LEAF = 5
NONE_TARGET = 70

# Reasons, in priority order (a row is kept under the first matching reason).
REASON_CONFLICT = "conflict"
REASON_LOW = "low_confidence"
REASON_NONE = "none"
REASON_HIGH = "high_audit"


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".parquet":
        import pandas as pd

        df = pd.read_parquet(path)
        return df.to_dict(orient="records")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _leaf_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("family") or "unknown"), str(row.get("leaf") or "unknown"))


def _stable_rng(seed: int):
    import random

    return random.Random(seed)


def select_review_rows(
    rows: list[dict[str, Any]],
    *,
    seed: int = DEFAULT_SEED,
    high_per_leaf: int = HIGH_PER_LEAF,
    none_target: int = NONE_TARGET,
) -> list[dict[str, Any]]:
    """Return catalog rows tagged with ``review_reason``, 300–400 items typical."""
    chosen: dict[str, dict[str, Any]] = {}  # file_id -> row with review_reason

    def take(row: dict[str, Any], reason: str) -> None:
        fid = row.get("file_id") or row["rel_path"]
        if fid in chosen:
            return
        tagged = dict(row)
        tagged["review_reason"] = reason
        chosen[fid] = tagged

    # 1. Every remaining conflict (highest priority).
    for row in rows:
        if row.get("conflict"):
            take(row, REASON_CONFLICT)

    # 2. Every low-confidence row not already included.
    for row in rows:
        if row.get("instrument_conf") == "low":
            take(row, REASON_LOW)

    # 3. Stratified sample of high-confidence rows: ~5 per family/leaf.
    rng = _stable_rng(seed)
    high_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("instrument_conf") == "high" and not row.get("conflict"):
            high_groups[_leaf_key(row)].append(row)
    for _key, group in sorted(high_groups.items()):
        group = sorted(group, key=lambda r: r.get("rel_path") or "")
        k = min(high_per_leaf, len(group))
        for row in rng.sample(group, k) if k else []:
            take(row, REASON_HIGH)

    # 4. Sample of none-confidence rows, stratified by path pattern.
    none_rows = [
        row for row in rows
        if row.get("instrument_conf") == "none" and not row.get("conflict")
    ]
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in none_rows:
        path = (row.get("rel_path") or "").lower()
        if "songstarter" in path:
            bucket = "songstarter"
        elif "stack" in path:
            bucket = "stack"
        elif "melodic" in path or "melody" in path:
            bucket = "melodic"
        else:
            bucket = "other"
        buckets[bucket].append(row)

    # Split the budget evenly across buckets that have rows, then top up.
    live = {k: v for k, v in buckets.items() if v}
    if live and none_target > 0:
        remaining = none_target
        names = sorted(live)
        for i, name in enumerate(names):
            share = remaining // (len(names) - i)
            group = sorted(live[name], key=lambda r: r.get("rel_path") or "")
            k = min(share, len(group))
            for row in rng.sample(group, k) if k else []:
                take(row, REASON_NONE)
            remaining -= k

    # Stable output order: reason priority, then path.
    reason_rank = {REASON_CONFLICT: 0, REASON_LOW: 1, REASON_NONE: 2, REASON_HIGH: 3}
    selected = list(chosen.values())
    selected.sort(key=lambda r: (reason_rank[r["review_reason"]], r.get("rel_path") or ""))
    return selected


def audio_path_for(rel_path: str) -> str:
    return f"{AUDIO_ROOT}/{rel_path}"


def confidence_for(row: dict[str, Any]) -> float:
    """Queue everything: conflicts/low/none are uncertain; high_audit is a check."""
    reason = row.get("review_reason")
    if reason == REASON_HIGH:
        return 0.35
    if reason == REASON_CONFLICT:
        return 0.15
    if reason == REASON_NONE:
        return 0.10
    return 0.20


def to_hil_record(row: dict[str, Any]) -> dict[str, Any]:
    rel = row["rel_path"]
    proposed = {
        "family": _json_val(row.get("family")),
        "leaf": _json_val(row.get("leaf")),
        "type": _json_val(row.get("type")),
        "bpm": _json_num(row.get("bpm")),
        "key": _json_val(row.get("key")),
        "key_root": _json_val(row.get("key_root")),
        "key_mode": _json_val(row.get("key_mode")),
        "instrument_conf": _json_val(row.get("instrument_conf")),
        "type_conf": _json_val(row.get("type_conf")),
        "conflict": bool(row.get("conflict")),
    }
    return {
        "project": PROJECT,
        "name": "label_review",
        "file_id": row.get("file_id"),
        "input": {
            "rel_path": rel,
            "pack": _json_val(row.get("pack")),
            "audio": audio_path_for(rel),
            "audio_available": False,
            "audio_note": (
                "Audio lives only on the Mac mini and must not be uploaded. "
                "Review is path- and label-based until hil-eval can play a remote path."
            ),
            "proposed": proposed,
            "instrument_source": _json_val(row.get("instrument_source")),
            "type_source": _json_val(row.get("type_source")),
            "matched_tokens": _json_list(row.get("matched_tokens")),
            "matched_rules": _json_list(row.get("matched_rules")),
            "size_bytes": _json_num(row.get("size_bytes")),
            "reason_selected": row["review_reason"],
        },
        "output": {
            "proposed": "correct",
            **proposed,
        },
        "options": list(OPTIONS),
        "confidence": confidence_for(row),
        "tags": {
            "reason": row["review_reason"],
            "family": _json_val(row.get("family")),
            "leaf": _json_val(row.get("leaf")),
            "rules_version": _json_val(row.get("rules_version")),
        },
    }


def _is_na(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, dict, set, str, bytes)):
        return False
    if getattr(value, "shape", None) is not None:
        return False  # numpy / pandas array — not a scalar NA
    if isinstance(value, float):
        import math
        return math.isnan(value)
    if type(value).__name__ in {"NAType", "NaTType"}:
        return True
    return False


def _json_val(value: Any) -> Any:
    """Turn pandas NA / NaN into None so the JSONL is strict JSON."""
    if _is_na(value):
        return None
    if hasattr(value, "item") and not isinstance(value, (bytes, str, list, dict)):
        try:
            value = value.item()
        except Exception:
            pass
        if _is_na(value):
            return None
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _json_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value is None:
        return []
    return [_json_val(v) for v in list(value)]


def _json_num(value: Any) -> int | float | None:
    value = _json_val(value)
    if value is None:
        return None
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, (int, float)):
        return value
    return value


def write_jsonl(records: Iterable[dict[str, Any]], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def composition_report(rows: list[dict[str, Any]]) -> str:
    total = len(rows)
    reasons = Counter(r["review_reason"] for r in rows)
    leaves = Counter(_leaf_key(r) for r in rows)
    lines = [
        f"review subset: {total} items",
        "by reason:",
    ]
    for reason in (REASON_CONFLICT, REASON_LOW, REASON_NONE, REASON_HIGH):
        n = reasons.get(reason, 0)
        lines.append(f"  {reason:16s} {n:5d}")
    lines.append("top family/leaf in subset:")
    for (family, leaf), n in leaves.most_common(15):
        lines.append(f"  {n:5d}  {family}/{leaf}")
    return "\n".join(lines)


def maybe_enqueue(records: list[dict[str, Any]], db: Path) -> int:
    """Queue items with hil.submit.  Does not attach audio assets (files are remote)."""
    try:
        import hil_eval as hil
    except ImportError as exc:
        raise SystemExit(
            "hil-eval is not installed. Run without --enqueue, or install the hil extra."
        ) from exc

    db.parent.mkdir(parents=True, exist_ok=True)
    hil.init(project=PROJECT, store=f"sqlite:///{db}")
    n = 0
    for rec in records:
        hil.submit(
            rec["input"],
            rec["output"],
            confidence=rec["confidence"],
            tags=rec["tags"],
            project=PROJECT,
            options=rec["options"],
            name=rec["name"],
        )
        n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--high-per-leaf", type=int, default=HIGH_PER_LEAF)
    parser.add_argument("--none-target", type=int, default=NONE_TARGET)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--enqueue",
        action="store_true",
        help="also queue items via hil.submit (no audio assets)",
    )
    parser.add_argument("--db", type=Path, default=Path(".hil/soundai-labels.db"))
    args = parser.parse_args(argv)

    catalog = load_rows(args.input)
    selected = select_review_rows(
        catalog,
        seed=args.seed,
        high_per_leaf=args.high_per_leaf,
        none_target=args.none_target,
    )
    records = [to_hil_record(row) for row in selected]
    write_jsonl(records, args.output)
    report = composition_report(selected)
    print(f"wrote {len(records)} items to {args.output}")
    print(report)
    if args.report is not None:
        args.report.write_text(report + "\n", encoding="utf-8")
    if args.enqueue:
        n = maybe_enqueue(records, args.db)
        print(f"queued {n} items in {args.db} (project={PROJECT}, no audio assets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
