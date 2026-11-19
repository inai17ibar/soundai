"""Tests for the hil-eval label-review subset builder."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.build_review_subset import (
    OPTIONS,
    PROJECT,
    REASON_CONFLICT,
    REASON_HIGH,
    REASON_LOW,
    REASON_NONE,
    composition_report,
    select_review_rows,
    to_hil_record,
)


def _row(
    path: str,
    *,
    family: str = "drums",
    leaf: str = "kick",
    conf: str = "high",
    conflict: bool = False,
    typ: str = "one_shot",
) -> dict:
    return {
        "file_id": path,
        "rel_path": path,
        "pack": path.split("/", 1)[0],
        "family": family,
        "leaf": leaf,
        "instrument_conf": conf,
        "instrument_source": "filename",
        "type": typ,
        "type_conf": "high",
        "bpm": 120,
        "key": "A minor",
        "key_root": "A",
        "key_mode": "minor",
        "conflict": conflict,
        "matched_tokens": [leaf],
        "matched_rules": [],
        "rules_version": "0.2.0",
        "size_bytes": 1000,
    }


def _catalog() -> list[dict]:
    rows = []
    # Conflicts and lows
    rows.append(_row("PackA/conflict_sax.wav", family="woodwind", leaf="woodwind", conf="low", conflict=True))
    rows.append(_row("PackA/low_only.wav", family="keys", leaf="piano", conf="low"))
    # High-confidence, two leaves
    for i in range(12):
        rows.append(_row(f"PackB/kick_{i:02d}.wav", family="drums", leaf="kick", conf="high"))
    for i in range(8):
        rows.append(_row(f"PackB/snare_{i:02d}.wav", family="drums", leaf="snare", conf="high"))
    # None, several patterns
    for i in range(10):
        rows.append(_row(f"PackC/songstarter_{i:02d}.wav", family="unknown", leaf="unknown", conf="none", typ="loop"))
    for i in range(6):
        rows.append(_row(f"PackC/stack_{i:02d}.wav", family="unknown", leaf="unknown", conf="none", typ="loop"))
    for i in range(6):
        rows.append(_row(f"PackC/melodic_{i:02d}.wav", family="unknown", leaf="unknown", conf="none", typ="loop"))
    for i in range(6):
        rows.append(_row(f"PackC/other_{i:02d}.wav", family="unknown", leaf="unknown", conf="none", typ="loop"))
    return rows


def test_selects_all_conflicts_and_lows_plus_stratified_samples():
    selected = select_review_rows(_catalog(), seed=1, high_per_leaf=5, none_target=8)
    reasons = {r["rel_path"]: r["review_reason"] for r in selected}
    assert reasons["PackA/conflict_sax.wav"] == REASON_CONFLICT
    assert reasons["PackA/low_only.wav"] == REASON_LOW
    assert sum(r["review_reason"] == REASON_HIGH for r in selected) == 10  # 5 kick + 5 snare
    assert sum(r["review_reason"] == REASON_NONE for r in selected) == 8
    # Conflict is not also labelled low
    assert reasons["PackA/conflict_sax.wav"] != REASON_LOW


def test_selection_is_deterministic():
    a = [r["rel_path"] for r in select_review_rows(_catalog(), seed=42, high_per_leaf=3, none_target=8)]
    b = [r["rel_path"] for r in select_review_rows(_catalog(), seed=42, high_per_leaf=3, none_target=8)]
    assert a == b
    c = [r["rel_path"] for r in select_review_rows(_catalog(), seed=99, high_per_leaf=3, none_target=8)]
    assert a != c


def test_hil_record_shape_and_no_audio_ingest():
    row = select_review_rows(_catalog(), seed=1, high_per_leaf=1, none_target=1)[0]
    rec = to_hil_record(row)
    assert rec["project"] == PROJECT
    assert rec["options"] == OPTIONS
    assert rec["output"]["proposed"] == "correct"
    assert rec["input"]["audio"].startswith("/Volumes/Extreme SSD/Splice/sounds/packs/")
    assert rec["input"]["audio_available"] is False
    assert rec["input"]["reason_selected"] == row["review_reason"]
    assert rec["input"]["proposed"]["family"] == row["family"]
    assert rec["input"]["proposed"]["leaf"] == row["leaf"]
    assert 0 < rec["confidence"] < 0.5
    # JSON serialisable
    json.dumps(rec)


def test_script_writes_jsonl(tmp_path: Path):
    import pandas as pd

    from scripts.build_review_subset import main

    src = tmp_path / "catalog.parquet"
    pd.DataFrame(_catalog()).to_parquet(src, index=False)
    dest = tmp_path / "subset.jsonl"
    assert main(["--input", str(src), "--output", str(dest), "--seed", "1", "--none-target", "8"]) == 0
    lines = dest.read_text(encoding="utf-8").strip().splitlines()
    assert 10 < len(lines) < 50
    rec = json.loads(lines[0])
    assert rec["project"] == PROJECT
    assert rec["input"]["reason_selected"] in {REASON_CONFLICT, REASON_LOW, REASON_NONE, REASON_HIGH}


def test_composition_report_mentions_counts():
    selected = select_review_rows(_catalog(), seed=1, high_per_leaf=5, none_target=8)
    text = composition_report(selected)
    assert "review subset:" in text
    assert REASON_CONFLICT in text
    assert str(len(selected)) in text


def test_nan_becomes_null():
    row = _row("PackA/no_key.wav", conf="low")
    row["key"] = float("nan")
    row["key_root"] = float("nan")
    row["key_mode"] = float("nan")
    row["bpm"] = float("nan")
    row["review_reason"] = REASON_LOW
    rec = to_hil_record(row)
    assert rec["input"]["proposed"]["key"] is None
    assert rec["input"]["proposed"]["bpm"] is None
    dumped = json.dumps(rec)
    assert "NaN" not in dumped
    assert "null" in dumped
