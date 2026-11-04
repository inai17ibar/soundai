"""Per-result ESC-50 review and evaluation (no model import required)."""
from __future__ import annotations

import math
from pathlib import Path
from urllib.parse import quote

import hil_eval as hil

PROJECT = "soundai-search"
OPTIONS = ["relevant", "not_relevant"]
REASON_CODES = ("wrong_instrument", "wrong_source", "wrong_mood", "low_quality", "duplicate", "partially_relevant")


RELEVANCE_TAU = 0.40  # cosine boundary chosen from ESC-50 demo labels


def esc50_confidence(score, margin, *, tau=RELEVANCE_TAU, k_score=20.0, k_margin=4.0):
    """Confidence in the proposed label: low near the tau boundary, high far from it."""
    z = k_score * abs(score - tau) + k_margin * max(margin, 0.0)
    # Stable even for extreme scores.
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


def judge_result(arguments, output):
    score, margin = float(arguments["score"]), float(arguments["margin"])
    return hil.Judgment(
        confidence=esc50_confidence(score, margin), label=output["proposed"],
        reason=f"cosine={score:.4f}, margin={margin:.4f}",
    )


def reviewed_search(searcher, query, k=5, *, project=PROJECT, tags=None, threshold=0.8, audit_rate=0.05):
    """Return original predictions; SDK queues one item per routed result.

    Default threshold=.8/audit_rate=.05 routes only uncertain results (plus a random audit).
    Use threshold=1.0 to queue every result for complete precision@k labels.
    Ground-truth categories are never passed to the judge.
    """
    @hil.review(judge=judge_result, threshold=threshold, audit_rate=audit_rate,
                inputs=["query", "sample_path", "rank", "audio"],
                options=OPTIONS, project=project, tags=tags,
                assets=["audio"], name="search_result")
    def search_result(query, sample_path, rank, audio, score, margin):
        return {"proposed": "relevant" if score >= RELEVANCE_TAU else "not_relevant"}

    results = searcher.search_by_text(query, k=k)
    outputs = []
    for i, result in enumerate(results):
        score = float(result["score"])
        margin = score - float(results[i + 1]["score"]) if i + 1 < len(results) else 0.0
        path = str(result["path"])
        outputs.append(search_result(query, path, i + 1, path, score, margin))
    return outputs


class HilClient:
    """HTTP only for operations not exposed by the public SDK."""
    def __init__(self, client):
        self.client = client

    def register_reason_codes(self, project, codes=REASON_CODES):
        for code in codes:
            response = self.client.post(f"/v1/projects/{quote(project, safe='')}/reason-codes",
                                        json={"code": code, "label": code.replace('_', ' ')})
            response.raise_for_status()

    def submit_review(self, item_id, **payload):
        response = self.client.post(f"/v1/items/{item_id}/review", json=payload)
        response.raise_for_status()
        return response.json()


def simulate_reviews(items, categories, targets, client, project=PROJECT):
    """Label pending SDK-routed items using metadata, without a second routing gate."""
    client.register_reason_codes(project)
    seen = set()
    count = 0
    for item in sorted(items, key=lambda x: (x.input["query"], x.input["rank"])):
        query, path = item.input["query"], item.input["sample_path"]
        category = categories[Path(path).name]
        choice = "relevant" if category == targets[query] else "not_relevant"
        duplicate = (query, path) in seen
        seen.add((query, path))
        if item.status.value != "pending":
            continue
        reasons = ["duplicate"] if duplicate else (["wrong_source"] if choice == "not_relevant" else [])
        client.submit_review(item.id, action="choose", choice=choice, reason_codes=reasons,
                             judge_correct=choice == item.output["proposed"], reviewer="esc50-simulator")
        count += 1
    return count


def precision_metrics(items, records, k=5):
    """Paired precision on reviewed ranks; full precision@k only with complete coverage."""
    if k < 1:
        raise ValueError("k must be positive")
    # Export metadata contains id; join by ID to keep repeated runs distinct.
    expected = {r["metadata"]["id"]: r["expected"] for r in records}
    groups = {}
    for item in items:
        if item.input["rank"] <= k:
            groups.setdefault(item.input["query"], []).append(item)
    rows = []
    for query, group in sorted(groups.items()):
        ranks = [i.input["rank"] for i in group]
        if len(ranks) != len(set(ranks)):
            raise ValueError("Duplicate query/rank: use a fresh database for each demo run")
        paired = [i for i in group if i.id in expected]
        n = len(paired)
        complete = sorted(i.input["rank"] for i in paired) == list(range(1, k + 1))
        rows.append({"query": query, "reviewed": n, "coverage": n / k, "complete": complete,
                     "judge_precision": sum(i.output["proposed"] == "relevant" for i in paired) / n if n else None,
                     "human_precision": sum(expected[i.id] == "relevant" for i in paired) / n if n else None})
    return {"k": k, "queries": rows, "macro": {
        key: sum(r[key] for r in rows if r[key] is not None) / sum(r[key] is not None for r in rows)
        if any(r[key] is not None for r in rows) else None
        for key in ("judge_precision", "human_precision")}}
