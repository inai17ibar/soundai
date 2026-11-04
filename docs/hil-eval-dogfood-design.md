# hil-eval dogfood design: ESC-50 audio search review

Goal: wire `soundai`'s CLAP search into `hil-eval` so every (query, top-5 result)
pair becomes one review item with an audio asset, simulate human review from
ESC-50 ground truth, and compute precision@k (judge vs. human) + run `tune`.

## Data

- `scripts/fetch_esc50_subset.py`: downloads the ESC-50 GitHub zip
  (`karolpiczak/ESC-50-master.zip`), extracts `meta/esc50.csv` and a subset of
  `audio/*.wav` (e.g. 10 categories × 20 clips, ≤200 files) into
  `data/samples/esc50/` (git-ignored). Writes `data/samples/esc50/meta.csv`
  trimmed to the subset for later ground-truth lookup (`filename -> category`).
- `data/hil_queries.yaml`: ~20 entries, each `{query: "<text>", category: "<esc50 category>"}`,
  e.g. `{query: "a dog barking", category: dog}`. `category` is the ground-truth
  label used only by the review simulator, never sent to the judge.

## Verified hil-eval API (read from `/workspace/hil-eval/src/hil_eval`, v0.1.1 local checkout)

- `hil_eval.init(project, store="sqlite:///.hil/hil.db", *, threshold=None, target_miss_rate=None, audit_rate=None)`
- `hil_eval.review(judge, threshold=0.8, audit_rate=None, inputs=None, images=None, tags=None, project=None)`
  — decorator; `judge(arguments: dict, output) -> Judgment | float`.
- `hil_eval.Judgment(confidence, label=None, reason=None, metadata={})`
- `hil_eval.judges.logprob(extractor, *, transform="sigmoid")` — reusable confidence shaping.
- `hil_eval.submit(input, output, *, confidence=None, ...)`, `hil_eval.list_items(...)`,
  `hil_eval.export(*, project=None, fmt="jsonl", statuses=(approved,corrected), include_images=True)`.
- `hil_eval.tuning.recommend_threshold(store, project, *, target_miss_rate=None, tag=None)`/
  `apply_tuning(...)` — need ≥30 usable reviews (`tuning.py:_MIN_SAMPLES`).
- HTTP (server/app.py): `POST /v1/items` (body has `project, input, output, confidence,
  judge_label, tags, options, assets:[{data_base64, mime?, filename?, video_ts?}]`),
  `POST /v1/items/{id}/review` (`action, choice, corrected_output, reason_codes, reason_text,
  judge_correct, reviewer`), `POST /v1/projects/{project}/reason-codes`,
  `GET /v1/blobs/{sha256}` (Range-capable, serves `audio/wav` etc. — confirmed in
  `blobs.py:sniff_mime` and ADR 0003 "v0.1.1 audio extension").

### Key constraint: decorator vs. HTTP API

`hil_eval.review`'s `Client.record()` (`context.py`) never sets `Item.options`,
and `hil_eval.submit()` has no `options` kwarg either — only
`CreateItemRequest` in `server/app.py` exposes `options`. Since every item
needs `options=["relevant","not_relevant"]` for a `choose`-style review
(mirroring the README's color-picker example), **items are created via direct
`POST /v1/items` calls, not the `@hil_eval.review` decorator.** The
decorator's confidence/threshold/audit math is reproduced by hand in
`soundai/hil.py` (same concepts as `decorator.py:42-104`) via the HTTP path
that also supports `options` + `assets`.

## Module layout

- `soundai/hil.py`
  - `esc50_confidence(score, margin) -> float` — calibrated sigmoid (below).
  - `judge_result(arguments: dict, output: dict) -> Judgment` — same shape as a
    hil-eval judge (`arguments={"query","sample_path","rank"}`,
    `output={"proposed": ...}`), for parity with the judge protocol.
  - `build_item_payload(query, sample_path, rank, judgment, tags, project) -> dict`
    — `CreateItemRequest`-shaped dict: `input={query, sample_path, rank}`,
    `output={"proposed": judgment.label}`, `confidence=judgment.confidence`,
    `options=["relevant","not_relevant"]`, `tags`, one `assets` entry
    (`data_base64` of the wav, `filename=Path(sample_path).name`).
  - `HilClient` — thin `httpx` wrapper: `create_item(payload) -> item_id`,
    `submit_review(item_id, action, choice=None, reason_codes=(), reviewer=...)`,
    `register_reason_codes(project, codes)`.
  - `reviewed_search(searcher, query, k, *, project, tags, hil_client) -> list[item_id]`
    — `searcher.search_by_text(query, k)`, per rank computes
    `margin = score[i] - score[i+1]` (`0.0` on the last rank), builds a
    `Judgment`, posts one item per result.
- `scripts/fetch_esc50_subset.py` — data fetch (above).
- `scripts/hil_run_queries.py` — loads `data/hil_queries.yaml` + the FAISS index
  from `data/embeddings`, calls `reviewed_search` per query with
  `project="soundai-search"`, `tags={"query_set": "esc50-demo-v1",
  "category_guess": ...}`. Prints created item ids.
- `scripts/hil_simulate_reviews.py` — reproduces the decorator's routing
  outside the decorator: review a pending item iff `confidence <
  threshold(=0.8)` or `random() < audit_rate(=0.05)`. Looks up ground truth via
  `data/samples/esc50/meta.csv` and the query's `category`; posts
  `action="choose"`, `choice="relevant"` iff categories match else
  `"not_relevant"` plus one reason code from `{wrong_instrument, wrong_source,
  wrong_mood, low_quality, duplicate, partially_relevant}` (`wrong_source`
  default; `duplicate` if `sample_path` repeats higher-ranked for the query).
  Registers all six reason codes via `register_reason_codes` **before** any
  review — the server only enforces the allowlist once ≥1 code is registered
  (`server/app.py:506-508`), so a partial registration would reject the rest.
- `scripts/hil_eval_metrics.py` — calls `hil_eval.export(project=...,
  fmt="eval-set")` for human labels (`expected`) and `hil_eval.list_items(...)`
  for the pre-review judge `output.proposed`; computes precision@k per query
  for both, grouped by `input["query"]`/`input["rank"]`. Then
  `hil_eval.tuning.recommend_threshold`/`apply_tuning`, or shells out to
  `hil-eval tune --project soundai-search --target-miss-rate 0.05 --apply`.

## Confidence formula

```python
def esc50_confidence(score: float, margin: float, *, k_score=8.0, offset=0.15, k_margin=4.0) -> float:
    z = k_score * (score - offset) + k_margin * margin
    return 1 / (1 + math.exp(-z))
```

LAION-CLAP text-audio cosine scores run small (~0.05-0.35), unlike CLIP;
`offset=0.15`/`k_score=8` are starting points centering the sigmoid there, so
a top-5 slice mixes low- and high-confidence items — verify empirically after
the first `hil_run_queries.py` run and retune so ≥30 items fall below the 0.8
threshold (`tuning.py`'s `_MIN_SAMPLES=30`, else `recommend_threshold` just
returns `reason="insufficient_data"`). `margin` rewards a result that stands
out from the next-ranked one; the last rank in each top-k gets `margin=0.0`.

## pyproject changes

```toml
[project.optional-dependencies]
hil = [
    "hil-eval[server,image] @ git+https://github.com/inai17ibar/hil-eval@v0.1.0",
    "httpx>=0.27",
]
```

`v0.1.0` has no audio sniffing (`blobs.py` audio signatures and the
`POST /v1/items` audio upload path are a `v0.1.1` addition — confirmed by this
checkout's `pyproject.toml` version and `docs/adr/0003-m3-contract.md`'s
"v0.1.1 audio extension" section). Until that's tagged upstream, override the
source for local dogfooding with `[tool.uv.sources] hil-eval = { path =
"/workspace/hil-eval", editable = true }`; swap back to the `v0.1.1` git tag
once cut, and drop the override once `hil-eval[server,image]>=0.1.1` ships.

## Tests (`tests/test_hil_integration.py`, pytest)

No CLAP model, no network:

- `FakeEmbedder` implementing `embed_texts`/`embed_files` (same signatures as
  `CLAPEmbedder`), returning deterministic unit vectors keyed by a
  `dict[str, np.ndarray]` fixture, so `SampleSearcher(embedder=FakeEmbedder(...))`
  runs against a tiny in-memory `EmbeddingIndex` (6-8 fake `.wav` paths under
  `tmp_path`, real small WAV bytes so `blobs.py` signature sniffing succeeds).
- Unit tests for `esc50_confidence` (monotonic in `score`/`margin`, bounded in
  `(0, 1)`).
- Integration test spins up the real hil-eval FastAPI app in-process against a
  `tmp_path` SQLite store (no running server needed) via
  `create_app(SQLiteStore(tmp_path/"hil.db"))` + `fastapi.testclient.TestClient`.
  Drives `reviewed_search` against it, asserts one item per result and that
  the asset blob is retrievable via `GET /v1/blobs/{sha256}`, then runs
  `hil_simulate_reviews`'s logic and `hil_eval_metrics`'s precision@k against
  known fake ground truth, asserting exact expected precision values.

## Open questions / risks

- Decorator bypass is deliberate, not a workaround to delete later — `options`
  genuinely isn't reachable through `hil_eval.review`/`submit` in this
  checkout; revisit if a future hil-eval version adds it there.
- `faiss-cpu`/`laion-clap`/`torch`/`torchvision` stay required deps; `hil`
  extra only adds `hil-eval` + `httpx`.
- ESC-50 is CC BY-NC 3.0 — subset stays under `data/samples/esc50/`
  (git-ignored), never committed, matching existing `data/` handling.
