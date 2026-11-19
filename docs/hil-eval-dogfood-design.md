# ESC-50 human review demo

`soundai.hil.reviewed_search` wraps a per-result function with `@hil.review`.
Each (query, result) has a query, sample path, rank, and audio `Path` captured
through `inputs=[...]`, and choices `relevant` / `not_relevant` supplied through
`options=`. The SDK stores the audio asset, scores the result, and routes it.
Items are never created by raw HTTP. Ground-truth categories are used only by
the simulator, never by the judge.

The judge proposes relevant when cosine >= 0.15; its confidence is
`sigmoid(8 * (score - 0.15) + 4 * margin)`. Margin is the score difference
from the next result, or zero for the last result. This is a heuristic starting
point, not an empirically calibrated probability.

The default threshold is 1.0, audit rate 0, to queue every top-5 result and
make full precision@5 measurable. `--threshold 0.8 --audit-rate 0.05` exercises
selective routing instead. The simulator reviews all pending queued items;
it must not repeat the SDK's threshold/audit gate. It registers all six reason
codes before reviewing and sends explicit judge correctness for tuning.
Matching categories receive relevant; mismatches receive wrong_source, and
repeated paths within a query receive duplicate. Other reason codes are
available for real human review.

Metrics join eval-set exports to original item outputs by item ID. Judge and
human precision use the same reviewed ranks. Coverage and completeness are
reported per query: partial coverage yields precision on reviewed ranks only,
not a full top-k estimate. Duplicate query/rank pairs fail explicitly; use a
fresh database for each demo run. Simulated category labels demonstrate the
pipeline and cannot substitute for independent human judgments.

## Run

Run from the repository root using the prepared environment with the hil extra
and editable hil-eval installed. No dependency sync or lock is needed.

```bash
export UV_CACHE_DIR=/tmp/soundai-uv-cache
uv run --no-sync pytest -q
uv run --no-sync python scripts/fetch_esc50_subset.py
# Offline data alternative: append --zip-cache /path/to/ESC-50-master.zip
# or --zip-cache /path/to/extracted/ESC-50-master
uv run --no-sync python scripts/ingest.py --samples data/samples/esc50 --out data/embeddings/esc50
uv run --no-sync python scripts/hil_run_queries.py --db .hil/esc50.db
uv run --no-sync hil-eval --db .hil/esc50.db serve
```

Leave the server running; in a second terminal at the repository root:

```bash
export UV_CACHE_DIR=/tmp/soundai-uv-cache
uv run --no-sync python scripts/hil_simulate_reviews.py --db .hil/esc50.db
uv run --no-sync python scripts/hil_eval_metrics.py --db .hil/esc50.db --output .hil/esc50-metrics.json
uv run --no-sync hil-eval --db .hil/esc50.db tune --project soundai-search --target-miss-rate 0.05 --apply
```

Open http://127.0.0.1:8765 to inspect audio and reviews. Start with an unused
`.hil/esc50.db`; do not append another query run to the same evaluation.
The default download fetches the full upstream ZIP before copying 200 clips
(20 categories × 10). CLAP ingestion may download its checkpoint on first use.
Audio/metadata and indexes stay ignored. See `data/LICENSES.md` for ESC-50's
CC BY-NC 3.0 terms and K. J. Piczak attribution. The optional hil dependency
uses the upstream v0.1.1 git source; the prepared environment may retain its
editable `/workspace/hil-eval` installation with `--no-sync`.

Tests use deterministic embeddings and a tiny FAISS index without loading a
CLAP model, real WAV bytes, a temporary SQLite SDK store, and an in-process
FastAPI client. Threshold tuning requires at least 30 usable reviews; the CLI
reports insufficient data when that requirement is unmet.
