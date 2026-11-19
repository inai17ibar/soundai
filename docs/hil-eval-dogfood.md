# hil-eval で soundai の検索結果を人間レビューする（dogfood ガイド）

テキスト検索の top-5 結果を 1 件ずつ hil-eval に送り、**自信の低い結果だけ**を人が聴いて「relevant / not_relevant」で判定します。判定は評価セットとして書き出し、precision@k の計測と閾値チューニングに使います。

## 0. 前提
- macOS、`uv` が入っていること
- hil-eval **v0.1.1 以上**（音声プレイヤー・`options=` 対応）。`pyproject.toml` の `[tool.uv.sources]` は `tag = "v0.1.1"` を参照（v0.1.1 の push/タグ前は、ローカルの clone を `uv pip install -e ../hil-eval[server,cli,hf]` で入れてください）
- CLAP チェックポイント `630k-audioset-best.pt`（約 330MB）は初回に自動 DL。DL できない場合は https://huggingface.co/lukewys/laion_clap から手動で laion_clap パッケージのディレクトリへ置く

## 1. セットアップ
```bash
uv sync --extra hil
uv run pytest -q                                   # モデル不要のテスト
uv run python scripts/fetch_esc50_subset.py        # ESC-50 から 20 クラス×10 = 200 クリップ（約 650MB の zip を DL）
uv run python scripts/ingest.py --samples data/samples/esc50 --out data/embeddings/esc50
```
音声データはコミットしません（`data/samples/` は .gitignore 済み）。ライセンスは `data/LICENSES.md`（ESC-50 は **CC BY-NC 3.0**、個人の評価用途のみ）。

## 2. クエリを流す
```bash
# 通常運用: confidence < 0.8 の結果 + 5% のランダム監査だけがキューへ
uv run python scripts/hil_run_queries.py --db .hil/esc50.db
# 全件ラベル付け（precision@k を完全に測るとき）
uv run python scripts/hil_run_queries.py --db .hil/esc50-full.db --threshold 1.0 --audit-rate 0
```
judge: CLAP コサイン類似度 `score` が 0.40 以上なら `relevant` を提案。confidence は境界 0.40 からの距離と次順位とのスコア差（margin）から算出（境界付近ほど低い = 人が聴くべき）。

## 3. レビュー（ここが人の作業）
```bash
uv run hil-eval --db .hil/esc50.db serve      # → http://127.0.0.1:8765
```
- 各アイテム: クエリ文、順位、サンプルのパス、**音声プレイヤー**（5 秒）
- クリップを聴いて `relevant` / `not_relevant` を選ぶ。外れの場合は理由コードを付ける:
  `wrong_source`（音源が違う）、`wrong_instrument`、`wrong_mood`、`low_quality`、`duplicate`、`partially_relevant`
- ショートカット: J/K で移動。URL `/items/<id>` で直接開ける
- デモ用に、ESC-50 の正解カテゴリで自動判定するシミュレータもあります: `uv run python scripts/hil_simulate_reviews.py --db .hil/esc50.db [--limit 5]`（serve 起動中に実行）

## 4. 結果を読む
```bash
uv run python scripts/hil_eval_metrics.py --db .hil/esc50.db --output .hil/metrics.json
uv run hil-eval --db .hil/esc50.db export --project soundai-search --format eval-set --output .hil/evalset.jsonl
uv run hil-eval --db .hil/esc50.db tune --project soundai-search --target-miss-rate 0.05   # --apply で保存
```
- metrics の `Human` 列 = 人のラベルでの precision@5（検索品質そのもの）。`Judge` 列 = judge が relevant と言った割合。両者の差が judge の楽観/悲観の度合い
- `Coverage` < 100% のクエリは、レビュー済みの順位だけで計算した暫定値
- eval-set の各行: `input`（query, sample_path, rank, 音声 asset）、`expected`（人の判定）、`metadata`（confidence, judge_correct, reason_codes）。検索モデルや judge を変えたときの回帰テストに使えます
- `tune`: 見逃し率（confidence ≥ 閾値なのに judge が誤り）の Wilson 上限が目標以下になる最小閾値を推奨。サンプル 30 未満は `insufficient_data`

## 5. 2026-11-05 の box での実行結果（参考）
- 200 クリップ / 20 クエリ / k=5、CPU で ingest 約 20 秒
- 全件ラベル（シミュレータ）: precision@5 = **0.85**（human）、judge の正解率 0.84、judge が relevant と言った割合 0.87
- 弱いクエリ: insects buzzing 0.2、cow mooing / ocean waves / wind blowing 0.4
- 通常運用（閾値 0.8, audit 5%）では 100 件中 **28 件**（low_conf 24 + audit 4）だけがキューへ
- tune（目標見逃し率 5%）: 推奨は 1.0 のまま（confidence ≥ 0.9 でも 45 件中 4 件の誤りがあり、≥ 0.95 で初めて誤り 0）。judge のキャリブレーション改善（tau の調整、クエリ別の正規化）が次の課題

## 6. 自分の音源で使う
`data/samples/` に自分の素材を置いて ingest し、`data/hil_queries.json` に `{"query": "...", "category": "..."}` を書きます（自前データでは category は使わず、人がレビューするだけで OK）。
