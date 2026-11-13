# hil-eval への依頼メモ（soundai-labels のラベル修正）

宛先: しずく。hil-eval 自体はこちらでは改変しない。

soundai ではパスから規則で付けた楽器・種別・BPM・キーを、人が確認するサブセットを `scripts/build_review_subset.py` で作る。JSONL は `hil.submit` 互換（プロジェクト名 `soundai-labels`、options は `correct` / `wrong_instrument` / `wrong_type` / `wrong_bpm_key` / `unsure`）。いまのレビューはパスと提案ラベルだけを見て行う。音声は Mac mini の `/Volumes/Extreme SSD/Splice/sounds/packs/...` にしかなく、外部にも box にもアップロードしない。

## お願いしたいこと

1. **ラベル修正 UI（最優先）**  
   いまの options は「正しい / どの属性が違う / わからない」までしか言えない。違うと判定したあと、正しい `family` / `leaf` / `type` / `bpm` / `key` を構造化して入力するフォームが欲しい。汎用の `edit`（出力 JSON を直接いじる）はあるが、カタログ用のフィールド編集には向かない。修正値は `labels_overrides.parquet`（file_id → 修正値）に残し、規則のバージョンが変わっても保持したい。

2. **リモートパスの音声再生**  
   エンキュー時点でローカルにファイルがなくても、アイテムに書いた絶対パス（Mac mini の SSD）をレビュー画面で再生できるようにしてほしい。いまは `assets=` がエンキュー時にファイルを読んでハッシュするため、box 上では失敗する。遅延参照（パスだけ記録し、serve するマシンで存在すれば再生）か、それに近い仕組みが必要。音声のアップロードや転送はしない。

3. **JSONL の一括取込（任意）**  
   要件ドキュメントにあった `hil-eval import` があると、今回の `review_subset.jsonl` を Mac mini 上の DB に流し込みやすい。なければ `hil.submit` を回す `--enqueue` で足りる。

## 使いかた（予定）

```bash
uv run python scripts/build_review_subset.py \
  --input /workspace/soundai-data/splice_metadata.parquet \
  --output /workspace/soundai-data/review_subset.jsonl
# Mac mini で hil-eval が使えるようになったら:
#   uv run python scripts/build_review_subset.py --enqueue --db .hil/soundai-labels.db
#   uv run hil-eval --db .hil/soundai-labels.db serve
```
