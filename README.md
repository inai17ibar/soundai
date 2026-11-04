# soundai

DAW音源素材データ × AI — 自然言語・類似音響による音源検索の実験リポジトリ

## アーキテクチャ

```
音源ファイル (.wav / .aiff / .mp3 …)
        ↓  CLAPEmbedder
   512次元 embedding
        ↓  EmbeddingIndex (FAISS)
  類似度検索 / テキスト検索
```

CLAP (Contrastive Language-Audio Pretraining) によって音声とテキストを同じ潜在空間に射影し、
テキストクエリ・音響クエリの両方で検索可能にする。

## セットアップ

```bash
# 依存ライブラリのインストール (開発用含む)
uv sync --extra dev

# JupyterLab 起動
uv run jupyter lab
```

## 使い方

### 1. 音源をインデックスに登録

```bash
# data/samples/ 以下の音源を再帰的にスキャンして embedding 生成・保存
uv run python scripts/ingest.py --samples data/samples --out data/embeddings
```

### 2. 検索 (Python API)

```python
from soundai import SampleSearcher

searcher = SampleSearcher(index_dir="data/embeddings")

# 自然言語で検索
results = searcher.search_by_text("bright acoustic guitar strum", k=10)
searcher.pretty_print(results)

# 音源ファイルで類似検索
results = searcher.search_by_file("data/samples/my_loop.wav", k=10)
searcher.pretty_print(results)
```

## フェーズロードマップ

| フェーズ | 内容 | 状態 |
|---|---|---|
| 1 | CLAP による自然言語・類似度検索 | 🚧 実装中 |
| 2 | ループ互換性スコアリング | 未着手 |
| 3 | パック構造を用いた表現学習 | 未着手 |

## データ配置

音源ファイルは `data/samples/` 以下に置く（git 管理外）。
フォルダ構造は自由だが、パック名をフォルダ名にしておくと後続の実験で活用しやすい。

```
data/samples/
├── kick_pack/
│   ├── kick_01.wav
│   └── kick_02.wav
└── bass_loops/
    ├── bass_loop_Am_90bpm.wav
    └── ...
```

## hil-eval による人間レビュー（dogfood）

検索結果を hil-eval のレビュー UI で耳で確認し、precision@k と judge の閾値を測る手順は [docs/hil-eval-dogfood.md](docs/hil-eval-dogfood.md) を参照。
