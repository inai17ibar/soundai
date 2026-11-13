# 条件ベース評価の設計案（ドラフト）

> 状態: 提案。SSD の調査（§4）が終わったら確定する。
> 前提: 音の類似度や好みは評価しない。評価するのは **「検索結果がクエリで明示した条件（楽器・種別・BPM・キーなど）を満たすか」** だけ。
> 制約: 音声はどこにもアップロードしない（hil-eval はローカルの `serve` だけで使う）。音声・メタデータ・インデックスは git の管理外（`data/`、`.hil/`）。

## 0. 既存実装との関係

| 既存 | 今回の扱い |
|---|---|
| `soundai/embed.py` CLAP（HTSAT-tiny、512 次元）、`index.py` FAISS IndexFlatIP | そのまま「CLAP ベースライン」として使う |
| `scripts/ingest.py`（rglob でスキャン、追加分だけ処理） | スキャン結果を `catalog.parquet` にも書き出すように拡張する（§2） |
| `soundai/hil.py` `reviewed_search`（結果 1 件ごとに `@hil.review`、relevant/not_relevant、理由コード 6 種） | 判定の単位を「クエリとの関連」から「**条件ごとの充足**」に変える（§3.4） |
| `precision_metrics`（ID で突き合わせ、coverage と complete の扱い） | 指標の土台として流用し、条件別の内訳を足す |
| `data/hil_queries.json` `{query, category}` | 構造化した条件を持つ `condition_queries.json` に置き換える |
| ESC-50 ランナー | 回帰テスト用に残す（公開データでのスモークテスト） |

注: judge の閾値がずれています。`docs/hil-eval-dogfood-design.md` は 0.15、コード（`RELEVANCE_TAU`）とガイドは 0.40 です。条件ベースの評価では judge を「メタデータとの一致」に置き換えるので（§3.4）、CLAP の閾値に頼る場面は減ります。

---

## 1. 楽器カテゴリ一覧（2 階層＋直交する属性）

### 1.1 instrument の taxonomy

`family > leaf` の 2 階層。leaf が決まらないときは `family/unknown` にする。キーワードは小文字に正規化してから、トークン単位で照合する（§2.2）。

| family | leaf | 日本語 | 典型的なキーワードと略語 |
|---|---|---|---|
| drums | kick | キック | kick, kik, kck, bd, bassdrum, "bass drum", kickdrum |
| drums | snare | スネア | snare, snr, sn, sd, rimshot, rim |
| drums | clap | クラップ | clap, clp, handclap |
| drums | hihat | ハイハット | hihat, hi-hat, hh, hat, chh, ohh, closedhat, openhat, "closed hat", "open hat" |
| drums | cymbal | シンバル | cymbal, cym, crash, ride, splash, china |
| drums | tom | タム | tom, toms, floortom, "hi tom", "lo tom" |
| drums | perc | パーカッション | perc, percussion, shaker, shkr, tamb, tambourine, conga, bongo, cowbell, clave, block, triangle |
| drums | full | ドラムループ一式 | drum loop, drums, beat, groove, top loop, "full drums" |
| bass | synth_bass | シンセベース | bass, bs, synth bass, reese, wobble |
| bass | sub_808 | 808／サブベース | 808（ベースとして使うもの）, sub, subbass |
| bass | acoustic_bass | エレキ／ウッドベース | electric bass, upright, double bass, slap, "bass guitar" |
| keys | piano | ピアノ | piano, pno, grand, upright piano |
| keys | epiano | エレピ | rhodes, wurli, wurlitzer, ep, epiano, "e-piano" |
| keys | organ | オルガン | organ, org, hammond, b3 |
| keys | other_keys | その他の鍵盤 | keys, clav, celesta, harpsichord |
| synth | lead | リード | lead, ld |
| synth | pad | パッド | pad, pd, atmos, atmosphere, drone |
| synth | pluck | プラック | pluck, plk |
| synth | arp | アルペジオ | arp, arpeggio, seq, sequence |
| synth | chord | シンセコード | chord, chords, stab |
| guitar | electric | エレキギター | gtr, guitar, egtr, electric, riff, "power chord" |
| guitar | acoustic | アコギ | acoustic, agtr, nylon, steel |
| strings | strings | ストリングス | strings, str, violin, vln, viola, cello, pizz, pizzicato, orchestra |
| brass | brass | ブラス | brass, horn, horns, trumpet, tpt, trombone, tbn, sax（※） |
| woodwind | woodwind | 木管 | flute, fl, clarinet, oboe, sax（※）, bassoon |
| mallet | mallet | マレット／ベル | bell, bells, marimba, vibes, vibraphone, glock, kalimba |
| vocal | vocal | ボーカル | vox, vocal, vocals, voc, acapella, chant, choir, adlib, phrase, spoken |
| fx | fx | 効果音 | fx, sfx, riser, uplifter, downlifter, sweep, impact, hit, whoosh, noise, foley, transition, reverse, rev |
| world | world | 民族楽器 | sitar, koto, shamisen, tabla, steel drum など |
| unknown | unknown | 不明 | — |

※ sax は文化的には木管（woodwind）。パック側でブラスとして分類されていることもあるので、`woodwind` を正にしてパック名で上書きできるようにする。

### 1.2 直交する属性

| 属性 | 値 | 取り出すパターン |
|---|---|---|
| type | loop / one_shot / unknown | `loop`, `lp`, `loops` → loop。`one shot`, `oneshot`, `os`, `shot`, `hit`, `single` → one_shot。BPM があって長さが 2 秒を超えるときは補助的に loop |
| bpm | 整数（40〜220） | `(\d{2,3})\s?bpm`、`_(\d{2,3})_` は文脈つきのときだけ（§2.3） |
| key | 例: `A minor`, `F# major` | `\b([A-G])(#\|b\|s\|sharp\|flat)?\s?(m\|min\|minor\|maj\|major)?\b`。`Am`、`Cmaj`、`F#m`、`Bbmin` を正規化する |
| genre | house / techno / hiphop / trap / lofi / dnb / ... | 主にパック名から取る。ファイル名から取れることは少ない |
| character（任意） | dry / wet / processed / layered | 評価には使わない（感性に寄るため） |

### 1.3 曖昧さの罠（ルールを明示して扱う）

- **`bass` と `bass drum`**: `bass` の直後が `drum` なら kick にする。`bd` は drums フォルダの中だけ kick と読む。
- **`808`**: 次のトークンや親フォルダで判断する。`808 kick`、`808_bd`、`Kicks/` の下なら kick。`808 bass`、`808_C`（キーがついている）、`Bass/` の下なら sub_808。どれにも当たらなければ `drums/unknown` や `bass/sub_808` を候補に並べ、confidence は low にする。`TR-808` は機材名なので楽器としては扱わない。
- **`sub`**: `sub bass` なら bass。`sub` 単独でも bass に寄せるが、`sub kick` は kick。`subfolder` のような部分一致はトークン単位にすることで防ぐ。
- **`pad`**: `drum pad`、`pad hit` は perc や drums。`pads/` フォルダや `pad` 単独なら synth/pad。
- **`hit`**: `orch hit`、`brass hit` は楽器を優先し type=one_shot にする。`hit` 単独なら fx/impact と one_shot。
- **`perc`**: drums/perc としてまとめる。細かい楽器（shaker など）は sub_keyword として残す。
- **`lead` vocal**: `vocal lead` は vocal。
- **`rev`**: reverse の意味でも reverb の意味でもあり得る。楽器の判定には使わず、タグとしてだけ残す。
- **`sn`、`ep`、`os`、`ld`、`fl` などの短い略語**: 誤検知しやすいので、親フォルダの family と合うときだけ採用する。
- **キー名とトークンの衝突**: `A`、`E`、`Bb` の 1 文字トークンは、`bpm` や `min/maj` と並んでいる場合か、ファイル名の末尾にある場合だけキーとして扱う。
- **数字**: `01`〜`99` の連番と BPM の取り違えに注意（§2.3）。

---

## 2. メタデータ抽出の設計

### 2.1 入力と処理の流れ

```
/Volumes/Extreme SSD/<root>/<pack>/<folder...>/<file>.<ext>
  → 正規化（NFC、小文字化、拡張子の除去）
  → path 要素ごとにトークン化
  → 辞書照合（§1）＋正規表現（bpm/key/type）
  → 信号の統合（優先度・矛盾の解決）
  → catalog.parquet（1 ファイル 1 行）
```

- 読み取り専用で、元のファイルには手を加えない。オーディオヘッダ（長さ・サンプルレート・チャンネル数）は soundfile で読み、波形の中身は使わない。
- macOS の `._*`（AppleDouble）、`.DS_Store`、`__MACOSX/` は除外する。
- トークナイザ: 区切りは `[ _\-\.\(\)\[\]]+` とし、camelCase と、数字と文字の境目でも分割する（`Kick01` → `kick`, `01`、`120BPM` → `120`, `bpm`）。2 語のフレーズ（`bass drum`、`open hat`、`one shot`）は bigram で照合する。

### 2.2 信号の優先度

instrument と type は、次の順で強い信号を採用する（上ほど強い）。

1. **ファイル名の明示的なキーワード**（`Kick_Hard_01.wav`）
2. **直接の親フォルダ**（`.../Kicks/01.wav`）
3. **祖先フォルダ**（`.../Drums/One Shots/...`）
4. **パック名**（`Lofi Keys Vol.2`）。パック全体が単一楽器の場合に限る
5. **CLAP によるゼロショット分類**（任意）。**ラベルを提案するだけで、gold には使わない**

- 矛盾したとき: ファイル名が親フォルダと異なる family を示した場合はファイル名を優先し、`conflict=true` を立てる（レビューの優先対象）。
- leaf が決まらず family だけ分かる場合は `leaf=unknown`。

### 2.3 BPM・キー・種別の規則

- BPM は、`bpm` の隣にある数値を最も強い根拠にする。次に、ファイル名の中で `_NNN_` が 60〜200 の範囲にあり、`loop` も含まれる場合。パックの命名規則（`PackName_120_Am_Bass.wav` など）が分かったら、パックごとのテンプレートを追加する。
- 連番（`01`、`002`）は BPM として採用しない。2 桁の数値は `bpm` が付いているときだけ BPM とする。
- キーは `minor` / `major` / `unknown` の mode とルート音に正規化する（`Bb` は `A#` 表記にそろえる）。
- 種別（type）: キーワードを最優先し、次にフォルダ名（`Loops/`、`One Shots/`）、最後に補助ヒューリスティック（BPM があり、長さ ≥ 2 秒）で判定する。

### 2.4 確信度

| level | 条件 |
|---|---|
| high | ファイル名か直接の親フォルダに明示キーワードがあり、矛盾がない |
| medium | 祖先フォルダかパック名から推定した。または短い略語を文脈つきで採用した |
| low | 曖昧語（808、sub、hit、pad）を既定値で解決した。矛盾がある。補助ヒューリスティックだけで決めた |
| none | 該当なし → `unknown` |

### 2.5 出力スキーマ（`data/meta/catalog.parquet`。同じ内容の JSONL も出力可）

| column | type | 説明 |
|---|---|---|
| file_id | str | パスの相対部分の sha1（安定 ID） |
| rel_path | str | SSD ルートからの相対パス |
| pack | str | 先頭のパックディレクトリ名 |
| ext / duration_s / sr / channels | str / float / int / int | ヘッダ情報 |
| family / leaf | str | 楽器（§1.1） |
| instrument_conf | str | high/medium/low/none |
| instrument_source | str | filename / parent / ancestor / pack / clap |
| type / type_conf | str | loop/one_shot/unknown |
| bpm / bpm_source | int? / str | |
| key_root / key_mode | str? | |
| genre | str? | |
| tokens | list[str] | 正規化後の全トークン（ベースラインの検索と、後から規則を作り直すため） |
| matched_rules | list[str] | どの規則が当たったか（デバッグ用） |
| conflict | bool | 信号が食い違ったか |
| rules_version | str | 抽出規則のバージョン |
| verified / verified_by | bool / str | 人が確認したか（§2.7） |

### 2.6 未知の扱い

- `unknown` も正式な値として残す（捨てない）。カバレッジ（§3.3）の分母に入れる。
- unknown が多いトークンを頻度順に出力する（`unknown_tokens.csv`）。上位から辞書に追加していく（ルールの改善ループ）。
- パック単位で unknown の率を出し、命名規則が独特なパックはパック別テンプレートで対応する。

### 2.7 hil-eval でラベルを修正する

- `@hil.review` の別プロジェクト `soundai-labels` で、1 ファイルを 1 アイテムとする。inputs は `rel_path`、`audio`（ローカルのパスを asset として使う）、提案ラベル。options は `correct` / `wrong_instrument` / `wrong_type` / `wrong_bpm_key` / `unsure`。
- 修正後の値はレビューのコメントか `edit` アクションで受け取る（hil-eval の機能に合わせて決める）。結果は `labels_overrides.parquet`（file_id → 修正値）に保存する。カタログは「規則の出力に上書きをマージしたもの」として再生成する（上書きは規則のバージョンが変わっても保持する）。
- レビューの対象: (a) low と conflict の全件、(b) high と medium から family ごとに層別ランダムで抽出（推定精度を測るため。例: 各 family 30 件）。high の監査で誤り率が高かった規則は見直す。
- 理由コードを追加する: `wrong_type`、`wrong_bpm`、`wrong_key`、`ambiguous_name`。

---

## 3. 条件ベース評価の設計

### 3.1 クエリセット（`data/condition_queries.json`）

```json
{"qid": "q001", "text": "punchy kick one shot",
 "conditions": {"leaf": "kick", "type": "one_shot"}}
{"qid": "q014", "text": "house bass loop around 124 bpm",
 "conditions": {"family": "bass", "type": "loop", "bpm": [120, 128]}}
{"qid": "q020", "text": "rhodes chords in A minor",
 "conditions": {"leaf": "epiano", "key": "A minor"}}
```

- `text` は CLAP 用の自然文で、`conditions` が評価の基準。感性語（punchy、warm）は `text` に含めてもよいが、**評価対象にはしない**。
- 構成（案: 60〜80 クエリ）: leaf を 1 つ指定（全 leaf のうち件数が 20 以上のもの）、family と type の組み合わせ、BPM の範囲つき、キーつき、条件を 3 つ組み合わせたもの。
- 候補件数（gold で条件を満たすファイルの数）が k より少ないクエリは除くか、`recall`/`R-precision` で評価する。

### 3.2 gold ラベル

- gold は §2 で抽出したメタデータを人が確認した後の値。
- 評価で使うのは `verified=true`、または high かつ監査済みの規則から得たものだけ。それ以外の結果は判定保留として hil-eval のレビューに回す（§3.4）。
- 1 つの結果が満たすかどうかは、条件ごとに判定する。`leaf=kick` は一致すれば満たす。`family=bass` は leaf が何でも満たす。`bpm` は範囲内で満たし、BPM が不明なら保留。

### 3.3 指標

- **condition precision@k**: 上位 k 件のうち、**すべての条件**を満たす割合。k = 5 と 10。クエリ間はマクロ平均。
- **条件別の内訳**: 条件の種類ごと（instrument / type / bpm / key）に部分的な precision を出す。どの条件で外しているかが分かる。
- **family 別の内訳**: drums は強いが keys は弱い、のような偏りを見る。
- **coverage**: (a) カタログの中で instrument が unknown でない割合、(b) 評価に使った結果のうち gold が確定していた割合。`precision_metrics` と同じく、coverage が 100% 未満のときは暫定値と明記する。
- 任意: 候補数が少ないクエリに限り `recall@k` / R-precision。
- 信頼区間は Wilson（hil-eval の tune と同じ考え方）。クエリ数が少ないうちは、差を断定しない。

### 3.4 hil-eval でのレビュー（サンプリング）

- 既存の `reviewed_search` を流用し、judge を **メタデータとの一致判定** に置き換える。gold が確定していれば confidence を高く（自動ラベル）、gold が不明か low なら confidence を低くしてキューに入れる。
- options は条件ごとにする。`meets_all` / `fails_instrument` / `fails_type` / `fails_bpm` / `fails_key` / `unsure`。理由コードの `wrong_instrument` はそのまま使う。
- 監査: 自動ラベルが付いた結果からも 5〜10% をランダムにキューへ入れ、gold そのものの誤り率を測る（`audit_rate`）。
- 各ベースラインの top-k を重複なしでまとめてからキューに入れる（同じクエリと同じファイルは 1 回だけレビューする）。
- 1 回の実行ごとに新しい DB を使う（既存の Duplicate チェックに従う）。例: `.hil/cond-v1.db`。

### 3.5 ベースラインの比較

| 名前 | 方法 |
|---|---|
| B0 filename | `tokens` に対するキーワードと BM25 の検索。条件を直接フィルタにした版も作る |
| B1 CLAP | 既存の `search_by_text(text)` |
| B2 hybrid | CLAP のスコアにメタデータのフィルタやブーストをかける（例: 条件を満たす候補だけを CLAP で並べ替え） |
| B3 CLAP+zero-shot | CLAP でラベルを予測し、その予測でフィルタする（ファイル名を使わない） |

**重要な注意（ラベルのリーク）**: gold はファイル名から作っているので、**B0 と B2 は構造上ほぼ満点になる**（評価しているのは抽出規則の正しさであって、検索の性能ではない）。そこで次のように分けて評価する。

1. **CLAP 単体の評価（B1/B3）** は、ファイル名を見ていないので、そのままの比較が公正になる。
2. **人が確認したサブセット**: 規則が unknown や low を出したファイル、またはファイル名が楽器を示さないファイル（`Loop_07.wav` など）を中心に人がラベルを付け、そこで全ベースラインを比べる。ファイル名検索が効かない領域での CLAP の価値を測る。
3. **ファイル名を隠すテスト**: 検索時に B0/B2 のインデックスから楽器のキーワードを除き、フォルダやパックだけから推定させる。どの信号に依存しているかを分解する。
4. B0 の結果は「メタデータの品質の上限値」として報告し、検索性能の比較表には含めないか、注記をつけて載せる。

---

## 4. 次のステップ（SSD の調査で必要なもの）

読み取りのみ、深さを制限、出力は件数と例だけ（中身はコピーしない）:

1. **ルート直下とパックの一覧**: `find "/Volumes/Extreme SSD" -maxdepth 2 -type d`。パック数と、ベンダーやジャンルによる階層の有無を見る。
2. **拡張子ごとの本数**: wav / aif(f) / mp3 / flac / rex / ncw / nki / als / mid など。オーディオ以外（Kontakt、プロジェクトファイル、MIDI）の比率を確かめ、対象範囲を決める。
3. **パックごとの代表的なファイル名**: 各パックから 10〜20 件程度。命名テンプレート（BPM・キー・楽器の位置）を確かめる。
4. **フォルダ名の頻度上位 200 件**（`Kicks`、`Loops`、`One Shots` など）。辞書の当たり具合を見積もる。
5. **トークンの頻度上位 300 件**。§1 の辞書にない語を洗い出す。
6. 重複（同じファイル名・同じサイズ）や、`._*` などのゴミファイルの量。

これを受けて、辞書とパック別テンプレートを確定し、`scripts/build_catalog.py` を実装する（テストは ESC-50 と同じく合成パスで行う）。そのうえでカタログの統計（unknown 率）を確認し、クエリセットの案と `soundai-labels` のレビューを始め、最後にベースラインの比較へ進む。

### 未確定の論点
- 対象の拡張子（rex や Kontakt の素材を含めるか。現状の `SUPPORTED_EXTENSIONS` は wav/aif(f)/mp3/flac/ogg）。
- 415GB を CLAP で一度に ingest するか、パックごとのサブセットから始めるか（Mac mini の CPU で何時間かかるか）。
- genre を条件に含めるか（パック名に頼るので精度が不安定）。
- hil-eval v0.1.1 でラベルの「修正値の入力」（edit）に対応しているか。
