> Recognition project root: run `cd textrec` from the repository root first.
> Set up the dedicated environment with `uv sync --frozen`.

# 日本語OCR学習用テキスト

画像バッチ生成・CNN/CTC学習・中断後の再開は [TRAINING.md](TRAINING.md) を参照。

日本語Wikipediaの本文用例を優先し、各字種について50件に足りない分を、
ランダムなひらがなに異なる不足字を最大5字種埋め込んだ文字列で補う。
画像は生成しない。文字列は20〜25文字。

## 現在の完成データ

[datasets/final_50_len20_25_hiragana_mix5/train.txt](datasets/final_50_len20_25_hiragana_mix5/train.txt)

- UTF-8、1行1件、339,839件。
- Wikipediaの自然文248,668件、ひらがなへの埋め込み生成91,171件。
- 対象16,057字種すべて50件以上。補充対象は10,219字種。
- 自然文をランダム置換する旧方式は、現在の標準生成には使わない。
- 50件は収集目標であり、認識精度を保証する値ではない。

| ファイル | 内容 |
| --- | --- |
| train.txt | シャッフル済み学習用テキスト |
| train.jsonl | 自然文の出典、生成文の方式・対象字・埋め込み位置 |
| synthetic.jsonl | 生成文のみ、生成順 |
| targets.jsonl | 対象字種 |
| coverage.jsonl | 字種別の自然文件数、当該字のための生成件数、最終収録件数 |
| manifest.json | 生成条件、自然文入力・字種のSHA256 |
| summary.json | 件数 |
| reports/verification.json | 全行の形式・重複・字種・生成条件・件数の検証結果 |

生成文にWikipedia由来の出典は付けない。自然文と生成文は synthetic フラグで区別する。
coverage の generated_for_character は、その字を埋め込むために作った件数。
ひらがなは他の対象字の背景にも現れるため、最終件数はそれより多くなる。

## 再生成

今回使用した自然文入力を保持している。既存の自然文は削減せず、すべて使う。

```sh
uv sync --frozen
uv run --frozen python build_training_data.py --natural-base datasets/direct_100_len20_25_diverse
```

引数なしでも上記の自然文入力を使用する。完成済みなら条件を確認して train.txt を再出力する。
同じ入力・シードで新しいディレクトリに作り直す場合：

```sh
uv run --frozen python build_training_data.py --natural-base datasets/direct_100_len20_25_diverse --name regenerated_50_hiragana
```

現在の完成ファイルを、保存済みの自然文と生成文から完全一致復元する場合：

```sh
uv run --frozen python rebuild_current.py --output datasets/rebuilt_current
```

current_dataset.json が入力・出力・付随ファイルのSHA256とシャッフルシードを固定する。
その inputs と metadata が指すファイルは削除しないこと。

Wikipediaから自然文の抽出もやり直す場合：

```sh
uv run --frozen python build_training_data.py --fresh-natural --goal 50 --name fresh_50_hiragana
```

新規抽出は50件を目標とするため、100件目標で集めた今回の自然文入力とは異なる。
データはGit管理外なので、別の環境で現行版を復元するには入力ファイルもコピーする。
環境は .python-version と uv.lock で固定する。

## 本文用例の条件

- 学習用記事のみ、希少字から検索する。
- 文字表、仮名の羅列、空白区切りの漢字表を除外する。
- 候補そのものに日本語の文法表現と3文字以上のひらがなを要求する。
- 対象字種だけを含む20〜25文字。漢字のIVSを除去する。
- 原文区間の重複と、共通する16文字の原文断片を除外する。
- 本文判定はヒューリスティックであり、すべての自然な用例を拾えるわけではない。

今回再利用する自然文集合は、100件未満の字について索引内の掲載記事を探索済み。
50件未満の字はその集合での不足数を補充する。
自然文は既存の出典・重複検証結果を保持し、今回の生成では全行の文字列と件数を再検証する。
予約済みの検証記事から固定評価用の文章を抽出できる（TRAINING.md参照）。実文書での精度評価はまだ行っていない。

## 生成文字列の条件

- 各対象字の自然文用例を数え、50件に足りない分だけ生成する。
- 背景は通常のひらがな（清音・濁音・半濁音）。小書き仮名、旧仮名、結合濁点は使わない。
- 長さ20〜25と対象字の位置を乱数で選ぶ。
- 不足字を最大5字種ランダムに選び、それぞれ1文字ずつ異なる位置へ埋め込む。対象字は背景から除く。
- 不足が解消した字は候補から外す。残る不足字が5字種未満なら、その数だけ埋め込む。
- --targets-per-line で1行の最大字種数を指定できる（既定5）。
- 完全重複を除外。固定シードで生成し、別の同シード乱数で最後にシャッフルする。
- 意味のある文として扱わず、字形学習用の人工文字列として明示する。

## ディレクトリ

| 場所 | 内容 |
| --- | --- |
| ../corpus/ | Wikipedia本文と取得元情報、Unicode資料 |
| cache/ | 原文索引、文字頻度集計 |
| charset/candidates/ | 字種の分類結果 |
| charset/selected/ | 対象字種 |
| datasets/direct_100_len20_25_diverse/ | 再利用する自然文と reports/ の検証記録 |
| datasets/final_50_len20_25_hiragana_mix5/ | 現行の完成データ |
| ogura/ | 工程別処理・共通モジュール |
| tests/ | テスト |

## 字種と元コーパス

| グループ | 字種数 |
| --- | ---: |
| 漢字 | 15,425 |
| 仮名・関連記号 | 268 |
| ISO-8859-1内のラテン文字 | 116 |
| 基本ギリシャ文字 | 49 |
| ロシア語の基本キリル文字 | 66 |
| その他 | 133 |
| 合計 | 16,057 |

Wikipedia: wikimedia/wikipedia、20231101.ja、
リビジョン b04c8d1ceb2f5cd4588862100d08de323dccfbaa、1,389,467記事。
詳細は ../corpus/wikipedia/source.json。
変体仮名・閩南語用カタカナ・全角英字・独立したIVS字種は対象外。

## スクリプトとテスト

ルートの実行窓口は build_training_data.py と rebuild_current.py。
内部処理は python -m ogura.モジュール名 で実行する。
generate_hiragana が現行の補充処理。synthesize_shortfalls は旧置換方式で、
build_training_data.py --synthesis replacement を明示した場合のみ使う。

```sh
uv run --frozen --extra train python -m unittest discover -s tests
```

verify_training_text は出典付き自然文・旧置換文用の検証器。
現行の人工文字列は generate_hiragana 内で検証し、Wikipedia本文の条件を適用しない。

## Package layout

- `ogura/textrec/`: recognition, corpus preparation, rendering, training and evaluation.
- `../ogura/textdet/`: detection (independent development).

Use `python -m ogura.textrec.training.train` and `python -m ogura.textrec.recognize`.
The shared `ogura` is a PEP 420 namespace package. Use `ogura.textrec.*`;
legacy `ogura.recognize` / `ogura.training.*` entry points are no longer provided.
Both projects can be installed in one environment without replacing each other.
All commands in this document run from `textrec/`. The `datasets/`, `../corpus/`,
`config/`, and `runs/` paths are relative to this project directory.
Existing weight/state-dict checkpoints remain readable. Resume accepts the
pre-relocation training-code hash (6306d30), while retaining all other identity
checks; this does not bypass checks for unrelated historical code changes.

## Moving an existing checkout

Stop training before moving local assets. After pulling this layout, run once
from the repository root (safe to repeat):

```sh
python3 textrec/scripts/migrate_assets.py
cd textrec
uv sync --frozen
```

This moves `datasets`, `runs`, `cache`, `charset`, and `previews`
without rewriting manifests/checkpoints, and removes old compatibility links. Shared fonts and source data remain in
`../corpus/`. It refuses to merge two existing directories. Existing
GPU run configurations using relative paths work from the new project root.
When importing runs from another machine, use the existing `--font-dir` option
where needed. Historical manifests retain their original provenance paths.
