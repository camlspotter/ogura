# 日本語OCR学習用テキスト

日本語Wikipediaの抽出済み本文を取得し、文字別の出現頻度を集計する。
本文の取得・集計、対象字種の整理、25文字以下の学習サンプル選択を行う。画像生成は行わない。

## 現在の長さ条件：20〜25文字

最小長を20文字に変更した。原文に十分な長さの区間がない希少字は、20〜25文字の原文への置換で補う。
原文抽出・置換補充・最終検証のすべてでこの条件を適用する。

```sh
uv run --frozen python build_training_data.py --goal 100 --min-length 20 --seed 20260915
```

再生成・検証済みの結果は780,503件（自然文564,622件、置換合成215,881件）。
全16,057字種で100件以上、全件が20〜25文字。
既定値も20文字。設定が同じ完成済みデータは再利用する。
新しい出力は `data/training_text_final_100_len20_25/`、レポートは `results/training_text_final_100_len20_25/`。
以前の1〜25文字版は履歴として残している。下記の641,489件という数は旧版の結果。
旧条件を再現する場合は `--min-length 1 --name old_length_run` のように指定する。

## 全工程の実行（共通手順・旧結果）

```sh
uv sync --frozen
uv run --frozen python build_training_data.py --goal 100 --seed 20260915
```

初回は固定したWikipediaスナップショットの取得・文字集計、チェックサム付きUnicode資料の取得、
字種選定、原文索引、希少字優先の抽出、不足字の置換補充、検証を順に実行する。
既に完了した工程は再利用する。既存の同設定データを上書きせず、再実行は完了確認となる。
初回の取得にはネット接続・curlと数GB以上の空き容量が必要。
`.python-version` と `uv.lock` で実行環境を固定している。

以前の1〜25文字版の完成データは `data/training_text_final_100/train.txt`（641,489件）。
自然文585,660件、置換合成55,829件で、全16,057字種が100件以上。
出典付き統合データは `train.jsonl`、合成分のみは `synthetic.jsonl`。
集計・検証は `results/training_text_final_100/`。

再生成・別設定では新しい名前を指定する。

```sh
uv run --frozen python build_training_data.py --goal 100 --seed 20260915 --name rerun100
uv run --frozen python build_training_data.py --goal 200 --seed 20260915 --name goal200
```

`--seed` は置換の乱数シード。記事の学習・検証・テスト分割のシードは固定の20260915。
同じ入力・環境・設定なら同じ文字列を作る。設定が異なる既存出力の再利用はエラーにする。
途中で失敗して未完成のtrain.jsonlが残った場合も、上書きせず停止するため、新しい `--name` を指定する。
字種やプログラムを変更した場合も、新しい名前で再生成する。

置換元は、学習用Wikipediaから直接抽出した自然文のうち、対応クラスの文字を含むものからランダムに選ぶ。
漢字は漢字の位置、`〄` はUnicode一般カテゴリSoの記号の位置に置換する。
1文字列につき1文字を置換し、不足字を優先して目標まで補う。対象字種のみ、25文字以内、完全重複なし。
全合成行に元テキスト・元サンプルID・置換位置・置換前後の文字を記録する。
検証工程では合成全件を元記事から復元して照合する。検証・テスト記事は合成元にも使わない。
学習の認識精度はまだ評価していない。100件は初期目標であり精度を保証する値ではない。

```sh
uv run --frozen python -m unittest test_synthesis
```

以下には途中工程・旧版の記録も残している。

## 現在の対象集合（範囲を絞った版）

現在は `results/charset_refined/targets.jsonl` の16,057項目を対象とする。
以前の18,413項目から2,422項目（IVS付き99項目を含む）を除き、基本キリル文字66項目を追加した。

| グループ | 項目数 |
| --- | ---: |
| 漢字 | 15,425 |
| 仮名・関連記号 | 268 |
| ISO-8859-1のラテン文字（ª・ºを含む） | 116 |
| 基本ギリシャ文字 | 49 |
| 基本キリル文字（ロシア語） | 66 |
| その他の数字・記号・空白 | 133 |
| 合計 | 16,057 |

仮名は通常のひらがな・カタカナ、半角カナ、カタカナ音声拡張と関連記号。
変体仮名・閩南語用カタカナ・仮名拡張の歴史的文字は対象外。
ラテン文字はISO-8859-1の範囲に限定し、全角英字も除外する。
ギリシャ文字は大小各24文字と語末シグマ（ς）。アクセント・記号用の別形は含めない。
キリル文字はロシア語の大小各33文字（Ё・ёを含む）。他言語の追加キリル文字は含めない。
仮名の共有記号という理由だけで「。」「、」「「」「」」等を除外しない。
漢字集合は維持し、IVS付き99項目は除外する。この範囲にWikipedia未出現の項目はない。
新集合による抽出では、漢字直後の異体字セレクタ（U+E0100〜U+E01EF）を除いて元の漢字にまとめる。
原文と開始・終了位置は元のまま保存し、照合時だけ同じ処理を行う。旧IVS集合を指定した場合は従来の処理を維持する。

再生成: `uv run --frozen python refine_charset.py`。
追加・除外の一覧と定義は `results/charset_refined/` に保存する。

既存の `data/training_text/` の1,907,115件は旧18,413項目版の成果物であり、
新集合の再抽出先は `data/training_text_refined/`、集計先は `results/training_text_refined/`。
抽出スクリプトの既定対象は新集合に変更済み。再抽出時は別の `--output` と `--report-dir` を指定する。
旧版の再現には `--targets data/training_text/targets.jsonl` を指定する。

```sh
uv run --frozen python extract_training_text.py --output data/training_text_refined --report-dir results/training_text_refined
uv run --frozen python verify_training_text.py --output data/training_text_refined --report-dir results/training_text_refined
```

### 新集合の再抽出結果

再抽出・検証は完了。重複しない1〜25文字の学習テキスト1,874,789件を
`data/training_text_refined/train.txt`（1行1件）に保存した。
出典と原文位置は同じディレクトリの `train.jsonl` にある。
16,057文字中、500件以上は4,335文字、1〜499件は11,075文字、採用0件は647文字。
全件の文字集合・長さ・重複・記事分割・集計等を検証し、
別工程で1,903件を元記事と照合した（IVS除去を適用）。
抽出時にも全件で原文との対応を確認している。
詳細は `results/training_text_refined/report.md`、
不足字は `results/training_text_refined/shortfalls.jsonl` を参照。

### Wikipedia内の不足字追加抽出

`supplement_training_text.py` で学習用記事から不足字を追加回収した。
合成は行わず、IVS除去以外は原文の連続部分文字列を使う。
同じ文脈・原文区間から異なる範囲を採ることと、記事ごとの主対象5件上限の解除を許可した。
そのため500件の異なる文字列が500件の独立した用例を意味するわけではない。

追加3,219,981件、統合後5,094,770件。
500件以上10,891文字、1〜499件4,985文字、採用0件181文字となった。
最新の統合データは `data/training_text_supplemented/train.txt` と `train.jsonl`、
追加分だけの出典付きデータは `additional.jsonl`。
詳細と不足字は `results/training_text_supplemented/report.md` と `shortfalls.jsonl`。

```sh
uv run --frozen python supplement_training_text.py
uv run --frozen python verify_training_text.py --output data/training_text_supplemented --report-dir results/training_text_supplemented
```

### 希少文字優先での圧縮（500件目標の旧版）

`compact_training_text.py` で候補の少ない文字から処理し、他の不足字も多く含む
文字列を優先して選択した。最後に収録要件を満たしたまま除去できるサンプルを削除した。
5,094,770件から3,028,140件に削減（40.6%減）。各文字の min(500, 以前の収録数) を維持した。
500件以上10,891文字、1〜499件4,985文字、未収録181文字は変わらない。
現在の候補と維持条件では少なくとも1,726,039件は必須。最小件数の保証はない。

最新の学習データは `data/training_text_compact/train.txt` と `train.jsonl`。
詳細は `results/training_text_compact/report.md`。

```sh
uv run --frozen python compact_training_text.py --goal 500 --output data/training_text_compact --report-dir results/training_text_compact
uv run --frozen python verify_training_text.py --output data/training_text_compact --report-dir results/training_text_compact
```

### 現在の学習用集合（100件目標）

初期目標を各字種100件に変更した。これは必要数の保証ではなく、学習評価後に追加する運用上の目標。
既存の全候補から希少文字優先で選び直し、528,127件にした。
100件以上13,277字種、1〜99件2,599字種、未収録181字種。
各字種 min(100, 候補内収録数) を維持し、合成は行っていない。
最新データは `data/training_text_100/train.txt` と `train.jsonl`。
集計・不足字は `results/training_text_100/`。

```sh
uv run --frozen python compact_training_text.py --goal 100
uv run --frozen python verify_training_text.py --output data/training_text_100 --report-dir results/training_text_100
```

### Wikipedia原文からの直接選択

`select_wikipedia_direct.py` で全記事を索引化し、学習用記事内の出現頻度が低い字種から原文を探索した。
旧候補に制限せず、各記事内で未充足字種を多く含む1〜25文字の区間を優先した。
結果は585,660件。100件以上13,277字種、1〜99件2,599字種、未収録181字種。
字種の収録状況は旧候補からの100件版と同じだが、総件数は57,533件増えた。
記事間の候補比較は行っておらず、希少順の貪欲選択に最小件数の保証はない。
データは `data/training_text_direct_100/`、詳細は `results/training_text_direct_100/report.md`。
原文索引は `data/wikipedia_source_index/` に保存し、再利用できる。

```sh
uv run --frozen python select_wikipedia_direct.py
uv run --frozen python verify_training_text.py --output data/training_text_direct_100 --report-dir results/training_text_direct_100
```

## 元データ

- 配布元: https://huggingface.co/datasets/wikimedia/wikipedia
- サブセット: `20231101.ja`（2023年11月1日版）
- 固定リビジョン: `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`
- Parquet 15ファイル、3,941,998,526 bytes。記事ID・URL・タイトル・本文を含む原データを保存する。
- 配布元でマークアップや参考文献等を除去した本文であり、Wikipedia原ダンプの全内容ではない。
- データセットカードのライセンス表記: CC BY-SA 3.0 / GFDL。出典情報を保持する。

## 実行

Python 3.12以上、uv、curlを使用する。

```sh
UV_CACHE_DIR=/tmp/ogura-uv-cache uv run --frozen python analyze_wikipedia.py
```

固定リビジョンから取得し、ファイルサイズと公開LFS SHA256を検証する。
取得済みファイルと各ファイルの集計キャッシュは再利用する。
集計条件を変更した場合は `data/counts/` の該当キャッシュを削除して再実行する。

## 出力

- `data/20231101.ja/`: 原データ。約3.94GB。
- `data/counts/`: ファイルごとの集計キャッシュ。
- `results/source.json`: 固定リビジョン、取得ファイル、サイズ、ハッシュ、集計定義。
- `results/character_counts.jsonl`: 頻度降順の文字別集計。
- `results/summary.json`: 記事数、総文字数、字種数、頻度帯別・Unicodeカテゴリ別の集計。

文字別集計の各行には `character`, `codepoint`, `name`, `category`,
`occurrences`（本文での総出現回数）, `article_count`（出現する記事数）を保存する。
同頻度ならコードポイント順。改行等はJSONのエスケープで表現する。

## 集計の定義

本文のUnicodeコードポイントをそのまま数える。タイトルは別途加算しない。
正規化・文字の除外・重複記事の除外は行わない。全角／半角、異体字、
結合文字、空白、改行、外国語の文字も区別して残す。
これは表示上の文字（書記素）の数やCTC辞書確定後のトークン数とは異なる。

500回以上出現しても、異なる学習サンプルを500件確保できるとは限らない。
未出現の文字は対象文字集合が未確定のため、この集計には含まれない。

## 後続のサンプル選択方針

- 1サンプル最大25文字。短い文字列も含める。
- 各文字を含む異なるサンプル500件を初期目標とする。
- 不足文字を優先し、頻出文字の超過は許容する。
- 重複や近似重複、同一記事・語句への集中を抑える。
- Wikipediaで不足する文字は記録し、補完方法は後で検討する。
- 学習・評価に使う場合は記事単位の分割をサンプル抽出前に行う。

## 全件集計結果（2026-09-15実行）

| 指標 | 件数 |
| --- | ---: |
| 記事数 | 1,389,467 |
| 空の本文 | 0 |
| 本文の総コードポイント数 | 2,658,082,408 |
| 異なるコードポイント数 | 22,723 |
| 1〜99回出現する字種 | 15,577 |
| 100〜499回出現する字種 | 2,066 |
| 500回以上出現する字種 | 5,080 |

参考として、Unicode名が `CJK UNIFIED IDEOGRAPH-` で始まる文字は15,362字種、
そのうち500回以上出現するものは4,186字種。互換漢字等を含む全漢字の集計ではない。

全15ファイルのサイズ・SHA256を照合済み。
Parquetメタデータの記事数と、Arrowの `utf8_length` による独立した全文字数集計が一致した。
文字別出現回数の合計、字種数、記事数の範囲も確認した。

## 字種候補の分類とIVS集計

```sh
mkdir -p data/charset_sources
curl -fL https://www.unicode.org/ivd/data/2026-08-03/IVD_Sequences.txt -o data/charset_sources/IVD_Sequences-2026-08-03.txt
curl -fL https://www.unicode.org/Public/15.0.0/ucd/Scripts.txt -o data/charset_sources/Scripts-15.0.0.txt
curl -fL https://www.unicode.org/Public/15.0.0/ucd/ScriptExtensions.txt -o data/charset_sources/ScriptExtensions-15.0.0.txt
UV_CACHE_DIR=/tmp/ogura-uv-cache uv run --frozen python classify_characters.py
```

結果は `results/charset/` に保存する。

| ファイル | 内容 |
| --- | --- |
| `candidates.jsonl` | 暫定候補18,413項目。頻度降順、IVSは組み合わせを1項目として扱う |
| `required_characters.jsonl` | 頻度に依存せず確保する2,881文字。関連グループを記録 |
| `required_unobserved.jsonl` | 網羅枠のうちWikipedia未出現の1,462文字。出現数・記事数とも0 |
| `review.jsonl` | その他の外国語・一般記号・結合文字・私用文字等、要確認5,817字種 |
| `excluded.jsonl` | 制御・書式文字等と単独セレクタ54字種。原データは保存したまま |
| `classified_codepoints.jsonl` | 全22,723コードポイントの分類・頻度 |
| `ivs.jsonl` | 登録済みIVS 99組、合計652出現。出現記事数と登録元も記録 |
| `unregistered_vs_sequences.jsonl` | IVDにない直前文字＋セレクタ等2種類、5出現。自動修正しない |
| `summary.json` | 分類別件数・集計条件・IVDのハッシュ |

暫定候補の内訳は、Wikipediaに出現した単独コードポイント16,852文字、
網羅枠の未出現文字1,462文字、登録済みIVS 99組。
漢字15,425文字は引き続き実出現から選んだ候補であり、漢字全体を網羅しない。
一般記号等には要確認のものが残るため、必要なものを選んで追加できる。
約18,000は目安であり、数合わせによる採用はしない。この18,413項目を初期対象として採用する。

仮名・ラテン文字・ギリシャ文字は頻度にかかわらず網羅枠として確保する。
網羅範囲はUnicode 15.0.0のScriptおよびScript_Extensionsで
Hiragana/Katakana/Latin/Greekが指定されたコードポイントの和集合。
ASCII可視文字・空白、全角英数字・記号も確保する。各文字に `required: true` を付ける。
小書き・半角・歴史的仮名、拡張ラテン文字、大小文字、関連する結合記号等も含む。
任意長の結合文字列や、Script=Commonの数学用装飾アルファベット全体を
網羅するものではない。他の文字体系全体はまだ網羅枠に指定していない。

関連グループ別では仮名754、ラテン1,510、ギリシャ522、基本英数字・記号199。
共有記号・結合文字等は複数グループに属するため、重複を除く合計は2,881文字。
分類元データのURL・SHA256は `summary.json` に記録する。

分類はPythonのUnicode 15.0.0の文字名・カテゴリとコードポイント範囲を使った
機械的な候補分けであり、文字の使用言語や文脈を判定したものではない。
特に漢字は日本語と中国語で共有されるため、中国語の引用等も候補に含まれる。
Unicodeで未割り当てと判定された文字や私用文字も要確認一覧に残す。
網羅枠の文字は元の分類グループを維持したまま候補へ昇格させる。
`classified_codepoints.jsonl` は観測された22,723文字のみを維持し、
未出現文字は網羅枠一覧と候補一覧に追加する。一般記号の採否は次の候補確定時に行う。

IVSは2026-08-03版IVD全体と照合して登録済みの組み合わせを数えた。
U+E0100〜U+E01EFの全出現が登録済み652回＋未登録等5回と一致することを検証した。
U+FE00〜U+FE0Fによる標準化異体字シーケンスや絵文字シーケンスは
今回のIVS集計には含めず、そのセレクタ単独も候補にしない。

`occurrences` は元のコードポイント出現数を維持し、
`standalone_occurrences` は直後に登録済みIVSセレクタが付いた分だけ差し引く。
候補の `ranking_occurrences` はこの単独出現数（IVS項目では組の出現数）を使う。
コードポイントの `article_count` は元の集計のままで、IVS付きの出現記事も含む。
候補の頻度帯は500回以上4,709項目、100〜499回1,590項目、100回未満12,114項目（未出現を含む）。
これらは異なる学習サンプルの件数ではない。

## 学習用テキストの抽出

```sh
UV_CACHE_DIR=/tmp/ogura-uv-cache uv run --frozen python extract_training_text.py
UV_CACHE_DIR=/tmp/ogura-uv-cache uv run --frozen python verify_training_text.py
```

`data/training_text/` に出力する。既存の `train.jsonl` がある場合は上書きせず停止する。
別の抽出実験には `--output` で空の出力先を指定する。
検証スクリプトは標準出力先を対象にする。

- `train.jsonl`: テキスト・長さ・記事ID・URL・タイトル・元本文の開始／終了位置・抽出対象字種。
- `train.txt`: 全件検証後に出力する、1行1サンプルのプレーンテキスト。
- `targets.jsonl`: 抽出開始時点の対象18,413項目の固定コピー。
- `article_splits.jsonl`: 全記事の学習／検証／評価の割り当て。元本文はParquetに保持。
- `coverage.jsonl`: 字種ごとの採用サンプル数・不足数・全コーパス出現数。
- `manifest.json`: コーパス版・対象集合ハッシュ・乱数種・抽出条件。
- `summary.json`: サンプル総数、実際の長さ分布、目標達成／不足の件数。

頻度表・設定・要約は `results/training_text/` にも保存する。

### 抽出条件

記事IDと固定seedのSHA256で90%／5%／5%の区画に割り当てる。
これは比率の目安で、件数を厳密に90:5:5に調整するものではない。
学習用の記事のみから抽出し、検証・評価は記事を予約する段階までとする。
異なる記事に転載された同一・類似の文章までは分割時に検出していないため、
後で評価サンプルを作る際には学習テキストとの重複も確認する。

各字種500サンプルを目指して不足字を含む箇所を選ぶ。
対象文字のみからなる元本文の連続部分を使用し、改行や対象外文字を飛び越えて連結しない。
最大25トークン。通常は1コードポイント1トークン、漢字＋IVSは1トークン。
元本文の切り出し位置はPython文字列のコードポイント単位、終了位置は範囲外。

長さは1〜25から提案し、採用済み件数の少ない長さの重みを高くする。
各主対象位置で最大3回試し、最後の試行は対象文字が連続する区間に収まる長さを提案する。
1文字サンプルには完全重複除去による件数の上限があるため、1文字への重み集中は制限する。
安全な境界や重複排除により最終的な長さ分布は一様とは限らず、実測分布を要約に記録する。
空白や結合記号で始まるサンプル、空白で終わるサンプル、直後の結合記号を
切り離す境界は採用しない。単語・文の途中での切り出しは許容する。

完全に同じ文字列はSHA256で重複除去する。
不足字の前後各6トークン以内の文脈を、その字種について重複させない。
同一行の採用区間は重ねず、同じ記事から同じ字種を主対象とする採用は最大5件。
副次的に含まれる頻出字の件数はこの上限を超えてよい。
これらは近似重複を抑える規則で、全テキスト間の意味的・編集距離的な重複排除ではない。

不足は、コーパス未出現・出現はあるが採用0件・採用あり500件未満に分ける。
探索は貪欲法であり、条件や切り出し方を変えても500件に達しないという証明ではない。
`sample_count` は主対象・副次対象を問わず、その文字を含む採用文字列の件数。
`anchor_article_count` はその字種を主対象に採用した記事数のみを表す。

### 初回抽出結果

1,907,115サンプルを作成した。対象18,413項目のうち、500件以上4,354項目、
1〜499件11,909項目、コーパス出現あり・採用0件688項目、コーパス未出現1,462項目。
2〜25文字は各72,030〜87,999件、1文字は9,710件。

全件について対象文字・長さ・重複・記事分割・元本文内区間の非重複・
記事ごとの主対象上限・字種別集計を検証した。
抽出時には全件で元本文との一致を確認し、別工程ではハッシュによって選んだ
1,852件を元記事から独立に照合して一致を確認した。

詳しいグループ別件数と長さ分布は `results/training_text/report.md`、
検証結果は `results/training_text/verification.json` にある。
