# Synth-JDocによるtextdetデータの試作

公式Synth-JDocのHTMLテンプレート・段組み・縦中横処理を利用する。
ソースの固定版とライセンスは `vendor/README.md` を参照。
公式の画像生成モデルやLLM処理は呼ばず、手元の文章とフォントからCPU上のChromiumで生成する。
ブラウザでは出力ディレクトリ内のHTMLとフォントだけを読み込み、外部ネットワーク要求は遮断する。ブラウザ本体の初回インストールのみダウンロードが必要。

## 実行

リポジトリルートで実行する。依存関係はtextrecと共通のpyproject/uv.lockで管理する。

```sh
uv sync
export PLAYWRIGHT_BROWSERS_PATH="$PWD/ogura/textdet/.cache/playwright"
uv run playwright install chromium
uv run python -m ogura.textdet.synth_jdoc --count 6
```

LinuxでChromiumのシステムライブラリが不足する場合は、そのマシン上で
`uv run playwright install-deps chromium` を実行する（OSパッケージの導入権限が必要）。
GPU、SSH、APIキーは不要。既存出力先は上書きしない。別の `--output` を指定する。

既定フォントは `corpus/fonts/NotoSansCJKjp-Regular.otf`。
別のローカルフォントは `--font PATH` で指定する。文章中の文字をフォントが持っているか検査する。
フォントは出力の `fonts/` に一度だけコピーしてHTML間で共有する。生成物の配布時もフォントのライセンスに従う。

最初の6ページは同じ架空の文章を横書き・縦書き、1〜3段で描いた確認用サンプル。
大量学習用の文章集ではない。フォントサイズは6ページごとに20・16・24pxを巡回する。
`--font-sizes 12 16 20 24`、`--width 1200 --height 1600` で変更できる。
縦書きで画像外へ文字が出た場合は失敗として記録する。文章を短くするか幅を広げる。

## 出力

既定の `ogura/textdet/outputs/synth-jdoc-pilot/` に保存する。

- `images/`: 原画像
- `labels.json`: docTR DetectionDataset形式、文字行の矩形4頂点（画像ピクセル座標）
- `review/`: XOR反転でbboxと番号を描いた確認用画像
- `metadata/`: 行テキスト・方向・文字位置・生成条件・元文章ID
- `html/`: 実際に描画したHTML
- `fonts/`: 共有するローカルフォントと同じディレクトリにあるLICENSEファイル
- `manifest.json`: 完了状態、seed、入力とフォントのハッシュ、ブラウザ版、公式コードの版

生成後も正解は `candidate_needs_visual_review`。行の分割・余白・縦中横を確認してから使う。
DOMの文字範囲は字体の黒画素そのものではなくフォントのレイアウト上の範囲。
先頭末尾の空白と空白のみの行を除外し、段落・段組みを越えて結合しない。
縦中横はブラウザのspanの位置で扱い、行の中に含める。
図表・挿絵・ノイズはまだ対象外。位置を変えるノイズを追加する場合はbboxも同時変換する必要がある。

## 自分の文章を使う

JSONLで1行1文書。HTMLは渡さず、通常の文字列を渡す（エスケープして描画）。

```json
{"id":"report-001","title":"見出し","paragraphs":["第一段落。","第二段落。"]}
```

```sh
uv run python -m ogura.textdet.synth_jdoc \
  --input ogura/textdet/outputs/source-texts.jsonl \
  --count 100 --output ogura/textdet/outputs/synth-jdoc-pilot-100
```

入力文書を順に使い、足りなければ巡回する。100枚の異なる文章を作るには入力文書も増やす。
自動でtrain/val/testへ分割はしない。同じsource_idのレイアウト違いを別splitへ混ぜないこと。
今回のJDocQAのval/testはそのまま固定し、合成データの学習効果の確認に使う。

## 検証

```sh
RUN_SYNTH_BROWSER_TESTS=1 uv run python -m unittest ogura.textdet.tests.test_synth_jdoc
```

ブラウザを使うテスト以外は通常の `unittest discover -s ogura/textdet/tests` でも実行する。


## 120ページのバリエーション試作

学習はdocTR配布の重みから合成データだけでやり直す。既存のJDocQA train 308ページは使わず、
val 39ページ・test 40ページは維持する。以下の画像は学習前の目視確認用であり、自動採用はしない。

手元のWikipedia Parquetから120件の異なる記事を選び、本文を抜粋する。
既存のローカルデータだけを使い、ダウンロードはしない。
各シャードの先頭128件までを候補にしてシャッフルする限定的な試作用サンプルで、全Wikipediaからの一様抽出ではない。
短い見出し行は本文から除き、原文の文字は置換しない。指定した全フォントで表示できる記事だけを選ぶ。
出典URL・記事ID・タイトル・元文章ハッシュ・抜粋の説明をJSONLに残す。
本文と生成画像はGit管理外。公開する場合は元のWikipediaデータとフォントのライセンス・帰属表示を維持する。

```sh
uv run python -m ogura.textdet.prepare_synth_texts \
  --font corpus/fonts/NotoSansCJKjp-Regular.otf \
  --font corpus/fonts/NotoSansCJKjp-Bold.otf \
  --font corpus/fonts/NotoSerifCJKjp-Regular.otf \
  --font corpus/fonts/NotoSerifCJKjp-Bold.otf

uv run python -m ogura.textdet.synth_jdoc \
  --input ogura/textdet/outputs/synth-texts-120.jsonl \
  --count 120 --vary-layout --fill-page --font-sizes 12 16 20 24 \
  --extra-font corpus/fonts/NotoSansCJKjp-Bold.otf \
  --extra-font corpus/fonts/NotoSerifCJKjp-Regular.otf \
  --extra-font corpus/fonts/NotoSerifCJKjp-Bold.otf \
  --output ogura/textdet/outputs/synth-jdoc-filled-120-v1
```

先に上記セットアップの `PLAYWRIGHT_BROWSERS_PATH` を設定する。
同じseedと入力でレイアウト選択を再現でき、ブラウザ版とコードのハッシュもmanifestに記録する。

- 横書き90、縦書き30（`--vertical-fraction 0.25`）
- 4書体 × 各30ページ
- 文字サイズ12・16・20・24px × 各30ページ
- 1〜3段組み × 各40ページ
- 行間1.5・1.7・2.0、字間0・0.03・0.08em × 各40ページ
- 原文の長さ上限400・700・1000・1400文字。文章の長さと段落数も異なる

各軸を別々にシャッフルするため、例えば「明朝は常に縦書き」のような固定対応にはしない。
すべての組み合わせを網羅する設計ではない。実際の分布はmanifestのdistributionsで確認する。
文字抽出結果が描画した本文と一致すること、すべてのbboxが画像内にあることを生成時に検査する。
ただし目視確認の代わりではなく、bboxは引き続きブラウザのレイアウト座標。
見出し・本文以外の表や注記専用配置、スキャン風ノイズは次段階とする。


## 紙面を埋める生成

大量生成には `--fill-page` を指定する。文字サイズを大きくして余白を隠すのではなく、
複数の記事の本文を順に追加して1ページを超える量を組版する。追加記事のタイトルも本文の段落として配置する。
同じページ内で記事を繰り返さず、入力文章集が足りなければエラーにする。

画像は指定した幅・高さに固定する。横書きは左の段から右へ、縦書きは上の段から下へ流し、
紙面の本文領域を超える最初の行以降をDOMから取り除いてから画像とbboxを作る。
行や縦中横の途中で画像を切らず、画像に文字があるのに正解bboxがない状態を避ける。
保存HTMLも切り詰め後の内容で、余った本文は出力画像・ラベルに含まれない。

既定では80%を入るだけ本文を置くページにする。
残り20%（`--partial-fraction 0.2`）は収容可能な本文行数の50〜85%で終了し、途中で終わるページも残す。
これは文字の黒画素による占有面積の指定ではなく、収容行数に対する割合。
各ページの `pagination` に収容行数・採用行数・本文領域を、`source_ids` に実際に使った記事IDを記録する。
metadataには結合候補の記事の出典と段落ごとの記事IDも残す。

`--fill-page` を省略すると従来の短い入力をそのまま描く試作モードになる。

## 学習用5,000ページの生成

設定を `scripts/synth_5000.sh` に固定してある。リポジトリルートで実行する。
画像生成はCPU上で行い、GPUは学習時だけ必要。以下は利用者が生成先のマシンで実行する手順。

```sh
uv sync --locked
export PLAYWRIGHT_BROWSERS_PATH="$PWD/ogura/textdet/.cache/playwright"
uv run --locked playwright install chromium
```

入力は `corpus/wikipedia/20231101.ja/*.parquet` と `corpus/fonts/` の4書体。
Wikipediaは `wikimedia/wikipedia`、版 `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`、
設定 `20231101.ja` の15シャードを使う。既存のtextrec用コーパス・フォントがあれば共用する。
未取得の場合は既存の取得コードを使える（Wikipedia取得コマンドは文字頻度の集計も行うため時間がかかる）。

```sh
uv run --locked python -m ogura.analyze_wikipedia
uv run --locked python -m ogura.download_training_fonts
```

生成は2段階。最初に5,000件の異なる記事から本文を用意し、次に画像と正解を作る。

```sh
bash ogura/textdet/scripts/synth_5000.sh prepare
bash ogura/textdet/scripts/synth_5000.sh generate
```

- 本文: `outputs/synth-texts-5000-v1.jsonl`。記事あたり最大2,000・4,000・8,000文字。
- 画像・正解: `outputs/synth-jdoc-5000-v1/images/` と `labels.json`。
- 横書き3,750、縦書き1,250。4書体・4文字サイズはそれぞれ1,250ページずつ。
- 紙面を埋める4,000ページ、途中で終了する1,000ページ。表は含めない。
- 短い記事は次の記事を続け、各ページ内では同じ記事を繰り返さない。
- 抽出候補は各シャードの先頭2,048件まで。全Wikipediaからの一様抽出ではない。

`manifest.json` の `status` が `complete` で5,000ページあることを確認してから学習する。
docTRの `--train_path` は `ogura/textdet/outputs/synth-jdoc-5000-v1` を指定する。
JDocQAの既存val 39ページ・test 40ページは固定し、合成データを分割してそこに混ぜない。
モデルはdocTR配布の事前学習済み重みから開始し、以前のJDocQA学習済み重みを再開に使わない。

両コマンドは既存出力を上書きせず、途中からの再開は未対応。
失敗時は原因を直して、CLIで別の出力先を指定するか、不要と確認した失敗出力だけを手動で片付ける。
`review/`・`metadata/`・`html/` も作るため、学習画像以外にもディスク容量が必要。
本文・画像・ラベル・HTML・フォントはGit管理外。コミット対象はコードとこの手順のみ。
OS・Chromium版・フォントが異なる場合は描画差があり得るので、manifestの実行環境・ハッシュも保存する。
