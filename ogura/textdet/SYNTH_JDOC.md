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
表は `--tables` で追加できる。グラフ・挿絵・ノイズはまだ対象外。位置を変えるノイズを追加する場合はbboxも同時変換する必要がある。

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
本文のみの生成に加えて、下記の表付き試作モードを利用できる。注記専用配置、スキャン風ノイズは未対応。


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

通常の生成は既存出力を上書きしない。失敗時は原因を直して次のコマンドで再開する。

```sh
bash ogura/textdet/scripts/synth_5000.sh resume
```

manifestに記録済みのページの画像・メタデータを検査し、そこまでは再描画せず、未完了ページから続行する。
入力本文・フォント・設定・Chromium版が異なる場合は停止する。修正前の固定5,000ページ設定も再開可能。
旧形式には画像ハッシュがないため、旧画像はデコード・寸法・bbox整合性で検査し、その制限を再開履歴に記録する。
新規ページでは画像ハッシュも保存・検査する。コード修正をまたいだ場合は旧コードのハッシュを再開履歴に残す。
境界には0.5pxの許容差を設けるが、bbox自体は切り詰めず、画像内の座標であることを検査する。
`review/`・`metadata/`・`html/` も作るため、学習画像以外にもディスク容量が必要。
本文・画像・ラベル・HTML・フォントはGit管理外。コミット対象はコードとこの手順のみ。
OS・Chromium版・フォントが異なる場合は描画差があり得るので、manifestの実行環境・ハッシュも保存する。


文字座標の初回取得・ページ切り詰め・再取得はブラウザ内でまとめて行い、
残す行の座標だけをPythonに返す。画像・ラベルの形式や生成設定は変更しないため、
高速化前の出力にも上記 `resume` を使える。実行中の生成を切り替える場合は、
先にCtrl+Cで停止し、終了を確認してからコードを更新して `resume` を実行する。
同じ出力先で2つの生成処理を同時に動かさない。


## 合成5,000ページでの学習

生成完了後、公式docTR v1.0.0の学習スクリプトを配置済みのGPUマシンで実行する。
初回の取得方法は [EXPERIMENT.md](EXPERIMENT.md) を参照。

```sh
bash ogura/textdet/scripts/synth_5000.sh train
```

モデル保存先 `ogura/textdet/outputs/db-resnet34-synth5000-v1` を学習開始前に作成する。
設定は事前学習済みdb_resnet34、5 epoch、batch 2、入力1024、学習率0.0001、AMP。
合成5,000ページを学習に、JDocQAの固定valを検証に使う。
`CUDA_VISIBLE_DEVICES` は未指定なら0で、指定済みならその値を使う。
`train` は新規学習の開始であり、`resume` は画像生成の再開専用。

`train` では旧CUDA AMP APIの `autocast`・`GradScaler` の非推奨FutureWarningだけを
Pythonの警告フィルタで抑制し、進捗バーへの割り込みを防ぐ。他の警告・エラーは表示する。
docTRの外部コードやAMPの計算方法は変更しない。起動済みの学習には反映されず、次回起動から有効。


## 表付き文書の試作

`--tables` で横書きの表を1つ配置する。既定の `--table-position both` は上部・下部をほぼ半数ずつにし、seedで配置順をシャッフルする。
`--table-position top` または `bottom` で片側に固定できる。下部の場合は本文を上に流し、表の高さを先に確保する。
この段階では横書きの表付きページだけを生成する。既存の縦書き本文ページとは別の出力先を使う。

```sh
uv run --locked python -m ogura.textdet.synth_jdoc \
  --input ogura/textdet/outputs/synth-texts-5000-v1.jsonl \
  --output ogura/textdet/outputs/synth-tables-pilot-v2 \
  --count 16 --vary-layout --fill-page --tables --table-position both --vertical-fraction 0 \
  --font-sizes 12 16 20 24 \
  --extra-font corpus/fonts/NotoSansCJKjp-Bold.otf \
  --extra-font corpus/fonts/NotoSerifCJKjp-Regular.otf \
  --extra-font corpus/fonts/NotoSerifCJKjp-Bold.otf
```

Chromiumの設定は本文生成と共通。表のラベル・数値は自作の架空データであり、本文記事の統計ではない。
本文の出典は従来どおり記録する。

- 罫線: 格子・横線・外枠のみ・なし。
- セル結合: 見出しの列結合、区分欄の行結合。結合なしの例も作る。
- 色付き見出し（濃色背景の白文字も含む）、右寄せの整数・小数・負数・割合。
- 空セル、複数行の備考。表の文字サイズ12・14・16・18px、本文は従来のサイズ指定。
- 6列固定、6・10・14行。配置・列構成は試作用の限定的なもの。

正解は各セルの文字行単位で、セルや表全体の矩形ではない。
空セル・罫線にはbboxを付けず、セルをまたいで行を結合しない。
表のすべての文字を抽出し、本文だけをページに収まるように切り詰める。
表が収まらない設定では、画像内のbbox検査でエラーにする。
metadataの `table`・`table_lines` とmanifestの `table_distributions` に表の条件を記録する。
`--resume` に同じ引数を追加すれば中断から再開でき、本文のみの生成との取り違えは設定検査で拒否する。

まず確認画像をレビューしてから大量生成へ進む。既存5,000ページや学習モデルは変更しない。

表は本文とは別領域に置くため、本文が途中で終わる例でも下部の表はページ下部に残る。
配置はmetadataの `table.position` とmanifestの `table_distributions.position` に記録する。
以前の上部のみの表データを再開する場合は `--table-position top` を明示する。
上下混在へ切り替える場合は新しい出力先を使う。

## 既存5,000ページ＋表付き2,000ページの実験

既存の `synth-jdoc-5000-v1` はそのまま保存し、次の手順を生成先のマシンで実行する。
既存の本文JSONL・フォント・Chromium・JDocQA検証データを使うため、本文抽出をやり直す必要はない。
表付きページの本文は既存と同じ記事集から取り、seedを20260925に変えて表と本文を組版する。
別の2,000記事を取得する処理ではなく、学習画像の追加である。

```sh
bash ogura/textdet/scripts/synth_tables_2000.sh generate
bash ogura/textdet/scripts/synth_tables_2000.sh combine
bash ogura/textdet/scripts/synth_tables_2000.sh train
```

生成が中断した場合だけ、最初のコマンドを `resume` に置き換える。

- 表付き2,000ページ: `outputs/synth-tables-2000-v1`。上部1,000・下部1,000、4書体各500。全て横書き。
- 結合した7,000ページ: `outputs/synth-mixed-7000-v1`。既存5,000＋追加2,000をそのまま各1回含める。
- 学習モデル: `outputs/db-resnet34-synth7000-v1/synth7000-db-resnet34-v1.pt`。

`combine` は両セットの完了状態・枚数・ページ種別を検査し、原画像だけをコピーする。
ファイル名に `text-` / `table-` を付けて同名衝突を避け、画像ハッシュを確認する。
元データの移動・変更は行わず、確認画像・詳細metadata・HTMLは複製しない。
結合先には原画像7,000枚分の追加容量が必要。学習ラベルと元ファイルの対応は結合先に保存する。
既存出力先への上書きと結合の途中再開は未対応。失敗した結合先は `status: failed` となり、学習を開始しない。

学習は元の事前学習済みモデルから5 epoch、1024入力で開始する。
JDocQAの固定val/testは維持し、既存の学習済みモデルは上書きしない。
1 epochの学習枚数が5,000から7,000へ増えるため、5 epoch同士の比較では更新回数も増える点に注意する。
本文のみのページと表付きページの混合比は5:2（表付き約28.6%）。

## タイトルのテキストボックス

生成コマンドに `--title-style mixed` を付けると、背景（なし・淡色・濃色）、枠（なし・実線・破線・二重線）、
帯（なし・書き出し側・末尾側）と配色・線幅をseedに従ってランダムに選ぶ。固定スタイルの個別指定は
`outline`, `dashed`, `double`, `tinted`, `dark`。
標準値 `upstream` は従来の見出し装飾を使うため、既存スクリプトの出力は変わらない。

これは本リポジトリ側のCSS追加で、Synth-JDoc本体は変更しない。
横書き・縦書き、表付きページにも使用可能。正解は既存のh1テキストから取得し、
枠・背景・内側余白をbboxに含めない。スタイル設定はページmetadataの
`style.title_box` と再開用manifest設定に記録する。
既存データとは別の出力先で生成し、再開時には同じ `--title-style` を指定する。

## 本文中の h2 / h3

`--fill-page --section-headings` で、つなぎ合わせる後続記事のタイトルを
h2/h3交互の見出しとして配置する。これは見た目の学習用の階層であり、
元の記事間に意味上の親子関係があることを示さない。
h2は本文の1.3倍、h3は1.1倍。背景・枠・帯・配色・線幅は見出しごとに
ランダムに選び、同じseedで再現する。濃色背景には白文字を使う。
選んだ装飾はmetadataの `style.section_headings` に段落IDごとに記録する。
`--title-style mixed` や `--tables` と併用できる。

入力JSONLに `heading_levels` を指定すれば、段落ごとの種別も指定できる。
`paragraphs` と同じ長さで、0は本文、2はh2、3はh3。省略時は全て本文。
例: `{"id":"example","title":"文書名","paragraphs":["節見出し","説明文","小見出し","詳細"],"heading_levels":[2,0,3,0]}`。
見出しの文字も行bboxを抽出する。ページ末尾に見出しだけ残る場合は除き、
枠線や背景はラベルに含めない。段落IDと出典対応は維持する。

見出しのランダム装飾では、角の半径も `0 / 0.2 / 0.4 / 0.6em` から選ぶ。
角のある枠と角丸の枠を混在させ、metadataには `border_radius_em` を保存する。
タイトルは `--title-style mixed`、h2/h3は見出しごとに適用する。

見出しの淡色・透明背景では、黒・紺・深緑・えんじ・紫の文字色をランダムに使う。
濃色背景では白文字を維持する。`--colored-text` を追加すると、本文もページ単位で
約30%を濃い有彩色、残りを従来の黒系にする。seedで再現でき、本文色は
metadataの `style.text_color` に保存する。

## 装飾付き見出し2,000枚の追加実験（合計9,000枚）

GPUマシンの `~/ogura` で実行する。既存の5,000枚・表2,000枚、入力JSONL、
JDocQAの固定val、docTR v1.0.0の学習コードを前提とする。追加のモデルや画像素材は不要。

```bash
uv sync
bash ogura/textdet/scripts/synth_headings_2000.sh generate
bash ogura/textdet/scripts/synth_headings_2000.sh combine
bash ogura/textdet/scripts/synth_headings_2000.sh train
bash ogura/textdet/scripts/synth_headings_2000.sh evaluate
bash ogura/textdet/scripts/synth_headings_2000.sh package
```

初回は `bash ogura/textdet/scripts/synth_headings_2000.sh all` でも順次実行できる。
生成中断時は `resume`（両セットを確認し、未着手のセットは新規生成）。
結合はコピー方式で約9,000枚分の追加容量が必要。結合の途中再開は未対応。
既存出力は上書きせず、学習先にcheckpointがあれば学習も停止する。

- 本文＋見出し1,000枚: `outputs/synth-headings-1000-v1`。縦書き25%。
- 表＋見出し1,000枚: `outputs/synth-heading-tables-1000-v1`。横書き、表は上部・下部各半数。
- 結合先: `outputs/synth-mixed-9000-v1`。既存7,000枚を保持する。
- 4フォント、文字サイズ12/16/20/24、段組み・行間・字間、見出し装飾・角丸・文字色を変える。
- h2/h3は後続記事がページ内に現れた場合に入る。長い本文ではh1のみのページも残る。
- 学習: 既存実験と同じ公式事前学習済み重みから5 epoch、入力1024。
  `outputs/db-resnet34-synth9000-v1/synth9000-db-resnet34-v1.pt` が最良valの重み。
- 検証: その重みを1024/1536入力で固定JDocQA valに評価。testは使わない。
  学習ページが増えるため、同じ5 epochでも更新回数は以前より多い。
- 検証の片方だけ実行する場合は `evaluate 1024` / `evaluate 1536`。
  既存評価先は上書きしないため、失敗した出力は内容を確認してから別名に退避する。
- `package` は完了済みの両検証ディレクトリを `outputs/validation-synth9000-v1.tar.gz`
  にまとめる。正解・予測・比較PNG、`summary.json`、`pages.csv`、`predictions.json`を含む。
  学習画像・モデル重みは含めない。

Mac側でダウンロード・展開する（スクリプト自身はSSHやscpを実行しない）。

```bash
scp dgx:~/ogura/ogura/textdet/outputs/validation-synth9000-v1.tar.gz ~/ogura/ogura/textdet/outputs/
tar -xzf ~/ogura/ogura/textdet/outputs/validation-synth9000-v1.tar.gz -C ~/ogura/ogura/textdet/outputs/
```

`download-command` でも上記コマンドを表示する。生成データ・検証画像はGit対象外。
