# Synth-JDocによるtextdetデータの試作

公式Synth-JDocのHTMLテンプレート・段組み・縦中横処理を利用する。
ソースの固定版とライセンスは `vendor/README.md` を参照。
公式の画像生成モデルやLLM処理は呼ばず、手元の文章とフォントからCPU上のChromiumで生成する。
ブラウザ内からのネットワーク要求は遮断する。ブラウザ本体の初回インストールのみダウンロードが必要。

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
フォントはHTMLに埋め込まれるため、生成物の配布時にもフォントのライセンスに従う。

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
- `html/`: 実際に描画したHTML（フォント埋め込みのためサイズは大きい）
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
