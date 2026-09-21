# PDFからtextdetの正解候補を作る

一次フィルタリングをPDF単位で行い、通過したPDFの**全ページ**について、原画像・行bbox付き画像・文字座標を保存するバッチツールです。Webアプリや二次選別UIはありません。元PDFと既存のtextrec環境は変更しません。

## セットアップ・実行

リポジトリのルートで実行します。Popplerの `pdffonts` が必要です（macOSでは `brew install poppler`）。

```sh
UV_CACHE_DIR=ogura/textdet/.cache/uv uv venv ogura/textdet/.venv
UV_CACHE_DIR=ogura/textdet/.cache/uv uv pip install --python ogura/textdet/.venv/bin/python -r ogura/textdet/requirements.txt
ogura/textdet/.venv/bin/python -m ogura.textdet.prepare \
  --source ~/mocrdown/tests/data \
  --min-chars 100 \
  --dpi 150
```

既定の出力先は `ogura/textdet/outputs/YYYYMMDD-HHMMSS/`。`--output ogura/textdet/outputs/名前` でも指定できます。混在を防ぐため、既存の出力ディレクトリは上書きしません。入力は指定ディレクトリ直下の `*.pdf` です。出力・仮想環境・キャッシュはすべてtextdet配下に置き、Git管理外にします。

## 一次フィルタリング

**画像・bboxを一切作らず一次フィルタだけを実行する場合**:

```sh
ogura/textdet/.venv/bin/python -m ogura.textdet.filter_only \
  --source ~/ogura/JDocQA_pdf_files \
  --min-chars 100 --workers 2
```

出力は `outputs/filter-日時/summary.json`（件数）、`documents.jsonl`（各PDFの判定）、`passed.json`（通過一覧）です。フォント条件で落ちたPDFは文字抽出を省略します。そのためスキップ理由は全原因を網羅せず、文字数は未検査の場合があります。判定条件は画像生成時と同じです。

以下のいずれかに該当すると、PDF**全体**をスキップします。ページ単位の自動採否は行いません。

- `pdffonts` の `uni` が `no` のフォントが一つ以上ある。
- 抽出文字に U+FFFD、私用文字、NULなどの制御文字がある。
- 全ページ合計の空白を除いた抽出文字数が `--min-chars` 未満（既定100）。
- フォントが一つもない。

`uni` は明示的なToUnicode対応表の有無です。標準エンコーディング等で正常に読めるフォントでも、対応表がなければ除外する**保守的な判定**です。逆に対応表があっても誤った変換や未抽出文字の不存在は保証できません。画像・アウトライン化された文字はこの判定では検出できないため、二次の目視確認が必要です。

文字数はPDF全体の合計です。`prepare` の全ページ出力では、通過PDF内に文字のないページがあっても出力します。`sample_pages` では下記のとおり文字数0のページを候補から除きます。フォントや文字の検査に失敗したPDFは `error` として記録し、他のPDFを処理した後、終了コード1を返します。通常のスキップのみなら終了コード0です。非UTF-8のフォント名のバイトはログでエスケープ表示します。

## 出力

一次フィルタ通過一覧から、抽出文字数が0のページを除き、残ったページの約1/3・2/3の位置を選んで画像化する場合:

```sh
ogura/textdet/.venv/bin/python -m ogura.textdet.sample_pages \
  --list ogura/textdet/outputs/jdocqa-filter-only/passed.json \
  --output ogura/textdet/outputs/jdocqa-latest
```

保存済みの `documents.jsonl`（既定では一覧と同じディレクトリ）からページ別文字数を読み、一次フィルタは再実行しません。文字情報のあるページが1ページだけなら、その1ページだけを出力します。ファイル名は `元PDF名_page-0001_bbox.png`、原画像は同じ接頭辞の `_page.png`、座標は `_labels.json` です。ページ番号はPDF内の1始まりの位置で、紙面に印字されたページ番号ではありません。`manifest.json` に選択ページと出力ファイルを記録します。

```text
outputs/実行名/
├── manifest.json                   全PDFの採否・理由・フォント・文字数・SHA-256
└── public_document00104/
    └── page-0001/
        ├── page.png                原ページ画像
        ├── bbox.png                行bboxと照合用IDを重ねた画像
        └── labels.json             行・文字の座標、文字列、フォント、書字方向候補
```

bboxとIDは、原画像のRGB値をXOR反転（各チャンネルを255から減算）して描画します。青い背景では黄色系、白では黒、黒では白になります。枠が交差する箇所も一度だけ反転します。縦横は `labels.json` に保持します。IDはJSONの行と対応し、読み順ではありません。読み順の推定やOCRは行いません。

行のグループ分けはPyMuPDFのPDF文字抽出結果を使います。ただし、同一PDFブロック内で連続する縦書き断片（`wmode=1`, `dir=(0,1)`）は、文字原点が同じ列にあり、サイズ比が1.25以内、縦方向の間隔が小さい方の文字サイズの1.6倍以内なら結合します。列のずれの許容値は小さい方の文字サイズの0.2倍です。別ブロック・別列・大きな間隔・サイズ差をまたいでは結合しません。縦中横の数字は独立したbboxとして残ることがあります。結合元は `source_line_ids` に記録します。本文の結合・分断や表の別セルの混在は、bbox画像を見て確認してください。この段階のラベルは `candidate_needs_visual_review` であり、確定した正解ではありません。

- `bbox`: `[xmin, ymin, xmax, ymax]`。描画されたCropBoxとページ回転を反映した左上原点、単位PDF point。
- `polygon`: 行の四隅。回転前の左上・右上・右下・左下を表示座標へ変換した順。
- `bbox_pixels`, `polygon_pixels`: 同時出力の画像に対応するピクセル座標。
- `chars`: 各文字の文字列・フォント名・サイズ・四隅座標。
- `orientation`: PDFの `wmode` による候補。縦書きを横書き文字の配置で実現したPDFでは誤る場合があります。画像の傾きとは別です。

## 検証

```sh
ogura/textdet/.venv/bin/python -m unittest discover -s ogura/textdet/tests
```

PyMuPDFはAGPL／商用ライセンスです。元PDFの権利は別です。

縦書きの座標補正: 埋め込みIdentity-Vフォントを一意に特定でき、描画履歴の字形ID・原点と照合できる直立文字および90度単位で回転した文字は、フォントのアウトラインからbboxを算出します。PyMuPDFの四隅復元による過大なbboxと、縦書きrawdictの位置ずれを避けるためです。照合できない文字や任意角度の回転文字は従来の座標を残します。各文字の `geometry_source` で `glyph_outline` / `pdf_font_metrics` を区別できます。装飾の影や縁取りは字形アウトラインに含めません。

回転した縦長の1文字（縦向きの「－」など）は、字形から座標が取れ、直下の縦書き列の先頭文字と中心のずれが文字サイズの0.25倍以内、間隔が0.6倍以内、サイズ比が1.25以内の場合、PDFブロックをまたいで結合します。候補が複数ある場合や、複数の接頭文字が同じ列を指す場合は結合しません。ページ回転を戻した座標で判定します。

通常の行抽出、縦書き断片の結合、回転文字の結合の後に、未結合の1文字断片だけを対象に配置から縦書きを推定します。同じPDFブロック内で同じフォント・近いサイズの文字が同じ列に3文字以上連続する場合に結合します。既存の複数文字の行は再解釈しません。推定結果は `orientation=vertical` と `orientation_source=isolated_character_positions` に記録し、`wmode` と `baseline` はPDFの元の設定を保持します。

行bbox・polygonは先頭と末尾の空白文字（半角・全角・タブ等）を除いた文字範囲から作成します。`text` と `chars` は元の空白を保持し、行の途中にある空白も保持します。空白だけの行は出力しません。回転した行は文字方向に沿って枠を計算します。
