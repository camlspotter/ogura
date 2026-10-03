# 表セル認識用の合成データ（JSON方式）

まず描画パラメータをJSONに保存し、使用時にPillowで表画像と横・縦の罫線マップを
メモリ上に生成する。通常のデータ作成では画像の作成・保存を行わない。
CNN学習・セル復元はまだ実装していない。

## JSONデータの作成

リポジトリルートから:

```sh
uv run --frozen python -m ogura.textdet.synth_table_cells \
  --output textdet/outputs/table-cells-v3/train --count 160 --split train
uv run --frozen python -m ogura.textdet.synth_table_cells \
  --output textdet/outputs/table-cells-v3/validation --count 40 --split validation
```

`--scale 2|4|8` で描画倍率を指定する（既定4）。倍率はJSONに保存される。
`--count 10000` などで大量生成できる。既存の出力ディレクトリは上書きしない。
`--seed` は既定で20261003。同じseedでもsplitが違えば別のサンプルになる。
同じseed・splitを使ったデータセット同士は、先頭から同じサンプルを含むので注意。

```text
train/
  manifest.json           設定・フォント・完了状態・構造別件数
  samples.jsonl           サンプルID・seed・JSONへの相対パス
  recipes/table-*.json    各表の具体的な描画パラメータとセル構造
```

PNGやJPEG、プレビューは自動保存しない。フォントは既存のファイルを参照する。
既定では共有の `corpus/fonts/NotoSansCJKjp-Regular.otf` と
`corpus/fonts/NotoSerifCJKjp-Regular.otf` を使う。
`--font /absolute/path/font.otf` を繰り返して別フォントを指定できる。
文字収録の検査・折り返しとサイズ調整のため、JSON生成時にもフォントが必要。

## メモリ上での描画

```python
import json
from pathlib import Path
from ogura.textdet.synth_table_cells import render_sample

recipe = json.loads(Path(
    'textdet/outputs/table-cells-v3/train/recipes/table-000000.json'
).read_text())
image, horizontal, vertical = render_sample(recipe)
# PIL.Image: RGB入力画像、L横罫線マップ、L縦罫線マップ
# ファイル出力はしない。必要ならこの場でテンソルに変換する。
```

`make_sample(seed, font_path, mode)` がランダムな構造・装飾・劣化と文字レイアウトを
確定してJSON化可能なdictを返す。`render_sample(recipe)` は乱数を使わず、渡された
パラメータから描画する。JSONのseedは生成履歴用であり、描画には使用しない。

JSONには画像サイズ、格子座標、セルの結合範囲とbbox、文字サイズ、背景色・文字色、
各描画行の文字列と位置（`text_runs`）、罫線の色と太さ、画像劣化の具体値を保存する。
フォントは名前・元の絶対パス・SHA-256を保持する。既定では縦横4倍の解像度で描き、縮小する。
新規生成はschema v3 / pillow-table-v3。旧schema v2のJSONは従来の2倍描画で
読み込める。非対応のschema・倍率は明示的に拒否する。

`text` / `lines` / `align` は内容と配置の記録であり、実際の文字描画には確定済みの
`text_runs` を使う。手で文字を変更する場合は `text_runs` と内容ラベルを揃える。
罫線は `cells` と格子座標から復元し、保存されたsegmentsは参照用。
構造を編集する場合はbbox・格子・spanも整合させる。

## 学習時の逐次読み込み

```python
from pathlib import Path
from ogura.textdet.synth_table_cells import TableCellDataset

dataset = TableCellDataset(Path('textdet/outputs/table-cells-v3/train'))
sample = dataset[0]  # この時点で1サンプルだけ描画
image = sample['image']
horizontal = sample['horizontal']
vertical = sample['vertical']
recipe = sample['recipe']
```

`TableCellDataset` は `__len__` / `__getitem__` を持つmap-style dataset。
画像のディスクキャッシュは作らず、JSON一覧だけをメモリに持つ。
PyTorch DataLoaderで使う場合、PIL画像をテンソルにする独自 `collate_fn` 等が必要。
画像サイズは可変なので、バッチ化時は画像と両マップに同じパディング・リサイズを行う。
画像のテンソル化・バッチ化・学習処理は本モジュールの対象外。

他のマシンでは `render_sample(recipe, font_dir=Path('/data/fonts'))` または
`TableCellDataset(root, font_dir=Path('/data/fonts'))` で参照先を変更できる。
保存されたファイル名で検索し、SHA-256で同一フォントか確認する。
同じファイル名で内容が異なるフォントを暗黙に使用しない。
同じフォント・レンダラ・ライブラリ環境での再現を想定し、環境をまたぐ画素一致は保証しない。

## 小数座標・線幅とアンチエイリアス

座標・線幅の単位は最終画像のピクセル。格子座標は1/scale px刻みで変化させ、
内罫線幅は0.5〜3px、外枠は0.75〜4pxの候補から選ぶ（0.75、1.25、1.5px等を含む）。
罫線を中心座標±線幅/2の帯として高解像度で描く。画素中心で帯の内外を判定し、
端点の包含によって線が余分に太くなることを防ぐ。細かさは1/scale pxに量子化される。
隣接セルの共有辺は同じ座標を使用する。

入力画像はLanczosで、教師マップは面積平均（BOX）で縮小する。教師の中間値は
罫線が画素を覆う割合の近似であり、ぼかしで人工的に線を広げた値ではない。
4倍描画の作業画像は2倍描画の4倍の画素数になるため、学習時はworker数にも注意する。
JSON生成時には作業画像は作らない。

## 正解ラベル

両マップはuint8の被覆率で、白が罫線、黒が背景。`value / 255` で教師値にする。
高解像度からBOX縮小するため中間値も含む。交点は両チャンネルが正になる。
画像劣化は入力のみに適用し、マップには劣化前の罫線を残す。3枚は常に同じサイズ。

セルには0始まりの `row / column`、1以上の `rowspan / colspan` を保存する。
`bbox=[left, top, right, bottom]` は入力画像のピクセル座標で、罫線の基準座標に
囲まれた論理領域。線幅を含む外接矩形ではなく、隣り合うセルは辺を共有する。
空セルも保存する。全区画を重複なく長方形のセルで埋め、全行・列の境界が
表のどこかで観測できる構造だけを生成する。

## 現在の生成範囲

- 4〜16行、3〜9列、可変の行高・列幅・余白。
- 結合なし・列結合・行結合・縦横別々の結合・縦横同時の矩形結合を順番に生成。
- 日本語の架空ラベル・英字・数値・割合・空セル、左右中央寄せ、文字サイズ・折り返し。
- 単線罫線の色・太さ、外枠の太さ、見出し背景・交互の行背景。
- 劣化なし、JPEG圧縮、軽いぼけ、縮小後の再拡大。傾きなし。

単線罫線・横書きのみ。二重線、点線、縦書き、斜線、罫線省略は対象外。
語彙も限定された架空テキストで、業務帳票の意味構造は再現しない。
検証splitは同じ合成分布からの独立サンプルなので、実PDF由来の別評価が必要。

## 検証

```sh
uv run --frozen python -m unittest discover -s textdet/tests -p test_synth_table_cells.py
```

セルの被覆・結合内部のマップ・交点・再現性に加え、JSON生成中に画像を作らないこと、
描画時に乱数やディスク出力を使わないこと、JSON往復、フォント移設・内容不一致を検証する。
描画テストにはローカルのNoto Sansが必要。
