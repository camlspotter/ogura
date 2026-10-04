# 表セル認識用の合成データ（JSON方式）

まず描画パラメータをJSONに保存し、使用時にPillowで表画像と横・縦の罫線マップを
メモリ上に生成する。通常のデータ作成では画像の作成・保存を行わない。
CNN学習は [TABLE_CNN.md](TABLE_CNN.md) を参照。セル復元はまだ実装していない。

## JSONデータの作成

リポジトリルートから:

```sh
uv run --frozen python -m ogura.tablerec.synth_table_cells \
  --output tablerec/outputs/table-cells-v3/train --count 160 --split train
uv run --frozen python -m ogura.tablerec.synth_table_cells \
  --output tablerec/outputs/table-cells-v3/validation --count 40 --split validation
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
from ogura.tablerec.synth_table_cells import render_sample

recipe = json.loads(Path(
    'tablerec/outputs/table-cells-v3/train/recipes/table-000000.json'
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
新規生成はschema v8 / pillow-table-v8。旧schema v2/v3/v4/v5/v6/v7のJSONも従来の描画で
読み込める。非対応のschema・倍率は明示的に拒否する。

`text` / `lines` / `align` は内容と配置の記録であり、実際の文字描画には確定済みの
`text_runs` を使う。手で文字を変更する場合は `text_runs` と内容ラベルを揃える。
罫線は `cells` と格子座標から復元し、保存されたsegmentsは参照用。
構造を編集する場合はbbox・格子・spanも整合させる。

## 学習時の逐次読み込み

```python
from pathlib import Path
from ogura.tablerec.synth_table_cells import TableCellDataset

dataset = TableCellDataset(Path('tablerec/outputs/table-cells-v3/train'))
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
uv run --frozen python -m unittest discover -s tablerec/tests -p test_synth_table_cells.py
```

セルの被覆・結合内部のマップ・交点・再現性に加え、JSON生成中に画像を作らないこと、
描画時に乱数やディスク出力を使わないこと、JSON往復、フォント移設・内容不一致を検証する。
描画テストにはローカルのNoto Sansが必要。

## 誤検出しやすいセル内容（v4）

v4のJSONは `content_profile: hard-negatives-v1` を持つ。v5も各セルの内容を次の確率で選ぶ。
通常の文章・数値・空欄を45%、直線や四角形の多い文字を35%、図形のチェックボックスを10%、
空セルの斜線を10%とする。割合はサンプル全体での期待値で、各表で固定しない。

文字候補には `上 1 I ー L U R 月 日 回`、カタカナ、`年　月　日`、`第1回`、
`□ 有　□ 無`、`△123` / `▲123` のような負数表記を含む。
文字サイズや配置は既存の文字レイアウトに従う。
図形のチェックボックスは9〜20pxで、空・チェック入り・塗りつぶしを混ぜる。
空欄の斜線は `/`・`\`・`×` を生成し、角に接する場合と内側に余白を持つ場合を混ぜる。

各セルの `content_kind` と `marks` に種類・座標・線幅・色・状態を保存する。
これらは入力画像だけに描き、縦横罫線の正解マスクには含めない。セル外周は従来通り正解とする。
描画時の乱数は使わず、画像は引き続き学習時にメモリ上で生成する。
既存のJSONは変わらないため、この内容で学習するには新しい出力先へJSONを生成する。

```sh
uv run --frozen --inexact python -m ogura.tablerec.synth_table_cells \
  --output tablerec/outputs/table-cells-hard-v1/train --count 2000 --split train --seed 20261004
uv run --frozen --inexact python -m ogura.tablerec.synth_table_cells \
  --output tablerec/outputs/table-cells-hard-v1/validation --count 200 --split validation --seed 20261004
```


## 破線の罫線（v5）

`し`・`一`・`ユーザーリスト`・`ユーザー`・`リスト`・`ユーザー名` も
誤検出しやすい文字候補に追加した。
v5データの `content_profile` は `hard-negatives-dashed-v2`。
約半数の表で内罫線の一部を破線にする。横境界は60%、縦境界は25%の確率で選び、
外枠は実線に保つ。実線だけの表も残す。
破線の線分長は1〜8px、隙間は1〜5pxから選び、同じ境界上では位相を揃える。
`line_styles` に方向・境界座標ごとの線分長、隙間、位相を記録する。
入力だけを破線にし、正解マスクは隙間を埋めた連続したセル境界とする。
結合セル内の存在しない境界は埋めない。旧JSONは従来通り描画する。

```sh
uv run --frozen --inexact python -m ogura.tablerec.synth_table_cells \
  --output tablerec/outputs/table-cells-dashed-v1/train --count 2000 --split train --seed 20261005
uv run --frozen --inexact python -m ogura.tablerec.synth_table_cells \
  --output tablerec/outputs/table-cells-dashed-v1/validation --count 200 --split validation --seed 20261005
```

既存のJSONは更新されない。破線を学習するにはこの新データを生成し、
`table-cnn-hard-v1/best.pt` を `--init-checkpoint` に指定して追加学習する。


## 実画像で判明した条件の生成ルール（v6）

v6の `content_profile` は `real-table-cases-v3`。以下の乱数選択をJSON生成時に
確定し、描画時には乱数を使わない。v2〜v5は既存JSONの従来の描画を保つ。

| 項目 | 生成アルゴリズム |
|---|---|
| 通常セル | 内容選択の35%。従来の文章・数値・空欄 |
| 紛らわしい文字 | 29%。し、一、ユーザーリスト、直線の多い英字・カタカナ、月・日・回、三角記号付き数値など |
| 空欄記号 | 8%。ー・―・−・－・—を中央配置。文字サイズ10〜28pxを選び、セル内へ縮小して収める |
| チェックボックス | 10%。空・チェック入り・塗りつぶし。狭いセルでは枠内に収まるサイズへ制限 |
| 空欄斜線 | 10%。/・逆斜線・×。セルの角へ接する例も含む |
| 矢印 | 8%。セル内の長い片矢印・両矢印。約25%は斜め。先端3〜9px以下、線幅0.5〜2px |
| 狭い横長セル | 表の40%を12〜28行にし、各行の80%を高さ14〜30px、残りを42〜80px。1列を幅250〜450pxにする |
| セル背景 | 従来・行単位・列単位・セル単位を等確率で選択。ピンク・橙・水色・緑・黄と白を混ぜる |
| 背景だけの境目 | 各セル10%でセル左半分の背景色を変更。セル構造や罫線マスクは変更しない |
| 罫線の色・太さ | 各内境界30%で黒・赤・青・灰と線幅0.5〜3pxを選択。同じ境界内では統一 |
| 破線 | v5と同じ。約半数の表で横・縦の内境界の一部を破線にする |
| 丸角 | 表の30%で外枠を丸角にする。半径3〜12pxを選び、最外側のセル寸法の1/3以下へ制限 |

確率は期待値で、各画像に全条件が必ず入るわけではない。実線・通常の行高・白背景の例も残す。
狭いセルでは余白と文字サイズを調整して、文字をセル内に収める。

正解は色に依存しない縦横2チャネルのセル境界。境界ごとの線幅は正解にも反映する。
破線の隙間と丸角の欠けた四隅は、連続した矩形のセル境界として補完する。
文字・空欄記号・矢印・チェックボックス・斜線・背景の色の境目は正解に含めない。
結合セル内に存在しない境界を作ることはしない。

```sh
uv run --frozen --inexact python -m ogura.tablerec.synth_table_cells \
  --output tablerec/outputs/table-cells-real-cases-v1/train --count 2000 --split train --seed 20261006
uv run --frozen --inexact python -m ogura.tablerec.synth_table_cells \
  --output tablerec/outputs/table-cells-real-cases-v1/validation --count 200 --split validation --seed 20261006
```

学習時は上記データを使い、初期重みに `table-cnn-hard-v1/best.pt` を指定する。
実画像での改善は学習後に確認する。追加した条件の存在だけでは改善は保証されない。


## 帳票の用途に沿った配置（v7）

v7の生成プロファイルは `natural-forms-v4`。セルごとの無関係な組み合わせを抑えるため、
まず表単位でテンプレートを選び、列の役割に応じてセル内容を配置する。

| 型 | 選択確率 | 配置 |
|---|---:|---|
| チェック表 | 40% | 番号・広い説明・はい・いいえの4列。回答列のみ着色し、チェックボックスを中央配置 |
| 帳簿 | 35% | 科目・前年度・当年度・増減の4列。数値は右揃え、空欄記号は中央。見出し・小計の位置・最終行を着色 |
| 記入用紙 | 20% | 項目と記入欄の組を2〜3組。空欄を中心に日付・区分チェック・少数の斜線や案内矢印を配置 |
| 極端な組み合わせ | 5% | v6のセル単位ランダム配置を補助例として残す |

チェック表と帳簿は10〜22行。狭い行は14〜28px、見出しは28〜40px。
チェック表の説明列は300〜450px、回答列は55〜75px。
帳簿の科目列は200〜300px、数値列は100〜145px。
狭いセルは余白を1pxにし、文字サイズを収まるまで調整する。
記入用紙は6〜12行、高さ35〜65pxを基本とし、70%で外枠を丸角にする。

通常の3型は罫線の色と太さを表内で統一する。
チェック表・記入用紙の50%で一部の横境界を同一パターンの破線にする。
帳簿の30%で数値列の区切り1本だけを赤にする。
文字・記号の正解除外、破線・丸角の矩形境界への補完はv6と同じ。
結合セルの構造選択は従来の5モードを維持する。

```sh
uv run --frozen --inexact python -m ogura.tablerec.synth_table_cells \
  --output tablerec/outputs/table-cells-natural-v1/train --count 2000 --split train --seed 20261007
uv run --frozen --inexact python -m ogura.tablerec.synth_table_cells \
  --output tablerec/outputs/table-cells-natural-v1/validation --count 200 --split validation --seed 20261007
```

既存の `table-cells-real-cases-v1` はv6の極端な配置のデータであり、この変更では更新しない。
新しい型を学習する場合は `table-cells-natural-v1` を生成して使う。


## 帳票例の体裁と内容の整合（v8）

最新プロファイルは `natural-forms-v5`。型の選択確率と入力・正解の関係はv7と同じ。
通常の帳票では本文の行高を表内で揃え、見出し・小計・合計を少し太い文字で描く。
チェック表・帳簿の標準行高は20〜28px。25%では密な行の例として14〜18pxを選ぶ。
文字サイズも行高に合わせる。記入用紙は38〜55pxの行高を表内で揃える。

チェック表の説明項目は重複なく選び、チェックボックスのサイズ・線幅も揃える。
回答は行単位で「はい」50%、「いいえ」40%、未選択10%から選ぶ。
帳簿は明細の前年度・当年度の値を生成し、増減を当年度−前年度として計算する。
小計は直前の明細群、合計は全明細を集計する。ゼロは中央の空欄記号、負の増減は△付きで表示する。
値の記録は `ledger_records` に保存する。構造上結合されたセルでは表示できる項目のみ表示する。
記入用紙の項目は氏名・所属・住所・連絡先・日付などを順に配置し、繰り返しを減らす。
項目欄は薄い灰色、入力欄は白を基本とする。日付欄と区分・確認・承認のチェック欄を役割に応じて配置する。

`text_stroke_width` で見出し等の文字の太さをJSONに保存する。
旧JSONの描画は変更しない。これまで作成したv6データは最新の生成規則を含まない。

```sh
uv run --frozen --inexact python -m ogura.tablerec.synth_table_cells \
  --output tablerec/outputs/table-cells-natural-v2/train --count 2000 --split train --seed 20261008
uv run --frozen --inexact python -m ogura.tablerec.synth_table_cells \
  --output tablerec/outputs/table-cells-natural-v2/validation --count 200 --split validation --seed 20261008
```

## 背景・余白の条件（オプション、v9）

`--background-context` を指定すると、schema v9として広い余白付きの表・文字だけ・白紙を
混ぜる。種別と全座標をJSONに保存する。文字だけ・白紙の正解はゼロ。
指定しない場合は従来通りschema v8。旧JSONの描画も維持する。
確率と旧データとの混合方法は [RECOVERY_TRAINING.md](RECOVERY_TRAINING.md) を参照。
