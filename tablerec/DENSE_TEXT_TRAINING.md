# 密集した文章への誤反応の評価

`--dense-text` は30%の例を文章量の多い帳票にする。3列（番号・確認項目・結果）、
5〜9行、本文行高65〜110px、文字10〜12px。広いセルに複数行の確認文を配置する。
セル結合は従来の5モードを使う。文字の幅・行送りから収まる行数を計算し、
実際に表示する文と位置をJSONに保存する。画像生成時の乱数や画像キャッシュはない。
`--background-context` と併用して余白・文字のみ・白紙の例も保持する。

GPU側のリポジトリルートで実行する（既存出力先の上書きは不可）。

```sh
uv run --frozen --inexact python -m ogura.tablerec.synth_table_cells \
  --output tablerec/outputs/table-cells-dense-v1/train \
  --count 2000 --split train --seed 20261010 --dense-text --background-context
uv run --frozen --inexact python -m ogura.tablerec.synth_table_cells \
  --output tablerec/outputs/table-cells-dense-v1/validation \
  --count 200 --split validation --seed 20261010 --dense-text --background-context
uv run --frozen --inexact python -m ogura.tablerec.mix_table_cells \
  --input tablerec/outputs/table-cells-hard-v1/train \
  --input tablerec/outputs/table-cells-dense-v1/train \
  --output tablerec/outputs/table-cells-dense-mixed-v1/train
uv run --frozen --inexact python -m ogura.tablerec.mix_table_cells \
  --input tablerec/outputs/table-cells-hard-v1/validation \
  --input tablerec/outputs/table-cells-dense-v1/validation \
  --output tablerec/outputs/table-cells-dense-mixed-v1/validation
```

まず同一の検証JSONで既存モデルを比較する。全200例を指定する。

```sh
uv run --frozen --inexact python -m ogura.tablerec.evaluate_table_regions \
  --data tablerec/outputs/table-cells-dense-v1/validation --count 200 \
  --checkpoint tablerec/outputs/table-cnn-hard-v1/best.pt \
  --checkpoint tablerec/outputs/table-cnn-recovery-pilot-v1/best.pt \
  --checkpoint tablerec/outputs/table-cnn-recovery-v1/best.pt \
  --output tablerec/outputs/table-cnn-dense-comparison-v1.json --device cuda
```

文字領域は各行のフォントbboxに1pxの余裕を加えた矩形（字形そのものではない）。
教師罫線の周囲4pxを除外する。白領域は文字領域と罫線周囲を除いたRGB各値250以上の画素。
横・縦別に文字/白領域の平均確率と0.5以上の画素率、正解罫線の0.5以上の再現率を出す。
集計は画素数で重み付けし、各例の値も保存する。小さい誤反応は平均確率で確認する。
正解のない実画像の誤検出率を測るものではない。実画像previewの目視確認も必要。

追加学習は比較を揃えるためhard-v1から250バッチに制限して始める。

```sh
uv run --frozen --inexact python -m ogura.tablerec.train_table_cnn \
  --train tablerec/outputs/table-cells-dense-mixed-v1/train \
  --validation tablerec/outputs/table-cells-dense-mixed-v1/validation \
  --init-checkpoint tablerec/outputs/table-cnn-hard-v1/best.pt \
  --output tablerec/outputs/table-cnn-dense-pilot-v1 \
  --device cuda --epochs 1 --batch-size 2 --workers 2 --lr 3e-5 --max-train-batches 250
uv run --frozen --inexact python -m ogura.tablerec.preview_actual_table_cnn \
  --images tablerec/tests/actual \
  --checkpoint tablerec/outputs/table-cnn-dense-pilot-v1/best.pt \
  --output tablerec/outputs/table-cnn-dense-pilot-v1/actual-preview-half \
  --device cuda --scale 0.5
```

新モデルも同じ検証JSONで領域評価し、文字誤反応・余白誤反応・罫線再現率の全てを比較する。
改善は未検証。GPU学習はユーザーが実行する。
