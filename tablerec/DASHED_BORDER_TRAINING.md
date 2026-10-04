# 破線の認識を改善する試行

`--dashed-borders` は内側の横・縦罫線に点線/破線を追加する。
横の内部境界を55%、縦を25%の確率で選び、少なくとも横1本を選ぶ。
表ごと・方向ごとに線種を揃える。dash/gapは画素単位で
(0.75,1.5), (1,2), (2,2), (3,3), (6,3), (10,5) から選ぶ。
幅0.5/0.75/1/1.25px、黒〜灰色3色、ランダム位相をJSONに保存する。
外枠は実線。結合セル内部の消えた境界は描かない。
文字密集・色付きセル・余白・文字のみ・白紙を従来のオプションと併用する。
教師は破線の隙間も含めた連続したセル境界。描画時の乱数や画像保存は不要。

## GPU側のデータ準備

```sh
cd ~/ogura
uv run --frozen --inexact python -m ogura.tablerec.synth_table_cells \
  --output tablerec/outputs/table-cells-dashed-v1/train \
  --count 2000 --split train --seed 20261011 \
  --dense-text --background-context --dashed-borders
uv run --frozen --inexact python -m ogura.tablerec.synth_table_cells \
  --output tablerec/outputs/table-cells-dashed-v1/validation \
  --count 200 --split validation --seed 20261011 \
  --dense-text --background-context --dashed-borders
uv run --frozen --inexact python -m ogura.tablerec.mix_table_cells \
  --input tablerec/outputs/table-cells-hard-v1/train \
  --input tablerec/outputs/table-cells-dashed-v1/train \
  --output tablerec/outputs/table-cells-dashed-mixed-v1/train
uv run --frozen --inexact python -m ogura.tablerec.mix_table_cells \
  --input tablerec/outputs/table-cells-hard-v1/validation \
  --input tablerec/outputs/table-cells-dashed-v1/validation \
  --output tablerec/outputs/table-cells-dashed-mixed-v1/validation
```

## 250バッチの試行と比較

今回のdense-pilot-v1から、低学習率で250バッチだけ追加する。

```sh
uv run --frozen --inexact python -m ogura.tablerec.train_table_cnn \
  --train tablerec/outputs/table-cells-dashed-mixed-v1/train \
  --validation tablerec/outputs/table-cells-dashed-mixed-v1/validation \
  --init-checkpoint tablerec/outputs/table-cnn-dense-pilot-v1/best.pt \
  --output tablerec/outputs/table-cnn-dashed-pilot-v1 \
  --device cuda --epochs 1 --batch-size 2 --workers 2 --lr 3e-5 \
  --max-train-batches 250
uv run --frozen --inexact python -m ogura.tablerec.evaluate_table_regions \
  --data tablerec/outputs/table-cells-dashed-v1/validation --count 200 \
  --checkpoint tablerec/outputs/table-cnn-dense-pilot-v1/best.pt \
  --checkpoint tablerec/outputs/table-cnn-dashed-pilot-v1/best.pt \
  --output tablerec/outputs/table-cnn-dashed-comparison-v1.json --device cuda
uv run --frozen --inexact python -m ogura.tablerec.evaluate_table_regions \
  --data tablerec/outputs/table-cells-dense-v1/validation --count 200 \
  --checkpoint tablerec/outputs/table-cnn-dense-pilot-v1/best.pt \
  --checkpoint tablerec/outputs/table-cnn-dashed-pilot-v1/best.pt \
  --output tablerec/outputs/table-cnn-dashed-dense-regression-v1.json --device cuda
uv run --frozen --inexact python -m ogura.tablerec.preview_actual_table_cnn \
  --images tablerec/tests/actual \
  --checkpoint tablerec/outputs/table-cnn-dashed-pilot-v1/best.pt \
  --output tablerec/outputs/table-cnn-dashed-pilot-v1/actual-preview-half \
  --device cuda --scale 0.5
```

## 評価の見方

横/縦別に `solid`（実線）、`dash_ink`（破線の描画部分）、
`dash_gap`（隙間）の平均確率・0.5閾値の再現率を出す。
分類は入力画像の色ではなく、JSONの線幅・周期・位相を使って劣化前の描画マスクから計算する。
教師coverage >= 0.5の画素を対象とし、描画マスクcoverage >= 0.5をink、<= 0.01をgapにする。
中間のアンチエイリアス画素はink/gap評価から除くが、全体boundary評価には含む。
色付きセルやJPEGでの色変化によって分類しない。文字のみ/白紙は罫線評価対象なし。
罫線分類ごとの対象画素数も確認する。

`dash_gap`が上がり、`text`/`white`の誤反応が増えず、`solid`が下がらないか確認する。
同じ文章密集検証200件での評価も併用し、破線特化による後退を確認する。
実画像の `anzen__henkou__page-0108-table-0001.png` も目視する。
この試行の改善は未検証。既存出力先は上書きしない。
