# 背景への誤反応を抑える追加学習

今回のnatural-v2は、一部の実画像の白い余白で罫線確率が約0.4になった。
良かったhard-v1からやり直し、旧データを保持しながら背景の条件を増やす。
学習・GPU側のコマンドはユーザーが実行する。

## GPU側でのデータ準備

リポジトリルート `~/ogura` で最新コードを取得する。
旧データは前回使った `table-cells-hard-v1` をそのまま使う。
以下では同じ件数の旧・新JSONを結合し、学習4,000件・検証400件にする。

```sh
uv run --frozen --inexact python -m ogura.tablerec.synth_table_cells \
  --output tablerec/outputs/table-cells-context-v1/train \
  --count 2000 --split train --seed 20261009 --background-context

uv run --frozen --inexact python -m ogura.tablerec.synth_table_cells \
  --output tablerec/outputs/table-cells-context-v1/validation \
  --count 200 --split validation --seed 20261009 --background-context

uv run --frozen --inexact python -m ogura.tablerec.mix_table_cells \
  --input tablerec/outputs/table-cells-hard-v1/train \
  --input tablerec/outputs/table-cells-context-v1/train \
  --output tablerec/outputs/table-cells-recovery-v1/train

uv run --frozen --inexact python -m ogura.tablerec.mix_table_cells \
  --input tablerec/outputs/table-cells-hard-v1/validation \
  --input tablerec/outputs/table-cells-context-v1/validation \
  --output tablerec/outputs/table-cells-recovery-v1/validation
```

新しいJSONはschema v9。`background_context` は表単位で次の確率で選ぶ。

- `table` 40%: 自然な帳票例を従来のサイズで描く。
- `page_table` 40%: 幅に0〜1/3、高さに1/3〜1倍の余白を追加し、表をページ内に配置する。
  表の下に広い余白が残る。全座標を移動して保存し、正解も同じ位置へ移動する。
- `text_only` 10%: 文字・記号を描くが、表の背景色と罫線を描かない。正解は両チャネルともゼロ。
- `blank` 10%: 白紙。正解は両チャネルともゼロ。

種別と位置はJSON生成時に確定する。描画時の乱数や学習画像の保存は使わない。
既存JSONは変更しない。混合処理はレシピをバイト列のままコピーし、IDに入力元の接頭辞を付ける。
入力件数とsplitの一致を要求し、重複seedは拒否する。
混合比は確率ではなく件数による正確な1:1。ミニバッチ内の比率は固定しない。

## まず短い試行

初期重みは `table-cnn-hard-v1/best.pt`。optimizerは新規、学習率は `3e-5` に下げる。
まず250バッチだけ学習して検証・保存する。これは今回の試行の上限であり、
自動的に本学習へ移行するものではない。

```sh
uv run --frozen --inexact python -m ogura.tablerec.train_table_cnn \
  --train tablerec/outputs/table-cells-recovery-v1/train \
  --validation tablerec/outputs/table-cells-recovery-v1/validation \
  --init-checkpoint tablerec/outputs/table-cnn-hard-v1/best.pt \
  --output tablerec/outputs/table-cnn-recovery-pilot-v1 \
  --device cuda --epochs 1 --batch-size 2 --workers 2 --lr 3e-5 \
  --max-train-batches 250

uv run --frozen --inexact python -m ogura.tablerec.preview_actual_table_cnn \
  --images tablerec/tests/actual \
  --checkpoint tablerec/outputs/table-cnn-recovery-pilot-v1/best.pt \
  --output tablerec/outputs/table-cnn-recovery-pilot-v1/actual-preview-half \
  --device cuda --scale 0.5
```

手元のマシンでコピーする。

```sh
rsync -av dgx:~/ogura/tablerec/outputs/table-cnn-recovery-pilot-v1/actual-preview-half/ \
  ~/ogura/tablerec/outputs/table-cnn-recovery-pilot-v1/actual-preview-half/
```

`anzen__tougou_siryou__page-0122-table-0002.png` の下部余白に広いピンク領域が出ないこと、
文字の誤検出が悪化しないこと、本物の罫線を失わないことを確認する。
正解のない実画像なので、ここで合成データの検証lossだけを改善判定に使わない。

## 試行が良ければ1エポック

比較を揃えるため、短い試行のモデルを引き継がず、同じhard-v1から始める。

```sh
uv run --frozen --inexact python -m ogura.tablerec.train_table_cnn \
  --train tablerec/outputs/table-cells-recovery-v1/train \
  --validation tablerec/outputs/table-cells-recovery-v1/validation \
  --init-checkpoint tablerec/outputs/table-cnn-hard-v1/best.pt \
  --output tablerec/outputs/table-cnn-recovery-v1 \
  --device cuda --epochs 1 --batch-size 2 --workers 2 --lr 3e-5
```

本学習後も別の出力先へpreviewを作って比較する。改善は実行後に確認する。
既存の出力先は上書きしない。データを既に生成済みなら生成・混合のコマンドは省略する。
