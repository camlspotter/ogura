# 表罫線CNN

JSONから必要時に画像と正解マップを描画し、小型U-Netで横罫線・縦罫線を推定する。
表領域の検出、OCR、罫線からのセル復元は含まない。

## モデル

- RGB入力 `[N,3,H,W]`、出力logits `[N,2,H,W]`。順番は横・縦。
- 幅は32→64→128→256→256。4段階で1/16まで縮小する。
- 各段は3×3畳み込み・GroupNorm・ReLUを2回。
- 拡大はbilinear補間、同じ解像度のencoder特徴を結合。
- 最後は1×1畳み込みで2チャンネル。推論で独立したsigmoidを適用。
- 外部の事前学習重み・ダウンロード不要。`--base-channels` で規模を変更可能。

画像を縮小せず、右・下に白い余白を足して16の倍数にする。
近い縦横比・面積のサンプルをまとめ、バッチ内の余分なパディングを抑える。
出力のサイズは入力と一致する。バッチの余白は損失・評価から除外する。
GroupNormや畳み込みの特徴はパディングの影響を受けるため、異なるバッチ形状での
予測が完全一致する保証はない。

## 学習

既存の環境を使用し、リポジトリルートから実行する。
最初の本学習用として学習2,000件・検証200件のJSONを作る。
生成済みJSONを転送してもよいが、GPU側で生成すれば絶対フォントパスも現地で解決できる。

```sh
uv run --frozen python -m ogura.tablerec.synth_table_cells \
  --output tablerec/outputs/table-cells-train-v1/train \
  --count 2000 --split train --scale 4
uv run --frozen python -m ogura.tablerec.synth_table_cells \
  --output tablerec/outputs/table-cells-train-v1/validation \
  --count 200 --split validation --scale 4
```

これらのJSONとモデルの実装が揃ったGPUマシンで、次を実行する。

```sh
uv run --frozen python -m ogura.tablerec.train_table_cnn \
  --train tablerec/outputs/table-cells-train-v1/train \
  --validation tablerec/outputs/table-cells-train-v1/validation \
  --output tablerec/outputs/table-cnn-v1 \
  --device cuda --epochs 20 --batch-size 2 --workers 2
```

`--device` は `cpu / cuda / mps / auto`。既定autoはCUDA→MPS→CPUの順。
これは実行マシンの選択であり、他マシンへの接続・学習投入は行わない。
既存の出力ディレクトリには上書きしない。
初期データ160/40件は動作確認用。2,000/200件は最初の学習設定であり、
必要なデータ量や精度が十分かどうかは、学習曲線と実画像で判断する。

教師はuint8マップを255で割り、アンチエイリアスによる中間値を保持する。
損失は画素BCEWithLogits + Soft Dice。サンプル・チャンネルごとに計算して平均する。
AdamW、既定学習率3e-4、weight decay 1e-4、勾配ノルム上限5。
現在はfloat32のみ。学習再開、AMP、学習率スケジューラは未実装。

`--workers` の既定は0。並列化する場合はspawnで描画workerを起動する。
高解像度描画は一時メモリを使うため、worker数とバッチサイズはメモリに合わせる。
`--font-dir /data/fonts` で別マシンの同一フォントに切り替えられる。
画像ファイル・予測画像は自動保存しない。

```text
出力ディレクトリ/
  config.json      実行設定、パラメータ数、データJSONのハッシュ
  metrics.jsonl    epochごとの学習・検証損失とチャンネル別評価
  best.pt          検証損失が最小のモデル
  last.pt          最終epochのモデル
```

両checkpointにはモデル設定、重み、optimizer状態、epoch、実行設定と指標を保存。
学習・検証に同じrecipe seedが含まれた場合は拒否する。
独立seedは未知の実帳票への汎化を保証しない。

## 評価

横・縦別に被覆率のMAE、soft Dice、閾値0.5のprecision/recall/F1を出す。
指標はサンプルごとに計算して平均する。
0.5未満の薄い正解罫線は二値指標に現れないことがあるので、F1だけでは判断しない。
罫線の途切れや結合セル内部の誤検出、セル構造の精度は今後の後処理評価で確認する。

## 推論（保存せず利用）

```python
import json
from pathlib import Path
from ogura.tablerec.synth_table_cells import render_sample
from ogura.tablerec.table_cnn import load_model, predict_image

recipe = json.loads(Path(
    'tablerec/outputs/table-cells-v3/validation/recipes/table-000000.json'
).read_text())
image, _, _ = render_sample(recipe)
model = load_model(Path('tablerec/outputs/table-cnn-v1/best.pt'), device='cpu')
probabilities = predict_image(model, image, device='cpu')
horizontal, vertical = probabilities  # CPU tensor [H,W], 0〜1
```

実画像なら `image` にPILの表画像を渡す。モデルと入力のdeviceは揃える。

## 短い動作確認

```sh
uv run --frozen python -m ogura.tablerec.train_table_cnn \
  --train tablerec/outputs/table-cells-v3/train \
  --validation tablerec/outputs/table-cells-v3/validation \
  --output tablerec/outputs/table-cnn-smoke \
  --device cpu --epochs 1 --batch-size 1 --base-channels 4 \
  --train-limit 2 --validation-limit 1 --threads 2
uv run --frozen python -m unittest discover -s tablerec/tests -p 'test_table_cnn.py'
```

`--train-limit / --validation-limit` は使用する先頭サンプル数を制限する。
`--max-train-batches / --max-validation-batches` でも1epochのバッチ数を制限できる。
制限付きの実行結果は動作確認であり、本学習の評価ではない。

## 予測の可視化（20件）

GPU側で学習済み `best.pt` を使い、検証データの少数例だけを書き出す。
リポジトリルートから:

```sh
uv run --frozen --inexact python -m ogura.tablerec.preview_table_cnn \
  --dataset tablerec/outputs/table-cells-train-v1/validation \
  --checkpoint tablerec/outputs/table-cnn-gpu-pilot-v1/best.pt \
  --output tablerec/outputs/table-cnn-gpu-pilot-v1/preview \
  --device cuda --count 20
```

`preview/index.html` をブラウザで開く。GPUマシンにブラウザがなければ
`preview/` ディレクトリ全体をローカルへコピーする。外部リソースは不要。
出力先が既存なら上書きせず終了する。`--font-dir` でフォントの参照先を変更できる。

各比較PNGは横並び2枚で、元の画素数を維持する。HTML上の縮小表示で細線が見えない
場合は画像をクリックして拡大する。

| 左 | 右 |
|---|---|
| 入力画像 | 位置をずらした予測の重ね合わせ |

重ね合わせは横が赤、縦が青。正解マップ・単独の確率マップ・誤差画像は表示しない。正解は指標の計算に使用する。右の元画像は白背景にalpha=0.3で合成し、予測線は薄くしない。
元の罫線を見せるため、横の予測を下へ3px、縦の予測を右へ3pxずらす。
`--overlay-offset 5` などで変更可能（0でずらさない）。原寸ピクセル単位で、
画像外にはみ出す分は切り捨てる。指標はずらさず計算する。
予測線の色の濃さは確率に対応し、二値化せず0〜1の中間値を使う。
`summary.json` に選択サンプル、線幅、画像劣化、MAE/F1、使用checkpointのSHA-256を保存する。

選択は5種類の表構造と細線（内罫線または外枠が1px未満）/通常線の組み合わせを
順に取り出す。該当例が十分なら20件で各組2件ずつ。seed固定で再現できる。
この選択例の指標を検証全体の平均として扱わない。

本コマンドは1画像ずつ推論する。GroupNormが余白の影響を受けるため、学習時の
検証バッチとパディング量が違う場合、指標が完全に一致するとは限らない。
現在のSoft Diceは分母が予測と教師の和であり、教師に中間値があると完全一致でも
1にならない。そのため、ここではMAEと閾値0.5のF1を併記する。
可視化以外の通常のデータ生成・学習では、引き続き画像を保存しない。
