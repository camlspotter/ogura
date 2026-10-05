# 文字の横線への誤反応を抑える試行

初期モデル: `table-cnn-dashed-pilot-v1/best.pt`。
学習率1e-5、250バッチ、batch size 2、追加文字ペナルティ係数1。
`--text-penalty-weight` の既定値0では従来の損失を維持する。
追加損失は文字領域をゼロ教師とするBCE（softplus(logits)）で、表・方向別に
対象画素数で正規化する。文字のない表は追加損失ゼロ。paddingは除く。
文字領域は評価と共通の各行bbox+1pxの矩形。正解罫線の周囲4pxを除外する。
字形そのものを厳密に切り抜く方式ではない。

生成オプション `--negative-heavy --background-context` で、
表40%、余白付き表20%、文字のみ30%、白紙10%にする（従来40/40/10/10）。
`--dense-text --dashed-borders`も併用し、従来の文字・記号・破線条件を保持する。
新規2000/200件（seed 20261012）をhard-v1の2000/200件と1:1混合する。
生成・描画は従来どおりJSONのみ保存、on-the-fly画像生成。

実行例（データ準備後、GPU側）：

```sh
uv run --frozen --inexact python -m ogura.tablerec.train_table_cnn \
  --train tablerec/outputs/table-cells-textguard-mixed-v1/train \
  --validation tablerec/outputs/table-cells-textguard-mixed-v1/validation \
  --init-checkpoint tablerec/outputs/table-cnn-dashed-pilot-v1/best.pt \
  --output tablerec/outputs/table-cnn-textguard-pilot-v1 \
  --device cuda --epochs 1 --batch-size 2 --workers 2 --lr 1e-5 \
  --text-penalty-weight 1 --max-train-batches 250
```

既存のdashed-v1/validationとdense-v1/validationの全200件ずつで新旧モデルを比較する。
文字の平均確率・0.5閾値誤反応率、余白への反応、実線・破線描画部・破線隙間の再現率を確認する。
actual 99枚もscale 0.5で比較する。追加損失込みのvalidation lossを過去のlossと直接比較しない。
出力名は `table-cnn-textguard-dashed-comparison-v1.json`、
`table-cnn-textguard-dense-comparison-v1.json`、
`table-cnn-textguard-pilot-v1/actual-preview-half/`。
この試行は負例比率と損失の両方を変更するため、各変更の寄与を独立に特定する実験ではない。
