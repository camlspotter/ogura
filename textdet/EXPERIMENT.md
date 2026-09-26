# textdet 初回実験

準備済みデータ: `outputs/experiment-v1/`。元データ・trash・除外リストは変更しない。

- train: 308ページ、31,773行
- val: 39ページ、3,544行
- test: 40ページ、5,601行
- 空白bboxの未解決候補がある4ページは実験だけから保留。対象はデータmanifestに記録。
- PDF文書と類似候補の連結グループで分割。除外された文書を介した関係も保持。
- 乱数seed: 20260921。画像は原画像のみ。bbox画像は入力に使わない。
- 座標は画像ピクセル単位の矩形を4頂点に変換し、画像外をクリップ。
- 行テキストの認識・読み順は今回の対象外。

## 再作成

```sh
uv run --locked python -m ogura.textdet.prepare_experiment \
  --output outputs/experiment-v1
```

既存の出力先は上書きしない。データmanifestに元ラベル・入力一覧のハッシュを記録。

## 実行予定

GPUマシンへのアクセスと学習実行はユーザーが行う。アシスタントはSSHを使用せず、ローカルでコード・データ・実行手順を準備する。
現時点では学習・モデル評価は未実施。

最初の候補は docTR 1.0.0 の `db_resnet34` を学習済み重みから微調整する。
データは公式 DetectionDataset 形式。単一クラスの文字行検出として扱う。
書字方向は元ラベルに残すが、この初回モデルの出力クラスにはしない。

1. ユーザーがGPUマシン上に共通環境を準備し、CUDA・モデル読み込み・1バッチの逆伝播を確認。
2. 学習前モデルをvalで評価（precision/recall/F1、IoU 0.5での行対応）。
3. 初期案は5 epoch、lr=1e-4、batch=2。入力解像度とバッチサイズはGPU容量と細字の縮小率を見て決定。
4. valの結果と予測画像を比較。横書き・縦書きの差、行の結合/分割も確認。
5. 設定を固定してからtestを最終評価に使う。testでハイパーパラメータを選ばない。

参考: https://github.com/mindee/doctr/tree/v1.0.0/references/detection

類似候補の分割制御は既存の候補一覧に基づくため、未知の類似文書の漏れは保証しない。
この実験の値は391ページの手動選別に由来する小規模データでの結果であり、一般の文書性能とは区別する。

別マシンのPDFから画像を生成する場合は [REGENERATE.md](REGENERATE.md) を参照。選別と分割を再決定せず、ポータブルな採用ページレシピから再生成できる。

## uvでの学習コマンド

以下はすべて `cd ~/ogura/textdet` の後に実行する。依存関係は
`textdet/pyproject.toml`、解決した版は `textdet/uv.lock` に記録する。
`uv sync --locked` で検出用の依存関係が入る。仮想環境のactivateは不要。

```sh
uv sync
uv run --locked python -c \
  'import torch; assert torch.cuda.is_available(), "CUDAが利用できません"; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
```

CUDA確認に失敗した場合は、GPU・ドライバ・CPUアーキテクチャに対応するPyTorch配布を確認する。
手動でpipを重ねるとuvの同期で戻るため、必要な配布元の変更もpyprojectとlockに反映する。
GPU上での学習は未検証。

画像は先に [REGENERATE.md](REGENERATE.md) の手順で生成する。
公式学習スクリプトは初回だけHTTPSで取得する。

```sh
mkdir -p .cache
git clone --depth 1 --branch v1.0.0 \
  https://github.com/mindee/doctr.git \
  .cache/doctr-v1.0.0
```

初回の学習例（入力1024、batch 2、5 epoch）:

```sh
mkdir -p outputs/db-resnet34-v1
CUDA_VISIBLE_DEVICES=0 \
uv run --locked python \
  .cache/doctr-v1.0.0/references/detection/train.py \
  db_resnet34 \
  --pretrained --device 0 \
  --train_path outputs/experiment-v1-regenerated/train \
  --val_path outputs/experiment-v1-regenerated/val \
  --output_dir outputs/db-resnet34-v1 \
  --name jdocqa-db-resnet34-v1 \
  --epochs 5 --batch_size 2 --input_size 1024 --lr 0.0001 \
  --workers 2 --amp --save-interval-epoch
```

GPUメモリ不足ならまずbatchを1に下げる。testは設定を固定した後の最終評価に残す。

## 学習前後の検証画像を比較

`textdet/`で実行する。公式trainerの最良validation lossの重み（epoch番号なし）を使用する。
第5epochそのものと比較する場合はcheckpointを `jdocqa-db-resnet34-v1_epoch5.pt` に変える。

```sh
CUDA_VISIBLE_DEVICES=0 uv run python -m ogura.textdet.compare_predictions \
  --data outputs/experiment-v1-regenerated/val \
  --checkpoint outputs/db-resnet34-v1/jdocqa-db-resnet34-v1.pt \
  --output outputs/validation-comparison-v1 \
  --device cuda:0 --input-size 1024 --amp
```

- `*_comparison.png`: 左から正解、学習前、学習後。正解は緑、予測は黒縁のマゼンタ。
- `*_gt.png`, `*_pretrained.png`, `*_finetuned.png`: 個別の拡大確認用画像。
- `summary.json`: 全体のRecall・Precision・F1・Mean IoU、重みのハッシュ、実行条件、完了状態。
- `pages.csv`: 同じ指標のページ別一覧。数値は0〜1。
- `predictions.json`: リサイズ・余白追加後の入力画像に対する正規化座標と予測スコア。

学習時と同じアスペクト比維持・中央余白追加・正規化を用い、実際にモデルへ入力した
1024角の画像に描画する。元のページ画像の座標ではない。
学習時と同じdocTRのIoU 0.5の対応付けで計算し、追加の予測スコアフィルタはかけない。
1ページずつ推論するため、バッチや数値誤差により学習ログの値とわずかに異なる場合がある。
lossは計算しない。学習前モデルの重みは初回にdocTRがダウンロードする。
モデルは順にロードする。既存の出力先は上書きしない。失敗時はsummaryのstatusがfailedになる。
`--limit 3` と別の出力先で試運転できるが、全39ページの比較時にはlimitを外す。
testデータはこの比較には使用しない。GPUマシンへのアクセスと実行はユーザーが行う。
