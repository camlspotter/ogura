# 画像素材の生成サンプル

本文と混在させるための、文字を含まない写真・イラスト素材の候補を10枚生成する。
葉・木目・布・岩石・機械・建物・林・食品・小鳥・道具を用意した。
プロンプトは `prompts/image_assets_sample.jsonl`。内容と本文の対応は不要。
生成モデルへのプロンプトは英語で、被写体・質感・構図を記述する。
日本語版で文字の混入が多かったため、長い禁止事項の列挙も外して比較する。
言語だけを変えた比較ではなく、改善の有無は生成画像の目視で確認する。

Synth-JDocが使用するZ-Image-Turboを、公式Diffusers APIで呼び出すローカル用スクリプト。
Synth-JDoc本体は変更しない。Flash Attentionは必須にせず、標準SDPAを使用する。
公式モデル: https://huggingface.co/Tongyi-MAI/Z-Image-Turbo
モデルrevisionはスクリプト内で固定。9 steps・guidance_scale=0、bfloat16を使用する。
生成サイズの既定値は512×512ピクセル。`--size` で変更できる。
依存パッケージは共通のルートpyproject.toml/uv.lockに含む。

GPU側の `~/ogura` で:

```bash
uv sync --locked
# 軽い確認。モデルのダウンロードや推論は行わない。
uv run --locked python -m ogura.textdet.generate_image_assets check
# モデル取得だけを先に行う（容量の大きなダウンロード）。
uv run --locked python -m ogura.textdet.generate_image_assets download
# 最初の1枚でGPU動作確認。10枚の出力とは分ける。
CUDA_VISIBLE_DEVICES=0 uv run --locked python -m ogura.textdet.generate_image_assets generate \
  --limit 1 --output ogura/textdet/outputs/image-assets-one-v1
# 全10枚。既に取得したモデルはキャッシュから利用する。
CUDA_VISIBLE_DEVICES=0 uv run --locked python -m ogura.textdet.generate_image_assets generate
```

`generate` は未取得のモデルを自動取得するため、`download` は省略可能。
モデルキャッシュは `ogura/textdet/.cache/z-image-turbo/`。
GPUメモリに余裕がない場合は `--cpu-offload` を追加する（CPUへの退避で速度は低下）。
中断後は同じ引数に `--resume` を追加。設定・プロンプト・生成済み画像のハッシュを検証する。
以前の1024×1024の出力は512×512として再開できないため、
`--output ogura/textdet/outputs/image-assets-512-v1` など別の出力先を指定する。
英語プロンプトでの再生成も、日本語版とは別の出力先
（例: `--output ogura/textdet/outputs/image-assets-512-en-v1`）を指定する。
生成処理はCUDAとbfloat16対応を必須とし、CPUでの意図しない長時間生成を行わない。

出力 `ogura/textdet/outputs/image-assets-sample-v1/`:

- `images/`: 10枚のPNG。ファイル名はプロンプトID。
- `contact-sheet.jpg`: ID付きの一覧画像。
- `manifest.json`: 全プロンプト、モデルrevision、seed、画像ハッシュ、実行状態。

画像は自動では学習データに入れない。原寸で文字・数字・ロゴ等がないことを目視確認し、
合格した画像だけを将来の文書合成に使う。文字なしの指定でも生成結果は保証されない。
一覧画像に付けたIDは確認用で、学習に使う画像は `images/` 内の原画像。
モデルと出力はGit対象外。SSHなどのリモート接続は行わない。

## 200枚の素材集

`prompts/image_assets_200.jsonl` は20種類×10パターンの英語プロンプト。
葉・花・樹皮・岩石・水面・雲・鳥・魚・昆虫・毛並み・食品・布・陶器・工具・
機械部品・建物外壁・室内・街並み・植物イラスト・日用品イラストを含む。
各種類で被写体の具体例も変え、構図・照明・背景・描画スタイルを組み合わせる。
写真180枚とイラスト20枚。細かい質感と余白のある構図の両方を含む。
既存の10枚用プロンプトと出力はそのまま残す。

```bash
uv run --locked python -m ogura.textdet.generate_image_assets generate \
  --prompts ogura/textdet/prompts/image_assets_200.jsonl \
  --size 512 \
  --output ogura/textdet/outputs/image-assets-200-v1
```

モデルは最初に一度読み込む。CUDAの先行初期化はスクリプト内で行う。
中断時は同じコマンドに `--resume` を追加する。
画像名は `leaves-01.png` から `object_illustrations-10.png` のように種類と番号を含む。
この素材集も生成後に文字・数字・ロゴの混入を目視確認してから採用する。

## 素材を使った文書ページ

確認済み200枚を `outputs/image-assets-200-v1/images/` に置き、
既存の `outputs/synth-texts-5000-v1.jsonl` と4種類のNotoフォントを使用する。
Synth-JDocの既存image要素を利用し、上流コードは変更しない。

```bash
# 本文中心8ページ（うち縦書き2）と表付き4ページ、計12ページのサンプル。
bash ogura/textdet/scripts/synth_image_pages.sh generate
# 中断から再開。件数・出力先を変えずに実行する。
bash ogura/textdet/scripts/synth_image_pages.sh resume
```

出力先は `outputs/synth-image-pages-pilot-v2/`。`text/` と `tables/` の各フォルダに
`images/`（ページ画像）、`review/`（文字bbox付き）、`html/`、`labels.json`、
`metadata/` と `manifest.json` を保存する。使用した素材とフォントもコピーする。
画像は1ページに1枚。既定では半数を本文内のfloatにして、文字を画像の周囲に回り込ませる。
floatは横書きでは左・右、縦書きでは上・下に寄せ、段の幅（縦書きでは高さ）の42%以下にする。
残り半数は横書きで上・下、縦書きで右・左に画像用の領域を確保し、大きさと寄せ方を変える。
`--image-float-fraction` でfloatの割合を指定できる（0〜1、既定0.5）。
`--image-position top/bottom` は通常配置に適用し、縦書きでは右/左に対応する。
旧v1とは配置設定が異なるので、再開せず別の出力先に生成する。
素材自体は文字bboxの対象にせず、画像の欠落・ページ外へのはみ出し・文字との重なりを検査する。
本文・見出し・表の文字は従来通り正解に含む。素材のファイル名・SHA-256・画像位置も記録する。
再開時は素材一覧とハッシュも照合し、変更があれば停止する。

件数と出力先は引数で指定できる。例えば600ページ（本文400・表付き200）なら:

```bash
bash ogura/textdet/scripts/synth_image_pages.sh generate 600 \
  ogura/textdet/outputs/synth-image-pages-600-v1
```

素材は各グループ内で200枚をランダム順に一巡してから再利用する。
画像生成モデルは呼ばず、ページの合成にGPUは不要。

## 画像入り2,000ページを追加して学習

`scripts/synth_images_2000.sh` で生成・結合・学習・比較・結果の梱包を行う。
画像入り本文1,334ページ（うち縦書き334）と、画像・表付き666ページを生成する。
float配置は計1,000ページ、通常配置は計1,000ページ。
既存5,000＋表2,000＋見出し1,000＋見出し・表1,000は保持し、
別の `outputs/synth-mixed-11000-v1/` に画像とラベルをコピーして結合する。
学習は以前と同じ公式pretrainedから5epoch、入力1024、batch 2、lr 0.0001で開始する。
GPU側のリポジトリ直下で順番に実行する:

```bash
bash ogura/textdet/scripts/synth_images_2000.sh generate
# 生成が中断した場合だけ、generateの代わりにresumeを使う。
bash ogura/textdet/scripts/synth_images_2000.sh combine
bash ogura/textdet/scripts/synth_images_2000.sh train
bash ogura/textdet/scripts/synth_images_2000.sh evaluate
bash ogura/textdet/scripts/synth_images_2000.sh package
bash ogura/textdet/scripts/synth_images_2000.sh download-command
```

比較対象は9,000ページ版と11,000ページ版。JDocQAの固定validation 39ページを使い、
後処理は従来選択した入力1536・二値化閾値0.5・unclip 1.0・box閾値0.1に固定する。
testセットはこのスクリプトでは使用しない。結果には両モデルのbbox画像・確率マップPNG・
集計値・予測JSONを含め、ダウンロード用tar.gzには学習画像・重み・確率配列npyを含めない。
まず11,000ページで効果を比較し、追加増量は結果を見て判断する。

Mac側へのコピー例:

```bash
scp -r dgx:~/ogura/ogura/textdet/outputs/image-assets-sample-v1 ~/ogura/ogura/textdet/outputs/
```
