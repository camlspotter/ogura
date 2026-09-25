# 画像素材の生成サンプル

本文と混在させるための、文字を含まない写真・イラスト素材の候補を10枚生成する。
葉・木目・布・岩石・機械・建物・林・食品・小鳥・道具を用意した。
プロンプトは `prompts/image_assets_sample.jsonl`。内容と本文の対応は不要。

Synth-JDocが使用するZ-Image-Turboを、公式Diffusers APIで呼び出すローカル用スクリプト。
Synth-JDoc本体は変更しない。Flash Attentionは必須にせず、標準SDPAを使用する。
公式モデル: https://huggingface.co/Tongyi-MAI/Z-Image-Turbo
モデルrevisionはスクリプト内で固定。9 steps・guidance_scale=0、bfloat16を使用する。
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
生成処理はCUDAとbfloat16対応を必須とし、CPUでの意図しない長時間生成を行わない。

出力 `ogura/textdet/outputs/image-assets-sample-v1/`:

- `images/`: 10枚のPNG。ファイル名はプロンプトID。
- `contact-sheet.jpg`: ID付きの一覧画像。
- `manifest.json`: 全プロンプト、モデルrevision、seed、画像ハッシュ、実行状態。

画像は自動では学習データに入れない。原寸で文字・数字・ロゴ等がないことを目視確認し、
合格した画像だけを将来の文書合成に使う。文字なしの指定でも生成結果は保証されない。
一覧画像に付けたIDは確認用で、学習に使う画像は `images/` 内の原画像。
モデルと出力はGit対象外。SSHなどのリモート接続は行わない。

Mac側へのコピー例:

```bash
scp -r dgx:~/ogura/ogura/textdet/outputs/image-assets-sample-v1 ~/ogura/ogura/textdet/outputs/
```
