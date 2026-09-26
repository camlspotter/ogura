# 別マシンのPDFから学習画像を再生成

ネットワーク接続・SSH・モデル学習は行わない。すべて実行したマシン上のCPUで処理する。
PDF自体をコピーする必要はなく、別マシンにある同一内容のPDFを利用できる。

## 転送するもの

- 最新の `ogura/textdet/` のコードとリポジトリルートの `pyproject.toml`・`uv.lock`（Git対象）。
- `ogura/textdet/outputs/experiment-v1-recipe.json`（Git管理対象）。

レシピは387ページの明示的な採用一覧と分割、PDF内容ハッシュを含む。
train 308、val 39、test 40。手動除外・重複除外・品質保留ページは含まない。
PDF・画像・ラベルはコミットしない。再生成用レシピはGitで管理する。転送とGPUマシンへのアクセスはユーザーが行う。

## GPUマシン側の準備

リポジトリルートで実行する。Python 3.12以上を使用する。
textrecとtextdetでルートの `.venv` を共用し、通常の `uv sync` で画像生成・学習の依存関係をまとめてインストールする。

```sh
uv sync
```

この再生成にはPoppler / pdffontsやCUDAは不要。PDF一次フィルタは再実行しない。

## 画像生成

Gitで取得した `ogura/textdet/outputs/experiment-v1-recipe.json` を使用する。
`/path/to/JDocQA_pdf_files` をPDFがあるディレクトリに置き換える。
PDFはそのディレクトリ直下に、元のファイル名で必要。

```sh
uv run --locked python -m ogura.textdet.regenerate_dataset generate \
  --recipe ogura/textdet/outputs/experiment-v1-recipe.json \
  --pdf-root /path/to/JDocQA_pdf_files \
  --output ogura/textdet/outputs/experiment-v1-regenerated \
  --workers 2
```

出力例:

```text
experiment-v1-regenerated/
  manifest.json
  train/
    images/       原画像のみ（学習入力）
    labels.json   docTR形式の行bbox
    review/       bbox画像と文字・書字方向を含む詳細ラベル
  val/            同じ構成
  test/           同じ構成
```

PDF内容、抽出コード、PyMuPDF/Pillowの版を照合し、不一致なら停止する。
版は `pyproject.toml` と `uv.lock` で固定する。PDFはファイル内容で照合するので、パス・更新日時は異なってもよい。
全対象PDFを確認してから画像生成を開始し、既存の出力先は上書きしない。
途中失敗時はmanifestが `failed` となる。原因を解消し、別の出力先でやり直す。

試運転だけなら `--limit 3` と別の `--output` を指定する。
この場合はmanifestの `partial_run=true` となり、学習用の完全なデータではない。
最終的に `status=complete` かつ `partial_run=false` とページ数を確認して利用する。

## 選別を変更した場合（元のレビュー環境で実行）

選別から実験データを再エクスポートした後、新しいファイル名のレシピを生成する。

```sh
uv run --locked python -m ogura.textdet.regenerate_dataset pack \
  --experiment ogura/textdet/outputs/experiment-v1 \
  --pdf-root JDocQA_pdf_files \
  --output ogura/textdet/outputs/experiment-v1-recipe.json
```

レシピの上書きはしない。抽出コードの変更時も、データ再エクスポートとレシピ再作成が必要。
