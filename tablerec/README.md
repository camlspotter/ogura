# Table recognition

表の構造・セル認識を扱う独立したPythonパッケージ `ogura-tablerec`。
`textdet` は文字行の検出、`textrec` は文字認識、`tablerec` は表内の罫線・セル構造を担当する。
Gitリポジトリは共通の `ogura` のまま。

- [合成データ作成](SYNTH_TABLE_CELLS.md): JSONに描画パラメータを保存し、画像・教師マップを必要時にメモリ上で生成する。
- [罫線CNN](TABLE_CNN.md): 横・縦の罫線マップをU-Netで学習する。セルへの復元処理は未実装。

## セットアップ

リポジトリルートから既存の統合環境を使う場合:

```sh
uv sync --locked --inexact
uv run --frozen --inexact python -m ogura.tablerec.synth_table_cells --help
uv run --frozen --inexact python -m ogura.tablerec.train_table_cnn --help
```

表認識だけの環境を作る場合:

```sh
cd tablerec
uv sync --locked
uv run --frozen python -m ogura.tablerec.train_table_cnn --help
```

リンク先の実行例は**リポジトリルート基準**。`tablerec/` から実行する場合は
引数の `tablerec/outputs/...` を `outputs/...` に読み替える。
共有フォントはリポジトリ直下の `corpus/fonts/` を参照する。
生成JSON・チェックポイントは `tablerec/outputs/` 配下に置き、Git管理しない。

## 旧配置からの移行

Pythonモジュールは `ogura.textdet.synth_table_cells / table_cnn / train_table_cnn` から
`ogura.tablerec.synth_table_cells / table_cnn / train_table_cnn` に移動した。
旧名の互換モジュールは置かない。`textdet` の文書ページ生成用 `synth_tables.py` はそのまま。

別マシンではGit更新だけでは既存のローカルデータは移動しない。
今回の表認識用 `textdet/outputs/table-cells-*` と `textdet/outputs/table-cnn-*` を、
同名の移動先がないことを確認して `tablerec/outputs/` に移すか、JSONを再生成する。
フォントの場所が異なる場合は `--font-dir` を使う。
移動済みJSON・チェックポイント内部の旧パスや生成コードハッシュは実行履歴として保持する。
既存の重みは `ogura.tablerec.table_cnn.load_model` で読み込める。

## 検証

リポジトリルートから:

```sh
uv run --frozen --inexact python -m unittest discover -s tablerec/tests
```
