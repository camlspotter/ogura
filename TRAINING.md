# 画像バッチとCNN/CTC学習

高さ48px、Noto Sans CJK JP Regular、黒文字・白背景で学習する最小構成。
画像はCPU上で必要なバッチだけ描画する。PNGを全件保存する必要はない。
CNNで横方向の特徴列を作り、CTCで学習する。モデルはランダム初期化。
これは学習・再開の実装であり、実文書での精度を確認したモデルではない。

## 準備

```sh
uv sync --frozen --extra train
```

GPUマシンへはコード・uv.lockと、少なくとも次のファイルをコピーする。

- datasets/final_50_len20_25_hiragana_mix5/train.txt
- datasets/final_50_len20_25_hiragana_mix5/targets.jsonl
- corpus/fonts/NotoSansCJKjp-Regular.otf と同じ場所の LICENSE

これらのデータとフォントはGit管理外。別の保存場所を使うときは
--text、--vocabulary、--font で指定できる。再開時は runs/ 内の実行ディレクトリも必要。
NVIDIAドライバとインストールしたPyTorchのCUDA対応をGPUマシンで確認する。

```sh
uv run --frozen --extra train python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

## バッチAPI

```python
from ogura.training.render import BatchRenderer, RenderParams, Sample, Vocabulary

vocabulary = Vocabulary.read('datasets/final_50_len20_25_hiragana_mix5/targets.jsonl')
render_batch = BatchRenderer(vocabulary)
params = RenderParams(font_path='corpus/fonts/NotoSansCJKjp-Regular.otf', font_size=40)
batch = render_batch([
    Sample(text='今日は晴れです', render_params=params, sample_id='example-1'),
    Sample(text='明日は雨の予報です', render_params=params, sample_id='example-2'),
])
```

| 属性 | 形式 |
| --- | --- |
| images | float32 [B, 1, 48, W]、黒0・白1 |
| image_widths | int64 [B]、バッチ用右余白を追加する前の幅 |
| targets | int64 [正解の総文字数]、各サンプルのID列を連結 |
| target_lengths | int64 [B] |
| texts、sample_ids | デバッグ・追跡用 |

文字IDはtargets.jsonlの順で1から採番し、0はCTC blank専用。
画像はグレースケール、左右に4px以上の余白。字形が高さ40pxに収まらない場合は、
フォントサイズを下げて縦横比を維持し、48px高の中央に配置する。
バッチ内の最大幅を8の倍数に切り上げ、右を白で埋める。
未収録文字は1文字ずつ通常のスペース（U+0020）に置き換える。描画・正解ID列・batch.textsは
すべて置換後の文字列を使う。連続空白や先頭・末尾の空白もそのまま保持する。
置換にはフォントと文字ID辞書の両方がスペースに対応している必要がある。

CNNの横方向縮小率は8で、有効出力長は ceil(image_width / 8)。
CTCに必要な長さ（文字数＋隣接する同一文字の数）を満たすことも検査する。

## 学習開始と再開

```sh
uv run --frozen --extra train python -m ogura.training.train \
  --device cuda --run-dir runs/noto48 --batch-size 32 --epochs 10 --workers 4
```

中断後は同じ設定に --resume を追加する。

```sh
uv run --frozen --extra train python -m ogura.training.train \
  --device cuda --run-dir runs/noto48 --batch-size 32 --epochs 10 --workers 4 --resume
```

既にチェックポイントがある場所では、--resume なしの起動は拒否する。
1つの実行ディレクトリを複数のプロセスで同時に更新することも拒否する。

主なオプション：

- --log-samples 3：ログ出力時にバッチ先頭3件の正解・予測を引用符付きで表示。0で無効。追加推論はしない。
- --save-every 500：成功したパラメータ更新500回ごとに保存。開始時とエポック終了時も保存。
- --batch-size 32：初期値。GPUメモリに応じて新しい実行で調整する。
- --learning-rate 0.001、--lr-decay 0.95：AdamW、エポックごとの指数減衰。
- --channels 32：CNN最初の層のチャンネル数。後段は最大4倍。
- --amp：CUDAで混合精度。CTCとlog-softmaxはfloat32のまま計算する。
- --font-size-min 40 --font-size-max 40：既定は固定。範囲指定時はサンプルごとに決定。
- --seed 20260915：サンプル順・モデル初期化・描画条件のシード。
- --workers 0：描画ワーカー数。GPU用にはCPUの余裕に合わせて増やす。
- --deterministic：PyTorchに決定的な演算を要求。GPU側で未対応の演算はエラーになる場合がある。
- --limit N：先頭N件だけ使用する動作確認用。
- --max-steps N：累計更新N回で保存して終了。再開時には増やすか省略する。

### 未対応字

初回・再開時とも全文を検査し、指定フォントにない文字だけをスペース（U+0020）に置き換える。
行は除外せず、描画画像と正解ラベルの両方に置換後の文字列を使用する。
元のtrain.txtや文字ID辞書は変更しない。サンプルIDは元テキストのSHA256を保持する。
置換によって同じ文字列になった行も削除しない。
run_dir/font_coverage.json に未対応文字・置換行番号・置換文字数・行数を記録する。
未対応字そのものはこのフォントで学習できないため、その字の50件を維持することにはならない。
最初のフォントだけで実装を検証する方針のため、フォントの自動切り替えはしない。

旧「行を除外する」方式のチェックポイントは学習コードが異なるため再開を拒否する。
新しいrun-dirで開始する。新方式で作成したチェックポイント同士では通常どおり再開できる。

## チェックポイントの保証範囲

- latest.pt と previous.pt の2世代を保持する。
- 一時ファイルへ書き、flush/fsync後に置き換える。保存途中で失敗しても前の世代を保持する。
- latest.pt が欠損・読み取り不能ならprevious.ptを試す。条件不一致を破損扱いで迂回しない。
- モデル、AdamW、学習率スケジューラ、AMPスケーラ、次のエポック・バッチ位置・更新回数を保存。
- Python、NumPy、PyTorch CPU/CUDAの乱数状態を保存。
- データ・字種辞書・フォント・学習コードのSHA256、主要設定・実行ライブラリの版を照合する。
- テキスト順はシードとエポック、描画条件はシード・エポック・サンプルIDから再現する。
- 先読み済みのバッチは保存しない。最後に保存した「次のバッチ」から再生成する。
- エラー・Ctrl-Cの際に途中のモデルを無理に保存せず、最後に完了した保存地点へ巻き戻る。

絶対パス、ワーカー数、ログ・保存間隔、終了ステップ、予定エポック数は変更可能。
バッチサイズ、学習率、文字辞書などを変えて継続する操作は、この再開機能の対象外。
CPUとCUDAの間の切り替えやライブラリの版変更も拒否する。
CPUの決定的なテストでは中断・再開と連続実行の重み・学習状態が完全一致することを確認する。
GPUや異なるハードウェア間でのビット単位の一致までは保証しない。

## 小規模な動作確認

```sh
uv run --frozen --extra train python -m ogura.training.train \
  --device cpu --run-dir runs/smoke --batch-size 2 --channels 4 \
  --limit 4 --epochs 1 --max-steps 1 --save-every 1 --threads 1

uv run --frozen --extra train python -m ogura.training.train \
  --device cpu --run-dir runs/smoke --batch-size 2 --channels 4 \
  --limit 4 --epochs 1 --max-steps 2 --save-every 1 --threads 1 --resume

uv run --frozen --extra train python -m unittest discover -s tests
```

テストには実際の描画、CTC逆伝播、保存中の失敗、破損からの復旧、
未保存更新後の中断・再開、ワーカー数を変えた再現性の検証が含まれる。
複数ワーカーのテストにはOSの共有メモリが必要。
独立した検証データでの評価は下記の手順で有効にする。推論専用CLIはまだ含まない。

## バッチ・エポックの正解率と時間

実行ディレクトリの metrics.jsonl に、全バッチと完了した各エポックを1行ずつ記録する。
端末には --log-every ごとのバッチと、すべてのエポックの結果を表示する。

| 項目 | 意味 |
| --- | --- |
| kind | batch または epoch |
| epoch、batch、step | 1始まりのエポック・バッチ番号、累計成功更新回数 |
| exact_accuracy | 文字列が正解と完全一致したサンプルの割合（0〜1） |
| cer | 挿入・削除・置換の最小回数 ÷ 正解の総文字数。1を超える場合もある |
| mean_loss | サンプル数で重み付けした平均CTC損失 |
| seconds | データ待ち・描画、転送、順伝播、逆伝播・更新、greedy復号の経過秒数 |
| samples_per_second | サンプル数 ÷ seconds |
| samples、exact_matches、character_errors、reference_characters | 集計の分子・分母 |
| optimizer_updated | AMPのオーバーフロー等で更新を飛ばしたか（バッチのみ） |

精度は学習バッチの順伝播出力をgreedy CTC復号し、未対応字をスペースに置換した正解と比較する。
そのバッチによる重み更新より前の出力を使用する。独立した検証データでの正解率ではない。
エポックの精度はバッチの率の単純平均ではなく、全文字列・全文字の件数から算出する。
エポックの時間は各バッチ時間の合計。保存・ログ書き込み、データ事前検査、モデル初期化、
停止していた時間は含まない。GPUでは同期してから時間を確定する。

途中までのエポック集計とログの確定位置をチェックポイントに保存する。
再開時は確定位置より後のログを切り戻して再計測し、巻き戻ったバッチを二重計上しない。
以前の世代へ復旧する場合も同じ。再開用に metrics.jsonl もチェックポイントと一緒に保持する。
計測機能の追加前のチェックポイントはコード・保存形式が違うため、新しい実行ディレクトリで開始する。

正解・予測の表示件数は再開時にも変更可能。データ・フォント・学習設定等は引き続き照合する。

## 固定検証データと最良モデル

Wikipedia取得済みの環境で、一度だけ次を実行する。

```sh
uv run --frozen python -m ogura.build_validation
```

予約済みのvalidation記事から1,000行を作り、datasets/validation/ に保存する。
1記事1行、20〜25文字。文字表を除外し、学習テキストと16文字連続で一致する候補も除外する。
全対象記事を走査し、固定シードで候補を選ぶ。train/test記事は採取しない。
validation.jsonl に出典と原文位置、manifest.json に入力と検証テキストのハッシュを記録する。
出力が既に存在する場合は上書きせず停止する。学習文や辞書を変える場合は
--output で新しい検証データの保存先を指定する。
--training-text、--targets、--corpus で既定以外の入力を指定できる。
検証データは自然文での精度を測るための集合であり、希少字を均等に含むものではない。

検証付きで全学習データの学習を新しく始める例：

```sh
uv run --frozen --extra train python -m ogura.training.train \
  --device cuda --run-dir runs/noto48-validation \
  --batch-size 32 --epochs 10 --workers 4 \
  --validation-text datasets/validation/validation.txt \
  --save-every 500 --log-every 20 --log-samples 3
```

再開時は同じコマンドに --resume を追加する。--limit は指定しない。
--validation-text を省略すると、従来どおり学習データの計測のみを行う。

各エポックの全更新完了後、model.eval() と勾配計算なしで検証する。
検証は毎回同じ順序・高さ48px・フォントサイズ40px（--validation-font-size で変更可能）。
学習のフォントサイズ範囲、乱数消費、AMPの設定によらず、固定条件のfloat32で評価する。
未対応文字は画像と正解の両方でスペースに置換する。検証用の置換結果も別に報告する。
学習データと検証データが置換後に完全一致する場合は起動を拒否する。
生成済みmanifestがある場合は、学習テキスト・辞書・検証テキストの対応も照合する。

metrics.jsonl に kind=validation として完全一致率、CER、損失、所要時間を保存する。
検証CERが過去最小を更新したときだけbestを更新する。同点なら先のモデルを保持する。
**best.pt は推論・比較用の重み、latest.pt / previous.pt は学習再開用**。
best.pt にはモデル、文字辞書、チャンネル設定、採用エポック・ステップ・検証値を含む。

最良時点の重みと成績は再開用チェックポイントにも格納する。
checkpointの保存完了後にbest.ptを書き出し、再開時にも保存済みのbestを復元する。
best.ptの書き込み中断やprevious.ptへの巻き戻しでも、評価ログと最良モデルを同じ地点へ戻せる。
検証が学習の乱数状態を変えないこと、中断再開で評価履歴と最良モデルを引き継げることをテストしている。

検証機能の追加でチェックポイント形式が変わったため、以前の過学習試験用チェックポイントから
この新しい学習を再開することはできない。過学習試験は残し、新しいrun-dirで開始する。


## エポックの経過時間・終了予想

--log-every のバッチログに以下の3つを表示する。

- elapsed=00:04:20：現在のエポックの経過時間。
- remaining=00:03:10：学習バッチが終わるまでの予想残り時間。
- finish_local=2026-09-16T15:32:10+09:00：実行マシンのローカル時刻での終了予想。UTCオフセット付き。

併せてbatch=6300/10620 (59.3%)のように進捗も表示する。
経過時間は描画待ち・学習・ログ・保存を含む実時間。エポック開始前の事前検査や、
エポック後の検証評価は含まない。予想は平均バッチ所要時間 × 残りバッチ数。
初期のワーカー起動や処理速度の変化で見積もりは変動する。
チェックポイントには経過時間も保存し、再開時は停止していた時間を含めず合算する。
保存の直前までの時間を記録するため、その保存処理自体の時間は巻き戻し時に失われる。
ETA表示追加直前の版のチェックポイントも再開可能。その場合、中断前の経過時間は
保存済みのバッチ処理時間合計から近似する（過去のログ・保存時間は含まれない）。
