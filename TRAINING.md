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
すべて置換後の文字列を使う。連続する半角空白は1文字にまとめ、先頭・末尾の半角・全角空白は除去する。
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

- --log-samples 3：ログ出力対象のバッチ内で、各行のCER（編集距離÷正解文字数）が高い順に3件のCER・正解・予測を表示。同率ならバッチ内の順序を維持。0で無効。追加推論はしない。
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

## 第1段階の描画拡張と追加学習

ゴシック・明朝のRegular/Bold、文字サイズ、余白、上下位置をサンプルごとに抽選する。
画像の高さは48pxのまま。文字が収まらない場合はサイズを下げ、上下移動は文字が
欠けない範囲に制限する。左右余白と上下の最低余白を、それぞれpaddingの範囲から独立に抽選する。
ノイズや傾きはこの段階では加えない。

まず固定リビジョンのNotoフォント4本とライセンスを取得する（約60〜70MB）。
既存ファイルが配布元と異なる場合は上書きせず停止する。

```sh
uv run --frozen python -m ogura.download_training_fonts
```

追加学習の開始例：

```sh
uv run --frozen --extra train python -m ogura.training.train \
  --device cuda --run-dir runs/noto48-augment1 \
  --init-from runs/noto48-validation/best.pt \
  --extra-font corpus/fonts/NotoSansCJKjp-Bold.otf \
  --extra-font corpus/fonts/NotoSerifCJKjp-Regular.otf \
  --extra-font corpus/fonts/NotoSerifCJKjp-Bold.otf \
  --font-size-min 28 --font-size-max 40 \
  --padding-min 2 --padding-max 6 --vertical-full-range \
  --clean-probability 0.25 \
  --learning-rate 0.0001 --batch-size 32 --epochs 10 --workers 4 \
  --validation-text datasets/validation/validation.txt --validation-augmented \
  --save-every 500 --log-every 100 --log-samples 1
```

これらの範囲・確率・学習率は初回実験の設定で、最適値として検証したものではない。
`--font`（既定はNoto Sans Regular）と各`--extra-font`を等確率で選ぶ。
ただし25%は元の条件（基準フォント、40px、余白4px、上下移動なし）に戻す。
次のエポックでは再抽選する。同じシード・エポック・サンプルID・設定なら
ワーカー数や中断にかかわらず同じ画像になる。

複数フォント時は元のテキストを保持し、選ばれたフォントで描けない文字だけを
バッチ作成時に空白へ置換する。正解も同じ空白にする。元のデータファイルは変更しない。
`font_coverage.json` の置換数は「いずれかのフォントで置換が必要な文字」の件数であり、
実際に各エポックで置換した件数ではない。

`--init-from` は旧版を含む `best.pt` の重みだけを読み込む。字種の順序とモデルの
channelsは一致必須。optimizer・学習率スケジューラ・エポック数・最良値の履歴は
新しく開始する。元の実行とは別の `--run-dir` を使う。
再開時は上のコマンドから `--init-from ...` を外して `--resume` を付ける。
新しい実行のフォント・描画条件・学習率は再開時にも同じ値にする。
この変更より前の `latest.pt` の厳密な再開には旧版コードが必要。
新しい描画条件への移行には `--init-from` を使う。

`--validation-augmented` 指定時は同じ検証テキストを2通りで評価する。

- `validation_baseline`：基準フォント、固定40px（`--validation-font-size`で変更可能）、余白4px。
- `validation`：学習と同じ範囲から固定シード・epoch=0で抽選。cleanへの分岐は使わず、
  毎回同じフォント・サイズ・余白の画像を評価する。こちらのCERで新しい `best.pt` を選ぶ。

元の実行のCERと比べるときは `validation_baseline` を見る。
揺らぎ付きの値は別の評価条件なので、元のCERと直接比較しない。
`--validation-augmented` を省略すると、従来と同じ基準条件の検証のみを行う。


## 同じ文字列の描画比較画像

学習用の描画関数を使い、基準1行と4フォント×3条件の計13行を1枚のPNGにまとめる。
白い矩形がモデルへの入力画像（高さ48px、拡大縮小なし）。ラベルと外側の灰色背景は比較用で、
学習画像には含まれない。左右余白は独立に抽選する。
各フォントが必ず載るようフォント別にサンプルを作り、サイズ・余白・上下位置は
学習時と同じ乱数生成関数で選ぶ。通常学習でのフォント出現比率を示す図ではない。

```sh
uv run --frozen --extra train python -m ogura.training.preview \
  --text '春の図書館で、日本語の文字をゆっくり読む。' \
  --output previews/augmentation.png
```

`--variants` はフォントごとの例数、`--seed` は乱数シード。
`--font` を繰り返してフォントを指定できる。`--size-min/max`、`--padding-min/max`、
`--vertical-jitter` で範囲も変更可能。同名のJSONには元テキストと各行の描画条件・画像幅・
空白置換後の正解を保存する。ラベルのsize/dyは要求値で、文字が欠けないよう実際には
サイズを下げたり上下移動を制限する場合がある。


描画拡張の設定例は28〜40px。`--vertical-full-range` は、上下に最低余白を
確保した上で、残る可動範囲全体から上位置を一様に選ぶ。この指定時は
`--vertical-jitter` を使わない。元の条件を選ぶclean分岐と基準条件の検証は中央配置。
比較画像も既定で28〜40px・全範囲の上下配置を使う。`Y range` は可動範囲内の位置
（0%=上端、100%=下端）。従来の±移動を確認する場合は `--vertical-mode jitter` を指定する。


## 5文字・80文字の長さ別検証テキスト

既存の20〜25文字セットを残して、各1,000件を別ディレクトリに作る。
長さはUnicodeコードポイント数。自然文の本文行から連続した部分を切り出すため、
5文字は単語・文として完結するとは限らない。短い抜粋には自然文判定を適用せず、
切り出し元の記事と本文行に適用する。文字表などはそこで除外する。

```sh
uv run --frozen python -m ogura.build_validation \
  --min-length 5 --max-length 5 --output datasets/validation_short5 \
  --exclude-validation datasets/validation/validation.jsonl

uv run --frozen python -m ogura.build_validation \
  --min-length 80 --max-length 80 --output datasets/validation_long80 \
  --exclude-validation datasets/validation/validation.jsonl \
  --exclude-validation datasets/validation_short5/validation.jsonl
```

両方とも予約済みのvalidation記事から1記事1件を採り、3セット間の記事も重複させない。
学習データとの共通16文字断片を除外する。5文字では一般的な語句が学習本文内にも
現れ得るが、学習記事からの切り出しではない。短い文字列の完全重複はセット内で除外する。
各ディレクトリの `validation.txt` は1行1件、`validation.jsonl` は記事ID・原文位置などの出典、
`manifest.json` は文字数・シード・入力ハッシュを記録する。
GPUマシンでは、そのマシンで作成した学習データを使って上記コマンドを実行する。
この手順はテキスト作成のみ。進行中の実行の `--validation-text` を差し替えると
再開の同一性チェックで拒否されるため、既存の検証設定は変更しない。

## 保存済みbest.ptの長さ別評価

```sh
uv run --frozen --extra train python -m ogura.evaluate_lengths \
  --checkpoint runs/noto48-augment1/best.pt --device cuda \
  --output runs/noto48-augment1/length-evaluation.json
```

既定で `datasets/validation_short5`、`datasets/validation`、`datasets/validation_long80` の
`validation.txt` を各々baseline（基準フォント・固定サイズ）とaugmented（固定した揺らぎ）で
評価し、6組の完全一致率・CER・時間を表示する。学習・重み更新はしない。
描画設定とシードはチェックポイントから読み、各セットともepoch=0固定。
フォントのパスは隣の `run_config.json` から読む。ファイル内容のハッシュも照合する。
移動後の実行では `--font-dir corpus/fonts` でフォントのディレクトリを変更できる。
`--run-config` で設定ファイルを明示することも可能。
各検証セットのmanifestと字種一覧も照合し、別の学習データから作ったセットは拒否する。
`--validation-text PATH` を繰り返せば評価対象を明示できる。
出力JSONは上書きしないので、再評価時は新しい `--output` を指定する。

## 毎エポックの追加監視

現在の追加学習コマンド（再開なら `--init-from` を外し `--resume`）に次を追加する。

```sh
--monitor-validation datasets/validation_short5/validation.txt \
--monitor-validation datasets/validation_long80/validation.txt
```

従来の `--validation-text` と `--validation-augmented` はそのままにする。
追加セットはbaseline/augmentedの両方を毎エポック評価し、`validation_length` として
コンソールと `metrics.jsonl` に記録する。JSONLにはepoch・step・セット名・描画条件名・
テキストのハッシュが入る。`best.pt` は従来の主検証セットのCERだけで選ぶ。
追加監視は乱数状態とモデルのtrain/eval状態を元に戻し、重み更新には影響しない。
このため監視セットは再開時に追加・削除できる（その後のエポックから計測）。
過去のエポックの追加評価は補完しない。監視対象を指定しなければ従来の評価のみ。

## CERによる早期終了

学習コマンドに `--early-stopping-patience 5` を追加すると、主検証セットのCERが
5エポック連続で最良値を更新しなかった時点で終了する。`--epochs` は引き続き
最大エポック数として働き、どちらかに到達すると正常終了する。
既定値0は早期終了を無効にする。正の値を指定するには `--validation-text` が必要。

判定は表示上の丸め前のCERで行い、同値は改善に含めない。少しでも低くなれば
待ち回数を0に戻す。`--validation-augmented` 指定時は揺らぎ付きの主検証CERを使い、
baselineや5文字・80文字の監視用評価は停止判定に使わない。
終了エポックの追加検証も完了してから、終了理由を `metrics.jsonl` の
`kind=early_stop` に記録し、`latest.pt` と `best.pt` を保存する。

実行中の学習に導入するときはCtrl+Cで中断し、更新後、従来の再開コマンドに
次を加える（その他の学習条件は変更しない）。

```sh
--epochs 60 --early-stopping-patience 5 --resume
```

待ち回数はチェックポイントの完了エポック数と最良エポックから復元するので、
旧版からの再開でもそれまでの非改善エポックを数える。再開時点で既に条件を
満たしていれば追加学習せず終了する。停止後さらに続けたい場合は、patienceを
増やすか0にして再開できる。patienceの変更は再開時に許可する。

## 深く・広くした残差CNN

`--model-type residual --channels 64` で新モデルを選ぶ。既定の `small` は従来の4層CNN。
新モデルの各段階は64/128/256/256チャネルで、縮小畳み込みの後に残差ブロックを
1/1/2/2個置く。各ブロックは3×3畳み込み→ReLU→3×3畳み込み→入力との加算→ReLU。
ブロック内では縮小せず、BatchNormやDropoutは加えない。
最後に1×1畳み込みで128チャネルへ変換してから、高さ平均・1次元CNN・文字分類を行う。
横方向の縮小率は従来と同じ1/8。16,058クラス時は8,235,834パラメータ、
FP32の重みは約32.94MB（メタデータ別）。実際のGPU速度は要計測。

モデル構造が変わるため、旧モデルの `--resume` や `--init-from` ではなく、
別のrun-dirで最初から学習する。以下は開始例。既存モデルを残して同じ検証条件で比較する。

```sh
uv run --frozen --extra train python -m ogura.training.train \
  --device cuda --run-dir runs/noto48-residual64 \
  --model-type residual --channels 64 \
  --extra-font corpus/fonts/NotoSansCJKjp-Bold.otf \
  --extra-font corpus/fonts/NotoSerifCJKjp-Regular.otf \
  --extra-font corpus/fonts/NotoSerifCJKjp-Bold.otf \
  --font-size-min 28 --font-size-max 40 \
  --padding-min 2 --padding-max 6 --vertical-full-range \
  --clean-probability 0.25 --learning-rate 0.001 \
  --batch-size 32 --epochs 60 --workers 4 \
  --validation-text datasets/validation/validation.txt --validation-augmented \
  --monitor-validation datasets/validation_short5/validation.txt \
  --monitor-validation datasets/validation_long80/validation.txt \
  --early-stopping-patience 5 \
  --save-every 500 --log-every 100 --log-samples 3
```

最初は別run-dirで `--limit 8 --max-steps 2` を追加した動作確認もできる。
その制限付き実行から全件学習へresumeはできないので、本学習は上記の新規run-dirで開始する。
新モデルの中断再開は同じ設定に `--resume` を追加する。
`best.pt` にはモデル種別も保存され、長さ別評価コマンドが対応するモデルを構築する。
旧版のsmallモデルは、更新後も従来の設定で再開・評価できる。

早期終了時は `Best checkpoint: epoch=... step=...` に続けて、最良モデルを得た
エポックの主検証・baseline・長さ別検証のaccuracyとCERを再表示する。
最終エポックの値ではなく、best更新時に保存した成績を使う。追加の推論は行わない。
旧版のチェックポイントの場合は、同じ実行の `metrics.jsonl` からbestのepoch/stepに
一致する記録を読み出す。その時点で未実施の監視セットは表示しない。
対応するログがなければ、チェックポイント内に残る主検証成績だけを表示する。
既に早期終了した実行でも、元と同じ設定とpatienceで `--resume` すれば、
追加学習せずに最良時点の成績を表示できる。

### 3種類の長さの平均CERで選ぶ

`--selection-metric mean-augmented-cer` は、揺らぎ付きの通常長
（20–25文字）、短文（5文字）、長文（80文字）のCERを等重みで平均し、
best更新とearly stoppingに使います。文字数を合算したCERではありません。
通常長には `--validation-augmented`、短文・長文にはそれぞれ
`--monitor-validation` が必要です。不足や異なる長さはエラーになります。
比較には丸める前の数値を使い、baselineは参考値として記録します。

判定基準を変更する場合は別のrun-dirを使います。`--init-from` は
`best.pt` に加えて `latest.pt` / `previous.pt` の重みにも対応します。
辞書・モデル構造が一致することを確認し、optimizerとepochは新しく開始します。
この平均基準でwarm startすると、更新前にも評価を行い、epoch=0の初期bestを
保存します。以降に改善しなければ、このモデルがbestとして残ります。
resume時には評価データのハッシュと判定基準も検証します。

例：旧runのepoch 16のlatest.ptから、学習率0.0001で追加学習します。
epoch数は新runでの上限です。途中再開は同じ指定から `--init-from ...` を外し、
`--resume` を追加してください。

```sh
uv run --frozen --extra train python -m ogura.training.train \
  --device cuda --run-dir runs/noto48-residual64-mean \
  --init-from runs/noto48-residual64/latest.pt \
  --model-type residual --channels 64 \
  --extra-font corpus/fonts/NotoSansCJKjp-Bold.otf \
  --extra-font corpus/fonts/NotoSerifCJKjp-Regular.otf \
  --extra-font corpus/fonts/NotoSerifCJKjp-Bold.otf \
  --font-size-min 28 --font-size-max 40 \
  --padding-min 2 --padding-max 6 --vertical-full-range \
  --clean-probability 0.25 --learning-rate 0.0001 \
  --batch-size 32 --epochs 20 --workers 4 \
  --validation-text datasets/validation/validation.txt --validation-augmented \
  --monitor-validation datasets/validation_short5/validation.txt \
  --monitor-validation datasets/validation_long80/validation.txt \
  --selection-metric mean-augmented-cer --early-stopping-patience 5 \
  --save-every 500 --log-every 100 --log-samples 3
```

best.ptには `selection_metric`、`selection_score`（0–1の率）と
`validation_results` を保存します。`metrics` は従来通り通常長の評価です。
早期終了時には、best時点の判定値と各検証データの成績を表示します。

### 実文書の行画像を認識する

手元のPDFから初回確認用の12行を切り出す例（Popplerが必要）：

```sh
uv run --frozen --extra train python scripts/prepare_real_samples.py \
  --source-dir ~/mocrdown/tests/data --output datasets/real_samples
```

画像はPDFの1ページ目を300dpiでレンダリングしたものから切り出す。
manifest.jsonlには元PDFのハッシュ、ページ、切り出し座標とPDFから抽出した
参照用テキストを保存する。参照用テキストは空白等の照合が必要であり、
そのまま確定正解やCER計算には使わない。contact-sheet.pngで一覧を確認できる。
出力先が既にある場合は上書きしない。

画像ディレクトリをGPUマシンにも置いてから：

```sh
uv run --frozen --extra train python -m ogura.recognize \
  --checkpoint runs/noto48-residual64-mean/best.pt --device cuda \
  --output runs/noto48-residual64-mean/real-predictions.jsonl \
  datasets/real_samples/line-*.png
```

入力は切り出し済みの横書き1行画像。縦横比を保って高さ48pxにリサイズし、
右を白で8の倍数幅まで補う。背景色はグレースケール化し、二値化はしない。
CTC greedy decodeによる認識文字列を表示・保存する。
これはPDFを画像化した資料での確認であり、スキャン由来の傾き・汚れ等は別途検証する。

二値化・階調補正の効果を同じモデルで比較するには、次を実行する：

```sh
uv run --frozen --extra train python -m ogura.recognize \
  --checkpoint runs/noto48-residual64-mean/best.pt --device cuda \
  --compare-preprocessing \
  --save-inputs runs/noto48-residual64-mean/real-inputs-comparison \
  --output runs/noto48-residual64-mean/real-predictions-comparison.jsonl \
  datasets/real_samples/line-*.png
```

各画像について `preprocessing=none`、`otsu`、`contrast` の3結果を並べて出力する。
contrastは二値化せず、元解像度で明るさの分布の両端を各1%除いて黒〜白へ線形に広げる。
中間階調を保ち、輪郭へのフィルターは使わない。`--contrast-cutoff` で両端の除外率を
0以上50未満で指定できる（0なら最小値〜最大値）。単色画像はそのままにする。
`--preprocessing contrast` で単独実行でき、結果JSONLにはcontrast_cutoffも記録する。
otsuは元解像度のグレースケール画像を大津法で二値化してから48pxに縮小する。
縮小時にはアンチエイリアスが入る。二値化は背景とともに文字の輪郭にも
影響するため、結果だけで字体と背景の影響を完全には分離できない。
`--save-inputs` でモデルに渡す画像（右余白を含む）も保存する。
単独で二値化する場合は `--preprocessing otsu` を使う。既存の出力ファイル・
画像保存ディレクトリは上書きしない。これらの画像もGitに追加しない。

### 細い書体・丸ゴシックを追加する

取得元のコミットとSHA-256を `ogura/training_fonts.json` に固定した。
次のコマンドで既存のNoto 4書体と、Noto Sans CJK JP Light、
Zen Maru Gothic Light・Regularの合計7書体およびライセンスを取得する。
M PLUS Roundedは今回の採用対象に含めない。

```sh
uv run --frozen python -m ogura.download_training_fonts --include-rounded
uv run --frozen --extra train python scripts/preview_font_candidates.py
```

取得先は `corpus/fonts/`。検証済みの既存ファイルは再取得せず、不一致なら
上書きせずエラーにする。ダウンロードした内容も固定ハッシュと照合する。
引数なしの取得コマンドは従来通り4書体のみ。
比較画像は `datasets/font_candidates/training-preview.png`、
描画パラメータと実際の正解文字列は同名のJSON、字種カバー数は
`training-preview.coverage.json` に保存する。
比較には実際の学習用レンダラーと高さ48px・サイズ28〜40pxの揺らぎを使う。
`--text`、`--variants`、`--font-dir`、`--vocabulary`、`--output` で変更可能。

追加書体を混ぜる学習例：

```sh
uv run --frozen --extra train python -m ogura.training.train \
  --device cuda --run-dir runs/noto48-residual64-rounded \
  --init-from runs/noto48-residual64-mean/best.pt \
  --model-type residual --channels 64 \
  --extra-font corpus/fonts/NotoSansCJKjp-Bold.otf \
  --extra-font corpus/fonts/NotoSerifCJKjp-Regular.otf \
  --extra-font corpus/fonts/NotoSerifCJKjp-Bold.otf \
  --extra-font corpus/fonts/NotoSansCJKjp-Light.otf \
  --extra-font corpus/fonts/ZenMaruGothic-Light.ttf \
  --extra-font corpus/fonts/ZenMaruGothic-Regular.ttf \
  --font-size-min 28 --font-size-max 40 \
  --padding-min 2 --padding-max 6 --vertical-full-range \
  --clean-probability 0.25 --learning-rate 0.0001 \
  --batch-size 32 --epochs 20 --workers 4 \
  --validation-text datasets/validation/validation.txt --validation-augmented \
  --monitor-validation datasets/validation_short5/validation.txt \
  --monitor-validation datasets/validation_long80/validation.txt \
  --selection-metric mean-augmented-cer --early-stopping-patience 5 \
  --save-every 500 --log-every 100 --log-samples 3
```

既存の学習コードが `--extra-font` を扱うため、モデル・学習処理の変更は不要。
cleanの25%は従来のNoto Sans Regular、残り75%は全7書体から一様に選ぶ。
新書体が描けない文字は、従来通りそのサンプルの画像と正解の両方で空白にする。
元のテキストは変更しない。Zenのカバー数は対象16,057字中7,277字のため、
希少字を学ぶ機会を残すためにも既存Noto書体を外さない。

別runとして初期評価から新しいbest判定を開始する。検証時のフォント集合も
増えるので、旧runのCERとの単純比較はしない。実文書の画像は学習に投入しない。
取得フォント・比較画像・実文書画像はGitに追加しない。

### 連続する半角スペースの正規化

描画不能文字をU+0020に置換した後、連続するU+0020を1文字にまとめる。
もともとテキストに含まれる連続半角スペースにも適用する。
例えば `日<描画不能><描画不能>  本` は `日 本` となり、
画像も正解もこの文字列から作成する。先頭・末尾の半角・全角空白は除去する。
行内の全角スペース（U+3000）は変更しない。すべて描画不能で空文字となるサンプルは、空白を正解として学習せずエラーにする。
コーパスファイル自体は変更しない。

学習・検証・プレビューに共通して適用する。5文字・80文字の検証セットの
区分とmanifestの長さチェックは正規化前の原文長を使う。
CERの分母とCTCの正解長は正規化後の文字列を使う。

教師データの意味が変わるため旧版のチェックポイントは `--resume` できない。
`--init-from .../best.pt`（またはlatest.pt）と新しい `--run-dir` で開始する。
例えば7書体学習の上記コマンドのrun-dirを
`runs/noto48-residual64-rounded-space1` に変更する。
旧版のbest評価値は引き継がず、初期評価から選び直す。

### 全文×全書体による検証とbest選択

`--selection-metric mean-font-cer` を指定すると、各検証文字列を
全書体で描画する。7書体・各長さ1,000文なら、揺らぎ付きは
3長さ×7書体×1,000文＝21,000画像。参考のbaselineは従来の
Noto Sans Regularによる3,000画像を追加で評価する。

各文字列・書体のサイズと位置は固定seed/epoch=0から決まり、毎エポック
同じ条件を使う。各フォントで欠字を空白に置換し、連続空白も正規化する。
21条件それぞれのCER（各条件の正規化後の総文字数が分母）を等重みで平均し、
best・early stoppingを判定する。baselineは平均に含めない。
書体別CERは3長さの平均、長さ別CERは全書体の平均として表示する。
`validation` の値は通常長の全書体平均となる。

各条件のaccuracy/CER/時間、書体別・長さ別平均をmetrics.jsonlとbest.ptに
保存する。`validation_total` はbaselineを含む一回の検証全体の実測秒数。
フォントの重複、条件の不足・重複や不正な長さはエラーにする。

実行中の旧方式から切り替えるときは、その最新チェックポイントの重みを
新runに引き継ぐ。optimizer・epoch・best履歴は新しく開始し、更新前の
epoch=0で全組み合わせを評価する。判定方式を変えてのresumeはできない。

```sh
uv run --frozen --extra train python -m ogura.training.train \
  --device cuda --run-dir runs/noto48-residual64-rounded-grid \
  --init-from runs/noto48-residual64-rounded-space1/latest.pt \
  --model-type residual --channels 64 \
  --extra-font corpus/fonts/NotoSansCJKjp-Bold.otf \
  --extra-font corpus/fonts/NotoSerifCJKjp-Regular.otf \
  --extra-font corpus/fonts/NotoSerifCJKjp-Bold.otf \
  --extra-font corpus/fonts/NotoSansCJKjp-Light.otf \
  --extra-font corpus/fonts/ZenMaruGothic-Light.ttf \
  --extra-font corpus/fonts/ZenMaruGothic-Regular.ttf \
  --font-size-min 28 --font-size-max 40 \
  --padding-min 2 --padding-max 6 --vertical-full-range \
  --clean-probability 0.25 --learning-rate 0.0001 \
  --batch-size 32 --epochs 20 --workers 4 \
  --validation-text datasets/validation/validation.txt --validation-augmented \
  --monitor-validation datasets/validation_short5/validation.txt \
  --monitor-validation datasets/validation_long80/validation.txt \
  --selection-metric mean-font-cer --early-stopping-patience 5 \
  --save-every 500 --log-every 100 --log-samples 3
```

学習を始める前に全組み合わせの評価だけを実行して負荷を測ることもできる：

```sh
uv run --frozen --extra train python -m ogura.evaluate_lengths \
  --checkpoint runs/noto48-residual64-rounded-space1/best.pt \
  --device cuda --all-fonts \
  --output runs/noto48-residual64-rounded-space1/font-grid-evaluation.json
```

この単独評価はcheckpointのrun_config.jsonに記録されたフォント集合を使う。
mean-font-cer方式のbest.ptでは `--all-fonts` を省略しても全組み合わせとなる。
旧方式の学習は引き続き `mean-augmented-cer` で実行できる。

### 欧文と日本語の混植

半角ラテン文字（Latin-1の文字を含む）・ASCII数字・ギリシャ文字・
キリル文字を欧文フォントで描くオプションを追加した。全角英数字は
変換せず、日本語フォントで描く。元コーパス・辞書は変更しない。

取得は固定コミットとSHA-256付き。Tinos Regular（セリフ）と
Arimo（サンセリフ、可変フォントの既定400）をライセンスとともに取得する：

```sh
uv run --frozen python -m ogura.download_training_fonts --include-rounded --include-western
uv run --frozen --extra train python -m ogura.training.preview \
  --font corpus/fonts/NotoSansCJKjp-Regular.otf \
  --western-font corpus/fonts/Tinos-Regular.ttf \
  --western-font 'corpus/fonts/Arimo[wght].ttf' \
  --text '日本語と ISO 15216、Mill 123 “AV” αβγ АБВ ＡＢＣ１２３' \
  --variants 2 --output datasets/font_candidates/mixed-preview.png
```

ASCII空白・ASCII記号・欧文引用符などは、欧文文字を含む連続区間で
欧文書体に合わせる。日本語の句読点・全角記号は日本語書体のまま。
各区間は共通ベースラインで配置し、欧文にはフォントの文字幅と
OpenType GPOS kernの横方向ペア調整を使う。合字は作らない。
これは選んだ左書き欧文の混植用であり、アラビア文字などの複雑な
シェーピングには対応しない。結合ダイアクリティカルマークは今回の
欧文振り分けには含めない。

欧文フォントにない文字は日本語フォントへ戻し、そこにもなければ
空白に置換する。連続半角空白の正規化は画像と正解に同じように適用する。
細い欧字の連続などでCTCに必要な出力位置数が不足する場合のみ、
行画像を必要幅まで横に伸ばす（高さ48pxは維持）。

学習では日本語用 `--extra-font` と別に `--western-font` を繰り返し指定する。
例えば直前の7書体学習コマンドを元に、次を変更・追加する：

```text
--run-dir runs/noto48-residual64-mixed
--init-from runs/noto48-residual64-rounded-grid/best.pt
--western-font corpus/fonts/Tinos-Regular.ttf
--western-font 'corpus/fonts/Arimo[wght].ttf'
```

欧文フォントはサンプルごとに1つ選び、その行の欧文区間で共通に使う。
clean-probabilityで選ばれたclean画像は従来どおりNoto Sans Regular単独。
欧文フォント未指定なら従来の描画を維持する。

mean-font-cer検証では、3長さ×7日本語書体×2欧文書体＝42条件を固定描画し、
そのCERを等重みで平均する。各長さ1,000文なら揺らぎ付き42,000画像に
従来のbaseline 3,000画像が加わる。ラベルには両フォント名を表示する。
欧文を含まない行も同じ条件集合に含まれるため、これは欧文専用スコアではない。
検証時間は従来より増える。evaluate_lengthsも保存した欧文フォントを復元する。

描画条件・検証条件が変わるので、新しいrun-dirにinit-fromで移行する。
欧文フォントの内容・順序をチェックポイントで検証し、異なる設定での
resumeは拒否する。画像・フォント本体はGitに追加しない。
この変更は描画方法の追加であり、欧文コーパスの増量は行わない。

### 同一視する文字の設定

`--character-aliases config/character_aliases.json` で代表元を指定できます。
同梱の設定には英数字・記号・カナ・空白・確認済み漢字の159組を登録しています。
オプション未指定時は統合しません。設定の形式は次のとおりです。

```json
{"version": 1, "groups": [{"representative": "A", "members": ["A", "Α", "А"]}]}
```

各集合は2文字以上で、代表元も集合内に含めます。集合間の重複、複数字からなる要素、改行などの制御文字は拒否します。
空白はU+0020とU+3000のみ許可し、代表元をU+0020に限定します。
元の語彙にない要素も設定できますが、使われる集合の代表元は元の語彙に必要です。
未使用の要素は出力クラスを増やしません。設定外の文字は従来どおり独立したクラスです。
描画は元の文字で行い、描画不可能文字の空白置換後、正解だけを代表元に変換します。
CTCの繰り返し文字数、学習・検証のCERと完全一致率も変換後の正解を使用します。
認識結果は代表元になり、元の文字体系を復元することはできません。コーパスは変更しません。

設定内容はチェックポイントのidentityに保存され、`best.pt`には元の語彙も保存されます。
単独評価は保存された設定を使用するため、設定ファイルを移動しても評価できます。
再開時には同じ内容の設定を指定してください。集合・代表元を変更した再開や
通常の `--init-from` は拒否します。新しいrunで `--init-from` に
`--migrate-aliases` を併用すれば、分類層を移行して追加学習できます。

代表元はASCII英字 `A–Z` と `a–z`、数字 `0–9`（ISO-8859-1にも含まれる）です。
全角英字 `Ａ–Ｚ・ａ–ｚ` と全角数字 `０–９` を、それぞれ対応するASCII文字と同一視します。
うち22組には、対応するギリシャ文字・キリル文字も含みます。英数字部分は156文字・62組です。
現在の語彙には全角英字がありませんが、全角数字はあります。全設定を適用後の文字クラス数は15,918です。
将来全角英字をコーパスと元の語彙に追加すれば、全角字形で描画し、ASCII英字を正解とします。
画素の完全一致を全書体に要求するものではなく、書体による微差は許容する方針です。
小文字 `o / ο / о` も含めます。`a / α`、`p / ρ`、`Y / У`、`K / К` は
形状差の判断を保留し、今回は統合しません。数字と英字（`0 / O`、`1 / I / l`）、大小文字、漢字と仮名は統合しません。
採用集合の正確なコードポイントは `config/character_aliases.json` のUnicode文字で定義しています。

追加した集合は次のとおりです。

- 全角ASCII記号32組：`！→!`、`（→(`、`％→%`、`＼→\` など。
- 半角カナ・付随記号63組（U+FF61–U+FF9F）：`ｱ→ア`、`｡→。`、`｢→「`、`ﾞ→゛` など。
- 空白1組：U+3000 → U+0020。
- CJK互換漢字1組：`淚`（U+F94D）→ `淚`（U+6DDA）。`涙` は別クラス。

同梱設定はversion 2で、`collapse_ascii_spaces: true` と
`compose_katakana_diacritics: true` を指定しています。集合で変換した後、
合成可能なカタカナ＋濁点・半濁点だけを `カ゛→ガ`、`ハ゜→パ` のように合成し、
連続するU+0020を1個にまとめます。ただし描画前の処理で先頭・末尾の空白は除去します。
単独の濁点や合成できない組は残し、テキスト全体へのNFKCは行いません。
合成結果の文字は語彙に必要です。描画は正規化前の字形と空白幅で行います。
正解・評価時の予測・recognize出力に同じ正規化を適用します。
version 1の設定は従来どおり文字集合の変換だけを行います。

### 同一視設定を変更して追加学習する

これまでの学習コマンドから `--resume` を外し、別の `--run-dir` にして、次を指定します。
モデルの種類・channelsは移行元と同じ値にし、フォント・揺らぎ・validationの指定も引き継いでください。

```text
--run-dir runs/noto48-aliases
--init-from runs/noto48-residual64-rounded-grid/best.pt
--character-aliases config/character_aliases.json
--migrate-aliases
```

特徴抽出部・横方向CNNの重みはそのままコピーします。分類層はblankと単独クラスを
そのままコピーし、統合クラスのweight・biasを旧クラスの算術平均で初期化します。
これは旧確率の和を厳密に再現する変換ではなく、追加学習の初期値です。
元チェックポイントは変更せず、optimizer・scheduler・epochは新規に開始します。
validation指定時は学習前のepoch 0も評価し、新しい評価基準でbestを選び直します。

移行元はbest.ptまたはlatest.ptです。クラス一覧を持たない旧latest.ptの場合は、
元のtargets.jsonlのハッシュ一致を確認して旧クラス順を復元します。
モデル構造・channels・元の文字集合が異なる移行や、統合済みクラスを再分割する移行は拒否します。
移行元SHA256、旧新設定、統合した各クラス、初期化方式を
run-dir/class_migration.jsonとlatest.pt/best.ptのinitializationに保存します。

移行後の中断再開は、同じ同一視設定と学習条件で `--resume` を使用します。
`--init-from` と `--migrate-aliases` は外してください。再開時に再度重みを統合しません。

### 丸ゴシック3書体を追加する

```sh
uv run --frozen python -m ogura.download_training_fonts \
  --include-rounded --include-western --include-rounded-extra
uv run --frozen --extra train python scripts/preview_rounded_fonts.py
```

追加書体はM PLUS Rounded 1c Thin/Light（フォント内にOFL 1.1の宣言あり）と
Kosugi Maru Regular（Apache 2.0、LICENSE-KosugiMaru.txtを取得）。
取得元の版・SHA256はtraining_fonts.jsonに固定し、バイナリはコミットしない。
比較画像はdatasets/font_candidates/rounded-comparison.pngに保存する。
M PLUS Roundedの元語彙カバー数は5,572、Kosugi Maruは7,212（元語彙16,057）。
描画不能文字は従来どおり描画・正解とも空白に置換する。

以下は欧文混合学習後のbest.ptを引き継ぐ。字種設定は同じなので重み移行は不要。
学習前のepoch 0を含め、3長さ×10日本語書体×2欧文書体の60条件を評価する。
別途baselineの3条件も評価する。前段階の42条件とは平均CERを直接比較しない。

```sh
uv run --frozen --extra train python -m ogura.training.train \
  --device cuda --run-dir runs/noto48-residual64-rounded-extra \
  --init-from runs/noto48-residual64-aliases-western/best.pt \
  --character-aliases config/character_aliases.json \
  --model-type residual --channels 64 \
  --extra-font corpus/fonts/NotoSansCJKjp-Bold.otf \
  --extra-font corpus/fonts/NotoSerifCJKjp-Regular.otf \
  --extra-font corpus/fonts/NotoSerifCJKjp-Bold.otf \
  --extra-font corpus/fonts/NotoSansCJKjp-Light.otf \
  --extra-font corpus/fonts/ZenMaruGothic-Light.ttf \
  --extra-font corpus/fonts/ZenMaruGothic-Regular.ttf \
  --extra-font corpus/fonts/MPLUSRounded1c-Thin.ttf \
  --extra-font corpus/fonts/MPLUSRounded1c-Light.ttf \
  --extra-font corpus/fonts/KosugiMaru-Regular.ttf \
  --western-font corpus/fonts/Tinos-Regular.ttf \
  --western-font 'corpus/fonts/Arimo[wght].ttf' \
  --font-size-min 28 --font-size-max 40 \
  --padding-min 2 --padding-max 6 --vertical-full-range \
  --clean-probability 0.25 --learning-rate 0.0001 \
  --batch-size 32 --epochs 20 --workers 4 \
  --validation-text datasets/validation/validation.txt --validation-augmented \
  --monitor-validation datasets/validation_short5/validation.txt \
  --monitor-validation datasets/validation_long80/validation.txt \
  --selection-metric mean-font-cer --early-stopping-patience 5 \
  --save-every 500 --log-every 100 --log-samples 3
```

### 誤認識を画像付きで診断する

```sh
uv run --frozen --extra train python -m ogura.diagnose_validation \
  --checkpoint runs/noto48-residual64-rounded-extra/best.pt \
  --device cuda --batch-size 32 --top 20 \
  --output runs/noto48-residual64-rounded-extra/validation-errors
```

保存済みの書体・揺らぎ・字種統合設定を復元し、全書体の組み合わせとbaselineを
固定の検証画像で評価する。現在の10日本語書体×2欧文書体なら60条件＋baseline3条件。
元のvalidationと同じ語彙・データ・フォントのハッシュを検証する。
`--validation-text` は繰り返し指定可能。省略時はshort5、通常長、long80を評価する。
移動した実行環境では `--run-config` と `--font-dir` で場所を指定できる。
学習用レンダラーの画像を診断するため、contrast等の実画像用前処理は加えない。

- `index.html`：ブラウザで開く。条件ごとの混同上位30組と、CERの悪い順の画像・正解・予測。
- `images/`：各条件で誤った例の上位 `--top` 件。実入力の高さ48px、バッチ用右パディングを除く。
- `errors.jsonl`：上位だけでなく、全誤認識例の正解・予測・編集操作・位置・描画パラメータ。
- `conditions.json`：条件一覧、全サンプルのCER・完全一致率、文字出現数と混同集計、上位例。
- `confusions.json`：全条件を合わせた置換・削除・挿入の集計。字のコードポイントも記録。
- `manifest.json`：チェックポイントSHA256・epoch・描画条件などの記録。

CER降順、同率はサンプル順。正しく読めた例は画像を保存しないが、文字出現数の分母には含める。
置換・削除の率は、その条件での正解文字の出現数を分母にする。挿入にはこの率を付けない。
同じ文を複数条件で描画した場合、それぞれ別の観測として数える。
編集位置は同一視・空白置換後の正解上の0始まりで、挿入は文字間の位置。
Levenshteinの最小編集対応が複数ある場合は対角・削除・挿入を優先して固定するため、
同じ文字の連続などでは位置や誤りの種類が一意に確定した証拠ではない。
統合済み文字同士の混同はこのモデルの出力からは復元できない。
既存の出力ディレクトリは上書きしない。生成したレポート・画像はGitに追加しない。

### モデルを変更せず、評価時だけ文字を同一視する

モデル用の `config/character_aliases.json` と、判定用の
`config/evaluation_aliases.json` は分けて管理する。後者には現在 `~ / ～ / 〜 → ~`
を登録している。正解と予測の双方を判定直前に変換し、CER・完全一致率・混同集計に適用する。
モデルの出力クラス、画像、CTCの正解ラベル・loss、recognizeの認識出力は変更しない。
判定用の集合が蓄積した段階で、別途モデル用設定に採用して分類層を移行する。

既存best.ptで評価する例（移行・追加学習不要）：

```sh
uv run --frozen --extra train python -m ogura.diagnose_validation \
  --checkpoint runs/noto48-residual64-rounded-extra/best.pt \
  --evaluation-aliases config/evaluation_aliases.json \
  --device cuda --top 20 \
  --output runs/noto48-residual64-rounded-extra/validation-errors-scored
```

`ogura.evaluate_lengths` も同じ `--evaluation-aliases` オプションを受け付ける。
単独評価では明示指定を優先し、省略時はチェックポイントに保存された判定設定を使う。
判定設定がない旧チェックポイントでは従来の判定になる。空集合の設定を明示すると無効化できる。
診断レポートには、判定用正解・予測と変換前の正解・予測の両方を保存する。
条件ごとの `accepted_by_aliases` は、判定用変換によって行全体が正解になった件数。

学習コマンドにも `--evaluation-aliases` を指定可能。学習時accuracy/CERとvalidation、
best選択・早期終了に適用し、CTC lossは従来のラベルのままにする。
判定設定はチェックポイントに保存し、異なる判定基準でのresumeは拒否する。
既存モデルで新しい判定基準の学習を開始する場合は新しいrunで `--init-from` を使う。
`--migrate-aliases` は不要。判定基準変更前後の成績は直接比較しない。
