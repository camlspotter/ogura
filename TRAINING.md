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

`training_errors.jsonl` には、各エポックの100、200、300…バッチ目で、
そのバッチ内のCERが高い誤認識を最大3例保存する。100バッチ全体からの選抜ではない。
同率ならバッチ内の順序を維持し、正解例・ランダム例・画像は保存しない。
これはコンソールの `--log-every` / `--log-samples` とは独立して常に有効。
1エポック11,567バッチなら最大345例で、誤認識が少なければさらに減る。

各行にはepoch・batch・step、サンプルID、描画元テキスト、正解・予測、
評価用同一視後のテキスト、CER、画像幅、確定済みの描画パラメータを保存する。
予測はそのバッチの重み更新前のもの。共通のフォントSHA256・実行環境・コード情報は
`training_errors_metadata.json` に保存する。
画像の再生成には、同じフォント・実行環境、保存した語彙・同一視設定を使い、
`input_text` と `render_params` から `Sample` を作って `BatchRenderer` に渡す。
CTC用の幅調整も必要なので、単独の描画関数だけでは再生成しない。

再開時はチェックポイントの保存位置まで誤認識ログも巻き戻す。
`training_errors.jsonl` とメタデータもチェックポイントと一緒に保持する。
この機能がなかった直前版のチェックポイントからも再開でき、その時点から記録を始める。

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

### 欧文テキストの追加コーパス

```bash
uv run --frozen python -m ogura.build_english
```

固定リビジョンの `wikimedia/wikipedia` の `20231101.en` から最初の1分割
（約420MB、SHA256検証あり）を `corpus/wikipedia/20231101.en/` に取得する。
英語Wikipedia全体からの無作為抽出ではなく、この分割内からの初期追加データである。

出力は `datasets/english_wikipedia_30k/`：

- `train.txt`：学習用30,000行、1行1エントリ。
- `validation.txt`：検証用1,000行。学習とは別の記事から抽出。
- `train.jsonl` / `validation.jsonl`：記事ID・タイトル・URL・原文内の開始終了位置。
- `manifest.json`：固定ソース、乱数seed、入力・出力ハッシュ、文字数・連続文字頻度。

20〜25文字を単語境界で切り出す。元の大文字・小文字や句読点を保存し、
既存の字種集合にない文字を含む候補は採らない。自然な段落を優先するヒューリスティックで、
一覧記事・コード・文字表などを除外するが、意味的な自然さを完全には保証しない。
1記事から1例のみ採用し、同じ16文字の断片が重なる例を除外する。
既存の日本語学習データと3種類の検証テキストにも同じ重複チェックを適用する。

既存の学習ファイルには混ぜず、追加テキストとして独立して作成する。
検証テキストを学習用に混ぜないこと。画像生成・追加学習はこのコマンドでは行わない。
既存出力は上書きせず、再生成比較時は `--output` で別ディレクトリを指定する。
再現には同じソース・除外データ・字種集合・seedが必要。
Wikipediaの原文抜粋であり、記事URL等の出典を保持し、配布時には元資料のライセンスに従う。

欧文追加学習用の混合データと、対応する検証manifestは次で作成する。

```bash
uv run --frozen python -m ogura.prepare_english_training
bash scripts/train_english_supplement.sh
```

`datasets/japanese_english_30k/` に日本語339,839行＋欧文30,000行を
seed固定で混ぜた `train.txt`、語彙、4種類の検証セットを作る。
元データのハッシュ、語彙、検証との16文字断片重複を確認してから生成する。
既存データやmanifestは変更しない。`build_english` と同様、出力の上書きはしない。

実行スクリプトはGPU用。`runs/noto48-residual64-rounded-extra/best.pt` から
重みを引き継ぎ、`runs/noto48-residual64-english30k/` に新しい学習を開始する。
分類層は変更せず、optimizer・epochはリセットする。10種類の日本語書体と
Tinos/Arimo、既存のサイズ・余白の揺らぎを使う。clean例25%では従来通り
Noto Sansのみで描くため、追加欧文のすべてが欧文フォントになるわけではない。

今回は揺らぎ付き欧文1,000行のCERをbest選択・早期終了の基準にする。
最大20epoch、5epoch改善がなければ終了する。日本語の通常・5文字・80文字は
baseline/augmentedで毎epoch監視するが、best選択には含めない。
各augmented検証は固定seedのフォント割当であり、全フォント直積の評価ではない。
日本語の悪化が見られたら継続を見直す。以前のmean-font-cerと数値を直接比較しない。

中断からの再開：

```bash
bash scripts/train_english_supplement.sh --resume
```

`--resume` 時はスクリプトが `--init-from` を外し、新runのlatestから再開する。
コマンド末尾に `--epochs 30` などを渡して上書き可能。
このリポジトリには学習済み重み・生成テキスト・フォントは含めない。
GPUマシンで同じコーパス生成コマンドを実行するか、生成ディレクトリをコピーする。

### 実文書の新旧予測の比較

ローカルの `datasets/real_samples/references.jsonl` は12行を画像とPDF文字情報で
照合した正解。画像SHA256・出典・判定上の注記を保持する。画像とともにGPUマシンの
同じディレクトリへコピーする。権利上、画像・正解・予測・レポートはコミットしない。
折り返された端末表示ではなく、`ogura.recognize --output` の元JSONLを入力する。

```bash
uv run --frozen --extra train python -m ogura.evaluate_real \
  --references datasets/real_samples/references.jsonl \
  --preprocessing contrast \
  --output runs/real-comparison.json \
  runs/noto48-residual64-rounded-extra/real-predictions-contrast.jsonl \
  runs/noto48-residual64-english30k/real-predictions-contrast.jsonl
```

1ファイルでも評価可能。複数指定時は先頭ファイルを基準に行ごとの誤り数差を出す。
全行の誤り数／全正解文字数のCERと完全一致率、空白に関係する編集数、文字ごとの編集を
保存し、CERの高い行から表示する。ファイルごとに同じ12画像の予測が揃っている必要があり、
欠落・重複・未確認正解・画像ハッシュ不一致はエラー。
既存のcharacter_aliasesとevaluation_aliasesを両側に適用し、元の文字列も残す。
予測の端の空白は除去しない。O/0・引用符などの新しい同一視は追加しない。
行内空白には字間との判別の曖昧さがあるため、空白誤り数も併せて確認する。
これら12行は繰り返し改善に使った診断用データであり、未見資料での精度推定とは区別する。

### 曲がった引用符を含む次の字種集合

`refine_charset` は `‘` (U+2018)、`’` (U+2019)、`“` (U+201C)、
`”` (U+201D) を必須文字に含める。ASCIIの引用符とは別の字種とし、
同一視設定には追加しない。既存モデル用の語彙を上書きしない生成例：

```bash
uv run --frozen python -m ogura.refine_charset --output charset/selected_quotes
```

現行集合に4文字だけを追加し、元の字種数は16,061、現在の同一視設定適用後は
15,922クラス（CTC blankを除く）となる。既存datasetsのtargetsやモデルは変更しない。
この集合を学習に使うには、対応する用例の抽出・検証manifest更新に加え、
モデルの分類層への新クラス追加が必要。追加時は `--migrate-aliases --allow-new-classes` を指定する。

新モデル用の `config/character_aliases_quotes.json` は半角・全角の既存の同一視を維持し、
`~`・`～`・`〜` を代表文字 `~` に統合する。引用符4文字はそれぞれ別クラス。
この設定と `charset/selected_quotes/targets.jsonl` の組み合わせは15,921クラス＋blank。
既存の `config/character_aliases.json` と評価用同一視設定は旧モデルの比較用に残す。

新しい語彙・学習用例・検証manifestを用意した後、旧bestから移行する際の追加指定：

```text
--init-from runs/noto48-residual64-english30k/best.pt
--character-aliases config/character_aliases_quotes.json
--migrate-aliases --allow-new-classes
```

変更のないクラスとblank、特徴抽出・文脈層はそのままコピーし、統合クラスは旧分類層の行を
平均する。新しい曲がった引用符は対応するASCII引用符の重みをコピーし、バイアスを5下げる。
対応元がない新クラスはblankをコピーして同じくバイアスを5下げる。移行直後は対応元より
スコアが低いため、新クラスのランダム出力で既存認識を壊さない。確率分布は厳密には変わり、
波線クラスの平均による変化は別に残る。移行記録には初期化元・マージン・追加文字・
初期化クラス・統合クラスを保存する。optimizerとepochは新規開始。旧クラスの分割、
元字種の削除、構造変更は拒否する。古いlatestで元語彙が保存されていない場合は新語彙から
対応を推測せずエラーにするため、元語彙が保存されたbestを使う。
途中再開は新runで `--resume` とし、`--init-from`・`--migrate-aliases`・
`--allow-new-classes` は外す。字種や設定は新runと一致させる。
既存の欧文追加学習スクリプトは旧語彙向けなので、この新語彙にはそのまま使用しない。

引用符の用例抽出と描画確認：

```bash
uv run --frozen python -m ogura.build_quotes
uv run --frozen --extra train python scripts/preview_quotes.py
```

`build_quotes` は既存英語コーパスの固定分割から、各引用符を含む20〜25文字の
原文抜粋を採る。既存英語学習・検証で使用した記事は除外し、記事hashで分割する。
既存混合学習、日本語と英語の検証、および新しい用例同士で16文字断片の重複を除外する。
既定は各文字を含む学習100行・検証25行以上（同じ行に複数の引用符を含められる）。
不足した場合は合成せずエラー。`’` はアポストロフィとしての自然な用例も含む。
出力は `datasets/quote_supplement/` のtrain/validation TXT、出典付きJSONL、targets、
頻度・入力ハッシュ付きmanifest。既存学習ファイルにはまだ混ぜない。

`preview_quotes` は10種の日本語フォント単独と、Tinos/Arimoとの組み合わせを28/40pxで
学習と同じレンダラーを使って描画する。`datasets/font_candidates/quote-review/` に
3枚の比較画像とcoverage.jsonを保存する。引用符とASCII引用符は別ラベルのまま、
全半角英字・カナ・波線の描画とラベル対応も表示する。この比較文字列は描画確認専用で、
学習データには入れない。

引用符データを既存の日本語＋欧文データに混合する：

```bash
uv run --frozen python -m ogura.prepare_quote_training
```

`datasets/japanese_english_quotes/` に369,839＋290＝370,129行の `train.txt` と
新しい `targets.jsonl` を作る。引用符用74行は学習に混ぜず `quotes/validation.txt` に保存。
`english`・`validation`・`validation_short5`・`validation_long80` の4検証セットは本文を
変えずコピーし、新しい学習データ・語彙のハッシュに結び直す。元のデータは上書きしない。
入力ハッシュ・字種・重複・全5検証セットとの16文字断片の重複を検査し、seed固定で混合する。

GPUで引用符追加・波線統合の学習を開始する：

```bash
uv run --frozen python -m ogura.refine_charset --output charset/selected_quotes
uv run --frozen python -m ogura.build_quotes
uv run --frozen python -m ogura.prepare_quote_training
bash scripts/train_quote_supplement.sh
```

前の欧文追加学習で作成済みの `datasets/english_wikipedia_30k` と
`datasets/japanese_english_30k`、元Wikipediaコーパス・charset生成元が必要。
生成ディレクトリをコピー済みの場合は、その生成コマンドを再実行しない（上書き拒否）。
学習は `runs/noto48-residual64-english30k/best.pt` から移行し、
`runs/noto48-residual64-quotes-v2` に保存する。移行直後のepoch=0にも検証して記録する。
今回は引用符専用74行の揺らぎ付きCERをbest・早期終了の基準とし、
従来の欧文1,000行、日本語の通常・短文・長文を毎epoch監視する。
専用検証は小規模なので、全体成績の代用とはしない。最大20epoch、patience=5。

中断後の再開：

```bash
bash scripts/train_quote_supplement.sh --resume
```

再開時は移行フラグとinit-fromを外し、新runのlatestを読み込む。

旧ランダム初期化で学習した `runs/noto48-residual64-quotes` は再開しない。
修正版スクリプトを `--resume` なしで実行し、欧文追加学習のbestから移行し直す。
新クラスは最初は選ばれないため、epoch=0で引用符の誤りが出ること自体は想定内。
既存の欧文・日本語の監視成績と、大量の引用符出力がないことを確認する。


### 引用符モデルから追加の同形字を統合する

`config/character_aliases_quotes_homoglyphs.json` は既存の引用符モデル用設定を引き継ぎ、
`Ë/Ё → Ë`、`ë/ё → ë`、`Π/П → Π`、`Φ/Ф → Φ`、`Γ/Г → Γ` をモデルの正解ラベルと出力クラスで統合する。
今回の追加は欧文の文字体系間に限定し、漢字・仮名や `З/3` は追加しない。
小文字の `п/π`、`ф/φ`、`г/γ` は統合しない。元の字形で描画し、正解ラベルを代表元に変換する。
以前の設定ファイルは旧runの再現用に保持する。

```sh
bash scripts/train_quote_homoglyphs.sh
# 同じ新runの中断後に再開する場合
bash scripts/train_quote_homoglyphs.sh --resume
```

`runs/noto48-residual64-quotes-v2/best.pt` を起点に、
新run `runs/noto48-residual64-quotes-homoglyphs` を開始する。
分類層の対応する重み・バイアスを平均して5クラス減らし、特徴抽出部分とその他の
クラスの重みを引き継ぐ。optimizerとepochは新規開始。旧runの `--resume` ではない。
これは追加学習用の初期化であり、統合直後の認識性能を保存するものではない。

### 評価セットの拡充と6セット平均による選択

```sh
uv run --frozen --extra train python -m ogura.build_evaluation_suite
```

`datasets/evaluation_v2` に、自然文から引用符・同形字の追加検証セットと、
学習中には使わない最終確認用の `test/` を作る。既存の引用符74例は新validationにも保持。
日本語Wikipediaは従来のvalidation/test記事分割、英語は従来の非学習記事分割を使い、
既存英語学習記事・既存引用符学習記事も除外する。
元テキストおよび同一視後の16文字断片で、学習・既存評価・新セット間の重複を除外する
（保持する既存引用符は意図的な再利用）。出典・切り出し位置・入力ハッシュを保存する。
引用符は新規各100用例、同形字は各50用例を目標にするが、自然文が不足した文字は
合成せず `manifest.json` の `shortfalls` に記録する。既存出力は上書きしない。
既にモデル比較に使用した70例の出典は `--prior-probe` で指定でき、testへの混入を防ぐ。
同じ成果物をGPUへコピーする場合は `datasets/evaluation_v2/` 全体をコピーする。

追加学習前に、best/latestを同じ固定条件で評価できる。

```sh
uv run --frozen --extra train python -m ogura.evaluate_suite \
  --checkpoint runs/noto48-residual64-quotes-homoglyphs/latest.pt \
  --font-dir corpus/fonts --device cuda \
  --output runs/noto48-residual64-quotes-homoglyphs/evaluation-v2-latest.json
```

`--checkpoint` と `--output` を変えればbestも比較できる。latestの読み込みでは、
run_configに記載された語彙ファイルが必要で、checkpointのハッシュと照合する。
このコマンドは `test/` を読み込まない。testのmanifestは `split=test` とし、
通常の検証ローダーへの誤投入も拒否する。testは引用符・同形字の最終確認用であり、
日本語全般の性能を測る包括的なテストセットではない。

```sh
bash scripts/train_evaluation_v2.sh
# 中断後
bash scripts/train_evaluation_v2.sh --resume
```

最新の統合モデルの `latest.pt` から、新run `runs/noto48-residual64-evaluation-v2` を開始する。
`--selection-metric mean-set-cer` は、通常長・短行・長行・英語・引用符・同形字の
**各セットの揺らぎ付きCERの単純平均**でbestとearly stoppingを判断する。
各セットを1/6ずつ扱い、行長・サンプル数で重み付けしない。baselineは監視用で平均には含めない。
各テキストの書体・サイズ・位置はseedとsample_idで固定し、毎epoch同じ画像で比較する。
全書体の直積評価ではなく、各行に固定された1組の書体を使う。
従来どおり個別のaccuracy/CERも記録し、開始時（epoch 0）にも全セットを評価する。
選択対象の変更は同じrunのresumeでは行わず、新runのinit-fromで行う。

### 引用符の混同・周辺の空白挿入への半減点

validationログ・JSONと診断HTMLに `weighted_CER`（JSONでは `weighted_cer`）を併記する。
通常のCER・完全一致率は変更しない。同一視設定適用後の文字列を比較し、
正解にも存在する引用符 `“ ” ‘ ’ " '` の直前・直後への余分なASCII空白挿入は
1文字あたり0.5とする（予測の引用符が同じグループの別コードでも適用）。
引用符の置換も、二重引用符 `“ ” "` 内、一重引用符 `‘ ’ '` 内では0.5とする。
同一コードなら0、二重と一重の間の置換は1。それ以外の挿入・削除・置換も1として
最小編集距離を計算する。書体によって左右・直線型の区別が困難なための部分点であり、
引用符のコードやモデルの分類クラスを統合するものではない。
分母は通常のCERと同じ正解文字数。例えば `a“b` → `a “ b` は通常2誤り、
重み付きでは1誤り相当になる。元からある空白の削除は半減しない。

Kosugi Maruなどでは引用符の左右の字面の余白が大きく、空白文字と見分けにくい。
原文どおりを最良としつつ、この曖昧さを部分点として扱う。CTC損失、描画、正解ラベル、
訓練バッチのCERは変更しない。評価基準を変えるだけではモデルへの勾配は変わらない。
改善しない場合には、原文を優先しつつ引用符周辺の空白や同グループの引用符の違いを減点付きで許す代替正解を扱う損失へ
拡張する方針を `weighted_edit_distance` にもコメントしている。

最良モデル選択・early stoppingにも半減点を使う場合は、新しいrunで明示する：

```sh
bash scripts/train_evaluation_v2.sh \
  --init-from runs/noto48-residual64-evaluation-v2/best.pt \
  --run-dir runs/noto48-residual64-evaluation-v2-weighted \
  --selection-metric mean-set-weighted-cer
```

6セットの揺らぎ付きweighted CERを等重みで平均する。既存の `mean-set-cer` は通常CERのまま。
選択基準が変わるため既存runへ `--resume` して切り替えない。
単独評価の `ogura.evaluate_suite` も `--selection-metric mean-set-weighted-cer` に対応する。

連続する一重引用符2個と二重引用符1個も、両方向とも組全体で0.5減点とする。
例えば `''` ↔ `"`、`‘‘` ↔ `“`、`’’` ↔ `”`。上記の一重・二重グループ内の
字形の組み合わせをすべて対象にする。`' '` のように間に空白がある場合は対象外。
編集距離に2文字対1文字・1文字対2文字の遷移を加え、消費した文字を重複して数えず、
3個以上の連続では余った引用符にも減点を適用する。完全一致は引き続き0減点。

### 細い欧文字の連続をWikipediaから収集

```sh
uv run --frozen python -m ogura.collect_latin_sequences \
  --output datasets/latin_sequences_wikipedia_lowercase
```

既存の固定された英語Wikipediaシャードから、`ff ll tt ii fi fl ffi ffl il li ij ji ft ti it lt tl`
を含むASCII単語を採取する。対象の小文字列に大小文字を区別して一致させ、元の綴りを保持する。
例えば `Hawaii` はiiの対象、`II`・`III` は対象外。
`words.jsonl` は本文中の単語一覧・出現数・出典例。英語版の本文に現れる固有名詞・外国語も
含み、辞書で英単語と認定した一覧ではない。アクセント付き単語や数字の途中は切り出さない。
`train.txt` は単語を途中で切らない20–25文字の自然文抜粋（1行1例）、`train.jsonl` は
記事URL・元本文の位置・対象単語・該当する組を保持する。合成は行わない。

既存の英語記事の分割seedを維持して学習側のみを利用し、datasets内のvalidation.jsonl/test.jsonl
および既存probeに登録された英語記事を除外する。評価テキストと既存学習テキストについて、
元文字列・同一視後の16文字断片の重複も除外する。既存学習データとの混合・学習は別工程。

`--goal`（既定1000）は各組を含む抜粋の目標数。同じ例が複数の組を満たしても重複保存しない。
1記事から同じ組を狙って何度も採取せず、不足はmanifestに記録する。共起によって目標数を
超える組もある。単語一覧は採択した抜粋だけでなく対象シャードの適格な学習側本文全体を走査する。
manifestに入力・除外ファイル・実装・出力のハッシュを記録し、既存出力の上書きは拒否する。
再生成時には同じ除外ファイル集合を用意する（必要なら `--exclude-jsonl` を繰り返し指定）。

### 欧文の連続文字データを混合して追加学習

```sh
# 抽出済みデータをコピーした場合は収集コマンド不要
uv run --frozen python -m ogura.collect_latin_sequences \
  --output datasets/latin_sequences_wikipedia_lowercase
uv run --frozen python -m ogura.prepare_latin_training
bash scripts/train_latin_sequences.sh
# 中断後
bash scripts/train_latin_sequences.sh --resume
```

`datasets/japanese_english_quotes` の370,129行に抽出した11,953行を1回ずつ加え、
固定seedでシャッフルした382,082行を `datasets/japanese_english_latin_sequences/train.txt`
に保存する。既存のデータは保持し、追加例を重複させるオーバーサンプリングは行わない。
語彙・同一視設定は維持し、描画不能字の処理も従来どおり。入力ハッシュ・重複を確認して生成する。

6評価セットの本文はそのままコピーし、新しい学習テキストのハッシュにmanifestを付け替える。
reserved testの2セットも `test/` に保持し、学習・epochごとの検証には使用しない。
追加学習は `runs/noto48-residual64-evaluation-v2/best.pt` から重みを読み込み、
新run `runs/noto48-residual64-latin-sequences` でoptimizerとepochを開始する。
`--selection-metric mean-set-weighted-cer` により、引用符の混同・空白の半減点を含む
6セットの平均でbestとearly stoppingを決める。通常CERも併記し、CTC損失は変更しない。

GPUへは `datasets/japanese_english_latin_sequences/` 全体をコピーすれば学習可能。
生成済みデータ、フォント、チェックポイント、実文書画像はGitへ追加しない。

### 欧文字列17組の専用評価

```sh
uv run --frozen python -m ogura.build_latin_validation
uv run --frozen --extra train python -m ogura.evaluate_latin_sequences \
  --checkpoint runs/noto48-residual64-latin-sequences/best.pt \
  --checkpoint runs/noto48-residual64-latin-sequences/latest.pt \
  --device cuda --font-dir corpus/fonts \
  --output runs/noto48-residual64-latin-sequences/sequence-evaluation
```

`datasets/validation_latin_sequences` に20–25文字の自然文を収集する。
各組200行が目安で、不足はmanifestに記録して合成しない。英語Wikipediaの既存分割を維持し、
学習側と予約済みテスト側を除外する。既存の学習・検証記事と元文字列・同一視後の16文字断片も
除外する。単語自体が学習と同じ場合はあるが、記事・文脈を別にする。検証本文を追加学習へ混ぜない。

評価時は同一runのbest/latestを指定し、画像はbaseline・固定揺らぎの各条件で一度描画したものを
両モデルに渡す。各組の全出現数、欠落を含む出現数、欠落文字数、置換数を計測する。
欠落率は「1文字以上欠落した組の出現数 / その組の全出現数」。完全に認識できた行も分母に含む。
重複する組（ff/ffi/fiなど）は独立に集計し、足し合わせない。編集位置が曖昧なときは通常の
最小編集距離の対角・削除・挿入の優先順に従うため、画像上の物理的な欠落位置の断定ではない。
全体の通常CER・weighted CER・完全一致率も残す。

`comparison.md` が横並びの比較、`report.json` が集計、`predictions.jsonl` が全例の
正解・予測・編集位置・再描画パラメータ。画像は保存しない。初期bestから改善していないrunでは
bestがepoch 0、latestが最後のepochとなる。生成済み検証データをGPUにコピーすれば収集処理は不要。
出力先が存在する場合は上書きせずエラーとする。通常のepoch検証やearly stoppingの設定は変更しない。

### 専用評価を毎epochとbest選択に含める

```sh
bash scripts/train_latin_validation.sh
# 新runの中断後
bash scripts/train_latin_validation.sh --resume
```

`datasets/validation_latin_sequences/` もGPUにコピーしてから実行する。
従来の6セットに専用評価を加えた7セットの揺らぎ付きweighted CERを等重みで平均する。
baselineは監視用。専用セットのCER・完全一致率は毎epochに表示し、組別欠落率の詳細比較は
`ogura.evaluate_latin_sequences` で行う。CTC損失や学習データは変更しない。

文字落ちが改善した `runs/noto48-residual64-latin-sequences/latest.pt` から重みを読み込み、
`runs/noto48-residual64-latin-validation` を新規作成する。optimizer・epochは新しく開始し、
epoch 0にも7セットで評価する。旧runへのresumeでは評価対象を変えない。
旧 `scripts/train_latin_sequences.sh` は6セットのまま保持し、旧runの再開に使える。

専用評価を加えても引用符・同形字の悪化を免除するわけではない。今回の保存値を使った参考計算
（従来6セットはGPU、専用セットはCPU）では、7セット平均は旧epoch 0が約0.1998%、
旧epoch 5が約0.2236%で、旧epoch 0が上位のまま。今回の新runの起点にlatestを選ぶのは、
欧文字列の改善を保持した状態から他の成績を回復できるか試すためであり、総合bestへの昇格ではない。
