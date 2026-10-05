# セル内の下線・浅い斜め線

生成オプション `--content-lines` で、表のセル内にだけ追加する。
文字のみ/白紙の例には追加しない。周囲の本物の罫線は維持する。

- 下線: 本文の各text runを40%の確率で選び、語句（30%/60%）または行全体に付ける。
  字形の下端から1〜3px下、太さ0.5/0.75/1/1.5px。
- 空欄セルの対角線: 幅/高さが3以上の本文セルを35%の確率で選び、
  文字・既存記号を除いて左上隅から右下隅まで線を引く。
  角度はセル形状から決まり、任意の角度や開始位置にはしない。
  横長セルでは数度の水平に近い線になる。太さ0.5/0.75/1/1.5px。
  空欄セルの対角線と文章の下線は同じセルに同時配置しない。

JSONに全座標・線幅・roleを保存する。既存のdiagonal描画を使い、教師の縦横罫線は変更しない。
`underline`/`shallow_diagonal`領域別に平均確率と0.5閾値の誤反応率を測る。
領域は線の描画マスクを元に1px周囲を含め、教師罫線の周囲4pxを除く。
縮小時は完全に覆われる評価画素を選ぶため、細い線の対象画素数も確認する。
文字ペナルティを有効にした場合、この2領域もゼロ教師の追加ペナルティ対象に含める。

GPU試行案（まだ未実行）:
新規2000/200件、seed 20261013、既存hard-v1と1:1混合。
`table-cnn-textguard-1024-pilot-v1`からlr1e-5、文字ペナルティ係数1、250バッチ。
学習・評価・実画像previewの長辺上限を1024に揃える。
新しい検証JSONで新旧モデルの下線/浅い斜め線/文字の誤反応、実線/破線の再現率を比較し、
既存dashed/dense検証でも後退を確認する。GPU作業一式はユーザーの事前許可後に行う。

JSON生成例:

```sh
uv run --frozen --inexact python -m ogura.tablerec.synth_table_cells \
  --output tablerec/outputs/table-cells-content-lines-v1/train \
  --count 2000 --split train --seed 20261013 \
  --dense-text --background-context --negative-heavy --dashed-borders --content-lines
```
