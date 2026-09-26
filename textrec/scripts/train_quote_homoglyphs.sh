#!/usr/bin/env bash
# Run from any directory. Additional arguments may override defaults below.
set -euo pipefail
cd "$(dirname "$0")/.."
resume=false
for arg in "$@"; do
  if [[ "$arg" == "--resume" ]]; then resume=true; fi
done
if [[ "$resume" == false ]]; then
  set -- --init-from runs/noto48-residual64-quotes-v2/best.pt --migrate-aliases "$@"
fi
exec uv run --frozen --extra train python -m ogura.textrec.training.train \
  --device cuda \
  --run-dir runs/noto48-residual64-quotes-homoglyphs \
  --text datasets/japanese_english_quotes/train.txt \
  --vocabulary datasets/japanese_english_quotes/targets.jsonl \
  --character-aliases config/character_aliases_quotes_homoglyphs.json \
  --evaluation-aliases config/evaluation_aliases.json \
  --model-type residual --channels 64 \
  --extra-font ../corpus/fonts/NotoSansCJKjp-Bold.otf \
  --extra-font ../corpus/fonts/NotoSerifCJKjp-Regular.otf \
  --extra-font ../corpus/fonts/NotoSerifCJKjp-Bold.otf \
  --extra-font ../corpus/fonts/NotoSansCJKjp-Light.otf \
  --extra-font ../corpus/fonts/ZenMaruGothic-Light.ttf \
  --extra-font ../corpus/fonts/ZenMaruGothic-Regular.ttf \
  --extra-font ../corpus/fonts/MPLUSRounded1c-Thin.ttf \
  --extra-font ../corpus/fonts/MPLUSRounded1c-Light.ttf \
  --extra-font ../corpus/fonts/KosugiMaru-Regular.ttf \
  --western-font ../corpus/fonts/Tinos-Regular.ttf \
  --western-font '../corpus/fonts/Arimo[wght].ttf' \
  --font-size-min 28 --font-size-max 40 \
  --padding-min 2 --padding-max 6 --vertical-full-range \
  --clean-probability 0.25 --learning-rate 0.0001 \
  --batch-size 32 --epochs 20 --workers 4 \
  --validation-text datasets/japanese_english_quotes/quotes/validation.txt \
  --validation-augmented --selection-metric validation-cer \
  --monitor-validation datasets/japanese_english_quotes/english/validation.txt \
  --monitor-validation datasets/japanese_english_quotes/validation/validation.txt \
  --monitor-validation datasets/japanese_english_quotes/validation_short5/validation.txt \
  --monitor-validation datasets/japanese_english_quotes/validation_long80/validation.txt \
  --early-stopping-patience 5 \
  --save-every 500 --log-every 100 --log-samples 3 "$@"
