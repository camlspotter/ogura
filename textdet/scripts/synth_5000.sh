#!/usr/bin/env bash
# Run from any working directory. Generated data stays in ignored directories.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$PWD/.cache/playwright}"
fonts=(
  ../corpus/fonts/NotoSansCJKjp-Regular.otf
  ../corpus/fonts/NotoSansCJKjp-Bold.otf
  ../corpus/fonts/NotoSerifCJKjp-Regular.otf
  ../corpus/fonts/NotoSerifCJKjp-Bold.otf
)
texts=outputs/synth-texts-5000-v1.jsonl
output=outputs/synth-jdoc-5000-v1
case "${1:-}" in
  prepare)
    font_args=()
    for font in "${fonts[@]}"; do font_args+=(--font "$font"); done
    uv run --locked python -m ogura.textdet.prepare_synth_texts \
      --source ../corpus/wikipedia/20231101.ja \
      --output "$texts" --count 5000 --rows-per-shard 2048 \
      --lengths 2000 4000 8000 --seed 20260924 "${font_args[@]}"
    ;;
  generate|resume)
    resume_args=()
    if [[ "$1" == resume ]]; then resume_args+=(--resume); fi
    uv run --locked python -m ogura.textdet.synth_jdoc \
      "${resume_args[@]}" --input "$texts" --output "$output" --count 5000 --seed 20260924 \
      --vary-layout --fill-page --partial-fraction 0.2 --vertical-fraction 0.25 \
      --width 1200 --height 1600 --font-sizes 12 16 20 24 \
      --line-heights 1.5 1.7 2.0 --letter-spacings 0 0.03 0.08 \
      --font "${fonts[0]}" --extra-font "${fonts[1]}" \
      --extra-font "${fonts[2]}" --extra-font "${fonts[3]}"
    ;;
  train)
    model_output=outputs/db-resnet34-synth5000-v1
    mkdir -p "$model_output"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" uv run --locked python \
      -W 'ignore:`torch.cuda.amp.autocast(args...)` is deprecated:FutureWarning' \
      -W 'ignore:`torch.cuda.amp.GradScaler(args...)` is deprecated:FutureWarning' \
      .cache/doctr-v1.0.0/references/detection/train.py \
      db_resnet34 --pretrained --device 0 \
      --train_path "$output" \
      --val_path outputs/experiment-v1-regenerated/val \
      --output_dir "$model_output" --name synth5000-db-resnet34-v1 \
      --epochs 5 --batch_size 2 --input_size 1024 --lr 0.0001 \
      --workers 2 --amp --save-interval-epoch
    ;;
  *) echo "Usage: bash scripts/synth_5000.sh prepare|generate|resume|train" >&2; exit 2 ;;
esac
