#!/usr/bin/env bash
# Run from repository root; existing text-only pages and models are preserved.
set -euo pipefail
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$PWD/ogura/textdet/.cache/playwright}"
tables=ogura/textdet/outputs/synth-tables-2000-v1
mixed=ogura/textdet/outputs/synth-mixed-7000-v1
case "${1:-}" in
  generate|resume)
    resume_args=()
    if [[ "$1" == resume ]]; then resume_args+=(--resume); fi
    uv run --locked python -m ogura.textdet.synth_jdoc "${resume_args[@]}" \
      --input ogura/textdet/outputs/synth-texts-5000-v1.jsonl \
      --output "$tables" --count 2000 --seed 20260925 \
      --vary-layout --fill-page --partial-fraction 0.2 \
      --tables --table-position both --vertical-fraction 0 \
      --width 1200 --height 1600 --font-sizes 12 16 20 24 \
      --line-heights 1.5 1.7 2.0 --letter-spacings 0 0.03 0.08 \
      --font corpus/fonts/NotoSansCJKjp-Regular.otf \
      --extra-font corpus/fonts/NotoSansCJKjp-Bold.otf \
      --extra-font corpus/fonts/NotoSerifCJKjp-Regular.otf \
      --extra-font corpus/fonts/NotoSerifCJKjp-Bold.otf
    ;;
  combine)
    uv run --locked python -m ogura.textdet.combine_synthetic \
      --text ogura/textdet/outputs/synth-jdoc-5000-v1 \
      --tables "$tables" --output "$mixed"
    ;;
  train)
    uv run --locked python -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); m=json.loads((p/"manifest.json").read_text()); assert m["status"]=="complete" and m["counts"]=={"text":5000,"table":2000,"total":7000}, "Complete combined dataset required"' "$mixed"
    model_output=ogura/textdet/outputs/db-resnet34-synth7000-v1
    mkdir -p "$model_output"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" uv run --locked python \
      -W 'ignore:`torch.cuda.amp.autocast(args...)` is deprecated:FutureWarning' \
      -W 'ignore:`torch.cuda.amp.GradScaler(args...)` is deprecated:FutureWarning' \
      ogura/textdet/.cache/doctr-v1.0.0/references/detection/train.py \
      db_resnet34 --pretrained --device 0 \
      --train_path "$mixed" \
      --val_path ogura/textdet/outputs/experiment-v1-regenerated/val \
      --output_dir "$model_output" --name synth7000-db-resnet34-v1 \
      --epochs 5 --batch_size 2 --input_size 1024 --lr 0.0001 \
      --workers 2 --amp --save-interval-epoch
    ;;
  *) echo 'Usage: bash ogura/textdet/scripts/synth_tables_2000.sh generate|resume|combine|train' >&2; exit 2 ;;
esac
