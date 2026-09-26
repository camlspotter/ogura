#!/usr/bin/env bash
# Execute on the GPU machine from any working directory. Never connects remotely.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$PWD/.cache/playwright}"
root=outputs
headings=$root/synth-headings-1000-v1
tables=$root/synth-heading-tables-1000-v1
mixed=$root/synth-mixed-9000-v1
model=$root/db-resnet34-synth9000-v1
checkpoint=$model/synth9000-db-resnet34-v1.pt
bundle=$root/validation-synth9000-v1.tar.gz
case "${1:-}" in
  generate|resume)
    for kind in text table; do
      extra=()
      if [[ "$kind" == text ]]; then
        output=$headings; seed=20260926; vertical=.25
      else
        output=$tables; seed=20260927; vertical=0
        extra+=(--tables --table-position both)
      fi
      if [[ "$1" == resume && -f "$output/manifest.json" ]]; then extra+=(--resume); fi
      uv run --locked python -m ogura.textdet.synth_jdoc "${extra[@]}" \
        --input "$root/synth-texts-5000-v1.jsonl" --output "$output" \
        --count 1000 --seed "$seed" --vary-layout --fill-page --partial-fraction .2 \
        --vertical-fraction "$vertical" --section-headings --title-style mixed --colored-text \
        --width 1200 --height 1600 --font-sizes 12 16 20 24 \
        --line-heights 1.5 1.7 2.0 --letter-spacings 0 .03 .08 \
        --font ../corpus/fonts/NotoSansCJKjp-Regular.otf \
        --extra-font ../corpus/fonts/NotoSansCJKjp-Bold.otf \
        --extra-font ../corpus/fonts/NotoSerifCJKjp-Regular.otf \
        --extra-font ../corpus/fonts/NotoSerifCJKjp-Bold.otf
    done
    ;;
  combine)
    uv run --locked python -m ogura.textdet.combine_synthetic \
      --text "$root/synth-jdoc-5000-v1" --tables "$root/synth-tables-2000-v1" \
      --headings "$headings" --heading-tables "$tables" --output "$mixed"
    ;;
  train)
    uv run --locked python -c 'import json,sys; from pathlib import Path; m=json.loads((Path(sys.argv[1])/"manifest.json").read_text()); assert m["status"]=="complete" and m["counts"]=={"text":5000,"table":2000,"heading":1000,"heading-table":1000,"total":9000}, "Complete combined dataset required"' "$mixed"
    if compgen -G "$model/*.pt" > /dev/null; then echo "Existing checkpoints: $model; refusing overwrite" >&2; exit 1; fi
    mkdir -p "$model"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" uv run --locked python \
      -W 'ignore:`torch.cuda.amp.autocast(args...)` is deprecated:FutureWarning' \
      -W 'ignore:`torch.cuda.amp.GradScaler(args...)` is deprecated:FutureWarning' \
      .cache/doctr-v1.0.0/references/detection/train.py \
      db_resnet34 --pretrained --device 0 --train_path "$mixed" \
      --val_path "$root/experiment-v1-regenerated/val" \
      --output_dir "$model" --name synth9000-db-resnet34-v1 \
      --epochs 5 --batch_size 2 --input_size 1024 --lr 0.0001 --workers 2 --amp --save-interval-epoch
    ;;
  evaluate)
    sizes=(1024 1536)
    if [[ -n "${2:-}" ]]; then
      case "$2" in 1024|1536) sizes=("$2");; *) echo "Expected 1024 or 1536" >&2; exit 2;; esac
    fi
    for size in "${sizes[@]}"; do
      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" uv run --locked python \
        -m ogura.textdet.compare_predictions --checkpoint "$checkpoint" \
        --data "$root/experiment-v1-regenerated/val" --device cuda:0 --amp \
        --input-size "$size" --output "$root/validation-synth9000-${size}-v1"
    done
    ;;
  package)
    for size in 1024 1536; do
      uv run --locked python -c 'import json,sys; from pathlib import Path; assert json.loads(Path(sys.argv[1]).read_text())["status"]=="complete", "Evaluation incomplete"' "$root/validation-synth9000-${size}-v1/summary.json"
      test -f "$root/validation-synth9000-${size}-v1/pages.csv"
    done
    if [[ -e "$bundle" ]]; then echo "Already exists: $bundle" >&2; exit 1; fi
    tar -czf "$bundle.tmp" -C "$root" validation-synth9000-1024-v1 validation-synth9000-1536-v1
    mv "$bundle.tmp" "$bundle"
    echo "Created $bundle (comparison images, predictions, metrics; no training images or model weights)"
    ;;
  download-command)
    echo 'Run on your Mac:'
    echo 'scp dgx:~/ogura/textdet/outputs/validation-synth9000-v1.tar.gz ~/ogura/textdet/outputs/'
    echo 'tar -xzf ~/ogura/textdet/outputs/validation-synth9000-v1.tar.gz -C ~/ogura/textdet/outputs/'
    ;;
  all)
    for action in generate combine train evaluate package; do bash "scripts/synth_headings_2000.sh" "$action"; done
    bash "scripts/synth_headings_2000.sh" download-command
    ;;
  *) echo "Usage: bash $0 generate|resume|combine|train|evaluate|package|download-command|all" >&2; exit 2 ;;
esac
