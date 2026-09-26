#!/usr/bin/env bash
# Run on the GPU machine from the repository root. Never connects remotely.
set -euo pipefail
root=ogura/textdet/outputs
pages=$root/synth-image-pages-2000-v1
mixed=$root/synth-mixed-11000-v1
model=$root/db-resnet34-synth11000-v1
checkpoint=$model/synth11000-db-resnet34-v1.pt
evaluation=$root/validation-synth11000-standard-v2
case "${1:-}" in
  generate|resume)
    bash ogura/textdet/scripts/synth_image_pages.sh "$1" 2000 "$pages"
    ;;
  combine)
    uv run --locked python -m ogura.textdet.combine_synthetic \
      --text "$root/synth-jdoc-5000-v1" --tables "$root/synth-tables-2000-v1" \
      --headings "$root/synth-headings-1000-v1" --heading-tables "$root/synth-heading-tables-1000-v1" \
      --image-text "$pages/text" --image-tables "$pages/tables" \
      --image-text-count 1334 --image-table-count 666 --output "$mixed"
    ;;
  train)
    uv run --locked python -c 'import json,sys; from pathlib import Path; m=json.loads((Path(sys.argv[1])/"manifest.json").read_text()); assert m["status"]=="complete" and m["counts"]=={"text":5000,"table":2000,"heading":1000,"heading-table":1000,"image":1334,"image-table":666,"total":11000}, "Complete 11000-page dataset required"' "$mixed"
    if compgen -G "$model/*.pt" > /dev/null; then echo "Existing checkpoints: $model; refusing overwrite" >&2; exit 1; fi
    uv run --locked python -m ogura.textdet.evaluation_scope
    mkdir -p "$model"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" uv run --locked python \
      -W 'ignore:`torch.cuda.amp.autocast(args...)` is deprecated:FutureWarning' \
      -W 'ignore:`torch.cuda.amp.GradScaler(args...)` is deprecated:FutureWarning' \
      ogura/textdet/.cache/doctr-v1.0.0/references/detection/train.py \
      db_resnet34 --pretrained --device 0 --train_path "$mixed" \
      --val_path "$root/experiment-standard-v2/val" \
      --output_dir "$model" --name synth11000-db-resnet34-v1 \
      --epochs 5 --batch_size 2 --input_size 1024 --lr 0.0001 --workers 2 --amp --save-interval-epoch
    ;;
  evaluate)
    # Compare both checkpoints on validation with the previously chosen settings.
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" uv run --locked python \
      -m ogura.textdet.diagnose_postprocess --amp --all-pages --expected-pages 35 \
      --exclude-documents ogura/textdet/evaluation_exclusions.json \
      --previous "$root/db-resnet34-synth9000-v1/synth9000-db-resnet34-v1.pt" --previous-name synth9000 \
      --current "$checkpoint" --current-name synth11000 --model both \
      --input-size 1536 --thresholds .5 --ratios 1 --box-threshold .1 --output "$evaluation"
    ;;
  package)
    uv run --locked python -c 'import json,sys; from pathlib import Path; m=json.loads((Path(sys.argv[1])/"summary.json").read_text()); assert m["status"]=="complete" and m["pages"]==35 and set(m["checkpoints"])=={"synth9000","synth11000"}' "$evaluation"
    if [[ -e "$evaluation.tar.gz" ]]; then echo "Already exists: $evaluation.tar.gz" >&2; exit 1; fi
    tar --exclude='*.npy' -czf "$evaluation.tar.gz.tmp" -C "$root" validation-synth11000-standard-v2
    mv "$evaluation.tar.gz.tmp" "$evaluation.tar.gz"
    ;;
  download-command)
    echo 'Run on your Mac:'
    echo 'scp dgx:~/ogura/ogura/textdet/outputs/validation-synth11000-standard-v2.tar.gz ~/ogura/ogura/textdet/outputs/'
    echo 'tar -xzf ~/ogura/ogura/textdet/outputs/validation-synth11000-standard-v2.tar.gz -C ~/ogura/ogura/textdet/outputs/'
    ;;
  all)
    for action in generate combine train evaluate package; do bash "$0" "$action"; done
    bash "$0" download-command
    ;;
  *) echo "Usage: bash $0 generate|resume|combine|train|evaluate|package|download-command|all" >&2; exit 2 ;;
esac
