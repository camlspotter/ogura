#!/usr/bin/env bash
# Runs locally on the user's GPU machine; never connects to another host.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
root=outputs
output=$root/postprocess-validation-v1
case "${1:-}" in
  run)
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" uv run --locked python \
      -m ogura.textdet.diagnose_postprocess --amp --all-pages \
      --expected-pages 39 --thresholds .3 .4 .5 --ratios 0 .5 1 1.5 --output "$output"
    ;;
  test)
    # Freeze the model and settings after reviewing validation; no test-set search.
    if [[ $# != 4 ]]; then echo "Usage: bash $0 test synth7000|synth9000 BIN_THRESHOLD UNCLIP_RATIO" >&2; exit 2; fi
    uv run --locked python -c 'import json,sys; from pathlib import Path; m=json.loads(Path(sys.argv[1]).read_text()); assert m["status"]=="complete" and m["all_pages"]; assert any(r["model"]==sys.argv[2] and r["bin_thresh"]==float(sys.argv[3]) and r["unclip_ratio"]==float(sys.argv[4]) for r in m["aggregate"]), "Select a setting evaluated on validation first"' "$output/summary.json" "$2" "$3" "$4"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" uv run --locked python \
      -m ogura.textdet.diagnose_postprocess --amp --all-pages \
      --expected-pages 40 --data "$root/experiment-v1-regenerated/test" --model "$2" \
      --thresholds "$3" --ratios "$4" --output "$root/postprocess-test-v1"
    ;;
  package)
    uv run --locked python -c 'import json,sys; from pathlib import Path; assert json.loads((Path(sys.argv[1])/"summary.json").read_text())["status"]=="complete"' "$output"
    if [[ -e "$output.tar.gz" ]]; then echo "Already exists: $output.tar.gz" >&2; exit 1; fi
    # Keep full precision maps on the GPU machine; copy PNGs and all metrics/predictions.
    tar --exclude='*.npy' -czf "$output.tar.gz.tmp" -C "$root" postprocess-validation-v1
    mv "$output.tar.gz.tmp" "$output.tar.gz"
    ;;
  download-command)
    echo 'scp dgx:~/ogura/textdet/outputs/postprocess-validation-v1.tar.gz ~/ogura/textdet/outputs/'
    echo 'tar -xzf ~/ogura/textdet/outputs/postprocess-validation-v1.tar.gz -C ~/ogura/textdet/outputs/'
    ;;
  *) echo "Usage: bash $0 run|package|download-command|test MODEL BIN_THRESHOLD UNCLIP_RATIO" >&2; exit 2 ;;
esac
