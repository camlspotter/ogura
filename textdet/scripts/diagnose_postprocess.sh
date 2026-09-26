#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
output=outputs/postprocess-diagnosis-v1
case "${1:-run}" in
  run)
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" uv run --locked python \
      -m ogura.textdet.diagnose_postprocess --amp --output "$output"
    ;;
  package)
    uv run --locked python -c 'import json,sys; from pathlib import Path; assert json.loads((Path(sys.argv[1])/"summary.json").read_text())["status"]=="complete"' "$output"
    if [[ -e "$output.tar.gz" ]]; then echo "Already exists: $output.tar.gz" >&2; exit 1; fi
    tar -czf "$output.tar.gz.tmp" -C outputs postprocess-diagnosis-v1
    mv "$output.tar.gz.tmp" "$output.tar.gz"
    ;;
  download-command)
    echo 'scp dgx:~/ogura/textdet/outputs/postprocess-diagnosis-v1.tar.gz ~/ogura/textdet/outputs/'
    echo 'tar -xzf ~/ogura/textdet/outputs/postprocess-diagnosis-v1.tar.gz -C ~/ogura/textdet/outputs/'
    ;;
  *) echo "Usage: bash $0 run|package|download-command" >&2; exit 2 ;;
esac
