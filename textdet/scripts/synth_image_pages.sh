#!/usr/bin/env bash
# Run from any working directory. Reuse reviewed assets; no remote access.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$PWD/.cache/playwright}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.cache/uv}"
action=${1:-generate}
count=${2:-12}
output=${3:-outputs/synth-image-pages-pilot-v2}
if [[ "$action" != generate && "$action" != resume ]]; then
  echo "Usage: bash $0 generate|resume [total-pages>=3] [output-root]" >&2; exit 2
fi
if [[ ! "$count" =~ ^[0-9]+$ ]] || ((count < 3)); then
  echo 'total-pages must be at least 3' >&2; exit 2
fi
for kind in text tables; do
  extra=()
  if [[ "$kind" == text ]]; then
    pages=$((count - count / 3)); seed=20260930; vertical=.25
  else
    pages=$((count / 3)); seed=20261001; vertical=0
    extra+=(--tables --table-position both)
  fi
  if [[ "$action" == resume && -f "$output/$kind/manifest.json" ]]; then extra+=(--resume); fi
  uv run --locked python -m ogura.textdet.synth_jdoc ${extra[@]+"${extra[@]}"} \
    --input outputs/synth-texts-5000-v1.jsonl \
    --image-assets outputs/image-assets-200-v1/images --image-position both --image-float-fraction .5 \
    --output "$output/$kind" --count "$pages" --seed "$seed" \
    --vary-layout --fill-page --partial-fraction .2 --vertical-fraction "$vertical" \
    --section-headings --title-style mixed --colored-text \
    --width 1200 --height 1600 --font-sizes 12 16 20 24 \
    --line-heights 1.5 1.7 2.0 --letter-spacings 0 .03 .08 \
    --font ../corpus/fonts/NotoSansCJKjp-Regular.otf \
    --extra-font ../corpus/fonts/NotoSansCJKjp-Bold.otf \
    --extra-font ../corpus/fonts/NotoSerifCJKjp-Regular.otf \
    --extra-font ../corpus/fonts/NotoSerifCJKjp-Bold.otf
done
