#!/usr/bin/env bash
# Run from the repository root. Generated data stays in ignored directories.
set -euo pipefail
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$PWD/ogura/textdet/.cache/playwright}"
fonts=(
  corpus/fonts/NotoSansCJKjp-Regular.otf
  corpus/fonts/NotoSansCJKjp-Bold.otf
  corpus/fonts/NotoSerifCJKjp-Regular.otf
  corpus/fonts/NotoSerifCJKjp-Bold.otf
)
texts=ogura/textdet/outputs/synth-texts-5000-v1.jsonl
output=ogura/textdet/outputs/synth-jdoc-5000-v1
case "${1:-}" in
  prepare)
    font_args=()
    for font in "${fonts[@]}"; do font_args+=(--font "$font"); done
    uv run --locked python -m ogura.textdet.prepare_synth_texts \
      --source corpus/wikipedia/20231101.ja \
      --output "$texts" --count 5000 --rows-per-shard 2048 \
      --lengths 2000 4000 8000 --seed 20260924 "${font_args[@]}"
    ;;
  generate)
    uv run --locked python -m ogura.textdet.synth_jdoc \
      --input "$texts" --output "$output" --count 5000 --seed 20260924 \
      --vary-layout --fill-page --partial-fraction 0.2 --vertical-fraction 0.25 \
      --width 1200 --height 1600 --font-sizes 12 16 20 24 \
      --line-heights 1.5 1.7 2.0 --letter-spacings 0 0.03 0.08 \
      --font "${fonts[0]}" --extra-font "${fonts[1]}" \
      --extra-font "${fonts[2]}" --extra-font "${fonts[3]}"
    ;;
  *) echo "Usage: bash ogura/textdet/scripts/synth_5000.sh prepare|generate" >&2; exit 2 ;;
esac
