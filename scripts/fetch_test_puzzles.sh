#!/usr/bin/env bash
# Download recent puzzles with xword-dl as a held-out test set.
# Usage: scripts/fetch_test_puzzles.sh OUT_DIR START_DATE END_DATE [outlet ...]
#   outlets default to: usa tny   (lat, nd, uni, wsj, atl failed in Oct 2026)
#   NYT needs a subscription: xword-dl nyt -u USER -p PASS -d DATE -o nyt_DATE.puz
set -euo pipefail
out="$1"; start="$2"; end="$3"; shift 3
outlets=("$@"); [ ${#outlets[@]} -eq 0 ] && outlets=(usa tny)
mkdir -p "$out"
d="$start"
while [[ "$d" < "$end" || "$d" == "$end" ]]; do
  for src in "${outlets[@]}"; do
    if timeout 60 xword-dl "$src" -d "$d" -o "$out/${src}_${d}.puz" >/dev/null 2>&1; then echo "ok   $src $d"; else echo "fail $src $d"; fi
  done
  d=$(date -I -d "$d + 1 day")
done
