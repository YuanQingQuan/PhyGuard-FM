#!/usr/bin/env bash
set -u
URL="$1"
OUT="$2"
TOTAL="${3:-}"
MAX="${4:-60}"
mkdir -p "$(dirname "$OUT")"
[ -n "$TOTAL" ] && echo "target size: $TOTAL bytes" || echo "target size: unknown (use curl exit status)"
for i in $(seq 1 "$MAX"); do
  cur=$(stat -c%s "$OUT" 2>/dev/null || echo 0)
  echo "[$i] $cur / ${TOTAL:-?}"
  if [ -n "$TOTAL" ] && [ "$cur" -ge "$TOTAL" ]; then
    echo "DONE: $OUT"
    exit 0
  fi
  if curl -sL -C - --max-time 540 --retry 3 --retry-delay 2 -o "$OUT" "$URL"; then
    cur=$(stat -c%s "$OUT" 2>/dev/null || echo 0)
    if [ -z "$TOTAL" ] || [ "$cur" -ge "$TOTAL" ]; then
      echo "DONE: $OUT"
      exit 0
    fi
  else
    echo "  connection dropped; retrying"
  fi
done
echo "EXHAUSTED attempts for $OUT (got $(stat -c%s "$OUT" 2>/dev/null || echo 0) bytes)"
exit 1
