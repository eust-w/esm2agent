#!/usr/bin/env bash
# Parallel chunked download of the ESM-2 650M safetensors weights.
#
# A single connection to huggingface.co sustains only ~54 kB/s from here;
# 16 connections aggregate to ~2.5 MB/s.
#
# Each attempt writes to a scratch file and is appended only after its length
# is checked against what was actually requested. An earlier version let curl
# retry internally while appending straight to the part file, so a retry
# re-sent the range from its start and duplicated bytes -- that assembled
# 4.17 GB for a 2.61 GB file.
set -uo pipefail

URL="https://huggingface.co/facebook/esm2_t33_650M_UR50D/resolve/main/model.safetensors"
OUT="hf/esm2_t33_650M_UR50D/model.safetensors"
PARTS=16
TMP="hf/esm2_t33_650M_UR50D/.parts"

cd "$(dirname "$0")"
mkdir -p "$TMP"

SIZE=$(curl -sIL "$URL" | tr -d '\r' | awk 'tolower($1)=="content-length:"{n=$2} END{print n}')
if [ -z "${SIZE:-}" ] || [ "$SIZE" -lt 1000000 ]; then
  echo "could not determine size (got '${SIZE:-}')" >&2
  exit 1
fi
echo "remote size: $SIZE bytes"
CHUNK=$(( (SIZE + PARTS - 1) / PARTS ))

part_path() { printf '%s/p%02d' "$TMP" "$1"; }

download_part() {
  local i=$1 start end want p have scratch got
  start=$(( i * CHUNK ))
  end=$(( start + CHUNK - 1 ))
  [ "$end" -ge "$SIZE" ] && end=$(( SIZE - 1 ))
  want=$(( end - start + 1 ))
  p=$(part_path "$i")
  scratch="$p.try"

  # A part left oversized by an earlier buggy run is cut back to its range.
  if [ -f "$p" ] && [ "$(stat -c%s "$p")" -gt "$want" ]; then
    truncate -s "$want" "$p"
  fi

  for _ in $(seq 1 80); do
    have=0
    [ -f "$p" ] && have=$(stat -c%s "$p")
    [ "$have" -ge "$want" ] && return 0

    rm -f "$scratch"
    # One attempt, no internal retry: a retry would restart the range and we
    # would append its bytes twice.
    curl -sL --connect-timeout 20 --max-time 600 \
         -r $(( start + have ))-$end -o "$scratch" "$URL" 2>/dev/null
    [ -f "$scratch" ] || { sleep 2; continue; }

    got=$(stat -c%s "$scratch")
    if [ "$got" -gt $(( want - have )) ]; then
      # Server ignored the Range header and sent more than asked; keep only
      # the bytes this part still needs.
      truncate -s $(( want - have )) "$scratch"
      got=$(( want - have ))
    fi
    if [ "$got" -gt 0 ]; then
      cat "$scratch" >> "$p"
    else
      sleep 2
    fi
    rm -f "$scratch"
  done

  have=0
  [ -f "$p" ] && have=$(stat -c%s "$p")
  echo "part $i incomplete: $have/$want" >&2
  return 1
}

PIDS=()
for i in $(seq 0 $((PARTS - 1))); do
  download_part "$i" &
  PIDS+=($!)
done

FAIL=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || FAIL=1
done

# Explicit numeric order: a p* glob sorts p10 before p2.
ORDER=()
for i in $(seq 0 $((PARTS - 1))); do ORDER+=("$(part_path "$i")"); done

TOTAL=0
for f in "${ORDER[@]}"; do
  [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }
  TOTAL=$(( TOTAL + $(stat -c%s "$f") ))
done

if [ "$FAIL" -ne 0 ] || [ "$TOTAL" -ne "$SIZE" ]; then
  echo "FAILED: assembled $TOTAL of $SIZE bytes" >&2
  exit 1
fi

cat "${ORDER[@]}" > "$OUT"
ACTUAL=$(stat -c%s "$OUT")
if [ "$ACTUAL" -ne "$SIZE" ]; then
  echo "FAILED: wrote $ACTUAL of $SIZE bytes" >&2
  exit 1
fi
echo "wrote $OUT ($ACTUAL bytes)"
rm -rf "$TMP"
echo "DONE"
