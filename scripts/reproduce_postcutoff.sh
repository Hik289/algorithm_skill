#!/usr/bin/env bash
# Reproduce paper Table 6 / §5.5 distribution shift.
#
# Prereq: you must first rebuild the post-cutoff corpus into
#   data/post_cutoff_v3_corpus.json
# from the public AtCoder problem pages (see README for the schema).
# We do not ship the corpus to avoid redistributing AtCoder problem text.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

CORPUS="$ROOT_DIR/data/post_cutoff_v3_corpus.json"
if [[ ! -f "$CORPUS" ]]; then
  echo "ERROR: $CORPUS not found."
  echo "Rebuild the corpus first (see README §'Reproducing §5.5 Distribution Shift')."
  exit 1
fi

RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/results/postcutoff}"
mkdir -p "$RESULTS_DIR" "$RESULTS_DIR/topt"

run_one () {
  local method="$1"
  local backbone="$2"
  local out="$RESULTS_DIR/${method}_${backbone}.json"
  if [[ -f "$out" ]]; then
    echo "  skip $(basename "$out") (exists)"
    return 0
  fi
  echo "  -> $(basename "$out")"
  python src/run_multitest.py \
    --corpus "$CORPUS" \
    --method "$method" --backbone "$backbone" \
    --out "$out"
}

judge_one () {
  local method="$1"
  local backbone="$2"
  local raw="$RESULTS_DIR/${method}_${backbone}.json"
  local out="$RESULTS_DIR/topt/${method}_${backbone}_judged.json"
  [[ -f "$raw" ]] || { echo "  skip judge (no raw): $(basename "$raw")"; return 0; }
  [[ -f "$out" ]] && { echo "  skip judge (exists): $(basename "$out")"; return 0; }
  echo "  judge -> $(basename "$out")"
  python src/run_topt_judge_hb_v3.py \
    --results "$raw" \
    --corpus  "$CORPUS" \
    --judge_backbone claude_haiku \
    --out "$out"
}

# ── Backbones used in paper Table 6 ────────────────────────────────────────
BACKBONES=(
  claude_haiku
  # bedrock_gpt_oss_120b
  # bedrock_gpt55
)

METHODS=( direct algoskill )

for bb in "${BACKBONES[@]}"; do
  echo "[$bb]"
  for m in "${METHODS[@]}"; do
    run_one "$m" "$bb"
  done
done

echo
echo "Running T-opt + S-opt judges (Haiku-judge)"
for bb in "${BACKBONES[@]}"; do
  for m in "${METHODS[@]}"; do
    judge_one "$m" "$bb"
  done
done

echo "Done. Results in $RESULTS_DIR/"
