#!/usr/bin/env bash
# Reproduce paper Tables 2 + 5 (v4-192 correctness + T-opt + S-opt).
#
# Configure concrete API providers locally through ALGOSKILL_* environment
# variables or ALGOSKILL_BACKEND_CONFIG. This script only refers to generic
# backend aliases.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/results/v4}"
mkdir -p "$RESULTS_DIR" "$RESULTS_DIR/topt"

CORPUS="$ROOT_DIR/data/rule_based_corpus_v4.json"

run_one () {
  local method="$1"
  local backbone="$2"
  local out="$RESULTS_DIR/${method}_${backbone}.json"
  if [[ -f "$out" ]]; then
    echo "  skip $(basename "$out") (exists)"
    return 0
  fi
  echo "  -> $(basename "$out")"
  python src/run_rule_based.py \
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
  python src/run_topt_judge.py \
    --results "$raw" \
    --corpus  "$CORPUS" \
    --judge_backbone judge \
    --out "$out"
}

# ── Generic backends to run (configure their real APIs locally) ─────────────
BACKBONES=(
  default
  # fast
  # strong
)

METHODS=( direct cot algoskill_g algoskill )

for bb in "${BACKBONES[@]}"; do
  echo "[$bb]"
  for m in "${METHODS[@]}"; do
    run_one "$m" "$bb"
  done
done

echo
echo "Running T-opt + S-opt judges"
for bb in "${BACKBONES[@]}"; do
  for m in "${METHODS[@]}"; do
    judge_one "$m" "$bb"
  done
done

echo "Done. Results in $RESULTS_DIR/"
