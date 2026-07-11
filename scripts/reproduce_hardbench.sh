#!/usr/bin/env bash
# Reproduce paper Table 3 (Hard Bench, 15 problems, pass@5).
# Runs Direct + CoT + AlgoSkill-Greedy + AlgoSkill-MCTS on selected backbones,
# then runs the T-opt + S-opt judge on each result.
#
# Configure concrete API providers locally through ALGOSKILL_* environment
# variables or ALGOSKILL_BACKEND_CONFIG. This script only refers to generic
# backend aliases.
#
# Note: AlgoSkill-MCTS can be expensive on high-latency reasoning backends.
# This script runs MCTS only on the affordable backend group by default.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/results/hardbench}"
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
  python src/run_hard_v3_unified.py \
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
    --corpus  "$ROOT_DIR/data/hard_bench_corpus.json" \
    --judge_backbone judge \
    --out "$out"
}

# ── Generic backends to run. Configure their real APIs locally. ─────────────
BACKBONES_CHEAP=(
  default
  # fast
)

BACKBONES_EXPENSIVE=(
  # strong
)

# ── 1) Direct / CoT / AlgoSkill-Greedy on all selected backbones ────────────
for bb in "${BACKBONES_CHEAP[@]}" "${BACKBONES_EXPENSIVE[@]}"; do
  echo "[$bb] running direct_v3, cot_v3, algoskill_g_v3"
  run_one direct_v3      "$bb"
  run_one cot_v3         "$bb"
  run_one algoskill_g_v3 "$bb"
done

# ── 2) AlgoSkill-MCTS only on backbones where it is affordable ──────────────
for bb in "${BACKBONES_CHEAP[@]}"; do
  echo "[$bb] running algoskill_v3 (AlgoSkill-MCTS, 10 trajectories)"
  run_one algoskill_v3 "$bb"
done

# High-latency backends: paper documents these as N/A on HB.
# Uncomment the loop below if you want to reproduce the kills:
# for bb in "${BACKBONES_EXPENSIVE[@]}"; do
#   echo "[$bb] running algoskill_v3 with --n_traj 3 (may hang on H01)"
#   python src/run_hard_v3_unified.py \
#     --method algoskill_v3 --backbone "$bb" --n_traj 3 \
#     --out "$RESULTS_DIR/algoskill_v3_${bb}.json"
# done

# ── 3) T-opt + S-opt judge ──────────────────────────────────────────────────
for bb in "${BACKBONES_CHEAP[@]}" "${BACKBONES_EXPENSIVE[@]}"; do
  for m in direct_v3 cot_v3 algoskill_g_v3 algoskill_v3; do
    judge_one "$m" "$bb"
  done
done

echo "Done. Results in $RESULTS_DIR/"
