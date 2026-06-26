#!/usr/bin/env bash
# Reproduce paper Tables 2 + 5 (v4-192 correctness + T-opt + S-opt).
#
# Required env vars (set before running, depending on which backbones you want):
#   ANTHROPIC_API_KEY      (claude_haiku, bedrock_claude_sonnet45)
#   BEDROCK_API_KEY        (gpt_oss_120b, gpt_oss_20b, llama33_70b, gpt55, sonnet on bedrock)
#   OPENAI_API_KEY         (gpt4o)
#   GEMINI_API_KEY         (gemini25flash)

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
    --judge_backbone claude_haiku \
    --out "$out"
}

# ── Backbones to run (comment / uncomment as you have keys) ─────────────────
BACKBONES=(
  claude_haiku
  # bedrock_claude_sonnet45
  # bedrock_gpt_oss_120b
  # bedrock_gpt_oss_20b
  # bedrock_llama33_70b
  # bedrock_gpt55
  # gpt4o
)

METHODS=( direct cot algoskill_g algoskill )

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
