#!/bin/bash
# Sequential model pipeline runner
# Runs each model one at a time to avoid API rate limits
# Usage: bash scripts/run_sequential_matrix.sh

set -e
cd "$(dirname "$0")/.."

OUT="logs/thesis/matrix_v2"
TIERS="1 2"
N_GOALS=5
SEEDS="0"

MODELS=(
  "anthropic_haiku"
  "deepseek_chat"
  "meta_llama_70b"
  "mistral_7b"
  "phi_mini"
)

echo "=== Sequential matrix run ==="
echo "Models: ${MODELS[*]}"
echo "Out: $OUT | Tiers: $TIERS | n-goals: $N_GOALS | Seeds: $SEEDS"
echo ""

FAILED=()
PASSED=()

for model in "${MODELS[@]}"; do
  echo ""
  echo ">>> Starting: $model"
  echo "    $(date)"
  
  python scripts/run_model_matrix.py \
    --only "$model" \
    --tiers $TIERS \
    --n-goals $N_GOALS \
    --seeds $SEEDS \
    --out "$OUT" \
    --no-aggregate \
    2>&1
  
  rc=$?
  if [ $rc -eq 0 ]; then
    PASSED+=("$model")
    echo "    >>> $model DONE"
  else
    FAILED+=("$model")
    echo "    >>> $model FAILED (rc=$rc)"
    echo "    Check log: $OUT/model_*/run.log"
  fi
  echo "    $(date)"
done

echo ""
echo "=== SEQUENTIAL RUN COMPLETE ==="
echo "PASSED: ${PASSED[*]}"
echo "FAILED: ${FAILED[*]}"

# Aggregate all results
if [ ${#PASSED[@]} -gt 0 ]; then
  echo ""
  echo "=== Aggregating results ==="
  python scripts/run_model_matrix.py \
    --out "$OUT" \
    --only openai_mini \
    --tiers $TIERS \
    --n-goals $N_GOALS \
    --seeds $SEEDS \
    2>&1 | grep -E "matrix|aggregate|SUMMARY" || true
fi
