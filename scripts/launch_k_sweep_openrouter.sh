#!/usr/bin/env bash
# Launch the PACE K-sweep ablation on OpenRouter.
# Reads K values and victim from configs/thesis_matrix.yaml.
#
set -euo pipefail
cd "$(dirname "$0")/.."

export OPENROUTER_ONLY=1
export PACE_FORCE_HASH_EMBED=1
export CALL_LOG_LEVEL=full

TIMESTAMP=$(date +%H%M-%d%m%Y)
RUN_DIR="logs/thesis/ksweep_${TIMESTAMP}"
mkdir -p "$RUN_DIR"

VICTIM=$(python3 -c "
import yaml
cfg = yaml.safe_load(open('configs/thesis_matrix.yaml'))
print(cfg.get('k_sweep', {}).get('victim', 'google/gemini-2.5-flash'))
")

K_VALUES=$(python3 -c "
import yaml
cfg = yaml.safe_load(open('configs/thesis_matrix.yaml'))
print(' '.join(str(k) for k in cfg.get('k_sweep', {}).get('K_values', [3,5,7,9])))
")

Q_RATIO=$(python3 -c "
import yaml
cfg = yaml.safe_load(open('configs/thesis_matrix.yaml'))
print(cfg.get('k_sweep', {}).get('q_ratio', 0.6))
")

SEEDS=$(python3 -c "
import yaml
cfg = yaml.safe_load(open('configs/thesis_matrix.yaml'))
print(' '.join(str(s) for s in cfg.get('k_sweep', {}).get('seeds', [0,1,2])))
")

TIERS=$(python3 -c "
import yaml
cfg = yaml.safe_load(open('configs/thesis_matrix.yaml'))
print(' '.join(str(t) for t in cfg.get('k_sweep', {}).get('tiers', [1,2])))
")

JUDGE_MODEL=$(python3 -c "
import yaml
cfg = yaml.safe_load(open('configs/thesis_matrix.yaml'))
print(cfg.get('judge', {}).get('model', 'google/gemini-2.5-flash'))
")

PACE_PLANNER=$(python3 -c "
import yaml
cfg = yaml.safe_load(open('configs/thesis_matrix.yaml'))
print(cfg.get('spq', {}).get('planner_model', 'openai/gpt-4o-mini'))
")

export OPENROUTER_VICTIM_MODEL="$VICTIM"
export OPENROUTER_JUDGE_MODEL="$JUDGE_MODEL"
export PACE_PLANNER_MODEL="$PACE_PLANNER"
export PACE_EXECUTOR_MODEL="$PACE_PLANNER"

echo "[k-sweep] victim=$VICTIM  K_values=$K_VALUES  q_ratio=$Q_RATIO"
echo "[k-sweep] run_dir=$RUN_DIR"

for K in $K_VALUES; do
    Q=$(python3 -c "import math; print(max(2, math.ceil($K * $Q_RATIO)))")
    echo "[k-sweep] K=$K q=$Q"

    export PACE_K="$K"
    export PACE_Q="$Q"

    CELL_DIR="${RUN_DIR}/spq_K${K}_q${Q}"
    mkdir -p "$CELL_DIR"

    python3 scripts/run_multiseed_matrix.py \
        --target-models "$VICTIM" \
        --seeds $SEEDS \
        --tiers $TIERS \
        --judge-backend openrouter \
        --out "$CELL_DIR" \
        2>&1 | tee "${CELL_DIR}/run.log"
done

echo "[k-sweep] running verification..."
python3 scripts/setup/verify_call_logs.py "$RUN_DIR" || true
python3 scripts/reporting/verify_pace_acceptance.py "$RUN_DIR" || true

echo "[k-sweep] DONE -> $RUN_DIR"
