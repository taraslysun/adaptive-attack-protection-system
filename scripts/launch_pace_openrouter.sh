#!/usr/bin/env bash
# Launch the PACE thesis experiment matrix on OpenRouter.
# Reads model list from configs/thesis_matrix.yaml.
#
# Usage:
#   bash scripts/launch_pace_openrouter.sh          # headline track
#   bash scripts/launch_pace_openrouter.sh --validate  # validation track (free models)
#
set -euo pipefail
cd "$(dirname "$0")/.."

export OPENROUTER_ONLY=1
export PACE_FORCE_HASH_EMBED=1
export CALL_LOG_LEVEL=full

TIMESTAMP=$(date +%H%M-%d%m%Y)
TRACK="${1:-headline}"

if [ "$TRACK" = "--validate" ] || [ "$TRACK" = "validate" ]; then
    TRACK="validation"
fi

RUN_DIR="logs/thesis/${TRACK}_${TIMESTAMP}"
mkdir -p "$RUN_DIR"
echo "[launch] track=$TRACK  run_dir=$RUN_DIR"

# Parse matrix YAML with Python (no external deps)
MODELS=$(python3 -c "
import yaml, sys
cfg = yaml.safe_load(open('configs/thesis_matrix.yaml'))
track = cfg.get('${TRACK}', {})
for m in track.get('models', []):
    print(m['slug'])
")

SEEDS=$(python3 -c "
import yaml
cfg = yaml.safe_load(open('configs/thesis_matrix.yaml'))
track = cfg.get('${TRACK}', {})
print(' '.join(str(s) for s in track.get('seeds', [0, 1, 2])))
")

TIERS=$(python3 -c "
import yaml
cfg = yaml.safe_load(open('configs/thesis_matrix.yaml'))
track = cfg.get('${TRACK}', {})
print(' '.join(str(t) for t in track.get('tiers', [1, 2, 3])))
")

IS_SMOKE=$(python3 -c "
import yaml
cfg = yaml.safe_load(open('configs/thesis_matrix.yaml'))
track = cfg.get('${TRACK}', {})
print('yes' if track.get('smoke', False) else 'no')
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

PACE_EXECUTOR=$(python3 -c "
import yaml
cfg = yaml.safe_load(open('configs/thesis_matrix.yaml'))
print(cfg.get('spq', {}).get('executor_model', 'openai/gpt-4o-mini'))
")

export OPENROUTER_JUDGE_MODEL="$JUDGE_MODEL"
export PACE_PLANNER_MODEL="$PACE_PLANNER"
export PACE_EXECUTOR_MODEL="$PACE_EXECUTOR"

PIDS=()
for MODEL in $MODELS; do
    SAFE=$(echo "$MODEL" | tr '/:' '__')
    OUT_DIR="${RUN_DIR}/model_${SAFE}"
    mkdir -p "$OUT_DIR"

    export OPENROUTER_VICTIM_MODEL="$MODEL"

    SMOKE_FLAG=""
    if [ "$IS_SMOKE" = "yes" ]; then
        SMOKE_FLAG="--smoke"
    fi

    echo "[launch] spawning $MODEL -> $OUT_DIR"
    nohup python3 scripts/run_multiseed_matrix.py \
        --target-models "$MODEL" \
        --seeds $SEEDS \
        --tiers $TIERS \
        --judge-backend openrouter \
        --out "$OUT_DIR" \
        $SMOKE_FLAG \
        > "${OUT_DIR}/run.log" 2>&1 &
    PIDS+=($!)
done

echo "[launch] waiting for ${#PIDS[@]} background jobs: ${PIDS[*]}"
FAILURES=0
for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        FAILURES=$((FAILURES + 1))
    fi
done

echo "[launch] all jobs done. failures=$FAILURES"

# Post-run verification
echo "[launch] running verify_no_ollama..."
python3 scripts/setup/verify_no_ollama.py || true

echo "[launch] running verify_call_logs..."
python3 scripts/setup/verify_call_logs.py "$RUN_DIR" || true

echo "[launch] running verify_pace_acceptance..."
python3 scripts/reporting/verify_pace_acceptance.py "$RUN_DIR" || true

echo "[launch] running verify_matrix_integrity..."
python3 scripts/setup/verify_matrix_integrity.py "$RUN_DIR" || true

echo "[launch] DONE -> $RUN_DIR"
exit $FAILURES
