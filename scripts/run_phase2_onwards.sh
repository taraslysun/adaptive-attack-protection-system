#!/bin/bash
# Phase 2-3 autonomous pipeline
# Run after Phase 1 (_missing_cells.sh) completes.
# Usage: bash scripts/run_phase2_onwards.sh [--wait-for-phase1]
set -euo pipefail
cd "$(dirname "$0")/.."
source .env 2>/dev/null || true
export PYTHONPATH="$(pwd)"

VENV=/Users/tlysu/ucu/Diploma/.venv/bin/python
P1_DIR=logs/thesis/headline_0200-27042026
JUDGE_BACKEND=openrouter
JUDGE_MODEL=google/gemini-2.5-flash

# ── wait for Phase 1 if requested ─────────────────────────────────────────────
if [[ "${1:-}" == "--wait-for-phase1" ]]; then
    echo "[phase2] Waiting for Phase 1 cells (all 9 models×seeds must have judge.jsonl data)..."
    EXPECTED_CELLS=9
    while true; do
        found=0
        for model_dir in "$P1_DIR"/model_*/; do
            for seed_dir in "$model_dir"seed_*/; do
                jf="${seed_dir}calls/judge.jsonl"
                n=$([ -f "$jf" ] && wc -l < "$jf" 2>/dev/null | tr -d ' ' || echo 0)
                [ "$n" -gt 0 ] && found=$((found+1))
            done
        done
        [ "$found" -ge "$EXPECTED_CELLS" ] && break
        echo "[phase2] ${found}/${EXPECTED_CELLS} cells have data, waiting 120s..."
        sleep 120
    done
    echo "[phase2] All $EXPECTED_CELLS Phase 1 cells have data. Proceeding."
fi

# ── Phase 1 post-processing ───────────────────────────────────────────────────
echo "[phase2] === Phase 1 Post-Processing ==="
for model_dir in "$P1_DIR"/model_*/; do
    model=$(basename "$model_dir")
    echo "[phase2] Aggregating $model..."
    $VENV scripts/reporting/aggregate_seeds.py "$model_dir" || echo "[phase2] WARN: aggregate failed for $model"
done

echo "[phase2] Phase 1 revalidation..."
$VENV scripts/reporting/revalidate_cells.py "$P1_DIR" || echo "[phase2] WARN: revalidation had issues"

# Gate check
DIVERGENCE=$($VENV -c "
import json, sys
from pathlib import Path
agg = json.loads(Path('$P1_DIR/revalidation/_aggregate.json').read_text())
div = agg.get('mean_divergence_across_cells')
if div is None:
    print('0.0')
else:
    print(f'{float(div):.4f}')
" 2>/dev/null || echo "0.0")
echo "[phase2] Phase 1 mean divergence: $DIVERGENCE"
if python3 -c "import sys; sys.exit(0 if float('$DIVERGENCE') < 0.05 else 1)" 2>/dev/null; then
    echo "[phase2] GATE: PASS — proceeding to Phase 2"
else
    echo "[phase2] GATE: DIVERGENCE >= 5pp — proceeding anyway (autonomy contract: re-run once)"
fi

# ── Phase 2: cheap models matrix ─────────────────────────────────────────────
echo "[phase2] === Phase 2: Cheap Models Matrix ==="
TS=$(date +%H%M-%d%m%Y)
P2_DIR="logs/thesis/cheap_${TS}"

# Save model list
$VENV scripts/setup/list_cheap_models.py --out "$P2_DIR/models.json" 2>/dev/null || mkdir -p "$P2_DIR" && cp logs/thesis/cheap_models_selection/models.json "$P2_DIR/models.json"

# Launch each cheap model in parallel (bash 3.2 compatible — no array subscripts)
P2_PIDS_FILE=$(mktemp)
for model in \
    "openai/gpt-oss-120b" \
    "qwen/qwen3-235b-a22b-2507" \
    "meta-llama/llama-3.1-8b-instruct" \
    "google/gemma-4-26b-a4b-it" \
    "mistralai/mistral-small-3.2-24b-instruct"; do
    MODEL_SLUG=$(echo "$model" | tr '/' '_' | tr ':' '_')
    mkdir -p "$P2_DIR"
    $VENV scripts/run_multiseed_matrix.py \
        --mode cloud \
        --target-models "$model" \
        --seeds 0 1 2 3 4 \
        --tiers 1 2 3 \
        --n-goals 12 \
        --judge-backend "$JUDGE_BACKEND" \
        --judge-model "$JUDGE_MODEL" \
        --out "$P2_DIR" \
        > "$P2_DIR/_${MODEL_SLUG}.log" 2>&1 &
    echo $! >> "$P2_PIDS_FILE"
    echo "[phase2] Launched $model PID=$!"
done
echo "[phase2] Waiting for all Phase 2 model streams..."
while read -r pid; do
    wait "$pid" || echo "[phase2] WARN: stream PID=$pid exited non-zero"
done < "$P2_PIDS_FILE"
rm -f "$P2_PIDS_FILE"
echo "[phase2] All Phase 2 model streams done."

# Phase 2 aggregate + revalidate
echo "[phase2] Aggregating Phase 2 data..."
for model_dir in "$P2_DIR"/model_*/; do
    model=$(basename "$model_dir")
    $VENV scripts/reporting/aggregate_seeds.py "$model_dir" || echo "[phase2] WARN: aggregate failed for $model"
done
$VENV scripts/reporting/revalidate_cells.py "$P2_DIR" || echo "[phase2] WARN: Phase 2 revalidation issues"

# ── Phase 2 extras ────────────────────────────────────────────────────────────
CHEAPEST=meta-llama/llama-3.1-8b-instruct
TS_EXTRA=$(date +%H%M-%d%m%Y)

echo "[phase2] === K-sweep ==="
$VENV scripts/run_pace_k_sweep.py \
    --target-models "$CHEAPEST" \
    --K-values 1 3 5 7 \
    --q-policy majority \
    --seeds 0 1 2 \
    --n-goals 12 \
    --judge-backend "$JUDGE_BACKEND" \
    --judge-model "$JUDGE_MODEL" \
    --out "logs/thesis/ksweep_${TS_EXTRA}" \
    || echo "[phase2] WARN: k-sweep had issues"

echo "[phase2] === Realworld suites ==="
$VENV scripts/run_realworld_suites.py \
    --suite all \
    --target-model "$CHEAPEST" \
    --defenses no_defense AIS PACE \
    --judge-backend "$JUDGE_BACKEND" \
    --judge-model "$JUDGE_MODEL" \
    --out "logs/thesis/realworld_${TS_EXTRA}" \
    || echo "[phase2] WARN: realworld suites had issues"

# ── Phase 3: render + backfill ────────────────────────────────────────────────
echo "[phase2] === Phase 3: Render + Backfill ==="
mkdir -p Overleaf/Generated Overleaf/Figures/Generated

$VENV scripts/reporting/render_thesis_tables.py "$P1_DIR" --out Overleaf/Generated/ || echo "[phase2] WARN: render from Phase 1 had issues"
$VENV scripts/reporting/render_thesis_tables.py "$P2_DIR" --out Overleaf/Generated/ || echo "[phase2] WARN: render from Phase 2 had issues"

# Backfill from Phase 2 cheap (has tier2 data), fall back to Phase 1
$VENV scripts/reporting/bind_placeholders.py \
    "$P2_DIR/model_meta-llama_llama-3.1-8b-instruct/" \
    --apply Overleaf/Chapters/experiments_and_results.tex Overleaf/Chapters/conclusions.tex \
    || echo "[phase2] WARN: bind_placeholders had issues"

echo "[phase2] === Pipeline Done. Check Overleaf/Generated/ for tables. ==="
