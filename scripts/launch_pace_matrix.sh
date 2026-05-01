#!/usr/bin/env bash
#
# PACE slimmed matrix launcher — nohup-safe.
#
# Plan §5.1-5.4: 2-3 victims x 3 seeds x tiers 1+2 x 4 benchmarks.
# Tier 1 includes: memory_poisoning, agentdojo, injecagent, agentharm, tau_bench.
# Tier 2 includes: adaptive attacks (GCG-IPI, RL, search, HumanRedTeam, PoisonedRAG).
#
# Usage:
#   nohup bash scripts/launch_pace_matrix.sh > logs/thesis/spq_matrix.log 2>&1 &
#   # or just:
#   bash scripts/launch_pace_matrix.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${VIRTUAL_ENV:-.venv}/bin/python"
TS=$(date +"%H%M-%d%m%Y")
RUN_DIR="logs/thesis/${TS}"
LOG_DIR="${RUN_DIR}/_logs"
mkdir -p "$LOG_DIR"

export PACE_K=5
export PACE_Q=3
export PACE_TRACE_PATH="${RUN_DIR}/pace_trace.jsonl"

MODELS="qwen3:8b llama3.1:8b"
SEEDS="0 1 2"

echo "[launch] $(date): starting PACE matrix"
echo "[launch] RUN_DIR=${RUN_DIR}"
echo "[launch] PACE_K=${PACE_K}, PACE_Q=${PACE_Q}"
echo "[launch] models: ${MODELS}"
echo "[launch] seeds: ${SEEDS}"

"$PYTHON" scripts/run_multiseed_matrix.py \
    --target-models $MODELS \
    --seeds $SEEDS \
    --tiers 1 2 \
    --n-goals 4 \
    --judge-backend keyword \
    --skip-search \
    --smoke \
    --out "$RUN_DIR" \
    2>&1 | tee "${LOG_DIR}/multiseed.stdout"

echo ""
echo "[launch] $(date): matrix done, running verifier..."

"$PYTHON" scripts/reporting/verify_pace_acceptance.py "$RUN_DIR" \
    2>&1 | tee "${LOG_DIR}/verify.stdout"

echo ""
echo "[launch] $(date): DONE — artefacts at ${RUN_DIR}"
