#!/usr/bin/env bash
# Full T1-T4 evaluation launcher.
# Main matrix only (no ablation table T2 was dropped; PACE now Planner+CFI+Quorum+NLI).
# 3 small tool-capable victim models × 3 seeds = 9 cells, batched.
# Tiers 1 + 2 (tier1 = T3 cost utility & FPR; tier2 = T1 attack×defense + T4 cross-vendor).
set -euo pipefail

cd "$(dirname "$0")/.."

MAX_CONCURRENT=${MAX_CONCURRENT:-3}  # 3 procs × ~12GB ML models = ~36GB; 9 OOMs system
TS=$(date +%H%M-%d%m%Y)
LOGROOT="logs/thesis/fulleval_${TS}"
mkdir -p "$LOGROOT"

log() { echo "[launcher] $*" | tee -a "$LOGROOT/launcher.log"; }

SEEDS=(0 1 2)
JUDGE="openai/gpt-4o-mini"

# Three small tool-capable victims confirmed by smoke (spqcheck_0428):
#   gemini-2.0-flash-lite : Δ=-0.067 (refusal-strong baseline)
#   llama-3.1-8b          : Δ=-0.133 (attack-prone baseline)
#   qwen3-8b              : Δ=-0.333 (most attack-prone baseline)
# Judge = gpt-4o-mini (distinct from every victim).
MODELS=(
  "google/gemini-2.0-flash-lite-001"
  "meta-llama/llama-3.1-8b-instruct"
  "qwen/qwen3-8b"
)

TASKS=()
for MODEL in "${MODELS[@]}"; do
  SLUG=$(echo "$MODEL" | tr '/' '_' | tr '-' '_' | tr '.' '_')
  for SEED in "${SEEDS[@]}"; do
    TASKS+=("main_${SLUG}_seed${SEED}|main/model_${SLUG}/seed_${SEED}|||$MODEL|$JUDGE|$SEED")
  done
done

log "Total tasks: ${#TASKS[@]}, batch size: $MAX_CONCURRENT"
log "Logs root: $LOGROOT"

run_task() {
  local label="$1" subdir="$2" env_prefix="$3" extra_flags="$4" model="$5" judge="$6" seed="$7"
  local out="$LOGROOT/$subdir"
  mkdir -p "$out"
  local logfile="$LOGROOT/${label}.log"
  # shellcheck disable=SC2086
  env $env_prefix python scripts/run_thesis_experiments.py \
    --target-model "$model" \
    --tiers 2 \
    --n-goals 10 \
    --seeds "$seed" \
    --mode cloud \
    --judge-backend openrouter \
    --judge-model "$judge" \
    $extra_flags \
    --out "$out" \
    >> "$logfile" 2>&1
}

BATCH_NUM=0
TOTAL=${#TASKS[@]}
IDX=0
ALL_PIDS=()

while [[ $IDX -lt $TOTAL ]]; do
  BATCH_NUM=$((BATCH_NUM + 1))
  BATCH_END=$((IDX + MAX_CONCURRENT))
  [[ $BATCH_END -gt $TOTAL ]] && BATCH_END=$TOTAL
  log "── Batch $BATCH_NUM: tasks $((IDX+1))..$BATCH_END of $TOTAL"
  BATCH_PIDS=()
  for (( j=IDX; j<BATCH_END; j++ )); do
    IFS='|' read -r LABEL SUBDIR ENVP EXTRA MODEL JUDGE_M SEED <<< "${TASKS[$j]}"
    run_task "$LABEL" "$SUBDIR" "$ENVP" "$EXTRA" "$MODEL" "$JUDGE_M" "$SEED" &
    PID=$!
    BATCH_PIDS+=($PID)
    ALL_PIDS+=($PID)
    log "  [$LABEL] pid=$PID"
  done
  log "Batch $BATCH_NUM launched (${#BATCH_PIDS[@]} procs); waiting ..."
  for PID in "${BATCH_PIDS[@]}"; do
    wait "$PID" || log "  pid=$PID exited non-zero"
  done
  log "Batch $BATCH_NUM done."
  IDX=$BATCH_END
done

printf '%s\n' "${ALL_PIDS[@]}" > "$LOGROOT/pids.txt"
echo "$LOGROOT" > logs/thesis/current_eval.txt
log "ALL DONE. Total: ${#ALL_PIDS[@]} cells. Logs: $LOGROOT"
