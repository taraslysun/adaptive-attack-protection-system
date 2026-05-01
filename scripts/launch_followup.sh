#!/usr/bin/env bash
# Two follow-up batches. Batch A heavy (full 7 defenses), batch sequentially=3 for memory.
# Batch B light (gpt-4o-mini × 3 seeds, parallel).
# Run sequentially: A then B.
set -euo pipefail
cd "$(dirname "$0")/.."

TS=$(date +%H%M-%d%m%Y)
LOGROOT="logs/thesis/followup_${TS}"
mkdir -p "$LOGROOT"
log() { echo "[followup] $*" | tee -a "$LOGROOT/launcher.log"; }

SEEDS=(0 1 2)
SMALL_MODELS=(
  "google/gemini-2.0-flash-lite-001"
  "meta-llama/llama-3.1-8b-instruct"
  "qwen/qwen3-8b"
)

# ── Batch A: Tier1 lite WITH all 6 baselines + PACE ──
# 9 cells, batch=3 sequential to avoid OOM (each ~12GB ML model load).
log "Batch A: tier1 lite (memory_poisoning + tau_bench, skip agentdojo/injecagent/agentharm) — full 7 defenses, batch=3"
TASKS_A=()
for MODEL in "${SMALL_MODELS[@]}"; do
  SLUG=$(echo "$MODEL" | tr '/' '_' | tr '-' '_' | tr '.' '_')
  for SEED in "${SEEDS[@]}"; do
    TASKS_A+=("$MODEL|$SLUG|$SEED")
  done
done

run_a_cell() {
  local model="$1" slug="$2" seed="$3"
  local out="$LOGROOT/tier1_lite/model_${slug}/seed_${seed}"
  mkdir -p "$out"
  python scripts/run_thesis_experiments.py \
    --target-model "$model" \
    --tiers 1 \
    --n-goals 8 \
    --seeds "$seed" \
    --mode cloud \
    --skip-agentdojo \
    --skip-injecagent \
    --skip-agentharm \
    --judge-backend openrouter \
    --judge-model "openai/gpt-4o-mini" \
    --out "$out" \
    >> "$LOGROOT/A_tier1_lite_${slug}_seed${seed}.log" 2>&1
}

IDX=0
TOTAL_A=${#TASKS_A[@]}
while [[ $IDX -lt $TOTAL_A ]]; do
  END=$((IDX + 3))
  [[ $END -gt $TOTAL_A ]] && END=$TOTAL_A
  log "  Batch A sub-batch tasks $((IDX+1))..$END of $TOTAL_A"
  PIDS_S=()
  for (( j=IDX; j<END; j++ )); do
    IFS='|' read -r MODEL SLUG SEED <<< "${TASKS_A[$j]}"
    run_a_cell "$MODEL" "$SLUG" "$SEED" &
    PIDS_S+=($!)
    log "    A: $SLUG seed=$SEED pid=$!"
  done
  for PID in "${PIDS_S[@]}"; do wait "$PID" || log "    pid=$PID failed"; done
  log "  sub-batch done"
  IDX=$END
done

# ── Batch B: gpt-4o-mini × 3 seeds tier2 slim5 (4th vendor, parallel since only 3 procs) ──
log "Batch B: gpt-4o-mini × 3 seeds × tier2 slim5 (4th vendor for T4)"
PIDS_B=()
for SEED in "${SEEDS[@]}"; do
  OUT="$LOGROOT/vendor4/model_openai_gpt_4o_mini/seed_${SEED}"
  mkdir -p "$OUT"
  python scripts/run_thesis_experiments.py \
    --target-model "openai/gpt-4o-mini" \
    --tiers 2 \
    --n-goals 10 \
    --seeds "$SEED" \
    --mode cloud \
    --judge-backend openrouter \
    --judge-model "google/gemini-2.0-flash-lite-001" \
    --out "$OUT" \
    >> "$LOGROOT/B_gpt4o_seed${SEED}.log" 2>&1 &
  PIDS_B+=($!)
  log "  B: gpt-4o-mini seed=$SEED pid=$!"
done
for PID in "${PIDS_B[@]}"; do wait "$PID" || log "  pid=$PID failed"; done

log "ALL DONE. Logs: $LOGROOT"
echo "$LOGROOT" > logs/thesis/current_followup.txt
