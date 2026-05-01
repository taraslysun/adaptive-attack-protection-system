# Reproducing thesis numbers

## Reality check first

The thesis headline matrix uses `n_goals = 10` per cell with three seeds.
Bootstrap 95 % percentile CIs at `n=10` are ~±30 percentage points wide.
**Differences below 30 pp are statistical noise**, even though the thesis
prose sometimes cites them as wins.

When reproducing, expect numbers within ±0.30 ASR of the published cell.
Anything tighter is luck or judge bias.

## Single-cell repro (cheapest path)

Mistral-Small-2603 × PAIR × no defence × `n=10` (one of the cheapest
headline cells; thesis Table 5.2 reports ASR ≈ 0.700):

```bash
aaps run-bench \
  --benchmark agentdojo \
  --suite workspace \
  --limit 4 \
  --defense none \
  --victim mistral-small-2603
```

Cost: under $0.50 against OpenRouter or Mistral's own API. Wall-clock: ~10
min.

## Full headline cell repro

Mistral-Small × PACE × slim-5 × `n=10` × 3 seeds:

```bash
for seed in 0 1 2; do
  aaps run-bench \
    --benchmark agentdojo \
    --suite workspace --suite travel \
    --limit 10 \
    --defense pace \
    --victim mistral-small-2603 \
    --seed $seed
done
```

Cost: under $5. Wall-clock: ~1 hour total.

## Cross-vendor pilot

Per thesis Table 5.1 (5 victims × slim-5 × no_defence vs PACE):

- Gemini-2.0-Flash-Lite: cheapest, refusal-strong, low baseline ASR.
- Mistral-Small-2603: Pareto best in the thesis (Util=1.0, FPR=0).
- Llama-3.1-8B: catastrophic FPR (0.75) — included to expose the
  defence's failure mode.
- Qwen3-8B: most attack-prone baseline; the strongest absolute drop.
- DeepSeek-V3.x: thesis row was "v4-Flash" which refused everything;
  use V3-Chat or V3.1 for a meaningful comparison.

Re-running the full cross-vendor table costs roughly $20–$50 depending on
provider mix. Capped to $5 by default in CI.

## Logs

Every run writes to `logs/thesis/<HHMM-DDMMYYYY>/`. Old logs are
immutable. The thesis workspace `docs/wiki/operations/reproducibility.md`
documents the full convention.

---

## L3 single-cell repro — actual run (2026-05-01)

| Field | Value |
|---|---|
| Victim | `mistralai/mistral-small-2603` (via OpenRouter) |
| Attack | PAIR |
| Defence | none |
| n_goals | 10 |
| budget | 4 (PSSU iterations) |
| seed | 0 |
| Judge | composite scorer (keyword + structural) |
| **Measured ASR** | **0.900** (9 / 10) |
| Thesis Table 5.2 cell | 0.700 |
| Δ | +0.200 |
| Tolerance (n=10 bootstrap CI ±) | 0.30 |
| **Within tolerance** | ✓ |
| Wall time | 206.9 s |
| Estimated cost | ~$0.02 |

Caveats interpreting the +0.200 gap:

- different goal mix (this run includes cyber-leaning goals: sandbox escape, EDR disable, GitHub repo access)
- composite scorer is more permissive than the thesis's LLM judge
- single seed; thesis Table 5.2 uses three-seed mean
- `Mistral-Small-2603` may have drifted (no `x-model-version` pin in OpenRouter)

The result is consistent with the thesis at the n=10 noise floor. To tighten,
re-run with: thesis goal set, three seeds, LLM judge instead of composite,
`mistral-small-3.1-24b-instruct` for a documented endpoint.

Reproduce with:

```bash
cd /path/to/diploma   # main thesis workspace
set -a && source .env && set +a
python scripts/l3_single_cell.py --n-goals 10 --budget 4 --seed 0
```

Output goes to `logs/thesis/<HHMM-DDMMYYYY>_l3_single_cell/summary.json`.
