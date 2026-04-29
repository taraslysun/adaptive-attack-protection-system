"""Summarize matrix_v2 results across all completed model runs.

Parses the ASR table printed by run_thesis_experiments.py for each model
and produces a combined cross-model comparison table.

Usage:
    python scripts/summarize_matrix_results.py [--out logs/thesis/matrix_v2]
"""
import re
import sys
from pathlib import Path

OUT_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("logs/thesis/matrix_v2")

ATTACK_COLS = ["RL", "HRT", "PAIR", "RAG", "SC"]
DEFENSES = ["no_defense", "StruQDefense", "DataSentinelDefense",
            "MELONDefense", "AMemGuard", "SmoothLLMDefense",
            "PromptGuard2Defense", "PACEDefense"]

def parse_log(log_path: Path):
    """Extract ASR table from a run.log."""
    text = log_path.read_text(errors="replace")
    rows = {}
    # Match lines like: |  no_defense  |  33.3%  |  0.0%  | ...
    for line in text.splitlines():
        m = re.match(r"\|\s*([\w]+)\s*\|(.+)", line)
        if not m:
            continue
        name = m.group(1).strip()
        vals = re.findall(r"(\d+\.\d+)%", m.group(2))
        if len(vals) >= 5:
            rows[name] = [float(v) for v in vals[:5]]
    return rows

def aggregate_asr(rows):
    """Compute mean ASR across attack families per defense."""
    result = {}
    for defense, vals in rows.items():
        result[defense] = sum(vals) / len(vals) if vals else 0.0
    return result

def main():
    all_models = {}
    for model_dir in sorted(OUT_ROOT.glob("model_*")):
        log = model_dir / "run.log"
        if not log.exists():
            continue
        if "artefacts written" not in log.read_text(errors="replace"):
            print(f"[skip] {model_dir.name} — not yet complete")
            continue
        rows = parse_log(log)
        if not rows:
            print(f"[skip] {model_dir.name} — no ASR table found")
            continue
        model_short = model_dir.name.replace("model_", "")
        all_models[model_short] = rows
        print(f"[ok]   {model_short} — {len(rows)} defense rows")

    if not all_models:
        print("No completed models found yet.")
        return

    # Print per-model tables
    for model, rows in all_models.items():
        print(f"\n{'='*70}")
        print(f"Model: {model}")
        print(f"{'Defense':<28} {'RL':>7} {'HRT':>7} {'PAIR':>7} {'RAG':>7} {'SC':>7} {'Mean':>7}")
        print("-"*70)
        for defense in DEFENSES:
            if defense not in rows:
                continue
            vals = rows[defense]
            mean = sum(vals) / len(vals)
            cols = "  ".join(f"{v:6.1f}%" for v in vals)
            print(f"  {defense:<26} {cols}  {mean:6.1f}%")

    # Cross-model PACE vs no_defense comparison
    print(f"\n{'='*70}")
    print("CROSS-MODEL: PACE ASR reduction vs no_defense (PAIR attack)")
    print(f"{'Model':<40} {'no_defense PAIR':>16} {'PACE PAIR':>10} {'Δ':>8}")
    print("-"*70)
    for model, rows in all_models.items():
        nd = rows.get("no_defense", [None]*3)
        spq = rows.get("PACEDefense", [None]*3)
        nd_pair = nd[2] if nd and len(nd) > 2 else None
        spq_pair = spq[2] if spq and len(spq) > 2 else None
        if nd_pair is not None and spq_pair is not None:
            delta = spq_pair - nd_pair
            print(f"  {model:<38} {nd_pair:>14.1f}%  {spq_pair:>8.1f}%  {delta:>+7.1f}%")

    # PACE vs best baseline
    print(f"\n{'='*70}")
    print("PACE mean ASR vs best baseline (DataSentinel) across models")
    print(f"{'Model':<40} {'PACE mean':>10} {'DataSent mean':>14} {'Δ':>8}")
    print("-"*70)
    for model, rows in all_models.items():
        spq = rows.get("PACEDefense", [])
        ds = rows.get("DataSentinelDefense", [])
        if spq and ds:
            spq_m = sum(spq) / len(spq)
            ds_m = sum(ds) / len(ds)
            print(f"  {model:<38} {spq_m:>9.1f}%  {ds_m:>13.1f}%  {spq_m - ds_m:>+7.1f}%")

if __name__ == "__main__":
    main()
