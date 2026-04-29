"""Parse PACE telemetry from unified.log.

Extracts per-attack-family counters:
- fire_rate: # PACE check_input / total inputs PACE saw
- cfi_blocks: # 'CFI VIOLATION'
- quorum_disagreements: # 'QUORUM FAILURE'
- nli_drops: # NLI filter drop events
- check_input p50 latency_ms

Usage:
    python scripts/reporting/parse_pace_telemetry.py <run_dir>
        Walks run_dir/**/unified.log, aggregates per cell, prints CSV + JSON.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Dict, List

ATTACK_MARKERS = [
    ("RLAttack", "RL"),
    ("HumanRedTeamAttack", "HRT"),
    ("PAIRAttack", "PAIR"),
    ("PoisonedRAGAttack", "PoisRAG"),
    ("SupplyChainAttack", "SC"),
]

LAT_RE = re.compile(r"latency_ms=(\d+(?:\.\d+)?)")
BENCH_ATTACK_RE = re.compile(r"\[bench\][^|]+\|\s*(\w+Attack)\s*\|")


def parse_log(log_path: Path) -> Dict:
    """Parse one unified.log; return per-attack telemetry dict."""
    if not log_path.exists():
        return {}

    per_attack: Dict[str, Dict] = {}
    for full, short in ATTACK_MARKERS:
        per_attack[short] = {
            "fire_rate_n_input": 0,
            "fire_rate_n_check": 0,
            "cfi_blocks": 0,
            "quorum_disagreements": 0,
            "quorum_passed": 0,
            "nli_drops": 0,
            "check_input_lat_ms": [],
        }

    current_attack: str | None = None
    with log_path.open() as fh:
        for line in fh:
            m = BENCH_ATTACK_RE.search(line)
            if m:
                full = m.group(1)
                current_attack = next(
                    (s for f, s in ATTACK_MARKERS if f == full), None
                )
                continue
            if current_attack is None:
                continue
            entry = per_attack[current_attack]

            # NOTE: pre-rename logs use "SPQ check_input"; post-rename logs use "PACE check_input".
            # We accept both so historical run dirs still parse cleanly.
            if "PACE check_input" in line or "SPQ check_input" in line:
                entry["fire_rate_n_input"] += 1
                entry["fire_rate_n_check"] += 1
                lm = LAT_RE.search(line)
                if lm:
                    entry["check_input_lat_ms"].append(float(lm.group(1)))
            elif "CFI VIOLATION" in line:
                entry["cfi_blocks"] += 1
            elif "QUORUM FAILURE" in line:
                entry["quorum_disagreements"] += 1
            elif "QUORUM PASSED" in line:
                entry["quorum_passed"] += 1
            elif "nli filter dropped" in line.lower() or "nli drop" in line.lower():
                entry["nli_drops"] += 1

    out: Dict[str, Dict] = {}
    for short, e in per_attack.items():
        lats = e["check_input_lat_ms"]
        out[short] = {
            "fire_rate": e["fire_rate_n_check"],
            "cfi_blocks": e["cfi_blocks"],
            "quorum_disagreements": e["quorum_disagreements"],
            "quorum_passed": e["quorum_passed"],
            "nli_drops": e["nli_drops"],
            "check_input_p50_ms": (statistics.median(lats) if lats else None),
        }
    return out


def aggregate_run(run_dir: Path) -> Dict:
    """Walk run_dir, find every unified.log, aggregate."""
    cells = []
    for p in sorted(run_dir.rglob("unified.log")):
        cell_id = str(p.relative_to(run_dir).parent)
        cells.append({"cell": cell_id, "telemetry": parse_log(p)})

    agg: Dict[str, Dict] = {}
    for short in [s for _, s in ATTACK_MARKERS]:
        agg[short] = {
            "fire_rate_total": 0,
            "cfi_blocks": 0,
            "quorum_disagreements": 0,
            "quorum_passed": 0,
            "nli_drops": 0,
            "lat_samples": [],
        }
    for c in cells:
        for short, e in c["telemetry"].items():
            a = agg[short]
            a["fire_rate_total"] += e["fire_rate"]
            a["cfi_blocks"] += e["cfi_blocks"]
            a["quorum_disagreements"] += e["quorum_disagreements"]
            a["quorum_passed"] += e["quorum_passed"]
            a["nli_drops"] += e["nli_drops"]
            if e["check_input_p50_ms"] is not None:
                a["lat_samples"].append(e["check_input_p50_ms"])
    summary = {}
    for short, a in agg.items():
        summary[short] = {
            "fire_rate_count": a["fire_rate_total"],
            "cfi_blocks": a["cfi_blocks"],
            "quorum_disagreements": a["quorum_disagreements"],
            "quorum_passed": a["quorum_passed"],
            "nli_drops": a["nli_drops"],
            "block_p50_ms": (
                round(statistics.median(a["lat_samples"]), 1)
                if a["lat_samples"] else None
            ),
        }
    return {"cells": cells, "summary": summary}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--out-json", default=None)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"ERROR: {run_dir} not found", file=sys.stderr)
        sys.exit(1)

    res = aggregate_run(run_dir)
    out = {"run_dir": str(run_dir), **res}
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.out_json}")
    s = res["summary"]
    print(f"{'attack':10s} fire_rate  cfi_blocks  quorum_disag.  nli_drops  block_p50_ms")
    for short in [s_ for _, s_ in ATTACK_MARKERS]:
        e = s[short]
        print(f"  {short:8s} {e['fire_rate_count']:>7d}      {e['cfi_blocks']:>3d}        {e['quorum_disagreements']:>3d}            {e['nli_drops']:>3d}        {e['block_p50_ms']}")


if __name__ == "__main__":
    main()
