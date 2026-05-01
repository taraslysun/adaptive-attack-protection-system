"""Verify PACE acceptance criteria across a multi-seed run.

Implements the verification half of acceptance criteria a-h listed in
``docs/design/spq.md`` §9 / the PACE research plan §7. Walks one or
more ``logs/thesis/<ts>/...`` directories, collects the per-cell
PACE telemetry written by
:func:`evaluation.defense_benchmark._spq_summary`, and emits an
ALL-GREEN / AMBER / RED status report.

Acceptance criteria checked:
  a. ``cfi_violation_count == 0`` across every (model, attack, PACE)
     cell. Any nonzero value is a bug.
  b. Quorum-margin histograms exist per cell.
  c. Bootstrap 95% CI tables exist for judged ASR per cell across
     the headline matrix {no_defense, struq, melon, smoothllm,
     prompt_guard2, data_sentinel, a_memguard, PACE@K=*} columns.
     (We check that the aggregate file has been written; the CI math
     itself is in ``aggregate_seeds.py``.)
  d. tau-bench utility delta is reported with a CI.
  e. AgentHarm tool-call success rate is reported with a CI per
     (attack, defense).
  f. K and q ablation directories are present (when run via
     ``run_pace_k_sweep.py``).

Exit code: 0 = GREEN, 1 = AMBER (partial), 2 = RED (CFI violated).

Usage::

    python scripts/reporting/verify_pace_acceptance.py logs/thesis/<ts>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Per-cell collectors
# ---------------------------------------------------------------------------


def _iter_run_jsons(root: Path) -> List[Path]:
    """Find every per-cell JSON written by run_thesis_experiments."""
    matches: List[Path] = []
    for sub in ("tier1", "tier2", "tier3"):
        for fp in root.rglob(f"{sub}/*.json"):
            matches.append(fp)
    return matches


def _collect_spq(matrix_json: Path) -> List[Dict[str, Any]]:
    try:
        data = json.loads(matrix_json.read_text())
    except Exception:
        return []
    out: List[Dict[str, Any]] = []
    spq = data.get("spq_metrics") or {}
    if not isinstance(spq, dict):
        return []
    for defense_name, summary in spq.items():
        if not isinstance(summary, dict):
            continue
        out.append(
            {
                "file": str(matrix_json),
                "defense": defense_name,
                "summary": summary,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Acceptance checks
# ---------------------------------------------------------------------------


def check_cfi(cells: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    """Acceptance criterion a: cfi_violation_count == 0 everywhere."""
    bad: List[str] = []
    legacy_ambiguous: List[str] = []
    n_checked = 0
    for c in cells:
        n_checked += 1
        summary = c["summary"]

        # New telemetry schema: explicit execution-level counter.
        if "cfi_execution_violation_count" in summary:
            cnt = summary.get("cfi_execution_violation_count", 0)
        else:
            # Legacy schema used cfi_violation_count for blocked attempts,
            # which is ambiguous for acceptance criterion (executed-only).
            legacy_cnt = summary.get("cfi_violation_count", 0)
            try:
                legacy_cnt = int(legacy_cnt)
            except Exception:
                legacy_cnt = 0
            if legacy_cnt != 0:
                legacy_ambiguous.append(
                    f"{c['file']} :: {c['defense']} -> legacy cfi={legacy_cnt}"
                )
            cnt = 0
        try:
            cnt = int(cnt)
        except Exception:
            cnt = 0
        if cnt != 0:
            bad.append(f"{c['file']} :: {c['defense']} -> cfi={cnt}")
    if not n_checked:
        return ("amber", ["no PACE cells found at all (criterion a vacuous)"])
    if bad:
        return ("red", bad)
    if legacy_ambiguous:
        msg = [
            "legacy PACE telemetry detected (cfi_violation_count used for blocked attempts)",
            "rerun cells with updated PACE trace schema for strict CFI acceptance",
        ]
        msg.extend(legacy_ambiguous[:8])
        if len(legacy_ambiguous) > 8:
            msg.append(f"... and {len(legacy_ambiguous) - 8} more")
        return ("amber", msg)
    return ("green", [f"{n_checked} PACE cells, all cfi_violation_count==0"])


def check_histograms(cells: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    """Acceptance criterion b: quorum-margin histogram per cell."""
    missing: List[str] = []
    n = 0
    for c in cells:
        n += 1
        hist = c["summary"].get("quorum_margin_histogram")
        if not isinstance(hist, dict) or not hist:
            missing.append(f"{c['file']} :: {c['defense']}")
    if missing:
        return ("amber", missing)
    if not n:
        return ("amber", ["no PACE cells found"])
    return ("green", [f"{n} cells, all carry quorum_margin_histogram"])


def check_aggregate(root: Path) -> Tuple[str, List[str]]:
    """Acceptance criterion c: bootstrap CI tables exist."""
    misses: List[str] = []
    found_any = False
    for sub in root.rglob("aggregate"):
        for fname in ("asr_with_ci.json", "psr_with_ci.json"):
            if (sub / fname).exists():
                found_any = True
            else:
                misses.append(f"missing {sub / fname}")
    if not found_any:
        return ("amber", ["no aggregate/ dirs found; run aggregate_seeds.py first"])
    if misses:
        return ("amber", misses)
    return ("green", ["aggregate ASR + PSR with CI present"])


def check_spq_aggregate(root: Path) -> Tuple[str, List[str]]:
    """Helper: spq_with_ci.json present when PACE ran."""
    found = list(root.rglob("aggregate/spq_with_ci.json"))
    if not found:
        return ("amber", ["no spq_with_ci.json found; rerun aggregate_seeds.py"])
    return ("green", [f"{len(found)} spq_with_ci.json file(s) present"])


def check_tau_bench(root: Path) -> Tuple[str, List[str]]:
    """Acceptance criterion d: tau-bench utility report exists."""
    files = list(root.rglob("tier1/tau_bench.json"))
    if not files:
        return ("amber", ["no tier1/tau_bench.json found"])
    return ("green", [f"{len(files)} tau_bench.json file(s) present"])


def check_agentharm(root: Path) -> Tuple[str, List[str]]:
    """Acceptance criterion e: AgentHarm tool-call success rate present."""
    files = list(root.rglob("tier1/agentharm.json"))
    if not files:
        return ("amber", ["no tier1/agentharm.json found"])
    return ("green", [f"{len(files)} agentharm.json file(s) present"])


def check_k_sweep(root: Path) -> Tuple[str, List[str]]:
    """Acceptance criterion f: K and q ablation dirs present."""
    sweep_dirs = list(root.glob("spq_K*_q*"))
    if not sweep_dirs:
        return ("amber", ["no spq_K*_q*/ ablation dirs (run scripts/run_pace_k_sweep.py)"])
    return ("green", [f"{len(sweep_dirs)} K/q ablation dirs: "
                      f"{sorted(d.name for d in sweep_dirs)}"])


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def verify(root: Path) -> int:
    print(f"[verify_spq] root = {root}")
    if not root.exists():
        print(f"[verify_spq] RED: root does not exist")
        return 2

    matrix_jsons = _iter_run_jsons(root)
    cells: List[Dict[str, Any]] = []
    for fp in matrix_jsons:
        cells.extend(_collect_spq(fp))
    print(f"[verify_spq] scanned {len(matrix_jsons)} matrix JSONs, "
          f"found {len(cells)} PACE cells")

    checks = [
        ("a. CFI invariant (cfi_violation_count == 0)", check_cfi(cells)),
        ("b. Quorum-margin histogram per cell",         check_histograms(cells)),
        ("c. Bootstrap CI aggregates present",          check_aggregate(root)),
        ("c'. PACE aggregate (spq_with_ci.json)",        check_spq_aggregate(root)),
        ("d. tau-bench utility report",                 check_tau_bench(root)),
        ("e. AgentHarm tool-call success report",       check_agentharm(root)),
        ("f. K / q ablation directories",               check_k_sweep(root)),
    ]

    overall = "green"
    for name, (status, msgs) in checks:
        marker = {"green": "[GREEN]", "amber": "[AMBER]", "red": "[RED  ]"}[status]
        print(f"\n{marker} {name}")
        for m in msgs[:10]:
            print(f"    {m}")
        if len(msgs) > 10:
            print(f"    ... and {len(msgs) - 10} more")
        if status == "red":
            overall = "red"
        elif status == "amber" and overall == "green":
            overall = "amber"

    print(f"\n[verify_spq] OVERALL: {overall.upper()}")
    return {"green": 0, "amber": 1, "red": 2}[overall]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path,
                   help="Path to logs/thesis/<ts>/")
    args = p.parse_args()
    sys.exit(verify(args.run_dir))


if __name__ == "__main__":
    main()
