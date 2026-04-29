#!/usr/bin/env python3
"""Run the thesis multi-model matrix.

Launches ``run_thesis_experiments.py`` once per victim in
:data:`attacks._core.model_registry.THESIS_MULTIMODEL_MATRIX`, each time
with the same defense set, tiers, seeds, and OpenRouter judge.

Usage::

    # Smoke / validation pass (2 goals, 1 seed, tiers 1+2)
    python scripts/run_model_matrix.py --smoke

    # Full matrix (10 goals, 3 seeds, all tiers)
    python scripts/run_model_matrix.py --tiers 1 2 3 --n-goals 10 --seeds 0 1 2

    # Single victim override (useful for debugging one row)
    python scripts/run_model_matrix.py --only openai_mini --smoke

The script writes each victim run into::

    logs/thesis/<timestamp>/model_<slug>/seed_<n>/

and prints a summary table at the end.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Ensure project root on path when invoked from any CWD
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Matrix definition
# ---------------------------------------------------------------------------

def _get_matrix() -> Dict[str, str]:
    from aaps.attacks._core.model_registry import THESIS_MULTIMODEL_MATRIX
    return THESIS_MULTIMODEL_MATRIX


def _get_judge() -> str:
    from aaps.attacks._core.model_registry import THESIS_JUDGE_MODEL
    return THESIS_JUDGE_MODEL


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _slug(model_id: str) -> str:
    return model_id.replace("/", "_").replace(":", "_").replace(".", "_")


def run_matrix(
    *,
    out_root: Path,
    tiers: List[int],
    n_goals: int,
    seeds: List[int],
    smoke: bool,
    only: Optional[str] = None,
    include_secondary: bool = False,
) -> Dict[str, int]:
    matrix = _get_matrix()
    judge = _get_judge()

    if only:
        if only not in matrix:
            print(f"[matrix] ERROR: --only '{only}' not in THESIS_MULTIMODEL_MATRIX")
            print(f"[matrix] Valid keys: {list(matrix.keys())}")
            sys.exit(1)
        matrix = {only: matrix[only]}

    results: Dict[str, int] = {}

    print(f"\n[matrix] Thesis multi-model matrix: {len(matrix)} victims")
    print(f"[matrix] Judge: {judge}")
    print(f"[matrix] Tiers: {tiers}  n_goals: {n_goals}  seeds: {seeds}")
    print(f"[matrix] Out root: {out_root}")
    if smoke:
        print("[matrix] SMOKE mode — minimal budgets")
    print()

    for role, model_id in matrix.items():
        slug = _slug(model_id)
        run_dir = out_root / f"model_{slug}"
        run_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(Path(__file__).parent / "run_thesis_experiments.py"),
            "--mode", "cloud",
            "--target-model", model_id,
            "--judge-backend", "openrouter",
            "--judge-model", judge,
            "--tiers", *[str(t) for t in tiers],
            "--n-goals", str(n_goals),
            "--seeds", *[str(s) for s in seeds],
            "--out", str(run_dir),
        ]
        if smoke:
            cmd.append("--smoke")
        if include_secondary:
            cmd.append("--include-secondary")

        print(f"[matrix] >>> {role}: {model_id}")
        print(f"[matrix]     out: {run_dir}")

        log_file = run_dir / "run.log"
        t0 = time.time()
        with open(log_file, "w") as lf:
            proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)

        elapsed = time.time() - t0
        rc = proc.returncode
        results[role] = rc
        status = "OK" if rc == 0 else f"FAILED (rc={rc})"
        print(f"[matrix]     {status} ({elapsed:.0f}s) -> log: {log_file}")

        # Print last few lines of log for quick diagnostics
        try:
            lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
            for line in lines[-6:]:
                print(f"    {line}")
        except Exception:
            pass
        print()

    # Summary
    ok = sum(1 for rc in results.values() if rc == 0)
    fail = len(results) - ok
    print(f"\n[matrix] === SUMMARY: {ok} OK / {fail} FAILED ===")
    for role, rc in results.items():
        status = "✓" if rc == 0 else "✗"
        print(f"  {status} {role} ({matrix[role]})")

    return results


# ---------------------------------------------------------------------------
# Aggregate + render
# ---------------------------------------------------------------------------

def aggregate_and_render(out_root: Path) -> None:
    """Run aggregate_seeds + render_thesis_tables + render_thesis_figures."""
    scripts = Path(__file__).parent

    print("\n[matrix] Aggregating seeds...")
    subprocess.run([
        sys.executable,
        str(scripts / "reporting" / "aggregate_seeds.py"),
        str(out_root),
    ])

    print("\n[matrix] Rendering tables...")
    subprocess.run([
        sys.executable,
        str(scripts / "reporting" / "render_thesis_tables.py"),
        str(out_root),
        "--out", "Overleaf/Generated",
    ])

    print("\n[matrix] Rendering figures...")
    subprocess.run([
        sys.executable,
        str(scripts / "reporting" / "render_thesis_figures.py"),
        str(out_root),
        "--out", "Overleaf/Figures/Generated",
    ])

    print("\n[matrix] Verifying PACE acceptance...")
    subprocess.run([
        sys.executable,
        str(scripts / "reporting" / "verify_pace_acceptance.py"),
        str(out_root),
    ])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out", type=Path, default=None,
                   help="Root output dir. Default: logs/thesis/<timestamp>/")
    p.add_argument("--tiers", nargs="+", type=int, default=[1, 2],
                   help="Tiers to run (default: 1 2)")
    p.add_argument("--n-goals", type=int, default=5,
                   help="Goals per scenario (default: 5)")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                   help="Seeds (default: 0 1 2)")
    p.add_argument("--smoke", action="store_true",
                   help="Minimal budget smoke pass (1 goal, 1 seed, tiers 1+2)")
    p.add_argument("--only", default=None,
                   help="Run only this one key from THESIS_MULTIMODEL_MATRIX")
    p.add_argument("--include-secondary", action="store_true",
                   help="Also include secondary baselines")
    p.add_argument("--no-aggregate", action="store_true",
                   help="Skip aggregation + rendering after run")
    args = p.parse_args()

    ts = datetime.now().strftime("%H%M-%d%m%Y")
    out_root = args.out or Path(f"logs/thesis/matrix_{ts}")
    out_root.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        tiers = [1, 2]
        n_goals = 1
        seeds = [0]
    else:
        tiers = args.tiers
        n_goals = args.n_goals
        seeds = args.seeds

    results = run_matrix(
        out_root=out_root,
        tiers=tiers,
        n_goals=n_goals,
        seeds=seeds,
        smoke=args.smoke,
        only=args.only,
        include_secondary=args.include_secondary,
    )

    all_ok = all(rc == 0 for rc in results.values())

    if not args.no_aggregate:
        aggregate_and_render(out_root)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
