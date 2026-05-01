"""PACE K-sweep wrapper for the multi-seed matrix.

Per the PACE research plan §5.5 (Ablations), the **strongest empirical
claim** the thesis can make about PACE is the *quorum cost-of-attack
curve* across ``K in {1, 3, 5, 7}``. K=1 is the no-quorum baseline
(plan-only CFI), K=3 is simple-majority-of-3, K=5 is the headline
configuration, K=7 is a stress upper-bound.

This wrapper invokes ``scripts/run_multiseed_matrix.py`` once per K
value, threading ``PACE_K`` and ``PACE_Q`` through the environment so
the registry in ``scripts/run_thesis_experiments.build_defense_registry``
picks them up. The aggregator at
``scripts/reporting/aggregate_seeds.py`` then naturally rolls each
K up into its own ``model_<id>/seed_<n>/`` cell.

Outputs land under
``logs/thesis/<HHmm-ddMMyyyy>/spq_K<k>/model_<id>/seed_<n>/``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MULTISEED = PROJECT_ROOT / "scripts" / "run_multiseed_matrix.py"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target-models", nargs="+", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument(
        "--K-values", nargs="+", type=int, default=[1, 3, 5, 7],
        help="Quorum sizes to sweep. Default: 1, 3, 5, 7 per plan §5.5.",
    )
    p.add_argument(
        "--q-policy", choices=["majority", "unanimous"], default="majority",
        help="majority => q = ceil(K/2); unanimous => q = K.",
    )
    p.add_argument("--tiers", nargs="+", default=["1"])
    p.add_argument("--n-goals", type=int, default=4)
    p.add_argument("--judge-backend", default="keyword")
    p.add_argument("--judge-model", default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--skip-search", action="store_true",
                   help="Forwarded to run_thesis_experiments.")
    p.add_argument("--mode", choices=["local", "cloud"], default="local",
                   help="Forwarded to run_multiseed_matrix --mode.")
    p.add_argument("--out", default=None,
                   help="Override base run dir (default logs/thesis/<ts>).")
    args = p.parse_args()

    timestamp = time.strftime("%H%M-%d%m%Y")
    base = Path(args.out) if args.out else PROJECT_ROOT / "logs" / "thesis" / timestamp
    base.mkdir(parents=True, exist_ok=True)
    print(f"[spq-k-sweep] base run dir: {base}")
    print(f"[spq-k-sweep] sweeping K={args.K_values}, q-policy={args.q_policy}")

    failures: List[str] = []
    for K in args.K_values:
        if args.q_policy == "majority":
            q = (K + 1) // 2
        else:
            q = K
        sweep_dir = base / f"spq_K{K}_q{q}"
        sweep_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["PACE_K"] = str(K)
        env["PACE_Q"] = str(q)
        env.setdefault("PACE_TRACE_PATH", str(sweep_dir / "pace_trace.jsonl"))

        cmd: List[str] = [
            sys.executable,
            str(MULTISEED),
            "--target-models", *args.target_models,
            "--seeds", *[str(s) for s in args.seeds],
            "--tiers", *args.tiers,
            "--n-goals", str(args.n_goals),
            "--judge-backend", args.judge_backend,
            "--mode", args.mode,
            "--out", str(sweep_dir),
        ]
        if args.judge_model:
            cmd.extend(["--judge-model", args.judge_model])
        if args.smoke:
            cmd.append("--smoke")
        if args.skip_search:
            cmd.append("--skip-search")

        print(f"\n[spq-k-sweep] >>> K={K} q={q} -> {sweep_dir}\n  {' '.join(cmd)}",
              flush=True)
        rc = subprocess.call(cmd, env=env, stderr=subprocess.STDOUT)
        if rc != 0:
            failures.append(f"K={K}/q={q} (rc={rc})")
            print(f"[spq-k-sweep] cell FAILED rc={rc}: K={K} q={q}", flush=True)

    summary = {
        "timestamp": timestamp,
        "base": str(base),
        "models": args.target_models,
        "seeds": args.seeds,
        "K_values": args.K_values,
        "q_policy": args.q_policy,
        "failures": failures,
    }
    (base / "spq_k_sweep_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    if failures:
        print(f"\n[spq-k-sweep] DONE with {len(failures)} failures: {failures}")
        sys.exit(1)
    print(f"\n[spq-k-sweep] DONE -> {base}")


if __name__ == "__main__":
    main()
