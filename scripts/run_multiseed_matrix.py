"""Multi-seed driver for `scripts/run_thesis_experiments.py`.

Thesis remediation ``run-full-matrix-with-stats``. Wraps the single-
seed runner to invoke it once per ``(seed, victim_model)`` pair and
then aggregates the per-seed runs into bootstrap-CI tables.

Convention: each invocation writes to
``logs/thesis/<HHmm-ddMMyyyy>/seed_<n>/`` so the matrix is grouped by
shared run timestamp and seed-disjoint by directory.

Example::

    python scripts/run_multiseed_matrix.py \\
        --target-models qwen3:8b llama3.1:8b \\
        --seeds 0 1 2 \\
        --tiers 1 3 \\
        --judge-backend keyword \\
        --smoke

After completion this script also calls
``scripts/reporting/aggregate_seeds.py`` to write
``logs/thesis/<ts>/aggregate/asr_with_ci.{json,md}``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


RUNNER = PROJECT_ROOT / "scripts" / "run_thesis_experiments.py"
AGG = PROJECT_ROOT / "scripts" / "reporting" / "aggregate_seeds.py"


def _model_dir(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target-models", nargs="+", required=True,
                   help="Victim models, ids from attacks/_core/model_registry.py")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--tiers", nargs="+", default=["1", "3"],
                   help="Tier subset to run.")
    p.add_argument("--n-goals", type=int, default=4)
    p.add_argument(
        "--judge-backend", default="keyword",
        choices=["auto", "openrouter", "openai", "ollama", "litellm", "keyword"],
    )
    p.add_argument("--judge-model", default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--skip-baselines", action="store_true")
    p.add_argument("--include-secondary", action="store_true",
                   help="Forwarded to run_thesis_experiments --include-secondary.")
    p.add_argument("--out", default=None,
                   help="Override output run dir (default logs/thesis/<ts>).")
    p.add_argument("--mode", choices=["local", "cloud"], default="local")
    args = p.parse_args()

    timestamp = time.strftime("%H%M-%d%m%Y")
    run_dir = Path(args.out) if args.out else (
        PROJECT_ROOT / "logs" / "thesis" / timestamp
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[multiseed] run dir: {run_dir}")

    plan = []
    for model in args.target_models:
        for seed in args.seeds:
            sub = run_dir / f"model_{_model_dir(model)}" / f"seed_{seed}"
            sub.mkdir(parents=True, exist_ok=True)
            plan.append((model, seed, sub))
    print(
        f"[multiseed] plan: {len(plan)} cells "
        f"({len(args.target_models)} models x {len(args.seeds)} seeds)"
    )

    failures: List[str] = []
    for model, seed, sub in plan:
        cmd: List[str] = [
            sys.executable,
            str(RUNNER),
            "--mode", args.mode,
            "--target-model", model,
            "--tiers", *args.tiers,
            "--n-goals", str(args.n_goals),
            "--judge-backend", args.judge_backend,
            "--seeds", str(seed),
            "--out", str(sub),
        ]
        if args.judge_model:
            cmd.extend(["--judge-model", args.judge_model])
        if args.smoke:
            cmd.append("--smoke")
        if args.skip_baselines:
            cmd.append("--skip-baselines")
        if args.include_secondary:
            cmd.append("--include-secondary")
        print(f"\n[multiseed] >>> {model} seed={seed}\n  {' '.join(cmd)}", flush=True)
        rc = subprocess.call(cmd, stderr=subprocess.STDOUT)
        if rc != 0:
            failures.append(f"{model}/seed_{seed} (rc={rc})")
            print(f"[multiseed] cell FAILED rc={rc}: {model} seed={seed}")

    for model in args.target_models:
        model_root = run_dir / f"model_{_model_dir(model)}"
        if not model_root.exists():
            continue
        print(f"\n[multiseed] aggregating seeds for {model_root}")
        rc = subprocess.call(
            [sys.executable, str(AGG), str(model_root)]
        )
        if rc != 0:
            failures.append(f"aggregate({model}) rc={rc}")

    summary = {
        "timestamp": timestamp,
        "run_dir": str(run_dir),
        "models": args.target_models,
        "seeds": args.seeds,
        "tiers": args.tiers,
        "smoke": args.smoke,
        "judge_backend": args.judge_backend,
        "judge_model": args.judge_model,
        "failures": failures,
    }
    (run_dir / "multiseed_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    if failures:
        print(f"\n[multiseed] DONE with {len(failures)} failures: {failures}")
        sys.exit(1)
    print(f"\n[multiseed] DONE -> {run_dir}")


if __name__ == "__main__":
    main()
