"""CLI runner: AgentDojo native benchmark for PACE defense.

Runs PACE inside AgentDojo's native AgentPipeline against workspace/banking/
travel/slack suites, producing utility_rate + security_rate comparable to the
published AgentDojo leaderboard.

Usage::

    # Quick smoke test (1 task, K=1, gpt-4o-mini)
    python scripts/run_agentdojo_suite.py \\
        --suites workspace \\
        --model openai/gpt-4o-mini \\
        --spq-k 1 --spq-q 1 \\
        --attacks ignore_previous_prompt \\
        --user-tasks user_task_0 \\
        --injection-tasks injection_task_0

    # Full headline run (all suites, PACE K=5 q=3)
    python scripts/run_agentdojo_suite.py \\
        --suites workspace banking travel slack \\
        --model openai/gpt-4o-mini \\
        --spq-k 5 --spq-q 3 \\
        --attacks important_instructions ignore_previous_prompt

    # Utility-only baseline (no injections)
    python scripts/run_agentdojo_suite.py \\
        --suites workspace \\
        --model openai/gpt-4o-mini \\
        --no-injection
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from aaps.attacks._core.config import OPENROUTER_VICTIM_MODEL
from aaps.attacks._core.logging_config import set_unified_log_path


_DEFAULT_SUITES = ["workspace", "banking", "travel", "slack"]
_DEFAULT_ATTACKS = ["important_instructions"]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AgentDojo native benchmark for PACE defense",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--suites",
        nargs="+",
        default=_DEFAULT_SUITES,
        choices=["workspace", "banking", "travel", "slack"],
        metavar="SUITE",
        help="AgentDojo task suites to run",
    )
    p.add_argument(
        "--model",
        default=OPENROUTER_VICTIM_MODEL,
        help="OpenRouter model ID for the victim LLM",
    )
    p.add_argument("--spq-k", type=int, default=5, help="PACE cluster count K")
    p.add_argument(
        "--spq-q",
        type=int,
        default=None,
        help="PACE quorum threshold q (default: K//2+1)",
    )
    p.add_argument(
        "--planner-model",
        default=os.environ.get("PACE_PLANNER_MODEL", "google/gemini-2.5-flash"),
        help="Planner LLM model ID (OpenRouter)",
    )
    p.add_argument(
        "--executor-model",
        default=None,
        help="Executor LLM model ID (defaults to planner-model)",
    )
    p.add_argument(
        "--attacks",
        nargs="+",
        default=_DEFAULT_ATTACKS,
        metavar="ATTACK",
        help="AgentDojo registered attack names",
    )
    p.add_argument(
        "--user-tasks",
        nargs="+",
        default=None,
        metavar="TASK_ID",
        help="Subset of user task IDs to run (default: all)",
    )
    p.add_argument(
        "--injection-tasks",
        nargs="+",
        default=None,
        metavar="TASK_ID",
        help="Subset of injection task IDs to run (default: all)",
    )
    p.add_argument(
        "--no-injection",
        action="store_true",
        help="Run utility-only (no injection benchmark)",
    )
    p.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip the no-defense baseline run",
    )
    p.add_argument(
        "--force-rerun",
        action="store_true",
        help="Re-run tasks even if log files exist",
    )
    p.add_argument(
        "--log-dir",
        default=None,
        help="Output directory (default: logs/thesis/<timestamp>/agentdojo)",
    )
    p.add_argument("--quiet", action="store_true", help="Suppress progress output")
    return p


def _default_log_dir() -> Path:
    ts = datetime.now().strftime("%H%M-%d%m%Y")
    return PROJECT_ROOT / "logs" / "thesis" / ts / "agentdojo"


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    log_dir = Path(args.log_dir) if args.log_dir else _default_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    set_unified_log_path(log_dir)

    verbose = not args.quiet
    if verbose:
        print(f"[run_agentdojo_suite] log_dir={log_dir}")
        print(f"  suites  : {args.suites}")
        print(f"  model   : {args.model}")
        print(f"  spq     : K={args.spq_k}  q={args.spq_q or 'auto'}")
        print(f"  attacks : {args.attacks}")

    if args.no_injection:
        _run_utility_only(args, log_dir, verbose)
        return

    spq_configs = [
        {
            "tag": f"spq_K{args.spq_k}_q{args.spq_q or (args.spq_k // 2 + 1)}",
            "K": args.spq_k,
            "q": args.spq_q,
            "planner_model": args.planner_model,
            "executor_model": args.executor_model,
        }
    ]

    from aaps.evaluation.agentdojo_native import run_agentdojo_native

    t0 = time.time()
    output = run_agentdojo_native(
        suite_names=args.suites,
        spq_configs=spq_configs,
        attack_names=args.attacks,
        victim_model_id=args.model,
        log_dir=log_dir,
        user_tasks=args.user_tasks,
        injection_tasks=args.injection_tasks,
        force_rerun=args.force_rerun,
        include_baseline=not args.no_baseline,
        verbose=verbose,
    )
    elapsed = time.time() - t0

    # Write JSON results
    out_json = log_dir / "agentdojo.json"
    with open(out_json, "w") as fh:
        json.dump(output, fh, indent=2)

    # Write markdown summary
    out_md = log_dir / "agentdojo.md"
    _write_markdown_summary(output["summary"], args, elapsed, out_md)

    if verbose:
        print(f"\n[run_agentdojo_suite] Done in {elapsed:.1f}s")
        print(f"  JSON : {out_json}")
        print(f"  MD   : {out_md}")
        _print_summary_table(output["summary"])


def _run_utility_only(args: argparse.Namespace, log_dir: Path, verbose: bool) -> None:
    """Utility-only run using benchmark_suite_without_injections."""
    from aaps.evaluation.agentdojo_native import build_baseline_pipeline, build_spq_pipeline
    from aaps.defenses.pace.pipeline import PACEDefense
    from agentdojo.benchmark import benchmark_suite_without_injections, SuiteResults
    from agentdojo.task_suite.load_suites import get_suite

    results = {}
    for suite_name in args.suites:
        suite = get_suite("v1", suite_name)
        suite_log = log_dir / suite_name
        suite_log.mkdir(parents=True, exist_ok=True)

        # Baseline
        if not args.no_baseline:
            bl_pipeline = build_baseline_pipeline(model_id=args.model)
            bl_sr: SuiteResults = benchmark_suite_without_injections(
                bl_pipeline, suite, suite_log / "no_defense", args.force_rerun,
                user_tasks=args.user_tasks,
            )
            n = len(bl_sr["utility_results"])
            bl_rate = sum(bl_sr["utility_results"].values()) / n if n else 0.0
            results.setdefault(suite_name, {})["no_defense"] = {"utility_rate": bl_rate}
            if verbose:
                print(f"[{suite_name}] no_defense utility={bl_rate:.2%}")

        # PACE
        spq = PACEDefense(
            planner_model=args.planner_model,
            executor_model=args.executor_model,
            K=args.spq_k,
            q=args.spq_q,
        )
        spq_pipeline = build_spq_pipeline(spq, args.model)
        tag = f"spq_K{args.spq_k}_q{args.spq_q or (args.spq_k // 2 + 1)}"
        spq_sr: SuiteResults = benchmark_suite_without_injections(
            spq_pipeline, suite, suite_log / tag, args.force_rerun,
            user_tasks=args.user_tasks,
        )
        n = len(spq_sr["utility_results"])
        spq_rate = sum(spq_sr["utility_results"].values()) / n if n else 0.0
        results.setdefault(suite_name, {})[tag] = {"utility_rate": spq_rate}
        if verbose:
            print(f"[{suite_name}] {tag} utility={spq_rate:.2%}")

    out_json = log_dir / "agentdojo_utility.json"
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2)
    if verbose:
        print(f"\nUtility results: {out_json}")


def _write_markdown_summary(
    summary: dict, args: argparse.Namespace, elapsed: float, out_path: Path
) -> None:
    lines = [
        "# AgentDojo Native Benchmark — PACE Defense",
        "",
        f"**Model**: `{args.model}`  ",
        f"**PACE**: K={args.spq_k}, q={args.spq_q or 'auto'}  ",
        f"**Suites**: {', '.join(args.suites)}  ",
        f"**Attacks**: {', '.join(args.attacks)}  ",
        f"**Elapsed**: {elapsed:.1f}s",
        "",
    ]
    for suite_name, suite_data in summary.items():
        lines.append(f"## {suite_name}")
        lines.append("")
        lines.append("| Defense | Attack | Utility | Security |")
        lines.append("|---------|--------|---------|----------|")
        for defense_tag, attack_data in suite_data.items():
            for attack_name, metrics in attack_data.items():
                u = metrics.get("utility_rate", 0.0)
                s = metrics.get("security_rate", 0.0)
                lines.append(f"| {defense_tag} | {attack_name} | {u:.2%} | {s:.2%} |")
        lines.append("")

    out_path.write_text("\n".join(lines))


def _print_summary_table(summary: dict) -> None:
    print("\n=== AgentDojo Results ===")
    print(f"{'Suite':<12} {'Defense':<20} {'Attack':<30} {'Utility':>8} {'Security':>9}")
    print("-" * 83)
    for suite_name, suite_data in summary.items():
        for defense_tag, attack_data in suite_data.items():
            for attack_name, metrics in attack_data.items():
                u = metrics.get("utility_rate", 0.0)
                s = metrics.get("security_rate", 0.0)
                print(
                    f"{suite_name:<12} {defense_tag:<20} {attack_name:<30} "
                    f"{u:>7.1%} {s:>8.1%}"
                )


if __name__ == "__main__":
    main()
