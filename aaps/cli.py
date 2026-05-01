"""aaps CLI. Three subcommands:

  aaps smoke                   — run import smoke tests programmatically.
  aaps run-attack ...          — single attack quick-run.
  aaps run-bench ...           — single small benchmark cell.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


CRITICAL_IMPORTS = [
    "aaps.defenses.pace.pipeline",
    "aaps.defenses.pace.planner",
    "aaps.defenses.pace.executor",
    "aaps.defenses.pace.agreement",
    "aaps.defenses.pace.clusters",
    "aaps.defenses.baselines",
    "aaps.attacks._core.base_attack",
    "aaps.attacks.slim5.pair.attack",
    "aaps.attacks.slim5.poisoned_rag.attack",
    "aaps.attacks.slim5.rl.attack",
    "aaps.attacks.slim5.human_redteam.attack",
    "aaps.attacks.slim5.supply_chain.attack",
    "aaps.evaluation.defense_benchmark",
    "aaps.evaluation.llm_judge",
    "aaps.agent.deep_agent",
]


def cmd_smoke(_args: argparse.Namespace) -> int:
    failed: list[tuple[str, str]] = []
    for mod in CRITICAL_IMPORTS:
        try:
            importlib.import_module(mod)
        except Exception as e:  # noqa: BLE001
            failed.append((mod, f"{type(e).__name__}: {e}"))
    if failed:
        print(json.dumps({"ok": False, "failed": failed}, indent=2))
        return 1
    print(json.dumps({"ok": True, "imports": len(CRITICAL_IMPORTS)}, indent=2))
    return 0


def cmd_run_attack(args: argparse.Namespace) -> int:
    from aaps.attacks._core.base_attack import AttackConfig

    family = args.family.lower()
    fam_map = {
        "pair": "aaps.attacks.slim5.pair.attack:PAIRAttack",
        "poisoned_rag": "aaps.attacks.slim5.poisoned_rag.attack:PoisonedRAGAttack",
        "rl": "aaps.attacks.slim5.rl.attack:RLAttack",
        "human_redteam": "aaps.attacks.slim5.human_redteam.attack:HumanRedTeamAttack",
        "supply_chain": "aaps.attacks.slim5.supply_chain.attack:SupplyChainAttack",
    }
    if family not in fam_map:
        print(f"unknown family {family!r}; choose from {list(fam_map)}", file=sys.stderr)
        return 2
    modpath, clsname = fam_map[family].split(":")
    cls = getattr(importlib.import_module(modpath), clsname)

    if args.victim != "mock":
        print("Only --victim mock is supported by the CLI quick-run. "
              "Use notebooks or scripts/ for real-model runs.", file=sys.stderr)
        return 2

    log_dir = Path(args.log_dir).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    out = log_dir / f"cli_{family}_mock.json"
    out.write_text(json.dumps({
        "family": family,
        "victim": args.victim,
        "n_goals": args.n_goals,
        "note": "CLI mock smoke run — no model calls executed.",
        "attack_class": f"{modpath}:{clsname}",
        "attack_config_default": AttackConfig().__dict__,
    }, default=str, indent=2))
    print(f"wrote {out}")
    return 0


def cmd_run_bench(args: argparse.Namespace) -> int:
    """Load scenarios for the chosen benchmark and report shape.

    Real defence/attack execution happens in scripts/run_thesis_experiments.py;
    this command is a deterministic, network-free smoke that verifies the
    benchmark adapter loads and returns a non-empty scenario list (when the
    backing dataset is available).
    """
    from pathlib import Path

    out: dict = {
        "benchmark": args.benchmark,
        "suite": args.suite,
        "limit": args.limit,
        "defense": args.defense,
        "victim": args.victim,
    }

    if args.benchmark == "agentdojo":
        try:
            from aaps.evaluation.external_benchmarks import load_agentdojo_scenarios
            scenarios = load_agentdojo_scenarios(
                suites=[args.suite],
                user_limit=args.limit,
                injection_limit=2,
            )
            out["scenarios_loaded"] = len(scenarios)
            out["sample_names"] = [s.name for s in scenarios[:3]]
        except ImportError as e:
            out["scenarios_loaded"] = 0
            out["error"] = f"agentdojo not installed: {e}"

    elif args.benchmark == "injecagent":
        try:
            from aaps.evaluation.external_benchmarks import load_injecagent_scenarios
            scenarios = load_injecagent_scenarios(
                splits=("dh_base",),
                limit_per_split=args.limit,
            )
            out["scenarios_loaded"] = len(scenarios)
            out["sample_names"] = [s.name for s in scenarios[:3]]
        except (ImportError, FileNotFoundError) as e:
            out["scenarios_loaded"] = 0
            out["error"] = str(e)

    elif args.benchmark in ("harmbench", "tau-bench"):
        out["scenarios_loaded"] = 0
        out["note"] = (
            f"{args.benchmark} adapter is in scripts/run_thesis_experiments.py; "
            "CLI wiring deferred to 0.2.x."
        )

    print(json.dumps(out, indent=2))
    return 0 if out.get("scenarios_loaded", 0) > 0 or "note" in out else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="aaps")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp_smoke = sub.add_parser("smoke", help="import smoke check")
    sp_smoke.set_defaults(func=cmd_smoke)

    sp_attack = sub.add_parser("run-attack", help="single attack quick-run")
    sp_attack.add_argument("--family", required=True,
                           choices=["pair", "poisoned_rag", "rl", "human_redteam", "supply_chain"])
    sp_attack.add_argument("--victim", default="mock")
    sp_attack.add_argument("--n-goals", type=int, default=2)
    sp_attack.add_argument("--log-dir", default="logs/cli")
    sp_attack.set_defaults(func=cmd_run_attack)

    sp_bench = sub.add_parser("run-bench", help="single small benchmark cell")
    sp_bench.add_argument("--benchmark", required=True,
                          choices=["agentdojo", "injecagent", "harmbench", "tau-bench"])
    sp_bench.add_argument("--suite", default="workspace")
    sp_bench.add_argument("--limit", type=int, default=2)
    sp_bench.add_argument("--defense", default="none",
                          choices=["none", "pace"])
    sp_bench.add_argument("--victim", default="mock")
    sp_bench.set_defaults(func=cmd_run_bench)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
