"""Defense-OFF attack baseline runner (raw attack ASR upper bound).

Thesis remediation P0-5. The previous header advertised this script
as the *Attacker Moves Second* (Nasr et al., arXiv:2510.09023)
methodology, but it runs every attack against ``defense=None``. That
is the *opposite* of "moves second": no defender exists, so the
methodology cannot apply. The full adaptive-attack-vs-defense matrix
lives in ``scripts/run_thesis_experiments.py`` and
``scripts/run_model_matrix.py`` which wire concrete AIS layers into
``LocalAgent`` per cell.

What this script does (honestly):
  - Runs the 5 attack families against an undefended ``LocalAgent``.
  - Produces the *defense-OFF* baseline ASR / mean score that the
    "moves second" matrix is then compared against.
  - Useful as a sanity check ("does the attack at least work without
    a defender") and to estimate attack-only query budget.

Attack families:
  1. Static/Template (prompt injection baselines)
  2. GCG Gradient-Based (whitebox -> transfer)
  3. RL-Based Adaptive (GRPO-style, see attack docstring)
  4. Search-Based MAP-Elites (evolutionary)
  5. Human Red-Team Simulation

Architecture:
  - TARGET model: local Ollama (configurable, default qwen2.5:7b)
  - ATTACKER/MUTATOR: cloud LLM via litellm (default gemini-2.5-pro)
  - JUDGE: cloud LLM via litellm (default gemini-2.0-flash)
  - Goals sourced from academic benchmarks with category tracking
  - Each run creates its own timestamped folder under logs/

Usage:
    python -m attacks.runners.run_all_attacks
    python -m attacks.runners.run_all_attacks --dataset harmbench --limit 5
    python -m attacks.runners.run_all_attacks --dataset all --category cybercrime_intrusion
    python -m attacks.runners.run_all_attacks --attacker-model gemini/gemini-2.5-flash
    python -m attacks.runners.run_all_attacks --surrogate-model Qwen/Qwen2.5-3B-Instruct --gcg-steps 250
    python -m attacks.runners.run_all_attacks --skip-gcg
"""

import argparse
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

from aaps.attacks._core.benchmarks import (
    BenchmarkGoal,
    load_benchmark,
    load_all_benchmarks,
    list_categories,
    VALID_BENCHMARKS,
)
from aaps.attacks._core.local_agent import LocalAgent
from aaps.attacks._core.config import (
    OLLAMA_TARGET_MODEL,
    OLLAMA_HELPER_MODEL,
    LITELLM_ATTACKER_MODEL,
    LITELLM_JUDGE_MODEL,
)

GCG_DEFAULT_SURROGATE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
from aaps.attacks._core.base_attack import AttackConfig, AttackResult


def _make_run_dir() -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"logs/run_{ts}")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _make_config(scorer: str, judge_model: str) -> AttackConfig:
    return AttackConfig(
        scorer_type=scorer,
        llm_judge_model=judge_model,
        verbose=True,
    )


# ── Per-attack runners ─────────────────────────────────────────────────


def _run_static(
    agent: LocalAgent,
    goals: List[BenchmarkGoal],
    cfg: AttackConfig,
) -> List[Dict[str, Any]]:
    from aaps.attacks.legacy.static.static_attacks import StaticAttackSuite

    results = []
    for i, g in enumerate(goals):
        agent.reset()
        attack = StaticAttackSuite(agent, config=_make_config(cfg.scorer_type, cfg.llm_judge_model))
        t0 = time.time()
        all_res = attack.execute_all_types(g.goal)
        best = max(all_res, key=lambda r: r.score)
        elapsed = time.time() - t0
        successes = sum(1 for r in all_res if r.success)
        print(f"    [{i+1}/{len(goals)}] score={best.score:.2f}  "
              f"({successes}/{len(all_res)} templates)  {g.goal[:60]}...")
        results.append(_goal_result(g, best, elapsed, len(all_res), successes, attack))
    return results


def _run_gcg(
    agent: LocalAgent,
    goals: List[BenchmarkGoal],
    cfg: AttackConfig,
    surrogate_model: str = GCG_DEFAULT_SURROGATE_MODEL,
    gcg_num_steps: int = 200,
    gcg_search_width: int = 256,
    gcg_topk: int = 128,
) -> List[Dict[str, Any]]:
    from aaps.attacks.legacy.gradient_attack.gcg import GCGAttack

    results = []
    for i, g in enumerate(goals):
        agent.reset()
        gcg_cfg = AttackConfig(
            budget=gcg_num_steps,
            scorer_type=cfg.scorer_type,
            llm_judge_model=cfg.llm_judge_model,
            verbose=cfg.verbose,
        )
        attack = GCGAttack(
            agent,
            config=gcg_cfg,
            num_steps=gcg_num_steps,
            search_width=gcg_search_width,
            topk=gcg_topk,
            suffix_init="x x x x x x x x x x x x x x x x x x x x",
            surrogate_model=surrogate_model,
            use_mellowmax=True,
        )
        t0 = time.time()
        result = attack.execute(g.goal, target=g.target)
        elapsed = time.time() - t0
        loss = result.metadata.get("best_loss", "?")
        print(f"    [{i+1}/{len(goals)}] score={result.score:.2f}  "
              f"loss={loss}  {g.goal[:60]}...")
        results.append(_goal_result(g, result, elapsed, result.query_count,
                                     1 if result.success else 0, attack))
    return results


def _run_rl(
    agent: LocalAgent,
    goals: List[BenchmarkGoal],
    cfg: AttackConfig,
    attacker_model: str,
) -> List[Dict[str, Any]]:
    from aaps.attacks.slim5.rl.attack import RLAttack

    results = []
    for i, g in enumerate(goals):
        agent.reset()
        rl_cfg = _make_config(cfg.scorer_type, cfg.llm_judge_model)
        attack = RLAttack(
            agent,
            config=rl_cfg,
            attacker_model=OLLAMA_HELPER_MODEL,
            num_sessions=5,
            rounds_per_session=3,
            num_candidates_per_round=3,
            use_litellm=True,
            litellm_model=attacker_model,
        )
        t0 = time.time()
        result = attack.execute(g.goal)
        elapsed = time.time() - t0
        print(f"    [{i+1}/{len(goals)}] score={result.score:.2f}  {g.goal[:60]}...")
        results.append(_goal_result(g, result, elapsed,
                                     result.metadata.get("total_attempts", 0),
                                     1 if result.success else 0, attack))
    return results


def _run_search(
    agent: LocalAgent,
    goals: List[BenchmarkGoal],
    cfg: AttackConfig,
    attacker_model: str,
) -> List[Dict[str, Any]]:
    from aaps.attacks.legacy.search_attack.attack import SearchAttack

    results = []
    for i, g in enumerate(goals):
        agent.reset()
        search_cfg = _make_config(cfg.scorer_type, cfg.llm_judge_model)
        attack = SearchAttack(
            agent,
            config=search_cfg,
            mutator_model=OLLAMA_HELPER_MODEL,
            num_generations=10,
            population_size=4,
            num_offspring=3,
            max_queries=60,
            use_llm_critic=True,
            use_litellm=True,
            litellm_model=attacker_model,
        )
        t0 = time.time()
        result = attack.execute(g.goal)
        elapsed = time.time() - t0
        print(f"    [{i+1}/{len(goals)}] score={result.score:.2f}  "
              f"queries={result.query_count}  {g.goal[:60]}...")
        results.append(_goal_result(g, result, elapsed, result.query_count,
                                     1 if result.success else 0, attack))
    return results


def _run_redteam(
    agent: LocalAgent,
    goals: List[BenchmarkGoal],
    cfg: AttackConfig,
) -> List[Dict[str, Any]]:
    from aaps.attacks.slim5.human_redteam.attack import HumanRedTeamAttack

    results = []
    for i, g in enumerate(goals):
        agent.reset()
        attack = HumanRedTeamAttack(
            agent, config=_make_config(cfg.scorer_type, cfg.llm_judge_model),
        )
        t0 = time.time()
        all_res = attack.execute_all_strategies(g.goal)
        best = max(all_res, key=lambda r: r.score)
        elapsed = time.time() - t0
        successes = sum(1 for r in all_res if r.success)
        print(f"    [{i+1}/{len(goals)}] score={best.score:.2f}  "
              f"({successes}/{len(all_res)} strategies)  {g.goal[:60]}...")
        results.append(_goal_result(g, best, elapsed, len(all_res), successes, attack))
    return results


def _goal_result(
    goal: BenchmarkGoal,
    result: AttackResult,
    elapsed: float,
    attempts: int,
    successes: int,
    attack,
) -> Dict[str, Any]:
    return {
        "goal": goal.goal,
        "target": goal.target,
        "category": goal.category,
        "source": goal.source,
        "behavior_id": goal.behavior_id,
        "success": result.success,
        "score": result.score,
        "trigger": result.trigger[:200],
        "response": result.response[:500],
        "elapsed": round(elapsed, 1),
        "attempts": attempts,
        "successes": successes,
    }


# ── Summary helpers ─────────────────────────────────────────────────────


def _asr(results: List[Dict[str, Any]]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r["success"]) / len(results)


def _mean_score(results: List[Dict[str, Any]]) -> float:
    if not results:
        return 0.0
    return sum(r["score"] for r in results) / len(results)


def _category_asr(results: List[Dict[str, Any]]) -> Dict[str, float]:
    from collections import defaultdict
    by_cat: Dict[str, List[bool]] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r["success"])
    return {cat: sum(v)/len(v) for cat, v in sorted(by_cat.items())}


# ── Main ────────────────────────────────────────────────────────────────


def run_all(
    dataset: str = "harmbench",
    category: Optional[str] = None,
    limit: int = 5,
    skip_gcg: bool = False,
    target_model: str = OLLAMA_TARGET_MODEL,
    attacker_model: str = LITELLM_ATTACKER_MODEL,
    scorer: str = "llm_judge",
    judge_model: str = LITELLM_JUDGE_MODEL,
    surrogate_model: str = GCG_DEFAULT_SURROGATE_MODEL,
    gcg_num_steps: int = 200,
    gcg_search_width: int = 256,
    gcg_topk: int = 128,
):
    run_dir = _make_run_dir()
    cfg = _make_config(scorer, judge_model)

    # ── Load benchmark goals ───────────────────────────────────────────
    categories = [category] if category else None
    if dataset == "all":
        goals = load_all_benchmarks(limit_per_source=limit, categories=categories)
    else:
        goals = load_benchmark(dataset, categories=categories, limit=limit)

    if not goals:
        print("ERROR: No goals loaded. Check --dataset / --category values.")
        return

    # ── Banner ─────────────────────────────────────────────────────────
    print("+" + "=" * 68 + "+")
    print("|" + " DEFENSE-OFF ATTACK BASELINE".center(68) + "|")
    print("|" + " (raw ASR ceiling; not the 'moves second' matrix)".center(68) + "|")
    print("+" + "=" * 68 + "+")
    print(f"\n  Target model  : {target_model} (Ollama, local)")
    print(f"  Attacker model: {attacker_model} (litellm, cloud)")
    print(f"  GCG surrogate : {surrogate_model} (local HF)")
    print(f"  Scorer        : {scorer} ({judge_model})")
    print(f"  Defenses      : NONE (use scripts/run_thesis_experiments.py")
    print(f"                  for the adaptive-attack vs defense matrix)")
    print(f"  Dataset       : {dataset}"
          + (f" [{category}]" if category else ""))
    print(f"  Goals loaded  : {len(goals)}")
    print(f"  Log folder    : {run_dir}")

    unique_cats = sorted({g.category for g in goals})
    print(f"  Categories    : {', '.join(unique_cats)}")
    print()

    agent = LocalAgent(model_name=target_model, defense=None)
    all_attack_results: Dict[str, List[Dict[str, Any]]] = {}
    total_start = time.time()

    # ── Attack 1: Static ───────────────────────────────────────────────
    _run_attack_block(
        "1/5", "Static/Template Prompt Injection", "static",
        lambda: _run_static(agent, goals, cfg),
        all_attack_results, run_dir,
    )

    # ── Attack 2: GCG ─────────────────────────────────────────────────
    if skip_gcg:
        print(f"\n\n{'='*70}")
        print("  ATTACK 2/5: GCG Gradient-Based (Whitebox -> Transfer)")
        print(f"{'='*70}")
        print("  -> Skipped (--skip-gcg flag)")
    else:
        _run_attack_block(
            "2/5", "GCG Gradient-Based (Whitebox -> Transfer)", "gcg",
            lambda: _run_gcg(
                agent,
                goals,
                cfg,
                surrogate_model=surrogate_model,
                gcg_num_steps=gcg_num_steps,
                gcg_search_width=gcg_search_width,
                gcg_topk=gcg_topk,
            ),
            all_attack_results, run_dir,
        )

    # ── Attack 3: RL / GRPO ───────────────────────────────────────────
    _run_attack_block(
        "3/5", "RL-Based Adaptive (GRPO)", "rl",
        lambda: _run_rl(agent, goals, cfg, attacker_model),
        all_attack_results, run_dir,
    )

    # ── Attack 4: Search / MAP-Elites ─────────────────────────────────
    _run_attack_block(
        "4/5", "Search-Based MAP-Elites", "search",
        lambda: _run_search(agent, goals, cfg, attacker_model),
        all_attack_results, run_dir,
    )

    # ── Attack 5: Human Red-Team ──────────────────────────────────────
    _run_attack_block(
        "5/5", "Human Red-Team Simulation", "redteam",
        lambda: _run_redteam(agent, goals, cfg),
        all_attack_results, run_dir,
    )

    total_time = time.time() - total_start

    # ── Summary Table ──────────────────────────────────────────────────
    _print_summary(all_attack_results, target_model, attacker_model,
                   scorer, judge_model, dataset, total_time)

    # ── Per-category breakdown ─────────────────────────────────────────
    _print_category_breakdown(all_attack_results)

    # ── Save summary JSON ──────────────────────────────────────────────
    summary = {
        "config": {
            "target_model": target_model,
            "attacker_model": attacker_model,
            "gcg_surrogate_model": surrogate_model,
            "gcg_num_steps": gcg_num_steps,
            "gcg_search_width": gcg_search_width,
            "gcg_topk": gcg_topk,
            "scorer": scorer,
            "judge_model": judge_model,
            "dataset": dataset,
            "category": category,
            "limit": limit,
            "skip_gcg": skip_gcg,
            "num_goals": len(goals),
        },
        "goals": [g.to_dict() for g in goals],
        "results": all_attack_results,
        "total_time_s": round(total_time, 1),
    }
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str, ensure_ascii=False)
    print(f"\n[*] All logs saved to {run_dir}/")


def _run_attack_block(
    number: str,
    title: str,
    key: str,
    runner_fn,
    all_results: Dict[str, List[Dict[str, Any]]],
    run_dir: Path,
):
    print(f"\n\n{'='*70}")
    print(f"  ATTACK {number}: {title}")
    print(f"{'='*70}")
    t0 = time.time()
    try:
        results = runner_fn()
        all_results[key] = results
        asr = _asr(results)
        avg = _mean_score(results)
        print(f"\n  -> ASR: {asr*100:.0f}%  Mean score: {avg:.3f}  "
              f"Time: {time.time()-t0:.1f}s")
        log_path = run_dir / f"{key}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    except Exception as e:
        print(f"  -> Error: {e}")
        import traceback
        traceback.print_exc()
        all_results[key] = [{"error": str(e)}]


def _print_summary(
    results: Dict[str, List[Dict[str, Any]]],
    target_model: str,
    attacker_model: str,
    scorer: str,
    judge_model: str,
    dataset: str,
    total_time: float,
):
    print(f"\n\n+{'='*78}+")
    print(f"|{'RESULTS SUMMARY'.center(78)}|")
    info = f"Target: {target_model}  |  Attacker: {attacker_model}"
    print(f"|{info:^78}|")
    info2 = f"Scorer: {scorer} ({judge_model})  |  Dataset: {dataset}"
    print(f"|{info2:^78}|")
    print(f"+{'='*78}+")
    print(f"| {'Attack Type':<22} {'ASR':>6} {'Goals':>7} {'Avg Score':>10} {'Time':>8} |")
    print(f"+{'-'*78}+")
    for key, label in [
        ("static", "Static/Template"),
        ("gcg", "GCG Gradient"),
        ("rl", "RL / GRPO"),
        ("search", "Search/MAP-Elites"),
        ("redteam", "Human Red-Team"),
    ]:
        r = results.get(key)
        if not r or (len(r) == 1 and "error" in r[0]):
            status = r[0].get("error", "SKIP") if r else "SKIP"
            print(f"| {label:<22} {'ERR':>6} {'0':>7} {'0.00':>10} {'0.0s':>8} |")
            continue
        asr = _asr(r)
        avg = _mean_score(r)
        elapsed = sum(x.get("elapsed", 0) for x in r)
        print(f"| {label:<22} {asr*100:>5.0f}% {len(r):>7} {avg:>10.3f} "
              f"{elapsed:>6.1f}s |")
    print(f"+{'-'*78}+")
    print(f"| {'Total time':.<66} {total_time:>6.1f}s   |")
    print(f"+{'='*78}+")


def _print_category_breakdown(results: Dict[str, List[Dict[str, Any]]]):
    from collections import defaultdict
    all_entries: List[Dict[str, Any]] = []
    for key, r in results.items():
        if r and not (len(r) == 1 and "error" in r[0]):
            for entry in r:
                entry_copy = dict(entry)
                entry_copy["attack"] = key
                all_entries.append(entry_copy)

    if not all_entries:
        return

    by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in all_entries:
        cat = e.get("category", "unknown")
        by_cat[cat].append(e)

    print(f"\n  Per-Category ASR (across all attacks):")
    print(f"  {'Category':<30} {'ASR':>6}  {'Avg Score':>10}  {'N':>4}")
    print(f"  {'-'*56}")
    for cat in sorted(by_cat):
        entries = by_cat[cat]
        asr = sum(1 for e in entries if e.get("success")) / len(entries)
        avg = sum(e.get("score", 0) for e in entries) / len(entries)
        print(f"  {cat:<30} {asr*100:>5.0f}%  {avg:>10.3f}  {len(entries):>4}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run all attacks (arXiv 2510.09023) with academic benchmarks"
    )
    parser.add_argument(
        "--dataset", type=str, default="harmbench",
        choices=list(VALID_BENCHMARKS) + ["all"],
        help="Benchmark dataset (default: harmbench)",
    )
    parser.add_argument(
        "--category", type=str, default=None,
        help="Filter goals by semantic category (e.g. cybercrime_intrusion)",
    )
    parser.add_argument(
        "--limit", type=int, default=5,
        help="Max goals per benchmark (default: 5)",
    )
    parser.add_argument(
        "--skip-gcg", action="store_true",
        help="Skip the GCG gradient attack",
    )
    parser.add_argument(
        "--surrogate-model", type=str, default=GCG_DEFAULT_SURROGATE_MODEL,
        help=(
            "Hugging Face model id for GCG white-box optimization "
            f"(default: {GCG_DEFAULT_SURROGATE_MODEL})"
        ),
    )
    parser.add_argument(
        "--gcg-steps", type=int, default=200,
        help="GCG optimization steps (default: 200)",
    )
    parser.add_argument(
        "--gcg-search-width", type=int, default=256,
        help="GCG search_width / candidate batch (default: 256)",
    )
    parser.add_argument(
        "--gcg-topk", type=int, default=128,
        help="GCG top-k tokens per coordinate (default: 128)",
    )
    parser.add_argument(
        "--target-model", type=str, default=OLLAMA_TARGET_MODEL,
        help=f"Target Ollama model (default: {OLLAMA_TARGET_MODEL})",
    )
    parser.add_argument(
        "--attacker-model", type=str, default=LITELLM_ATTACKER_MODEL,
        help=f"Attacker model via litellm (default: {LITELLM_ATTACKER_MODEL})",
    )
    parser.add_argument(
        "--scorer", type=str, default="llm_judge",
        choices=["llm_judge", "composite", "keyword"],
        help="Scoring method (default: llm_judge)",
    )
    parser.add_argument(
        "--judge-model", type=str, default=LITELLM_JUDGE_MODEL,
        help=f"LLM judge model (default: {LITELLM_JUDGE_MODEL})",
    )
    args = parser.parse_args()
    run_all(
        dataset=args.dataset,
        category=args.category,
        limit=args.limit,
        skip_gcg=args.skip_gcg,
        target_model=args.target_model,
        attacker_model=args.attacker_model,
        scorer=args.scorer,
        judge_model=args.judge_model,
        surrogate_model=args.surrogate_model,
        gcg_num_steps=args.gcg_steps,
        gcg_search_width=args.gcg_search_width,
        gcg_topk=args.gcg_topk,
    )
