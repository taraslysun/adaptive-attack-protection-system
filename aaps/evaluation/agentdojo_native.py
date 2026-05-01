"""Native AgentDojo benchmark runner for PACE defense.

Builds a proper ``AgentPipeline`` with PACE elements and calls
``agentdojo.benchmark.benchmark_suite_with_injections()`` so that results
are directly comparable to the published AgentDojo leaderboard.

Contrast with ``evaluation/external_benchmarks.py`` which uses AgentDojo
suites only as scenario *sources* fed into ``DefenseBenchmark``.  This
module runs PACE inside AgentDojo's own execution engine.

Metrics produced per (suite, defense_tag, attack):
  - utility_rate   : fraction of user tasks completed correctly
  - security_rate  : fraction of injection tasks NOT achieved by attacker
  - n_user_tasks   : number of user tasks evaluated
  - n_injection_tasks : number of injection task / user task pairs
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    import openai
    from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline
    from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage
    from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM
    from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop, ToolsExecutor
    from agentdojo.aaps.attacks.base_attacks import BaseAttack as AgentDojoBaseAttack
    from agentdojo.benchmark import (
        SuiteResults,
        benchmark_suite_with_injections,
        benchmark_suite_without_injections,
    )
    from agentdojo.task_suite.load_suites import get_suite

    _AGENTDOJO_AVAILABLE = True
except ImportError as _e:
    _AGENTDOJO_AVAILABLE = False
    _AGENTDOJO_IMPORT_ERROR = str(_e)

from aaps.attacks._core.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL
from aaps.defenses.pace.pipeline import PACEDefense

_SUITE_NAMES = ["workspace", "banking", "travel", "slack"]
_SUITE_VERSION = "v1"


def _require_agentdojo() -> None:
    if not _AGENTDOJO_AVAILABLE:
        raise ImportError(
            f"agentdojo is required for native benchmarking: {_AGENTDOJO_IMPORT_ERROR}"
        )


def _openrouter_client(api_key: Optional[str] = None) -> "openai.OpenAI":
    key = api_key or OPENROUTER_API_KEY
    if not key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY not set. Export it or pass api_key= explicitly."
        )
    return openai.OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)


def build_spq_pipeline(
    spq_defense: "PACEDefense",
    model_id: str,
    system_message: str = "",
    *,
    openrouter_api_key: Optional[str] = None,
    max_iters: int = 15,
) -> "AgentPipeline":
    """Return an AgentPipeline with PACE elements wired in.

    Args:
        spq_defense: Initialised PACEDefense instance.
        model_id: OpenRouter model ID (e.g. ``"openai/gpt-4o-mini"``).
        system_message: Suite-level system prompt (can be empty).
        openrouter_api_key: Override for OPENROUTER_API_KEY env var.
        max_iters: Max tool execution iterations per task.
    """
    _require_agentdojo()

    from aaps.defenses.pace.agentdojo_pipeline import (
        PACEEvidenceElement,
        PACEGateElement,
        PACEInitElement,
    )

    client = _openrouter_client(openrouter_api_key)
    llm = OpenAILLM(client, model_id)

    # Correct AgentDojo pipeline structure:
    # LLM is called once OUTSIDE the loop (initial response), then inside the
    # loop after each tool execution round (ToolsExecutor → LLM → check again).
    tools_loop = ToolsExecutionLoop(
        [
            PACEGateElement(spq_defense),
            ToolsExecutor(),
            PACEEvidenceElement(spq_defense),
            llm,
        ],
        max_iters=max_iters,
    )

    pipeline_elements = []
    if system_message:
        pipeline_elements.append(SystemMessage(system_message))
    pipeline_elements.append(InitQuery())
    pipeline_elements.append(PACEInitElement(spq_defense))
    pipeline_elements.append(llm)
    pipeline_elements.append(tools_loop)

    return AgentPipeline(pipeline_elements)


def build_baseline_pipeline(
    model_id: str,
    system_message: str = "",
    *,
    openrouter_api_key: Optional[str] = None,
    max_iters: int = 15,
) -> "AgentPipeline":
    """Plain pipeline with no defense — utility baseline."""
    _require_agentdojo()
    client = _openrouter_client(openrouter_api_key)
    llm = OpenAILLM(client, model_id)
    elements: list = []
    if system_message:
        elements.append(SystemMessage(system_message))
    elements.append(InitQuery())
    elements.append(llm)
    elements.append(ToolsExecutionLoop([ToolsExecutor(), llm], max_iters=max_iters))
    return AgentPipeline(elements)


def _suite_results_to_dict(suite_results: "SuiteResults") -> Dict[str, Any]:
    utility = suite_results["utility_results"]
    security = suite_results["security_results"]
    n_u = len(utility)
    n_s = len(security)
    return {
        "utility_rate": sum(utility.values()) / n_u if n_u else 0.0,
        "security_rate": sum(security.values()) / n_s if n_s else 0.0,
        "n_user_task_pairs": n_u,
        "n_injection_task_pairs": n_s,
        "utility_results": {str(k): v for k, v in utility.items()},
        "security_results": {str(k): v for k, v in security.items()},
    }


def run_agentdojo_native(
    suite_names: Sequence[str],
    spq_configs: Sequence[Dict[str, Any]],
    attack_names: Sequence[str],
    victim_model_id: str,
    log_dir: Path,
    *,
    user_tasks: Optional[Sequence[str]] = None,
    injection_tasks: Optional[Sequence[str]] = None,
    openrouter_api_key: Optional[str] = None,
    force_rerun: bool = False,
    include_baseline: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run AgentDojo benchmark natively with PACE pipeline.

    Args:
        suite_names: AgentDojo suite names, e.g. ``["workspace", "banking"]``.
        spq_configs: List of dicts, each defining one PACE configuration::

            {"tag": "spq_k5q3", "K": 5, "q": 3,
             "planner_model": "google/gemini-2.5-flash",
             "executor_model": "openai/gpt-4o-mini"}

        attack_names: Registered AgentDojo attack names to evaluate.
        victim_model_id: OpenRouter model ID for the victim LLM.
        log_dir: Directory for result JSON logs (created if absent).
        user_tasks: Subset of user task IDs; ``None`` = all.
        injection_tasks: Subset of injection task IDs; ``None`` = all.
        openrouter_api_key: Override for OPENROUTER_API_KEY.
        force_rerun: Re-run even if log files exist.
        include_baseline: Also run without defense for utility comparison.
        verbose: Print progress.

    Returns:
        Nested dict::

            {
              "results": {
                suite: {
                  defense_tag: {
                    attack: SuiteResults-as-dict
                  }
                }
              },
              "summary": {
                suite: {
                  defense_tag: {
                    attack: {utility_rate, security_rate, ...}
                  }
                }
              }
            }
    """
    _require_agentdojo()

    from agentdojo.aaps.attacks.base_attacks import FixedJailbreakAttack

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, Any] = {}
    all_summary: Dict[str, Any] = {}

    for suite_name in suite_names:
        if verbose:
            print(f"\n[agentdojo_native] Suite: {suite_name}")

        suite = get_suite(_SUITE_VERSION, suite_name)
        suite_log_dir = log_dir / suite_name
        suite_log_dir.mkdir(parents=True, exist_ok=True)

        suite_results: Dict[str, Any] = {}
        suite_summary: Dict[str, Any] = {}

        # ----------------------------------------------------------------
        # PACE configurations
        # ----------------------------------------------------------------
        configs_to_run = list(spq_configs)
        if include_baseline:
            configs_to_run = [{"tag": "no_defense", "_baseline": True}] + configs_to_run

        for cfg in configs_to_run:
            tag = cfg.get("tag", "spq")
            is_baseline = cfg.get("_baseline", False)

            if verbose:
                print(f"  Defense: {tag}")

            if is_baseline:
                pipeline = build_baseline_pipeline(
                    model_id=victim_model_id,
                    openrouter_api_key=openrouter_api_key,
                )
            else:
                spq = PACEDefense(
                    planner_model=cfg.get(
                        "planner_model",
                        os.environ.get("PACE_PLANNER_MODEL", "google/gemini-2.5-flash"),
                    ),
                    executor_model=cfg.get("executor_model"),
                    K=cfg.get("K", 5),
                    q=cfg.get("q"),
                )
                pipeline = build_spq_pipeline(
                    spq_defense=spq,
                    model_id=victim_model_id,
                    openrouter_api_key=openrouter_api_key,
                )

            tag_results: Dict[str, Any] = {}
            tag_summary: Dict[str, Any] = {}

            for attack_name in attack_names:
                if verbose:
                    print(f"    Attack: {attack_name}")

                try:
                    from agentdojo.aaps.attacks import load_attack
                    attack = load_attack(attack_name, suite, pipeline)
                except KeyError:
                    if verbose:
                        print(f"    [WARN] attack '{attack_name}' not found in registry; skipping")
                    continue

                attack_log_dir = suite_log_dir / tag / attack_name
                attack_log_dir.mkdir(parents=True, exist_ok=True)

                from agentdojo.logging import Logger, OutputLogger

                with OutputLogger(str(attack_log_dir)) as logger:
                    with logger:
                        sr: SuiteResults = benchmark_suite_with_injections(
                            agent_pipeline=pipeline,
                            suite=suite,
                            attack=attack,
                            logdir=attack_log_dir,
                            force_rerun=force_rerun,
                            user_tasks=list(user_tasks) if user_tasks else None,
                            injection_tasks=list(injection_tasks) if injection_tasks else None,
                            verbose=verbose,
                        )

                tag_results[attack_name] = {
                    "utility_results": {str(k): v for k, v in sr["utility_results"].items()},
                    "security_results": {str(k): v for k, v in sr["security_results"].items()},
                }
                tag_summary[attack_name] = _suite_results_to_dict(sr)

                if verbose:
                    s = tag_summary[attack_name]
                    print(
                        f"      utility={s['utility_rate']:.2%}  "
                        f"security={s['security_rate']:.2%}  "
                        f"n={s['n_injection_task_pairs']}"
                    )

            suite_results[tag] = tag_results
            suite_summary[tag] = tag_summary

        all_results[suite_name] = suite_results
        all_summary[suite_name] = suite_summary

    return {"results": all_results, "summary": all_summary}
