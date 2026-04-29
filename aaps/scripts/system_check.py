#!/usr/bin/env python3
"""One-shot thesis-grade system-check dashboard.

Runs in order:
  1. CI-safe pytest layer (tests/test_smoke.py, test_l0_contracts.py,
     test_l0_runnability.py, test_e2e_mock.py).
  2. External-service health probe (Qdrant, Ollama, sandbox, API keys).
  3. Per-attack instantiation + 1 PSSU cycle vs MockAgent.
  4. Per-defence instantiation + check_input vs benign string.

Final summary: green / yellow (gated) / red.

Usage::

    python -m aaps.scripts.system_check
    python -m aaps.scripts.system_check --json
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class CheckResult:
    name: str
    status: str  # "ok" | "fail" | "skip"
    detail: str = ""


def _run_pytest_default() -> CheckResult:
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_smoke.py",
        "tests/test_l0_contracts.py",
        "tests/test_l0_runnability.py",
        "tests/test_e2e_mock.py",
        "-q", "--tb=line",
    ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
    last = (proc.stdout or "").strip().splitlines()[-1] if proc.stdout else ""
    if proc.returncode == 0:
        return CheckResult("pytest_default", "ok", last)
    return CheckResult("pytest_default", "fail", last)


def _run_check_services() -> CheckResult:
    """External services are environment-dependent; report as skip not fail.
    Real RED conditions are caught by per-attack / per-defence checks below.
    """
    try:
        from aaps.scripts import check_services as svc
    except Exception as e:
        return CheckResult("services_dashboard", "skip", f"import error: {e}")
    results = [fn() for fn in svc.PROBES]
    failed = [r.name for r in results if r.status == "fail"]
    ok_n = sum(1 for r in results if r.status == "ok")
    return CheckResult(
        "services_dashboard",
        "ok" if ok_n > 0 else "skip",
        f"{ok_n}/{len(results)} reachable; "
        f"unreachable: {','.join(failed) or 'none'}",
    )


# ---------------------------------------------------------------------------
# Per-attack PSSU smoke
# ---------------------------------------------------------------------------

ATTACKS = [
    ("aaps.attacks.slim5.pair.attack", "PAIRAttack"),
    ("aaps.attacks.slim5.poisoned_rag.attack", "PoisonedRAGAttack"),
    ("aaps.attacks.slim5.rl.attack", "RLAttack"),
    ("aaps.attacks.slim5.human_redteam.attack", "HumanRedTeamAttack"),
    ("aaps.attacks.slim5.supply_chain.attack", "SupplyChainAttack"),
    ("aaps.attacks.legacy.tap.attack", "TAPAttack"),
    ("aaps.attacks.legacy.crescendo.attack", "CrescendoAttack"),
    ("aaps.attacks.legacy.advprompter.attack", "AdvPrompterAttack"),
]


class _MockAgent:
    model_name = "mock"
    system_prompt = ""
    defense = None
    tool_call_log: list = []
    memory: list = []

    def process_query(self, q, **_):
        return {"answer": f"[mock] {q[:80]}", "context_used": {}, "metadata": {}, "session_id": "default"}

    def reset(self):
        pass

    def start_session(self, session_id="default"):
        pass


def _attack_check(modpath: str, clsname: str) -> CheckResult:
    try:
        mod = importlib.import_module(modpath)
        cls = getattr(mod, clsname)
        from aaps.attacks._core.base_attack import AttackConfig
        cfg = AttackConfig(budget=2, success_threshold=0.5, verbose=False)
        inst = cls(agent=_MockAgent(), config=cfg)
        # one cycle
        cands = inst.propose(target_goal="test", iteration=0)
        if cands:
            inst.score(cands[:1], target_goal="test")
        return CheckResult(f"attack:{clsname}", "ok")
    except Exception as e:
        return CheckResult(f"attack:{clsname}", "fail", str(e)[:120])


# ---------------------------------------------------------------------------
# Per-defence smoke
# ---------------------------------------------------------------------------

DEFENCES = [
    "StruQDefense", "SecAlignDefense", "MELONDefense", "AMemGuard",
    "SmoothLLMDefense", "Spotlighting", "PromptSandwiching", "RPODefense",
    "CircuitBreakerDefense", "DataSentinelDefense", "RAGuard",
    "PromptGuard2Defense", "PromptGuardFilter", "LlamaFirewall",
    "WildGuardDefense",
    "LlamaGuardDefense", "GraniteGuardianDefense", "ConstitutionalClassifiersDefense",
]


def _defence_check(clsname: str) -> CheckResult:
    try:
        import aaps.defenses.baselines as b
        cls = getattr(b, clsname, None)
        if cls is None:
            return CheckResult(f"defence:{clsname}", "fail", "not exported")
        inst = cls()
        from aaps.defenses.base_defense import DefenseResult
        res = inst.check_input("hello world")
        if not isinstance(res, DefenseResult):
            return CheckResult(f"defence:{clsname}", "fail", f"check_input returned {type(res).__name__}")
        avail = getattr(inst, "_available", True)
        return CheckResult(
            f"defence:{clsname}",
            "ok" if avail else "skip",
            "" if avail else "_available=False (degraded)",
        )
    except Exception as e:
        return CheckResult(f"defence:{clsname}", "fail", str(e)[:120])


# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--json", action="store_true")
    p.add_argument("--quick", action="store_true",
                   help="skip pytest layer (~75 s saved)")
    args = p.parse_args()

    results: List[CheckResult] = []

    if not args.quick:
        results.append(_run_pytest_default())
    results.append(_run_check_services())

    for modpath, clsname in ATTACKS:
        results.append(_attack_check(modpath, clsname))

    for clsname in DEFENCES:
        results.append(_defence_check(clsname))

    failed = [r for r in results if r.status == "fail"]
    skipped = [r for r in results if r.status == "skip"]
    ok = [r for r in results if r.status == "ok"]

    if args.json:
        print(json.dumps({
            "ok": [asdict(r) for r in ok],
            "skip": [asdict(r) for r in skipped],
            "fail": [asdict(r) for r in failed],
            "summary": {"ok": len(ok), "skip": len(skipped), "fail": len(failed)},
        }, indent=2))
    else:
        widths = max((len(r.name) for r in results), default=20)
        print(f"\n{'check':<{widths}}  status  detail")
        print(f"{'-' * widths}  ------  ------")
        for r in results:
            mark = {"ok": "✓ ok ", "fail": "✗ FAIL", "skip": "·skip"}[r.status]
            print(f"{r.name:<{widths}}  {mark:<6}  {r.detail}")
        print()
        print(f"SUMMARY: {len(ok)} ok, {len(skipped)} skipped, {len(failed)} failed")
        if failed:
            print("OVERALL: ✗ RED")
        elif skipped:
            print("OVERALL: ✓ GREEN with documented skips (yellow services)")
        else:
            print("OVERALL: ✓ GREEN")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
