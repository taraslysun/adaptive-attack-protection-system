"""Import smoke tests. Must pass on a clean install with core deps only."""
from __future__ import annotations

import importlib

import pytest


def test_import_root():
    import aaps  # noqa: F401
    assert aaps.__version__


def test_import_pace():
    from aaps.defenses.pace.pipeline import PACEDefense  # noqa: F401
    from aaps.defenses.pace.planner import Planner  # noqa: F401
    from aaps.defenses.pace.executor import Executor  # noqa: F401
    from aaps.defenses.pace.agreement import AgreementVoter  # noqa: F401
    from aaps.defenses.pace.clusters import EvidencePool  # noqa: F401
    from aaps.defenses.pace.plan import PACEPlan  # noqa: F401


def test_import_baselines():
    # Just checking the package imports — individual baselines may need optional deps.
    import aaps.defenses.baselines  # noqa: F401


def test_import_attacks_core():
    from aaps.attacks._core.base_attack import (  # noqa: F401
        BaseAttack,
        AttackConfig,
        AttackResult,
    )


@pytest.mark.parametrize(
    "modpath",
    [
        "aaps.attacks.slim5.pair.attack",
        "aaps.attacks.slim5.poisoned_rag.attack",
        "aaps.attacks.slim5.rl.attack",
        "aaps.attacks.slim5.human_redteam.attack",
        "aaps.attacks.slim5.supply_chain.attack",
    ],
)
def test_import_slim5(modpath: str):
    importlib.import_module(modpath)


def test_import_evaluation():
    from aaps.evaluation.defense_benchmark import DefenseBenchmark  # noqa: F401
    import aaps.evaluation.llm_judge  # noqa: F401


def test_import_agent():
    import aaps.agent.deep_agent  # noqa: F401


def test_compat_aliases():
    """`from defenses.pace.pipeline import PACEDefense` style still works."""
    import aaps  # ensure aliases installed
    import defenses.pace.pipeline  # noqa: F401
    import attacks._core.base_attack  # noqa: F401


def test_cli_smoke():
    from aaps.cli import main as cli_main
    rc = cli_main(["smoke"])
    assert rc == 0
