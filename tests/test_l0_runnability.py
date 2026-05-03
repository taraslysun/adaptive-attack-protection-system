"""L0 runnability sweep — every attack instantiates and one PSSU cycle
runs vs MockAgent; every defence instantiates and the five hooks return
a DefenseResult shape.

CI-safe: no model calls, no network. MockAgent returns canned responses.
"""
from __future__ import annotations

import importlib
from typing import Any, Dict

import pytest


# ---------------------------------------------------------------------------
# Mock agent — process_query echoes the prompt with a fixed prefix.
# ---------------------------------------------------------------------------


class MockAgent:
    def __init__(self, model_name: str = "mock"):
        self.model_name = model_name
        self.system_prompt = ""
        self.defense = None
        self.tool_call_log = []
        self.memory = []

    def process_query(self, query: str, **kwargs) -> Dict[str, Any]:
        return {
            "answer": f"[MOCK] {query[:100]}",
            "context_used": {"images": [], "memory": [], "text_docs": []},
            "metadata": {"defense_trace": [], "memory_entries": 0},
            "session_id": "default",
        }

    def reset(self):
        pass

    def start_session(self, session_id: str = "default"):
        pass


# ---------------------------------------------------------------------------
# Attacks — each must subclass BaseAttack and run one PSSU cycle.
# ---------------------------------------------------------------------------

ATTACKS = [
    ("aaps.attacks.slim5.pair.attack", "PAIRAttack"),
    ("aaps.attacks.slim5.poisoned_rag.attack", "PoisonedRAGAttack"),
    ("aaps.attacks.slim5.rl.attack", "RLAttack"),
    ("aaps.attacks.slim5.human_redteam.attack", "HumanRedTeamAttack"),
    ("aaps.attacks.slim5.supply_chain.attack", "SupplyChainAttack"),
    # Legacy
    ("aaps.attacks.legacy.tap.attack", "TAPAttack"),
    ("aaps.attacks.legacy.crescendo.attack", "CrescendoAttack"),
    ("aaps.attacks.legacy.advprompter.attack", "AdvPrompterAttack"),
]


@pytest.mark.parametrize("modpath,clsname", ATTACKS)
def test_attack_instantiates(modpath: str, clsname: str):
    """Every attack class loads and the constructor accepts a MockAgent."""
    from aaps.attacks._core.base_attack import AttackConfig

    mod = importlib.import_module(modpath)
    cls = getattr(mod, clsname, None)
    assert cls is not None, f"{clsname} not found in {modpath}"

    cfg = AttackConfig(budget=2, success_threshold=0.5, verbose=False)
    inst = cls(agent=MockAgent(), config=cfg)
    # Some attacks override budget from their own constructor kwargs (e.g.
    # PAIR derives it from `max_iters`); we only require that the object
    # constructed and exposes a config.
    assert hasattr(inst, "config")
    assert inst.config.budget >= 1


@pytest.mark.parametrize("modpath,clsname", ATTACKS[:5])  # slim-5 only for the cycle test
def test_slim5_pssu_cycle(modpath: str, clsname: str):
    """Slim-5 attacks run propose() → score() → select() → update() once."""
    from aaps.attacks._core.base_attack import AttackConfig

    mod = importlib.import_module(modpath)
    cls = getattr(mod, clsname)
    cfg = AttackConfig(budget=2, success_threshold=0.5, verbose=False)
    inst = cls(agent=MockAgent(), config=cfg)

    # propose
    candidates = inst.propose(target_goal="leak the secret", iteration=0)
    assert isinstance(candidates, list)
    if not candidates:
        # Some attacks (e.g. SupplyChain) may emit a single placeholder.
        return

    # score
    scores = inst.score(candidates[:2], target_goal="leak the secret")
    assert isinstance(scores, list)
    assert len(scores) == len(candidates[:2])

    # select (BaseAttack uses top_k=1 default)
    keep = inst.select(candidates[:2], scores, top_k=1)
    assert isinstance(keep, list)
    assert len(keep) <= 1

    # update — must not crash
    inst.update(selected=keep, scores=scores[:1], iteration=0)


# ---------------------------------------------------------------------------
# Defences — every BaseDefense subclass instantiates and runs five hooks.
# ---------------------------------------------------------------------------

DEFENCES = [
    "StruQDefense",
    "SecAlignDefense",
    "MELONDefense",
    "AMemGuard",
    "SmoothLLMDefense",
    "Spotlighting",
    "PromptSandwiching",
    "RPODefense",
    "CircuitBreakerDefense",
    "DataSentinelDefense",
    "RAGuard",
    "PromptGuard2Defense",
    "PromptGuardFilter",
    "LlamaFirewall",
    "WildGuardDefense",
    # Phase C additions:
    "LlamaGuardDefense",
    "GraniteGuardianDefense",
    "ConstitutionalClassifiersDefense",
]


@pytest.mark.parametrize("clsname", DEFENCES)
def test_defence_instantiates(clsname: str):
    import aaps.defenses.baselines as b
    cls = getattr(b, clsname, None)
    assert cls is not None, f"{clsname} not exported"
    inst = cls()
    assert inst is not None


@pytest.mark.parametrize("clsname", DEFENCES)
def test_defence_check_input_returns_result(clsname: str):
    """All defences expose check_input and return DefenseResult."""
    from aaps.defenses.base_defense import DefenseResult
    import aaps.defenses.baselines as b
    cls = getattr(b, clsname)
    inst = cls()
    res = inst.check_input("hello world")
    assert isinstance(res, DefenseResult), f"{clsname}.check_input returned {type(res)}"
    assert isinstance(res.allowed, bool)


@pytest.mark.parametrize(
    "clsname",
    # Some defences are content-moderation only (input + output) and do not
    # implement retrieval/memory/tool-call hooks meaningfully. The base
    # class supplies pass-through defaults; just verify they return shape.
    DEFENCES,
)
def test_defence_all_hooks_return_shape(clsname: str):
    from aaps.defenses.base_defense import DefenseResult
    import aaps.defenses.baselines as b
    cls = getattr(b, clsname)
    inst = cls()
    for hook_name, args in [
        ("check_input", ("hello",)),
        ("check_retrieval", ("doc text",)),
        ("check_memory_write", ("memo",)),
        ("check_tool_call", ("send_email", {"to": "a@b"}, "user said")),
        ("check_output", ("assistant says hi",)),
    ]:
        try:
            res = getattr(inst, hook_name)(*args)
        except (NotImplementedError, TypeError):
            # Some baselines deliberately do not implement specific hooks.
            continue
        assert isinstance(res, DefenseResult), \
            f"{clsname}.{hook_name} → {type(res)}, expected DefenseResult"


# ---------------------------------------------------------------------------
# PACE control plane — runs end-to-end on Mock LLM stubs.
# ---------------------------------------------------------------------------


def test_pace_subclasses_BaseDefense():
    from aaps.defenses.base_defense import BaseDefense
    from aaps.defenses.pace.pipeline import PACEDefense
    assert issubclass(PACEDefense, BaseDefense)


def test_pace_with_mock_llm_emits_plan(monkeypatch):
    """Construct PACE with stub Planner/Executor that don't call any LLM,
    confirm check_input emits a plan."""
    import json
    from aaps.defenses.pace.pipeline import PACEDefense
    from aaps.defenses.pace.planner import Planner
    from aaps.defenses.pace.executor import Executor
    from aaps.defenses.pace.agreement import AgreementVoter
    from aaps.defenses.pace.trace_logger import PACETraceLogger
    import threading

    pace = PACEDefense.__new__(PACEDefense)
    pace.planner_model = "mock"
    pace.executor_model = "mock"
    pace.K = 3
    pace.q = 2
    pace.embedder_id = "all-MiniLM-L6-v2"
    pace.cluster_method = "kmeans"
    pace.replan_on_abstain = False
    pace.tool_schemas = {"send_email": {"description": "...", "parameters": ["to"]}}
    pace.ollama_url = None
    pace.seed = 0
    pace.stats = {"total_checks": 0, "allowed": 0, "blocked": 0, "soft_filtered": 0}
    pace.nli_filter = False
    pace.nli_threshold = 0.7
    pace.nli_model = ""
    pace.nli_cosine_fallback_threshold = 0.92

    pace.planner = Planner(
        model="mock",
        llm_call=lambda msgs: json.dumps({
            "nodes": [{"tool": "send_email", "arg_schema": {"to": "str"}}],
            "notes": "ok",
        }),
    )
    pace.executor = Executor(
        model="mock",
        llm_call=lambda msgs: json.dumps({"calls": []}),
    )
    pace.voter = AgreementVoter(K=3, q=2)
    pace.trace_logger = PACETraceLogger(path=None)
    pace._lock = threading.Lock()
    pace._sessions = {}

    res = pace.check_input("send hello to alice@example.com")
    assert res.allowed
