"""End-to-end mock tests. No model calls, no network. Exercise the wiring of:

- AgreementVoter quorum gate (PACE core invariant)
- PACEPlan / canonicalise_args
- BaseAttack scaffolding
- DefenseBenchmark instantiation
"""
from __future__ import annotations

from aaps.defenses.pace.agreement import AgreementVoter
from aaps.defenses.pace.plan import (
    PACEPlan,
    PACEPlanNode,
    ProposedToolCall,
    canonicalise_args,
)


# ---------------------------------------------------------------------------
# AgreementVoter — the headline PACE invariant
# ---------------------------------------------------------------------------


def _call(tool: str, args: dict, cluster_id: int = 0) -> ProposedToolCall:
    return ProposedToolCall(tool=tool, args=args, cluster_id=cluster_id)


def test_voter_unanimous_fires():
    """K=5 clusters all propose same call → quorum met."""
    voter = AgreementVoter(K=5, q=3)
    same = _call("send_email", {"to": "alice@example.com", "body": "hi"})
    decisions = voter.vote([[same]] * 5)
    assert len(decisions) == 1
    d = decisions[0]
    assert d.agreement == 5 >= d.q
    assert d.tool == "send_email"


def test_voter_split_below_quorum_logs_dissent():
    """K=5, q=3. Two clusters propose A, three propose B → only B fires."""
    voter = AgreementVoter(K=5, q=3)
    a = _call("send_email", {"to": "attacker@evil.io"})
    b = _call("send_email", {"to": "alice@example.com"})
    decisions = voter.vote([[a], [a], [b], [b], [b]])
    # Two distinct canonical keys.
    by_tool = {d.args_canonical: d for d in decisions}
    assert len(by_tool) == 2
    # B has 3 supporters, A has 2.
    assert {d.agreement for d in decisions} == {2, 3}
    fires = [d for d in decisions if d.agreement >= d.q]
    assert len(fires) == 1
    assert fires[0].args["to"] == "alice@example.com"


def test_voter_rejects_wrong_K():
    voter = AgreementVoter(K=5, q=3)
    same = _call("noop", {})
    try:
        voter.vote([[same]] * 4)
    except ValueError:
        return
    raise AssertionError("expected ValueError on K mismatch")


def test_canonicalise_args_is_order_insensitive():
    assert canonicalise_args("t", {"a": 1, "b": 2}) == canonicalise_args("t", {"b": 2, "a": 1})
    assert canonicalise_args("t", {"a": 1}) != canonicalise_args("t", {"a": 2})


# ---------------------------------------------------------------------------
# PACEPlan — CFI shape
# ---------------------------------------------------------------------------


def test_pace_plan_admits_listed_tool():
    plan = PACEPlan(
        nodes=[PACEPlanNode(tool="send_email", arg_schema={"to": "str", "body": "str"})],
    )
    node = plan.find_node("send_email", {"to": "alice@example.com", "body": "hi"})
    assert node is not None and node.tool == "send_email"


def test_pace_plan_rejects_unlisted_tool():
    plan = PACEPlan(
        nodes=[PACEPlanNode(tool="send_email", arg_schema={"to": "str"})],
    )
    assert plan.find_node("execute_shell", {"cmd": "rm -rf /"}) is None


# ---------------------------------------------------------------------------
# Attack scaffolding — BaseAttack import + AttackConfig defaults
# ---------------------------------------------------------------------------


def test_attack_config_round_trip():
    from aaps.attacks._core.base_attack import AttackConfig

    cfg = AttackConfig(budget=2, success_threshold=0.5)
    d = cfg.__dict__
    assert d["budget"] == 2
    assert d["success_threshold"] == 0.5


def test_pair_attack_class_loads():
    from aaps.attacks.slim5.pair.attack import PAIRAttack
    assert PAIRAttack.__name__ == "PAIRAttack"


# ---------------------------------------------------------------------------
# DefenseBenchmark — instantiation only (no model calls)
# ---------------------------------------------------------------------------


def test_defense_benchmark_class_loads():
    from aaps.evaluation.defense_benchmark import DefenseBenchmark
    assert DefenseBenchmark.__name__ == "DefenseBenchmark"
