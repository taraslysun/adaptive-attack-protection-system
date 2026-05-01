"""Typed PACEPlan DSL and tool-call canonicalisation.

A PACEPlan is a directed acyclic graph (DAG) of allowed tool calls
emitted by the Planner LLM (see ``defenses/pace/planner.py``) **before**
any untrusted byte enters the agent loop. The CFI gate
(``defenses/pace/pipeline.py::PACEDefense.check_tool_call``) validates
every executed tool call against this plan; the Quorum gate
(``defenses/pace/agreement.py``) requires that at least ``q`` of ``K``
Executor cluster-fillings agree on a call before it fires.

The DSL is intentionally narrow:

* a node is a triple ``(tool_name, arg_schema, post_condition)``;
* ``arg_schema`` is a JSON-schema-like ``{name -> type_string}`` map;
* ``post_condition`` is free-form text the planner uses to remind itself
  of the expected effect (not enforced; logged for audit).

Edges express *ordering hints* (``after``); PACE does not currently
schedule based on them but keeps them for future re-planning.

The class is deliberately small and depends only on the standard
library so it can be imported from any process (the trace logger
serialises it, the harness diffs it, the report writer reads it).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


_ARG_TYPES = {"str", "int", "float", "bool", "list[str]", "list[int]", "dict", "any"}


@dataclass(frozen=True)
class PACEPlanNode:
    """One allowed tool call in a PACEPlan."""

    tool: str
    arg_schema: Dict[str, str]
    post_condition: str = ""
    after: Sequence[str] = field(default_factory=tuple)

    def matches_call(self, tool: str, args: Dict[str, Any]) -> bool:
        """Return True if ``(tool, args)`` is admissible under this node."""
        if tool != self.tool:
            return False
        for arg_name, arg_type in self.arg_schema.items():
            if arg_name not in args:
                return False
            if not _type_compatible(args[arg_name], arg_type):
                return False
        return True


def _type_compatible(value: Any, declared: str) -> bool:
    """Permissive type-compatibility used by the CFI gate.

    Strings that JSON-parse to the declared type are accepted (the
    LocalAgent regex-based tool detector emits all arg values as raw
    strings; the CFI invariant in ``docs/design/spq.md`` §2.1 is
    primarily about *tool name + arg-name presence*, with type
    matching as a soft signal). The Quorum gate carries the
    semantic enforcement weight.
    """
    declared = declared.strip().lower()
    if declared == "any":
        return True
    if declared == "str":
        return True  # everything stringifies; reject only via Quorum/canonicalise
    if declared == "int":
        if isinstance(value, int) and not isinstance(value, bool):
            return True
        return _string_parses_to(value, int)
    if declared == "float":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        return _string_parses_to(value, float)
    if declared == "bool":
        if isinstance(value, bool):
            return True
        if isinstance(value, str):
            return value.strip().lower() in {"true", "false"}
        return False
    if declared == "dict":
        if isinstance(value, dict):
            return True
        if isinstance(value, str):
            return _string_parses_to(value, dict)
        return False
    if declared.startswith("list["):
        inner = declared[5:-1]
        if isinstance(value, list):
            return all(_type_compatible(v, inner) for v in value)
        if isinstance(value, str):
            import json as _json
            try:
                parsed = _json.loads(value)
            except Exception:
                return True  # heuristic detector dumped a raw string; accept softly
            if isinstance(parsed, list):
                return all(_type_compatible(v, inner) for v in parsed)
            return True
        return False
    return True


def _string_parses_to(value: Any, target_type: type) -> bool:
    if not isinstance(value, str):
        return False
    import json as _json
    try:
        parsed = _json.loads(value)
    except Exception:
        try:
            parsed = target_type(value)
            return isinstance(parsed, target_type)
        except Exception:
            return False
    return isinstance(parsed, target_type)


@dataclass
class PACEPlan:
    """A typed DAG of allowed tool calls produced by the Planner.

    The plan is content-addressed via :py:attr:`plan_id` so trace
    records can reference it without re-emitting the full DAG.
    """

    nodes: List[PACEPlanNode] = field(default_factory=list)
    user_request: str = ""
    notes: str = ""

    @property
    def plan_id(self) -> str:
        payload = {
            "nodes": [
                {
                    "tool": n.tool,
                    "arg_schema": n.arg_schema,
                    "post_condition": n.post_condition,
                    "after": list(n.after),
                }
                for n in self.nodes
            ],
            "user_request": self.user_request,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        return digest[:12]

    def find_node(self, tool: str, args: Dict[str, Any]) -> Optional[PACEPlanNode]:
        for node in self.nodes:
            if node.matches_call(tool, args):
                return node
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "user_request": self.user_request,
            "notes": self.notes,
            "nodes": [
                {
                    "tool": n.tool,
                    "arg_schema": n.arg_schema,
                    "post_condition": n.post_condition,
                    "after": list(n.after),
                }
                for n in self.nodes
            ],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PACEPlan":
        nodes = [
            PACEPlanNode(
                tool=str(n["tool"]),
                arg_schema={str(k): str(v) for k, v in (n.get("arg_schema") or {}).items()},
                post_condition=str(n.get("post_condition", "")),
                after=tuple(n.get("after", []) or []),
            )
            for n in payload.get("nodes", [])
        ]
        return cls(
            nodes=nodes,
            user_request=str(payload.get("user_request", "")),
            notes=str(payload.get("notes", "")),
        )


@dataclass(frozen=True)
class ProposedToolCall:
    """A concrete tool call proposed by an Executor for one cluster."""

    tool: str
    args: Dict[str, Any]
    cluster_id: int
    rationale: str = ""

    @property
    def args_canonical(self) -> str:
        return canonicalise_args(self.tool, self.args)


def canonicalise_args(tool: str, args: Dict[str, Any]) -> str:
    """Return a stable, comparable canonical form for ``(tool, args)``.

    Two tool calls vote for the same action iff their canonical forms
    are equal. The canonicaliser:

    * lower-cases tool names,
    * sorts dict keys,
    * trims and collapses whitespace in string values,
    * sorts list values (set semantics — order should not matter for
      recipients lists, etc.).

    This is deliberately conservative; over-collapsing could make
    quorum trivially achievable, so we err on the side of "different
    string -> different vote".
    """
    norm: Dict[str, Any] = {}
    for key in sorted(args.keys()):
        norm[str(key).lower()] = _normalise_value(args[key])
    payload = {"tool": tool.strip().lower(), "args": norm}
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _normalise_value(v: Any) -> Any:
    """Recursive value normaliser used by :func:`canonicalise_args`.

    Strings that JSON-parse to a list / dict are re-interpreted as the
    parsed value so a regex-based tool detector emitting
    ``recipients='["alice@x"]'`` votes equivalently to a structured
    Executor emitting ``recipients=['alice@x']``. Without this, the
    Quorum gate would treat them as different votes purely on source
    format, which is a paper bug not a security feature.
    """
    if isinstance(v, str):
        stripped = v.strip()
        if stripped.startswith(("[", "{")):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, (list, dict)):
                    return _normalise_value(parsed)
            except Exception:
                pass
        return re.sub(r"\s+", " ", stripped)
    if isinstance(v, list):
        normalised = [_normalise_value(item) for item in v]
        try:
            return sorted(normalised, key=lambda x: json.dumps(x, sort_keys=True))
        except TypeError:
            return normalised
    if isinstance(v, dict):
        return {str(k).lower(): _normalise_value(val) for k, val in sorted(v.items())}
    return v
