"""Supply-chain attack simulation: malicious MCP servers and skills."""

from aaps.attacks.slim5.supply_chain.mcp_code_search import CodeSearchMCP
from aaps.attacks.slim5.supply_chain.mcp_project_metrics import ProjectMetricsMCP
from aaps.attacks.slim5.supply_chain.attack import SupplyChainAttack
from aaps.attacks.slim5.supply_chain.skill_attack import SkillInjectionAttack

__all__ = [
    "CodeSearchMCP",
    "ProjectMetricsMCP",
    "SupplyChainAttack",
    "SkillInjectionAttack",
]
