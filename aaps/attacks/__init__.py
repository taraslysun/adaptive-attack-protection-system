"""Attack implementations for testing agent security.

Headline adaptive-attack families (slim-5 for the PACE thesis):

* RL/GRPO (black-box)                  -- :mod:`attacks.adaptive.rl_attack`
* Human red-team simulation            -- :mod:`attacks.adaptive.human_redteam`
* PAIR (Chao 2023, arXiv:2310.08419)   -- :mod:`attacks.adaptive.pair`
* PoisonedRAG (Zou 2024, arXiv:2402.07867) -- :mod:`attacks.adaptive.poisoned_rag`
* Supply-chain (MCP + skills)          -- :mod:`attacks.supply_chain`

Legacy / optional (moved to ``attacks._legacy``):

* Gradient-based GCG (white-box)       -- :mod:`attacks._legacy.gradient_attack`
* Adaptive GCG variants                -- :mod:`attacks._legacy.gcg_variants`
* Search/MAP-Elites                    -- :mod:`attacks._legacy.search_attack`

Still importable but not in headline matrix:

* Static prompts (baseline)            -- :mod:`attacks.static.static_attacks`
* TAP (Mehrotra 2024)                  -- :mod:`attacks.adaptive.tap`
* Crescendo (Russinovich 2024)         -- :mod:`attacks.adaptive.crescendo`
* AdvPrompter (Paulus 2024)            -- :mod:`attacks.adaptive.advprompter`

Shared infrastructure lives under :mod:`attacks._core` and is
re-exported here for backwards compatibility.
"""

from aaps.attacks._core.base_attack import AttackConfig, AttackResult, BaseAttack
from aaps.attacks._core.local_agent import LocalAgent
from aaps.attacks._core.config import (
    LITELLM_ATTACKER_MODEL,
    LITELLM_JUDGE_MODEL,
    OLLAMA_HELPER_MODEL,
    OLLAMA_JUDGE_MODEL,
    OLLAMA_TARGET_MODEL,
    OLLAMA_URL,
    OPENROUTER_API_KEY,
    OPENROUTER_ATTACKER_MODEL,
    OPENROUTER_JUDGE_MODEL,
    OPENROUTER_VICTIM_MODEL,
    USE_LITELLM,
)
from aaps.attacks._core.model_registry import (
    ALL_MODELS,
    NON_TOOL,
    TOOL_CAPABLE,
    get_available_models,
    get_model_endpoint,
    is_remote,
    is_tool_capable,
    load_models_config,
    register_model,
)
from aaps.attacks.legacy.static.static_attacks import StaticAttackSuite
from aaps.attacks.slim5.rl.attack import RLAttack
from aaps.attacks.slim5.human_redteam.attack import HumanRedTeamAttack

try:
    from aaps.attacks.slim5.pair.attack import PAIRAttack
except ImportError:
    PAIRAttack = None

try:
    from aaps.attacks.legacy.tap.attack import TAPAttack
except ImportError:
    TAPAttack = None

try:
    from aaps.attacks.slim5.poisoned_rag.attack import PoisonedRAGAttack
except ImportError:
    PoisonedRAGAttack = None

try:
    from aaps.attacks.legacy.crescendo.attack import CrescendoAttack
except ImportError:
    CrescendoAttack = None

try:
    from aaps.attacks.legacy.advprompter.attack import AdvPrompterAttack
except ImportError:
    AdvPrompterAttack = None

try:
    from aaps.attacks.slim5.supply_chain.attack import SupplyChainAttack
except ImportError:
    SupplyChainAttack = None

__all__ = [
    "BaseAttack",
    "AttackResult",
    "AttackConfig",
    "LocalAgent",
    "StaticAttackSuite",
    "RLAttack",
    "HumanRedTeamAttack",
    "PAIRAttack",
    "TAPAttack",
    "PoisonedRAGAttack",
    "CrescendoAttack",
    "AdvPrompterAttack",
    "SupplyChainAttack",
    "LITELLM_ATTACKER_MODEL",
    "LITELLM_JUDGE_MODEL",
    "OLLAMA_HELPER_MODEL",
    "OLLAMA_JUDGE_MODEL",
    "OLLAMA_TARGET_MODEL",
    "OLLAMA_URL",
    "OPENROUTER_API_KEY",
    "OPENROUTER_ATTACKER_MODEL",
    "OPENROUTER_JUDGE_MODEL",
    "OPENROUTER_VICTIM_MODEL",
    "USE_LITELLM",
    "ALL_MODELS",
    "NON_TOOL",
    "TOOL_CAPABLE",
    "get_available_models",
    "get_model_endpoint",
    "is_remote",
    "is_tool_capable",
    "load_models_config",
    "register_model",
]
