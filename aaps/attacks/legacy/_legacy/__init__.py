"""Archived / legacy attack modules.

Moved here as part of the PACE OpenRouter-only refactor (2026-04-25).
Compat shims at the old import paths re-export from here so existing
code continues to work (with a DeprecationWarning).

Contents:

* ``gradient_attack/`` — GCG white-box attack (requires torch + HF surrogate)
* ``gcg_variants/``     — AdaptiveGCGIPI, MultiObjective, TwoStage GCG
* ``search_attack/``    — SearchAttack / MAP-Elites (mutator LLM via Ollama)
* ``rl_attack/``        — archived RL helpers (policy.py, trainer.py)
* ``run_gcg_attack.py`` — standalone GCG runner
* ``run_search_attack.py`` — standalone search runner

Do not depend on this package from new code.
"""
