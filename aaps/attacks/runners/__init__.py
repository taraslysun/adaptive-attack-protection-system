"""CLI entry points for individual attack families.

Each ``run_*.py`` module is intended to be invoked with ``python -m``
or as a script.  They share argument-parsing and logging conventions
described in ``docs/wiki/runners/``.

Coverage map
------------
The following attack families have a dedicated single-family CLI here:

* ``run_gcg_attack.py``         -- :class:`attacks.adaptive.gradient_attack.gcg.GCGAttack`
* ``run_rl_attack.py``          -- :class:`attacks.adaptive.rl_attack.attack.RLAttack`
* ``run_search_attack.py``      -- :class:`attacks.adaptive.search_attack.attack.SearchAttack`
* ``run_static_attack.py``      -- :class:`attacks.static.static_attacks.StaticAttackSuite`
* ``run_human_redteam_attack.py`` -- :class:`attacks.adaptive.human_redteam.attack.HumanRedTeamAttack`
* ``run_adaptive_attacks.py``   -- pipeline runner for adaptive families
* ``run_all_attacks.py``        -- defense-OFF baseline matrix

The remaining adaptive families documented in the bibliography --
:class:`attacks.adaptive.pair.attack.PAIRAttack`,
:class:`attacks.adaptive.tap.attack.TAPAttack`,
:class:`attacks.adaptive.crescendo.attack.CrescendoAttack`,
:class:`attacks.adaptive.advprompter.attack.AdvPrompterAttack`, and
:class:`attacks.adaptive.poisoned_rag.attack.PoisonedRAGAttack` --
**do not have per-family CLIs in this folder yet** (thesis remediation
``codebase-add-runner-clis`` is intentionally deferred). They are
exercised exclusively via the matrix scripts:

* ``scripts/run_thesis_experiments.py``        (the defense-vs-attack
  matrix used for Ch.5 tables)
* ``scripts/run_realworld_suites.py``          (AgentDojo + InjecAgent)
* ``scripts/run_model_matrix.py``              (per-victim-model sweep)
* ``scripts/run_multiseed_matrix.py``          (multi-seed CIs)

To run any of those five families standalone for ad-hoc debugging,
import the class and call its :meth:`execute` method directly; thesis
prose may NOT cite a non-existent ``attacks/runners/run_pair_attack.py``
or similar file.
"""
