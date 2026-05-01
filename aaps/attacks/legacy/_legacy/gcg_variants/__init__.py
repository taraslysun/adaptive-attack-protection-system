"""Adaptive GCG variants targeting the Agent Integrity Stack.

Three attack variants whose design follows the line of work on adaptive
GCG against indirect-prompt-injection (IPI) defenses described as
"Zhan et al. 2025" in our internal notes (paraphrase + injection
two-stage GCG with explicit awareness of structured-prompt and probe
defenses). The actual published reference for that line of work is
NOT yet in ``Overleaf/bibliography.bib``; cite it as
``% TODO[bib]: zhan2025adaptive-gcg-ipi`` (see
``docs/bibliography_justification.md`` "Forecast"). Until the bibkey
is added, prose may also use the umbrella adaptive-attack
methodology of Nasr *et al.* "The Attacker Moves Second"
(arXiv:2510.09023, bibkey ``nasr2025attacker``) as the citation
anchor:

* :class:`AdaptiveGCGIPIAttack`     -- targets L1/L2 (channel + probe)
* :class:`MultiObjectiveGCGAttack`  -- targets L2 (probe-bypass + behavior)
* :class:`TwoStageGCGAttack`        -- targets L1 (paraphrase-then-inject)

All three subclass :class:`attacks.adaptive.gradient_attack.gcg.GCGAttack` so
that they share the same surrogate-loading, candidate-sampling and
transfer machinery.  Each variant overrides only what is necessary:

* prompt template (``initial_prompt``)
* loss (``_builtin_forward_loss``)
* transfer vector (user query vs. memory plant vs. paraphrase prefix)

The variants intentionally force the built-in GCG fallback path
(``prefer_builtin = True``) so that their custom losses are honoured
even when nanoGCG is installed.
"""

try:
    from aaps.attacks.legacy.gcg_variants.adaptive_gcg_ipi import AdaptiveGCGIPIAttack
    from aaps.attacks.legacy.gcg_variants.multi_objective_gcg import MultiObjectiveGCGAttack
    from aaps.attacks.legacy.gcg_variants.two_stage_gcg import TwoStageGCGAttack

    __all__ = [
        "AdaptiveGCGIPIAttack",
        "MultiObjectiveGCGAttack",
        "TwoStageGCGAttack",
    ]
except ImportError:  # pragma: no cover - torch missing
    __all__ = []
