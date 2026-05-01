"""CLI orchestration scripts.

Two primary entry points stay at the top level:

* :mod:`scripts.run_thesis_experiments` -- 3-tier paper pipeline
* :mod:`scripts.run_model_matrix`       -- cross-model attack matrix

Helpers live under sub-packages:

* :mod:`scripts.setup`     -- preflight checks and KB seeding
* :mod:`scripts.reporting` -- thesis figure/table generation
* :mod:`scripts._legacy`   -- earlier orchestrators kept for reference
"""
