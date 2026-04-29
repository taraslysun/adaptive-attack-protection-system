"""PACE-era external benchmark loaders.

The legacy AgentDojo and InjecAgent loaders live in
``evaluation/external_benchmarks.py``. This sub-package adds the new
PACE-relevant tool/skill-focused benchmarks the thesis matrix needs:

* :mod:`evaluation.benchmarks.agentharm` -- Andriushchenko et al. 2024,
  arXiv:2410.09377. Harmful-tool-call success-rate eval against an LLM
  agent.
* :mod:`evaluation.benchmarks.tau_bench` -- Yao et al. 2024,
  arXiv:2406.12045. Long-horizon tool-use utility benchmark; we use
  it as the **cost-of-defense** signal (does PACE tank legitimate
  task completion?).

Both loaders return :class:`evaluation.defense_benchmark.MultiSessionScenario`
objects so the existing matrix runner can consume them transparently.
Network access, large dataset downloads and external dependencies are
all *optional* -- if the dataset is missing the loader returns an empty
list rather than raising, the same convention as
``evaluation/external_benchmarks.py``.
"""

from aaps.evaluation.benchmarks.agentharm import load_agentharm_scenarios
from aaps.evaluation.benchmarks.tau_bench import load_tau_bench_scenarios

__all__ = ["load_agentharm_scenarios", "load_tau_bench_scenarios"]
