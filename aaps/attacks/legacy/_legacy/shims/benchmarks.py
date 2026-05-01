"""Backward-compatibility shim. Real module: :mod:`attacks._core.benchmarks`."""

from aaps.attacks._core.benchmarks import *  # noqa: F401,F403
from aaps.attacks._core.benchmarks import (  # noqa: F401
    VALID_BENCHMARKS,
    BenchmarkGoal,
    list_categories,
    load_all_benchmarks,
    load_benchmark,
)
