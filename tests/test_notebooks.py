"""Notebook execution smoke. Marked pytest.mark.notebooks, opt-in.

Every working notebook executes end-to-end with a 60 s per-cell timeout.
Notebooks that require paid APIs, prior log directories, or removed
modules are explicitly skipped with documented reasons.

Run:
    pytest -m notebooks tests/test_notebooks.py -q
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.notebooks

REPO_ROOT = Path(__file__).resolve().parent.parent
NB_DIR = REPO_ROOT / "notebooks"


# (filename, status, reason)
NOTEBOOKS = [
    ("00_setup_and_agent.ipynb", "run", None),
    ("01_static_attacks.ipynb", "run", None),
    ("02_pair_attack.ipynb", "run", None),
    ("03_poisoned_rag_attack.ipynb", "skip", "depends on removed feature_extractors / label_trust; awaiting rewrite"),
    ("04_supply_chain_attack.ipynb", "skip", "calls Ollama / network — exceeds 60s nbconvert timeout"),
    ("05_pace_defense.ipynb", "run", None),
    ("06_benchmark_comparison.ipynb", "skip", "loads precomputed log path that does not exist in fresh clone"),
    ("07_deep_agent_demo.ipynb", "skip", "depends on removed MIMGate; awaiting rewrite"),
    ("08_agentdojo_benchmark.ipynb", "skip", "paid (OpenRouter); opt-in via L3 runner not pytest"),
    ("09_adaptive_budget_sweep.ipynb", "skip", "DefenseBenchmark API drifted; awaiting rewrite"),
    ("10_deepagent.ipynb", "skip", "AgentConfig export drifted; awaiting rewrite"),
    ("99_mim_LEGACY.ipynb", "skip", "legacy MIM stack; preserved for history"),
]


@pytest.mark.parametrize("filename,status,reason", NOTEBOOKS)
def test_notebook(filename: str, status: str, reason: str | None):
    if status == "skip":
        pytest.skip(reason or "documented skip")
    nb_path = NB_DIR / filename
    assert nb_path.exists(), f"missing {filename}"

    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute", str(nb_path),
        "--output-dir", "/tmp",
        "--ExecutePreprocessor.timeout=60",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, (
        f"{filename} failed (rc={proc.returncode}):\n"
        f"STDOUT: {proc.stdout[-500:]}\n"
        f"STDERR: {proc.stderr[-500:]}"
    )
