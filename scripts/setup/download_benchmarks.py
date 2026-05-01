"""One-shot offline benchmark fetcher.

Pulls the four academic-benchmark assets the PACE thesis needs:

* AgentDojo (Debenedetti 2024): pip-installed package; we just verify it imports.
* AgentHarm (Andriushchenko 2024): HuggingFace dataset, mirrored to
  ``external/agentharm/``.
* tau-bench (Yao 2024): git submodule under ``external/tau_bench/``.
* HarmBench / AdvBench / JailbreakBench (supplementary): HuggingFace datasets,
  mirrored to ``external/<name>/``.

Run this once per machine before the first thesis-grade run; subsequent runs
read from the local mirror and require no network.

Modes:

    --check     Verify presence of every benchmark; non-zero exit if any miss.
    --download  Best-effort fetch via huggingface-cli / pip / git as available.
    (default = --check)

Environment:

    HF_TOKEN    Required for gated HuggingFace repos (AgentHarm, some HarmBench mirrors).

Plan reference: P2 #8 in ``/Users/tlysu/.claude/plans/groovy-dreaming-popcorn.md``.
"""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL = REPO_ROOT / "external"

# Each entry: (name, kind, target, hf_repo_or_url, gated)
# kind ∈ {"pip", "hf", "submodule"}
BENCHMARKS: List[Tuple[str, str, Path, str, bool]] = [
    ("agentdojo",   "pip",       EXTERNAL / "_pip_marker_agentdojo",  "agentdojo",                          False),
    ("agentharm",   "hf",        EXTERNAL / "agentharm",              "ai-safety-institute/AgentHarm",      True),
    ("poisonedrag", "submodule", EXTERNAL / "poisonedrag",            "https://github.com/sleeepeer/PoisonedRAG", False),
    ("tau_bench",   "submodule", EXTERNAL / "tau_bench",              "https://github.com/sierra-research/tau-bench",  False),
    ("harmbench",   "hf",        EXTERNAL / "harmbench",              "walledai/HarmBench",                 False),
    ("advbench",    "hf",        EXTERNAL / "advbench",               "walledai/AdvBench",                  False),
    ("jailbreakbench","hf",      EXTERNAL / "jailbreakbench",         "walledai/JailbreakBench",            False),
]


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------

OK = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
SKIP = "\033[33m·\033[0m"


def _exists_pip(pkg: str) -> bool:
    try:
        importlib.import_module(pkg)
        return True
    except ImportError:
        return False


def _exists_hf(target: Path) -> bool:
    """A HF dataset mirror exists if the target dir has any data files."""
    return target.exists() and any(target.iterdir())


def _exists_submodule(target: Path) -> bool:
    """A submodule is initialized if its dir has a .git file/dir or any files."""
    return target.exists() and any(target.iterdir())


# ---------------------------------------------------------------------------
# Check
# ---------------------------------------------------------------------------


def check_all(verbose: bool = True) -> int:
    """Print presence status; return number of missing benchmarks."""
    missing = 0
    for name, kind, target, _ref, gated in BENCHMARKS:
        if kind == "pip":
            present = _exists_pip(name)
        elif kind == "hf":
            present = _exists_hf(target)
        elif kind == "submodule":
            present = _exists_submodule(target)
        else:
            present = False
        marker = OK if present else FAIL
        if not present:
            missing += 1
        if verbose:
            extra = " (gated; needs HF_TOKEN)" if gated and not present else ""
            print(f"  {marker} {name:14s}  ({kind:9s})  {target.relative_to(REPO_ROOT) if kind != 'pip' else 'pip-installed'}{extra}")
    if verbose:
        print()
        if missing == 0:
            print(f"  All {len(BENCHMARKS)} benchmarks present.")
        else:
            print(f"  {missing}/{len(BENCHMARKS)} missing — run `python {sys.argv[0]} --download` to fetch.")
    return missing


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def _run(cmd: List[str], cwd: Path | None = None) -> bool:
    print(f"  $ {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)
        return True
    except subprocess.CalledProcessError as e:
        print(f"    FAILED: {e}")
        return False
    except FileNotFoundError as e:
        print(f"    NOT FOUND: {e}")
        return False


def download_pip(pkg: str) -> bool:
    if _exists_pip(pkg):
        print(f"  {SKIP} {pkg} already importable")
        return True
    return _run([sys.executable, "-m", "pip", "install", pkg])


def download_hf(target: Path, hf_repo: str, gated: bool) -> bool:
    if _exists_hf(target):
        print(f"  {SKIP} {target.name} already mirrored at {target}")
        return True
    if gated and not os.getenv("HF_TOKEN"):
        print(f"  {FAIL} {target.name} is gated; set HF_TOKEN before retrying")
        return False
    target.mkdir(parents=True, exist_ok=True)
    cli = shutil.which("huggingface-cli")
    if cli is None:
        return _run([sys.executable, "-m", "huggingface_hub.commands.huggingface_cli",
                     "download", "--repo-type", "dataset", hf_repo,
                     "--local-dir", str(target)])
    return _run([cli, "download", "--repo-type", "dataset", hf_repo,
                 "--local-dir", str(target)])


def download_submodule(target: Path, url: str) -> bool:
    if _exists_submodule(target):
        print(f"  {SKIP} {target.name} already present at {target}")
        return True
    target.parent.mkdir(parents=True, exist_ok=True)
    # Try git submodule add; fall back to plain clone if not in a git tree.
    if (REPO_ROOT / ".git").exists():
        rel = str(target.relative_to(REPO_ROOT))
        if _run(["git", "submodule", "add", "--depth", "1", url, rel], cwd=REPO_ROOT):
            return True
        # Already added but not initialized?
        return _run(["git", "submodule", "update", "--init", "--recursive", rel], cwd=REPO_ROOT)
    return _run(["git", "clone", "--depth", "1", url, str(target)])


def download_all() -> int:
    """Best-effort fetch every benchmark. Returns number of failures."""
    failures = 0
    for name, kind, target, ref, gated in BENCHMARKS:
        print(f"\n--- {name} ({kind}) ---")
        ok = False
        if kind == "pip":
            ok = download_pip(ref)
        elif kind == "hf":
            ok = download_hf(target, ref, gated)
        elif kind == "submodule":
            ok = download_submodule(target, ref)
        if not ok:
            failures += 1
    return failures


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description="Fetch / verify thesis benchmarks.")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true", default=True,
                   help="Default. Print presence status; non-zero exit if any missing.")
    g.add_argument("--download", action="store_true",
                   help="Best-effort fetch every missing benchmark.")
    args = p.parse_args()

    EXTERNAL.mkdir(parents=True, exist_ok=True)

    if args.download:
        failures = download_all()
        print()
        missing = check_all(verbose=True)
        return 1 if (failures or missing) else 0

    missing = check_all(verbose=True)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
