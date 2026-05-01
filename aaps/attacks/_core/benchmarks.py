"""Academic benchmark loader for red-teaming evaluations.

Downloads and caches prompts from three standardized open-source datasets:
  - AdvBench  (Zou et al., 2023)  — 520 harmful behaviors
  - HarmBench (Mazeika et al., 2024) — ~400 standard text behaviors
  - JailbreakBench / JBB-Behaviors (Chao et al., 2024) — 100 harmful + 100 benign

Each dataset is downloaded once and cached under ``data/benchmarks/``.
All entries are normalised into a unified ``BenchmarkGoal`` dataclass.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "benchmarks"

_ADVBENCH_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks"
    "/main/data/advbench/harmful_behaviors.csv"
)
_HARMBENCH_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench"
    "/main/data/behavior_datasets/harmbench_behaviors_text_test.csv"
)

VALID_BENCHMARKS = ("advbench", "harmbench", "jailbreakbench")


@dataclass
class BenchmarkGoal:
    """A single red-teaming goal from an academic benchmark."""

    goal: str
    target: str
    category: str
    source: str
    behavior_id: str

    def to_dict(self) -> dict:
        return asdict(self)


def _ensure_cache_dir() -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


def _download(url: str, dest: Path) -> Path:
    """Download *url* to *dest* if it does not already exist."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    _ensure_cache_dir()
    logger.info("Downloading %s -> %s", url, dest)
    urllib.request.urlretrieve(url, dest)
    return dest


# ── AdvBench ────────────────────────────────────────────────────────────


def _load_advbench() -> List[BenchmarkGoal]:
    path = _download(_ADVBENCH_URL, _CACHE_DIR / "advbench.csv")
    goals: List[BenchmarkGoal] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            goals.append(
                BenchmarkGoal(
                    goal=row["goal"].strip(),
                    target=row["target"].strip(),
                    category="harmful",
                    source="advbench",
                    behavior_id=f"advbench_{idx}",
                )
            )
    return goals


# ── HarmBench ───────────────────────────────────────────────────────────


def _load_harmbench() -> List[BenchmarkGoal]:
    path = _download(_HARMBENCH_URL, _CACHE_DIR / "harmbench.csv")
    goals: List[BenchmarkGoal] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("FunctionalCategory", "standard") != "standard":
                continue
            behavior = row["Behavior"].strip()
            bid = row.get("BehaviorID", "").strip()
            cat = row.get("SemanticCategory", "harmful").strip()
            target = _behavior_to_target(behavior)
            goals.append(
                BenchmarkGoal(
                    goal=behavior,
                    target=target,
                    category=cat,
                    source="harmbench",
                    behavior_id=bid or f"harmbench_{len(goals)}",
                )
            )
    return goals


_IMPERATIVE_VERBS = (
    "write ", "create ", "generate ", "produce ", "draft ",
    "give ", "provide ", "explain ", "describe ", "outline ",
    "develop ", "design ", "compose ", "make ", "build ",
    "list ", "show ", "tell ", "teach ", "demonstrate ",
)


def _behavior_to_target(text: str) -> str:
    """Build an affirmative-prefix target from a HarmBench behavior string.

    Strips leading imperative verbs so that ``"Write a phishing email"``
    becomes ``"Sure, here is a phishing email"``.
    """
    stripped = text.strip()
    lower = stripped.lower()
    for verb in _IMPERATIVE_VERBS:
        if lower.startswith(verb):
            remainder = stripped[len(verb):]
            return f"Sure, here is {remainder}"
    if lower.startswith(("a ", "an ", "the ")):
        return f"Sure, here is {lower}"
    return f"Sure, here is {stripped[0].lower() + stripped[1:]}" if stripped else text


# ── JailbreakBench ──────────────────────────────────────────────────────


def _load_jailbreakbench() -> List[BenchmarkGoal]:
    cache_path = _CACHE_DIR / "jailbreakbench.json"
    if cache_path.exists() and cache_path.stat().st_size > 0:
        with open(cache_path, encoding="utf-8") as f:
            rows = json.load(f)
        return [BenchmarkGoal(**r) for r in rows]

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The `datasets` package is required for JailbreakBench. "
            "Install it with: pip install datasets"
        ) from exc

    _ensure_cache_dir()
    ds = load_dataset(
        "JailbreakBench/JBB-Behaviors", "behaviors", split="harmful"
    )

    goals: List[BenchmarkGoal] = []
    for row in ds:
        goals.append(
            BenchmarkGoal(
                goal=row["Goal"].strip(),
                target=row["Target"].strip(),
                category=row["Category"].strip(),
                source="jailbreakbench",
                behavior_id=f"jbb_{row['Index']}",
            )
        )

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump([g.to_dict() for g in goals], f, indent=2)

    return goals


# ── Public API ──────────────────────────────────────────────────────────

_LOADERS = {
    "advbench": _load_advbench,
    "harmbench": _load_harmbench,
    "jailbreakbench": _load_jailbreakbench,
}


def load_benchmark(
    name: str,
    *,
    categories: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
) -> List[BenchmarkGoal]:
    """Load goals from a single benchmark.

    Parameters
    ----------
    name : str
        One of ``"advbench"``, ``"harmbench"``, or ``"jailbreakbench"``.
    categories : sequence of str, optional
        Keep only goals whose ``category`` is in this set (case-insensitive).
    limit : int, optional
        Maximum number of goals to return.
    """
    name = name.lower().strip()
    if name not in _LOADERS:
        raise ValueError(
            f"Unknown benchmark {name!r}. Choose from {list(_LOADERS)}"
        )
    goals = _LOADERS[name]()

    if categories:
        cats_lower = {c.lower() for c in categories}
        goals = [g for g in goals if g.category.lower() in cats_lower]

    if limit is not None and limit > 0:
        goals = goals[:limit]

    logger.info("Loaded %d goals from %s", len(goals), name)
    return goals


def load_all_benchmarks(
    *,
    limit_per_source: Optional[int] = None,
    categories: Optional[Sequence[str]] = None,
) -> List[BenchmarkGoal]:
    """Load goals from **all** benchmarks, concatenated."""
    all_goals: List[BenchmarkGoal] = []
    for name in _LOADERS:
        try:
            goals = load_benchmark(
                name, categories=categories, limit=limit_per_source
            )
            all_goals.extend(goals)
        except Exception:
            logger.warning("Failed to load %s, skipping", name, exc_info=True)
    return all_goals


def list_categories(name: str) -> List[str]:
    """Return the sorted unique categories in a benchmark."""
    goals = load_benchmark(name)
    return sorted({g.category for g in goals})
