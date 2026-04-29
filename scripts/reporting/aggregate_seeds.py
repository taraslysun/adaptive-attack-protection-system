"""Aggregate per-seed thesis runs into a single matrix with bootstrap CIs.

Thesis remediation ``run-full-matrix-with-stats``. The base runner
(``scripts/run_thesis_experiments.py``) consumes only ``seeds[0]`` per
invocation. To get multi-seed point estimates with bootstrap 95%
confidence intervals, the workflow is:

  1. Invoke the runner once per seed, e.g. ::

         for s in 0 1 2; do \\
             python scripts/run_thesis_experiments.py \\
                 --target-model qwen3:8b \\
                 --judge-backend keyword \\
                 --seeds $s \\
                 --out logs/thesis/$(date +%H%M-%d%m%Y)/seed_$s ; \\
         done

  2. Run this aggregator over the parent dir, e.g. ::

         python scripts/reporting/aggregate_seeds.py \\
             logs/thesis/<ts>/

It scans ``<run_dir>/seed_*/tier{1,2,3}/*.json`` for two formats and
builds two multi-seed tables:

* ``asr_matrix`` (tier 2 + tier 3 + quickmatrix) -- ``defense -> attack -> mean ASR``.
* ``psr`` / ``psr_keyword`` / ``psr_judge`` (tier 1 multi-session) --
  ``defense -> mean PSR`` (no attack dimension; the attack is implicit
  per scenario suite).

Outputs::

  <run_dir>/aggregate/asr_with_ci.json
  <run_dir>/aggregate/asr_with_ci.md
  <run_dir>/aggregate/psr_with_ci.json
  <run_dir>/aggregate/psr_with_ci.md
  <run_dir>/aggregate/run_metadata.json   # union of per-seed metadata

The CIs are percentile bootstrap 95% computed across seeds at the
``mean ASR / PSR`` level (each seed contributes one point estimate per
cell). With 3 seeds this is wide on purpose; that *is* the honest
answer.
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _bootstrap_ci(
    points: List[float],
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    rng: Optional[random.Random] = None,
) -> Dict[str, float]:
    if not points:
        return {"point": 0.0, "low": 0.0, "high": 0.0, "n": 0}
    rng = rng or random.Random(0)
    n = len(points)
    point = sum(points) / n
    if n == 1:
        return {"point": point, "low": point, "high": point, "n": 1}
    means: List[float] = []
    for _ in range(n_resamples):
        means.append(
            sum(points[rng.randrange(n)] for _ in range(n)) / n
        )
    means.sort()
    alpha = (1.0 - confidence) / 2.0
    lo_idx = max(0, int(math.floor(alpha * n_resamples)))
    hi_idx = min(
        n_resamples - 1, int(math.ceil((1.0 - alpha) * n_resamples)) - 1
    )
    return {
        "point": float(point),
        "low": float(means[lo_idx]),
        "high": float(means[hi_idx]),
        "n": int(n),
    }


def _load_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _iter_tier_files(seed_dir: Path) -> Iterable[Tuple[str, dict]]:
    for path in seed_dir.rglob("*.json"):
        if path.name in {"run_metadata.json", "summary.json"}:
            continue
        if path.parent.name not in {"tier1", "tier2", "tier3"}:
            continue
        data = _load_json(path)
        if not isinstance(data, dict):
            continue
        rel = path.relative_to(seed_dir)
        yield str(rel), data


def _matrix_to_points(
    seed_data: List[Tuple[int, str, dict]],
) -> Dict[Tuple[str, str, str], List[float]]:
    """Returns {(rel_path, defense, attack): [seed0_value, seed1_value, ...]}.

    Only consumes records that have an ``asr_matrix`` key (tier 2/3,
    quickmatrix). PSR-style tier1 records are aggregated separately
    via :func:`_psr_to_points`.
    """
    bag: Dict[Tuple[str, str, str], List[float]] = {}
    for seed, rel, data in seed_data:
        asr = data.get("asr_matrix") or {}
        if not isinstance(asr, dict):
            continue
        for defense, row in asr.items():
            if not isinstance(row, dict):
                continue
            for attack, val in row.items():
                try:
                    bag.setdefault((rel, defense, attack), []).append(float(val))
                except (TypeError, ValueError):
                    continue
    return bag


def _psr_to_points(
    seed_data: List[Tuple[int, str, dict]],
) -> Dict[Tuple[str, str, str], List[float]]:
    """Returns {(rel_path, defense, metric): [seed0_value, seed1_value, ...]}.

    Consumes the tier1 multi-session schema (``psr``, ``psr_keyword``,
    ``psr_judge``); each is a ``defense -> float`` map.
    """
    bag: Dict[Tuple[str, str, str], List[float]] = {}
    metric_keys = ("psr", "psr_keyword", "psr_judge")
    for seed, rel, data in seed_data:
        if not any(k in data for k in metric_keys):
            continue
        for metric in metric_keys:
            row = data.get(metric)
            if not isinstance(row, dict):
                continue
            for defense, val in row.items():
                try:
                    bag.setdefault((rel, defense, metric), []).append(float(val))
                except (TypeError, ValueError):
                    continue
    return bag


def _spq_to_points(
    seed_data: List[Tuple[int, str, dict]],
) -> Dict[Tuple[str, str, str], List[float]]:
    """Returns {(rel_path, defense, spq_metric): [seed0, seed1, ...]}.

    Consumes the new ``spq_metrics`` block written by
    :func:`evaluation.defense_benchmark._spq_summary`. Histogram bins
    are flattened to ``quorum_margin_<bin>`` keys so the per-seed
    aggregation reuses the same scalar machinery as ASR/PSR.
    """
    bag: Dict[Tuple[str, str, str], List[float]] = {}
    scalar_keys = (
        "cfi_violation_count",
        "abstain_rate",
        "replan_rate",
        "fire_rate",
        "quorum_margin_mean",
        "plan_deviation_rate",
        "n_calls",
    )
    for seed, rel, data in seed_data:
        spq = data.get("spq_metrics") or {}
        if not isinstance(spq, dict):
            continue
        for defense, summary in spq.items():
            if not isinstance(summary, dict):
                continue
            for k in scalar_keys:
                v = summary.get(k)
                if v is None:
                    continue
                try:
                    bag.setdefault((rel, defense, k), []).append(float(v))
                except (TypeError, ValueError):
                    continue
            hist = summary.get("quorum_margin_histogram") or {}
            if isinstance(hist, dict):
                for bin_name, count in hist.items():
                    try:
                        bag.setdefault(
                            (rel, defense, f"quorum_margin_{bin_name}"),
                            [],
                        ).append(float(count))
                    except (TypeError, ValueError):
                        continue
    return bag


def aggregate(run_dir: Path) -> Dict[str, Any]:
    seed_dirs = sorted(p for p in run_dir.glob("seed_*") if p.is_dir())
    if not seed_dirs:
        raise RuntimeError(
            f"No seed_* subdirs found under {run_dir}. Run the thesis "
            "runner once per seed with --out <run_dir>/seed_<n>/."
        )
    seed_data: List[Tuple[int, str, dict]] = []
    metadata: Dict[str, Any] = {"seeds": [], "per_seed_metadata": {}}
    for seed_dir in seed_dirs:
        try:
            seed = int(seed_dir.name.split("_", 1)[1])
        except Exception:
            continue
        meta_path = seed_dir / "run_metadata.json"
        if meta_path.exists():
            metadata["per_seed_metadata"][seed] = _load_json(meta_path) or {}
        for rel, data in _iter_tier_files(seed_dir):
            seed_data.append((seed, rel, data))
        metadata["seeds"].append(seed)

    asr_bag = _matrix_to_points(seed_data)
    psr_bag = _psr_to_points(seed_data)
    spq_bag = _spq_to_points(seed_data)

    asr_grouped: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for (rel, defense, attack), points in asr_bag.items():
        ci = _bootstrap_ci(points)
        asr_grouped.setdefault(rel, {}).setdefault(defense, {})[attack] = {
            "point": ci["point"],
            "ci95_low": ci["low"],
            "ci95_high": ci["high"],
            "n_seeds": ci["n"],
            "seed_points": points,
        }
    psr_grouped: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for (rel, defense, metric), points in psr_bag.items():
        ci = _bootstrap_ci(points)
        psr_grouped.setdefault(rel, {}).setdefault(defense, {})[metric] = {
            "point": ci["point"],
            "ci95_low": ci["low"],
            "ci95_high": ci["high"],
            "n_seeds": ci["n"],
            "seed_points": points,
        }

    spq_grouped: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for (rel, defense, metric), points in spq_bag.items():
        ci = _bootstrap_ci(points)
        spq_grouped.setdefault(rel, {}).setdefault(defense, {})[metric] = {
            "point": ci["point"],
            "ci95_low": ci["low"],
            "ci95_high": ci["high"],
            "n_seeds": ci["n"],
            "seed_points": points,
        }

    return {
        "metadata": metadata,
        "asr_with_ci": asr_grouped,
        "psr_with_ci": psr_grouped,
        "spq_with_ci": spq_grouped,
    }


def _format_cell(cell: Optional[Dict[str, Any]]) -> str:
    if cell is None:
        return "-"
    return (
        f"{cell['point']*100:.1f}% "
        f"[{cell['ci95_low']*100:.1f}, "
        f"{cell['ci95_high']*100:.1f}]"
    )


def write_markdown(agg: Dict[str, Any], out_md: Path) -> None:
    lines: List[str] = [
        "# Thesis matrix -- per-seed mean ASR with 95% bootstrap CI\n",
        "Each cell is `point [low, high]` over seeds; `n` = number of "
        "seeds; one section per benchmark file.",
        "",
        f"Seeds: {agg['metadata']['seeds']}",
        "",
    ]
    for rel, defs in agg["asr_with_ci"].items():
        lines.append(f"## {rel}")
        lines.append("")
        attacks = sorted({a for row in defs.values() for a in row.keys()})
        if not attacks:
            continue
        header = "| Defense | n | " + " | ".join(attacks) + " |"
        sep = "|---" * (2 + len(attacks)) + "|"
        lines.append(header)
        lines.append(sep)
        for defense in sorted(defs.keys()):
            row = defs[defense]
            n = max(
                (cell.get("n_seeds", 0) for cell in row.values()), default=0
            )
            cells = [_format_cell(row.get(a)) for a in attacks]
            lines.append(
                f"| {defense} | {n} | " + " | ".join(cells) + " |"
            )
        lines.append("")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))


def write_psr_markdown(agg: Dict[str, Any], out_md: Path) -> None:
    lines: List[str] = [
        "# Tier 1 multi-session -- per-seed mean PSR with 95% bootstrap CI\n",
        "Three definitions per cell: `psr` (canonical = keyword OR judge),"
        " `psr_keyword` (substring), `psr_judge` (LLM-judge). Each cell is"
        " `point [low, high]` over seeds; `n` = number of seeds. PSR is"
        " *attack success* per the AIS convention -- higher = worse for"
        " the defense.",
        "",
        f"Seeds: {agg['metadata']['seeds']}",
        "",
    ]
    for rel, defs in agg.get("psr_with_ci", {}).items():
        lines.append(f"## {rel}")
        lines.append("")
        metrics = ("psr", "psr_keyword", "psr_judge")
        header = "| Defense | n | " + " | ".join(metrics) + " |"
        sep = "|---" * (2 + len(metrics)) + "|"
        lines.append(header)
        lines.append(sep)
        for defense in sorted(defs.keys()):
            row = defs[defense]
            n = max(
                (cell.get("n_seeds", 0) for cell in row.values()), default=0
            )
            cells = [_format_cell(row.get(m)) for m in metrics]
            lines.append(
                f"| {defense} | {n} | " + " | ".join(cells) + " |"
            )
        lines.append("")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))


def write_spq_markdown(agg: Dict[str, Any], out_md: Path) -> None:
    """Emit the PACE telemetry table.

    The first column the thesis cares about is
    ``cfi_violation_count`` (acceptance criterion a in
    ``docs/design/spq.md`` §9: must be 0 across the matrix).
    The remaining columns are the quorum-margin histogram bins plus
    the rate decompositions (fire / abstain / replan).
    """
    lines: List[str] = [
        "# PACE telemetry -- per-seed mean with 95% bootstrap CI\n",
        "Headline: `cfi_violation_count` MUST be 0 (acceptance criterion a).",
        "Quorum-margin histogram is the per-cell distribution of `agreement / K`",
        "across executed tool-call decisions; high mass at 0.8-1.0 means",
        "broad cluster agreement, low mass means contested decisions.",
        "",
        f"Seeds: {agg['metadata']['seeds']}",
        "",
    ]
    for rel, defs in agg.get("spq_with_ci", {}).items():
        lines.append(f"## {rel}")
        lines.append("")
        metrics = sorted({m for row in defs.values() for m in row.keys()})
        if not metrics:
            continue
        header = "| Defense | n | " + " | ".join(metrics) + " |"
        sep = "|---" * (2 + len(metrics)) + "|"
        lines.append(header)
        lines.append(sep)
        for defense in sorted(defs.keys()):
            row = defs[defense]
            n = max(
                (cell.get("n_seeds", 0) for cell in row.values()), default=0
            )
            cells = [_format_cell(row.get(m)) for m in metrics]
            lines.append(
                f"| {defense} | {n} | " + " | ".join(cells) + " |"
            )
        lines.append("")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "usage: aggregate_seeds.py <run_dir>  "
            "(<run_dir>/seed_0/, seed_1/, ...)"
        )
        sys.exit(2)
    run_dir = Path(sys.argv[1]).resolve()
    if not run_dir.exists():
        print(f"run dir not found: {run_dir}")
        sys.exit(1)
    agg = aggregate(run_dir)
    out_dir = run_dir / "aggregate"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "asr_with_ci.json").write_text(
        json.dumps(
            {"metadata": agg["metadata"], "asr_with_ci": agg["asr_with_ci"]},
            indent=2,
            default=str,
        )
    )
    write_markdown(agg, out_dir / "asr_with_ci.md")
    if agg.get("psr_with_ci"):
        (out_dir / "psr_with_ci.json").write_text(
            json.dumps(
                {
                    "metadata": agg["metadata"],
                    "psr_with_ci": agg["psr_with_ci"],
                },
                indent=2,
                default=str,
            )
        )
        write_psr_markdown(agg, out_dir / "psr_with_ci.md")
    if agg.get("spq_with_ci"):
        (out_dir / "spq_with_ci.json").write_text(
            json.dumps(
                {
                    "metadata": agg["metadata"],
                    "spq_with_ci": agg["spq_with_ci"],
                },
                indent=2,
                default=str,
            )
        )
        write_spq_markdown(agg, out_dir / "spq_with_ci.md")
    (out_dir / "run_metadata.json").write_text(
        json.dumps(agg["metadata"], indent=2, default=str)
    )
    print(
        f"[aggregate] wrote {out_dir}/asr_with_ci.{{json,md}}"
        + (" + psr_with_ci.{json,md}" if agg.get("psr_with_ci") else "")
        + (" + spq_with_ci.{json,md}" if agg.get("spq_with_ci") else "")
    )


if __name__ == "__main__":
    main()
