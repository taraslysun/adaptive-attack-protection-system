#!/usr/bin/env python3
"""Verify thesis matrix completeness and aggregation integrity.

Checks:
1. Expected (model, seed, tier) cells from configs/thesis_matrix.yaml exist.
2. Each discovered seed has run_metadata.json and required tier JSON outputs.
3. Optional aggregate check ensures aggregate/*.json exists per model root.

Exit codes:
    0 GREEN  - complete
    1 AMBER  - partial / missing non-critical outputs
    2 RED    - structural gaps (missing model/seed cells)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MATRIX_PATH = PROJECT_ROOT / "configs" / "thesis_matrix.yaml"


def _safe_model(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def _track_name(run_dir: Path) -> str:
    name = run_dir.name
    if name.startswith("headline_"):
        return "headline"
    if name.startswith("validation_"):
        return "validation"
    if name.startswith("ksweep_"):
        return "k_sweep"
    return ""


def _required_tier_files(tier: str) -> List[str]:
    if tier == "1":
        # agentharm.json is optional (requires HF mirror or local data)
        return [
            "memory_poisoning.json",
            "agentdojo.json",
            "injecagent.json",
            "tau_bench.json",
        ]
    if tier == "2":
        # slim5_adaptive.json is the actual output name from run_thesis_experiments
        return ["slim5_adaptive.json"]
    if tier == "3":
        # jailbreakbench goals are sampled into advbench run, not a separate file
        return ["harmbench.json", "advbench.json"]
    return []


def verify(run_dir: Path, require_aggregate: bool = False) -> Tuple[str, List[str]]:
    msgs: List[str] = []
    status = "GREEN"

    if not run_dir.exists():
        return "RED", [f"run dir does not exist: {run_dir}"]

    track = _track_name(run_dir)
    if not track:
        return "AMBER", [f"cannot infer track from dir name: {run_dir.name}"]

    cfg = yaml.safe_load(MATRIX_PATH.read_text(encoding="utf-8"))
    track_cfg = cfg.get(track, {})
    models = [m["slug"] for m in track_cfg.get("models", [])]
    seeds = [int(s) for s in track_cfg.get("seeds", [0])]
    tiers = [str(t) for t in track_cfg.get("tiers", [1, 2, 3])]
    k_dirs: List[Path] = []

    if track == "k_sweep":
        victim = track_cfg.get("victim")
        if victim:
            models = [victim]
        k_dirs = sorted(p for p in run_dir.glob("spq_K*_q*") if p.is_dir())
        if not k_dirs:
            return "RED", [f"track={track}: no spq_K*_q* directories under {run_dir}"]

    if not models:
        return "RED", [f"no models found for track '{track}' in {MATRIX_PATH}"]

    roots_to_check: List[Tuple[str, Path]] = []
    for model in models:
        safe = _safe_model(model)
        if track == "k_sweep":
            for kd in k_dirs:
                roots_to_check.append((model, kd / f"model_{safe}"))
        else:
            roots_to_check.append((model, run_dir / f"model_{safe}"))

    for model, model_root in roots_to_check:
        safe = _safe_model(model)
        if not model_root.exists():
            status = "RED"
            msgs.append(f"missing model root: {model_root}")
            continue

        for seed in seeds:
            nested_seed_dir = model_root / f"model_{safe}" / f"seed_{seed}"
            flat_seed_dir = model_root / f"seed_{seed}"
            seed_dir = nested_seed_dir if nested_seed_dir.exists() else flat_seed_dir
            if not seed_dir.exists():
                status = "RED"
                msgs.append(
                    f"missing seed cell: {nested_seed_dir} (or {flat_seed_dir})"
                )
                continue

            md = seed_dir / "run_metadata.json"
            if not md.exists():
                if status != "RED":
                    status = "AMBER"
                msgs.append(f"missing run_metadata.json: {seed_dir}")

            for tier in tiers:
                tier_dir = seed_dir / f"tier{tier}"
                if not tier_dir.exists():
                    status = "RED"
                    msgs.append(f"missing tier dir: {tier_dir}")
                    continue
                for req in _required_tier_files(tier):
                    fp = tier_dir / req
                    if not fp.exists():
                        if status != "RED":
                            status = "AMBER"
                        msgs.append(f"missing tier artifact: {fp}")

        if require_aggregate and track != "k_sweep":
            agg = model_root / "aggregate"
            for req in ("asr_with_ci.json", "psr_with_ci.json", "spq_with_ci.json"):
                fp = agg / req
                if not fp.exists():
                    if status == "GREEN":
                        status = "AMBER"
                    msgs.append(f"missing aggregate artifact: {fp}")

    if not msgs:
        msgs.append(
            f"track={track}: models={len(models)} seeds={len(seeds)} tiers={tiers} complete"
        )

    return status, msgs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path, help="Path to logs/thesis/<track_timestamp>")
    p.add_argument(
        "--require-aggregate",
        action="store_true",
        help="Also require per-model aggregate/*.json outputs",
    )
    args = p.parse_args()

    status, msgs = verify(args.run_dir, require_aggregate=args.require_aggregate)
    print(f"[verify_matrix_integrity] {status}")
    for m in msgs[:200]:
        print(f"  - {m}")
    if len(msgs) > 200:
        print(f"  - ... and {len(msgs) - 200} more")

    code = {"GREEN": 0, "AMBER": 1, "RED": 2}[status]
    sys.exit(code)


if __name__ == "__main__":
    main()
