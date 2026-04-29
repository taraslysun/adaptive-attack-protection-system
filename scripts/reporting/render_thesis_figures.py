#!/usr/bin/env python3
"""Render thesis-ready figures from aggregated experiment data.

Usage::

    python scripts/reporting/render_thesis_figures.py logs/thesis/<ts>/ \\
        --out Overleaf/Figures/Generated/

Produces PDF figures:
    F1  asr_bars.pdf            — Slim-5 ASR bar chart (defense x attack)
    F2  latency_bars.pdf        — p50 latency per defense
    F3  fpr_bars.pdf            — FPR per defense on benign tasks
    F4  k_sweep_heatmap.pdf     — K-sweep ASR heatmap (K x q)
    F5  quorum_histogram.pdf    — Quorum-margin distribution
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _find(run_dir: Path, relpath: str) -> Optional[Path]:
    d = run_dir / relpath
    if d.exists():
        return d
    for md in sorted(run_dir.glob("model_*")):
        c = md / relpath
        if c.exists():
            return c
    return None


def render_f1_asr_bars(run_dir: Path, out_dir: Path) -> bool:
    """F1: Slim-5 ASR grouped bars."""
    src = _find(run_dir, "aggregate/asr_with_ci.json")
    if src is None:
        return False
    data = _load_json(src)
    if not data:
        return False

    defenses: List[str] = []
    attacks: List[str] = []
    matrix: Dict[str, Dict[str, float]] = {}

    for file_key, file_data in data.items():
        if not isinstance(file_data, dict):
            continue
        for defense, dinfo in file_data.items():
            if isinstance(dinfo, dict) and "asr" in dinfo:
                defenses.append(defense)
                attacks.append(file_key)
                matrix.setdefault(defense, {})[file_key] = dinfo["asr"]

    defenses = sorted(set(defenses))
    attacks = sorted(set(attacks))
    if not defenses or not attacks:
        return False

    x = np.arange(len(defenses))
    width = 0.8 / max(len(attacks), 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, atk in enumerate(attacks):
        vals = [matrix.get(d, {}).get(atk, 0) * 100 for d in defenses]
        ax.bar(x + i * width, vals, width, label=atk.split("/")[-1])

    ax.set_ylabel("ASR (%)")
    ax.set_xticks(x + width * len(attacks) / 2)
    ax.set_xticklabels(defenses, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("Slim-5 Adaptive Attack Success Rate")
    fig.tight_layout()
    fig.savefig(out_dir / "asr_bars.pdf")
    plt.close(fig)
    return True


def render_f2_latency(run_dir: Path, out_dir: Path) -> bool:
    """F2: Latency bars per defense."""
    src = _find(run_dir, "cost_of_defense.json")
    if src is None:
        return False
    data = _load_json(src)
    if not data:
        return False

    defenses_data = data.get("defenses", data)
    if not isinstance(defenses_data, dict):
        return False

    names = sorted(defenses_data.keys())
    vals = [defenses_data[n].get("p50_latency", 0) if isinstance(defenses_data[n], dict) else 0
            for n in names]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(names, vals)
    ax.set_xlabel("p50 Latency (s)")
    ax.set_title("Defense Latency Overhead")
    fig.tight_layout()
    fig.savefig(out_dir / "latency_bars.pdf")
    plt.close(fig)
    return True


def render_f3_fpr(run_dir: Path, out_dir: Path) -> bool:
    """F3: FPR bars."""
    src = _find(run_dir, "cost_of_defense.json")
    if src is None:
        return False
    data = _load_json(src)
    if not data:
        return False

    defenses_data = data.get("defenses", data)
    if not isinstance(defenses_data, dict):
        return False

    names = sorted(defenses_data.keys())
    vals = [defenses_data[n].get("fpr", 0) * 100 if isinstance(defenses_data[n], dict) else 0
            for n in names]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(names, vals)
    ax.set_xlabel("FPR (%)")
    ax.set_title("False Positive Rate on Benign Tasks")
    fig.tight_layout()
    fig.savefig(out_dir / "fpr_bars.pdf")
    plt.close(fig)
    return True


def render_f4_k_sweep(run_dir: Path, out_dir: Path) -> bool:
    """F4: K-sweep heatmap."""
    src = _find(run_dir, "spq_k_sweep_summary.json")
    if src is None:
        return False
    data = _load_json(src)
    if not data:
        return False

    cells = data.get("cells", [])
    if not isinstance(cells, list) or not cells:
        return False

    ks = sorted(set(c.get("K", 0) for c in cells))
    qs = sorted(set(c.get("q", 0) for c in cells))
    grid = np.zeros((len(ks), len(qs)))
    for c in cells:
        ki = ks.index(c["K"]) if c["K"] in ks else 0
        qi = qs.index(c["q"]) if c["q"] in qs else 0
        grid[ki][qi] = c.get("asr", 0) * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(len(qs)))
    ax.set_xticklabels([str(q) for q in qs])
    ax.set_yticks(range(len(ks)))
    ax.set_yticklabels([str(k) for k in ks])
    ax.set_xlabel("q (quorum)")
    ax.set_ylabel("K (clusters)")
    ax.set_title("ASR (%) by K and q")
    for i in range(len(ks)):
        for j in range(len(qs)):
            ax.text(j, i, f"{grid[i][j]:.0f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="ASR (%)")
    fig.tight_layout()
    fig.savefig(out_dir / "k_sweep_heatmap.pdf")
    plt.close(fig)
    return True


def render_f5_quorum_histogram(run_dir: Path, out_dir: Path) -> bool:
    """F5: Quorum-margin histogram."""
    src = _find(run_dir, "aggregate/spq_with_ci.json")
    if src is None:
        return False
    data = _load_json(src)
    if not data:
        return False

    margins: List[float] = []
    for victim, info in data.items():
        if isinstance(info, dict):
            hist = info.get("quorum_margin_histogram", {})
            if isinstance(hist, dict):
                for bin_label, count in hist.items():
                    try:
                        val = float(bin_label.replace("quorum_margin_", ""))
                        margins.extend([val] * int(count))
                    except (ValueError, TypeError):
                        pass

    if not margins:
        return False

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(margins, bins=20, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Quorum Margin")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Quorum Margins Across All Cells")
    fig.tight_layout()
    fig.savefig(out_dir / "quorum_histogram.pdf")
    plt.close(fig)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out_dir = args.out or Path("Overleaf/Figures/Generated")
    out_dir.mkdir(parents=True, exist_ok=True)

    renderers = [
        ("F1 asr_bars", render_f1_asr_bars),
        ("F2 latency_bars", render_f2_latency),
        ("F3 fpr_bars", render_f3_fpr),
        ("F4 k_sweep_heatmap", render_f4_k_sweep),
        ("F5 quorum_histogram", render_f5_quorum_histogram),
    ]

    ok = 0
    for name, fn in renderers:
        if fn(args.run_dir, out_dir):
            print(f"  [{name}] rendered to {out_dir}")
            ok += 1
        else:
            print(f"  [{name}] SKIPPED (data not found)")

    print(f"\n[render_thesis_figures] {ok}/{len(renderers)} figures rendered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
