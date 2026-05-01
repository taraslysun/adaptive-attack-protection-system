"""Render thesis figures and tables from a thesis-runner artefact dir.

Reads ``logs/thesis/<timestamp>/`` produced by
``scripts/run_thesis_experiments.py`` and writes::

  figures/
    asr_heatmap_<tier>.png
    qfs_curve_<tier>.png
    ablation_table.md
    latency_table.md
    fpr_table.md

Matplotlib is the only hard dependency.  When it is missing the
script still emits the markdown tables and skips the PNG plots.

Usage::

    python scripts/make_thesis_figures.py                # latest run
    python scripts/make_thesis_figures.py path/to/run    # specific dir
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _latest_run() -> Optional[Path]:
    base = PROJECT_ROOT / "logs" / "thesis"
    if not base.exists():
        return None
    runs = sorted(p for p in base.iterdir() if p.is_dir())
    return runs[-1] if runs else None


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# ASR heatmap (matplotlib optional)
# ---------------------------------------------------------------------------


def _try_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as exc:
        print(f"[figures] matplotlib unavailable, skipping plots ({exc})")
        return None


def _matrix_to_arrays(
    asr_matrix: Dict[str, Dict[str, float]],
) -> Tuple[List[str], List[str], List[List[float]]]:
    defenses = list(asr_matrix.keys())
    attacks = sorted({a for row in asr_matrix.values() for a in row.keys()})
    grid: List[List[float]] = []
    for d in defenses:
        grid.append([asr_matrix[d].get(a, 0.0) * 100 for a in attacks])
    return defenses, attacks, grid


def render_asr_heatmap(
    asr_matrix: Dict[str, Dict[str, float]],
    out_png: Path,
    *,
    title: str,
) -> None:
    plt = _try_matplotlib()
    if plt is None or not asr_matrix:
        return
    defenses, attacks, grid = _matrix_to_arrays(asr_matrix)
    if not grid or not grid[0]:
        return
    fig, ax = plt.subplots(
        figsize=(max(6, len(attacks) * 1.4), max(3, len(defenses) * 0.5))
    )
    im = ax.imshow(grid, cmap="Reds", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(attacks)))
    ax.set_xticklabels(attacks, rotation=30, ha="right")
    ax.set_yticks(range(len(defenses)))
    ax.set_yticklabels(defenses)
    for i in range(len(defenses)):
        for j in range(len(attacks)):
            v = grid[i][j]
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                    color="white" if v >= 55 else "black", fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="judged ASR (%)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"  -> {out_png}")


# ---------------------------------------------------------------------------
# Queries-to-first-success curve
# ---------------------------------------------------------------------------


def render_qfs_curve(
    detailed: Dict[str, List[dict]],
    out_png: Path,
    *,
    title: str,
) -> None:
    plt = _try_matplotlib()
    if plt is None or not detailed:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for d_name, rows in detailed.items():
        succ = sorted(
            r["query_count"] for r in rows
            if r.get("judged_success") and r.get("query_count")
        )
        if not succ:
            continue
        xs = list(range(1, len(succ) + 1))
        ax.step(succ, xs, label=d_name, where="post")
    ax.set_xlabel("queries used")
    ax.set_ylabel("# attacks judged successful")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"  -> {out_png}")


# ---------------------------------------------------------------------------
# Per-layer ablation table (Tier 2)
# ---------------------------------------------------------------------------


def write_ablation_table(
    asr_matrix: Dict[str, Dict[str, float]],
    out_md: Path,
) -> None:
    base_rows: Dict[str, Dict[str, float]] = {}
    abl_rows: Dict[str, Dict[str, Dict[str, float]]] = {}
    for d_name, row in asr_matrix.items():
        if "-no_" in d_name:
            base, layer = d_name.rsplit("-no_", 1)
            abl_rows.setdefault(base, {})[layer] = row
        else:
            base_rows[d_name] = row
    if not abl_rows:
        return
    attacks = sorted({a for row in asr_matrix.values() for a in row.keys()})

    lines = ["# Per-layer ablation (judged ASR)\n",
             "Lower is better. Removing a layer should not collapse robustness.\n",
             "| Defense | Layer removed | " + " | ".join(attacks) + " |"]
    lines.append("|---------|----------------|" + "|".join(["----"] * len(attacks)) + "|")
    for base, base_row in base_rows.items():
        lines.append(
            f"| {base} | (none) | "
            + " | ".join(f"{base_row.get(a, 0.0)*100:.1f}%" for a in attacks)
            + " |"
        )
        for layer, row in abl_rows.get(base, {}).items():
            lines.append(
                f"| {base} | {layer} | "
                + " | ".join(f"{row.get(a, 0.0)*100:.1f}%" for a in attacks)
                + " |"
            )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))
    print(f"  -> {out_md}")


# ---------------------------------------------------------------------------
# Latency and FPR tables (cross-tier aggregate)
# ---------------------------------------------------------------------------


def write_latency_table(
    tiers: Dict[str, dict],
    out_md: Path,
) -> None:
    lines = ["# Per-layer latency (mean / p50 / p95 ms)\n",
             "| Tier | Defense | Layer | n | mean | p50 | p95 |",
             "|------|---------|-------|---|------|-----|-----|"]
    any_row = False
    for tier_name, tier in tiers.items():
        if not isinstance(tier, dict):
            continue
        per_def = tier.get("latency_per_layer") or {}
        for d_name, per_layer in per_def.items():
            if not per_layer:
                continue
            for layer, stats in per_layer.items():
                lines.append(
                    f"| {tier_name} | {d_name} | {layer} | "
                    f"{int(stats['n'])} | {stats['mean']:.2f} | "
                    f"{stats['p50']:.1f} | {stats['p95']:.1f} |"
                )
                any_row = True
    if any_row:
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text("\n".join(lines))
        print(f"  -> {out_md}")


def write_fpr_table(
    tiers: Dict[str, dict],
    out_md: Path,
) -> None:
    lines = ["# False positive rate / utility per defense\n",
             "| Tier | Defense | Utility | FPR |",
             "|------|---------|---------|-----|"]
    any_row = False
    for tier_name, tier in tiers.items():
        if not isinstance(tier, dict):
            continue
        util = tier.get("utility") or {}
        fpr = tier.get("false_positive_rate") or {}
        keys = sorted(set(util.keys()) | set(fpr.keys()))
        for d in keys:
            u = util.get(d)
            f = fpr.get(d)
            lines.append(
                f"| {tier_name} | {d} | "
                f"{u*100:.1f}% | {f*100:.1f}% |"
                if (u is not None and f is not None)
                else f"| {tier_name} | {d} | "
                     f"{u*100:.1f}% |  - |"
                     if u is not None
                     else f"| {tier_name} | {d} |  - | "
                          f"{f*100:.1f}% |"
            )
            any_row = True
    if any_row:
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text("\n".join(lines))
        print(f"  -> {out_md}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def render(run_dir: Path) -> None:
    figures = run_dir / "figures"
    figures.mkdir(exist_ok=True)
    print(f"[figures] reading {run_dir}")

    tiers: Dict[str, dict] = {}
    for path in run_dir.rglob("*.json"):
        if path.name == "summary.json" or "skipped" in path.name:
            continue
        try:
            data = _load_json(path)
        except Exception:
            continue
        rel = path.relative_to(run_dir)
        tiers[str(rel)] = data
        title = path.stem
        if "asr_matrix" in data:
            render_asr_heatmap(
                data["asr_matrix"],
                figures / f"asr_heatmap_{title}.png",
                title=f"Judged ASR -- {title}",
            )
            if data.get("detailed_results"):
                render_qfs_curve(
                    data["detailed_results"],
                    figures / f"qfs_curve_{title}.png",
                    title=f"Queries-to-first-success -- {title}",
                )

    write_latency_table(tiers, figures / "latency_table.md")
    write_fpr_table(tiers, figures / "fpr_table.md")

    tier2 = next(
        (v for k, v in tiers.items()
         if "tier2" in k and isinstance(v, dict) and "asr_matrix" in v),
        None,
    )
    if tier2:
        write_ablation_table(tier2["asr_matrix"], figures / "ablation_table.md")

    print(f"[figures] done -> {figures}")


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_dir = Path(arg) if arg else _latest_run()
    if run_dir is None or not run_dir.exists():
        print("No thesis run directory found.")
        sys.exit(1)
    render(run_dir)


if __name__ == "__main__":
    main()
