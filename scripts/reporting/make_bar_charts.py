"""Generate bar charts for thesis results section.

Reads from logs/thesis/vendor_* and logs/thesis/t1_* directories.
Writes PNG figures to Overleaf/Figures/Generated/.

Usage:
    python scripts/reporting/make_bar_charts.py
    python scripts/reporting/make_bar_charts.py --prelim   # use n=3 partial data too
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "Overleaf" / "Figures" / "Generated"
LOGS = ROOT / "logs" / "thesis"

# ── model display names ────────────────────────────────────────────────────
VENDOR_DIRS = {
    # n=14 headline runs (preferred when available)
    "Gemini\nFlash Lite": "vendor_gemini_n14",
    "Llama\n3.1 8B":      "vendor_llama_n14",
    # Qwen n=14 run still in progress; use n=10 until it completes
    "Qwen3\n8B":          "vendor_qwen3_8b",
    # Mistral: v2 run has fixed judge + arg provenance gate; fall back to old if not ready
    "Mistral\nSmall":     "vendor_mistral_pace_v2",
    "DeepSeek\nFlash":    "vendor_deepseek_flash",
}
T1_DIRS = {
    "Gemini\nFlash Lite": "t1_gemini",
    "Llama\n3.1 8B":      "t1_llama",
    "Qwen3\n8B":          "t1_qwen",
    "Mistral\nSmall":     "t1_mistral",
    "DeepSeek\nFlash":    "t1_deepseek",
}
ATTACK_LABELS = {
    "RLAttack":            "RL",
    "HumanRedTeamAttack":  "HRT",
    "PAIRAttack":          "PAIR",
    "PoisonedRAGAttack":   "PoisRAG",
    "SupplyChainAttack":   "SupplyChain",
}
DEFENSE_LABELS = {
    "no_defense":          "No Defense",
    "PACEDefense":         "PACE",
    "StruQDefense":        "StruQ",
    "DataSentinelDefense": "DataSentinel",
    "MELONDefense":        "MELON",
    "AMemGuard":           "AMemGuard",
    "SmoothLLMDefense":    "SmoothLLM",
    "PromptGuard2Defense": "PromptGuard2",
}

# UCU thesis color palette
COLORS = {
    "No Defense":    "#E74C3C",
    "PACE":          "#2ECC71",
    "StruQ":         "#3498DB",
    "DataSentinel":  "#9B59B6",
    "MELON":         "#E67E22",
    "AMemGuard":     "#1ABC9C",
    "SmoothLLM":     "#F39C12",
    "PromptGuard2":  "#95A5A6",
}
ATTACK_COLORS = ["#E74C3C", "#E67E22", "#3498DB", "#9B59B6", "#2ECC71"]


def _load_json(path: Path) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _get_n(data: dict, defense: str = "no_defense") -> int:
    ci = data.get("asr_ci", {}).get(defense, {})
    if not ci:
        return 0
    vals = list(ci.values())
    return vals[0].get("n", 0) if vals else 0


# ── matplotlib setup ───────────────────────────────────────────────────────

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    })
    return plt


# ── Figure 1: ASR per attack, grouped by model ────────────────────────────

def fig_asr_per_attack_by_model(min_n: int = 5) -> None:
    """Grouped bar chart: x=attack, groups=models, bar=no_defense ASR."""
    plt = _plt()
    import numpy as np

    models: List[str] = []
    asr_by_model: Dict[str, Dict[str, float]] = {}

    for label, d in VENDOR_DIRS.items():
        path = LOGS / d / "tier2" / "slim5_adaptive.json"
        data = _load_json(path)
        if data is None:
            continue
        n = _get_n(data)
        if n < min_n:
            continue
        models.append(label.replace("\n", " "))
        nd = data["asr_matrix"].get("no_defense", {})
        asr_by_model[label.replace("\n", " ")] = nd

    if not models:
        print(f"[fig1] No models with n>={min_n}. Run with --prelim to use partial data.")
        return

    attacks = list(ATTACK_LABELS.keys())
    atk_labels = [ATTACK_LABELS[a] for a in attacks]
    x = np.arange(len(attacks))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model in enumerate(models):
        vals = [asr_by_model[model].get(a, 0.0) * 100 for a in attacks]
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9,
                      label=model, color=ATTACK_COLORS[i % len(ATTACK_COLORS)],
                      alpha=0.85)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{v:.0f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(atk_labels)
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_title("ASR per Attack Type — No Defense Baseline")
    ax.set_ylim(0, 110)
    ax.legend(loc="upper right", fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    out = OUT_DIR / "fig_asr_per_attack_by_model.pdf"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"[fig1] saved → {out.name}")


# ── Figure 2: ASR no_defense vs PACE per model, per attack ────────────────

def fig_pace_reduction_by_model(min_n: int = 5) -> None:
    """Grouped bar chart: x=model, bars=no_defense+PACE per attack."""
    plt = _plt()
    import numpy as np

    rows: List[Tuple[str, str, float, float]] = []  # (model, attack, no_def, pace)

    for label, d in VENDOR_DIRS.items():
        path = LOGS / d / "tier2" / "slim5_adaptive.json"
        data = _load_json(path)
        if data is None:
            continue
        n = _get_n(data)
        if n < min_n:
            continue
        model = label.replace("\n", " ")
        nd = data["asr_matrix"].get("no_defense", {})
        pace = data["asr_matrix"].get("PACEDefense", {})
        for atk, albl in ATTACK_LABELS.items():
            rows.append((model, albl, nd.get(atk, 0.0), pace.get(atk, 0.0)))

    if not rows:
        print(f"[fig2] No models with n>={min_n}.")
        return

    models_seen = list(dict.fromkeys(r[0] for r in rows))
    attacks_seen = list(ATTACK_LABELS.values())
    x = np.arange(len(models_seen))
    n_atk = len(attacks_seen)
    width = 0.35

    fig, axes = plt.subplots(1, n_atk, figsize=(4 * n_atk, 5), sharey=True)
    if n_atk == 1:
        axes = [axes]

    for ai, atk_lbl in enumerate(attacks_seen):
        ax = axes[ai]
        nd_vals = [next((r[2] for r in rows if r[0] == m and r[1] == atk_lbl), 0.0) * 100
                   for m in models_seen]
        pace_vals = [next((r[3] for r in rows if r[0] == m and r[1] == atk_lbl), 0.0) * 100
                     for m in models_seen]
        ax.bar(x - width / 2, nd_vals, width, label="No Defense",
               color=COLORS["No Defense"], alpha=0.85)
        ax.bar(x + width / 2, pace_vals, width, label="PACE",
               color=COLORS["PACE"], alpha=0.85)
        ax.set_title(atk_lbl, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(models_seen, rotation=30, ha="right", fontsize=8)
        if ai == 0:
            ax.set_ylabel("ASR (%)")
        ax.set_ylim(0, 110)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS["No Defense"], alpha=0.85),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["PACE"], alpha=0.85),
    ]
    fig.legend(handles, ["No Defense", "PACE"], loc="upper center",
               ncol=2, fontsize=10, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("PACE vs No-Defense ASR per Attack and Model", y=1.05, fontsize=12)
    fig.tight_layout()

    out = OUT_DIR / "fig_pace_reduction_by_model.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"[fig2] saved → {out.name}")


# ── Figure 3: Memory poisoning PSR across models and defenses ─────────────

def fig_memory_poisoning_psr() -> None:
    """Grouped bar: x=defense, bars=models."""
    plt = _plt()
    import numpy as np

    models_data: Dict[str, Dict[str, float]] = {}
    for label, d in T1_DIRS.items():
        path = LOGS / d / "tier1" / "memory_poisoning.json"
        data = _load_json(path)
        if data is None:
            continue
        models_data[label.replace("\n", " ")] = data.get("psr", {})

    if not models_data:
        print("[fig3] No memory poisoning data found.")
        return

    defenses = list(DEFENSE_LABELS.keys())
    def_labels = [DEFENSE_LABELS[d] for d in defenses]
    models = list(models_data.keys())
    x = np.arange(len(defenses))
    width = 0.8 / max(len(models), 1)

    bar_colors = ["#E74C3C", "#3498DB", "#E67E22", "#9B59B6", "#1ABC9C"]
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, model in enumerate(models):
        vals = [models_data[model].get(d, 0.0) * 100 for d in defenses]
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9, label=model,
                      color=bar_colors[i % len(bar_colors)], alpha=0.85)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{v:.0f}%", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(def_labels, rotation=20, ha="right")
    ax.set_ylabel("Poison Success Rate (%)")
    ax.set_title("Memory Poisoning PSR by Defense and Model")
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    out = OUT_DIR / "fig_memory_poisoning_psr.pdf"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"[fig3] saved → {out.name}")


# ── Figure 4b: ASR per model — bars=attacks (no_defense) ─────────────────

def fig_asr_per_model_by_attack(min_n: int = 5) -> None:
    """Grouped bar chart: x=model, bars=attack families, height=no_defense ASR.

    Lets the reader see which attack is most effective (highest bar) and
    least effective (lowest/zero bar) for each victim model independently.
    Bars sorted by descending ASR within each model group.
    """
    plt = _plt()
    import numpy as np

    asr_by_model: Dict[str, Dict[str, float]] = {}
    model_order: List[str] = []

    for label, d in VENDOR_DIRS.items():
        path = LOGS / d / "tier2" / "slim5_adaptive.json"
        data = _load_json(path)
        if data is None:
            continue
        n = _get_n(data)
        if n < min_n:
            continue
        ml = label.replace("\n", " ")
        model_order.append(ml)
        asr_by_model[ml] = data["asr_matrix"].get("no_defense", {})

    if not model_order:
        print(f"[fig4b] No models with n>={min_n}.")
        return

    attacks = list(ATTACK_LABELS.keys())
    atk_labels = [ATTACK_LABELS[a] for a in attacks]
    n_atk = len(attacks)
    n_models = len(model_order)
    width = 0.8 / n_atk

    # Color per attack family
    atk_colors = ["#E74C3C", "#E67E22", "#3498DB", "#9B59B6", "#2ECC71"]

    x = np.arange(n_models)
    fig, ax = plt.subplots(figsize=(max(8, n_models * 2.2), 5))

    for ai, (atk, albl) in enumerate(zip(attacks, atk_labels)):
        vals = [asr_by_model[m].get(atk, 0.0) * 100 for m in model_order]
        offset = (ai - n_atk / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9, label=albl,
                      color=atk_colors[ai % len(atk_colors)], alpha=0.85)
        for bar, v in zip(bars, vals):
            if v > 2:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.8,
                        f"{v:.0f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(model_order, rotation=15, ha="right")
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_title("No-Defense ASR per Victim Model — Attack Families Compared")
    ax.set_ylim(0, 115)
    ax.legend(title="Attack", loc="upper right", fontsize=9, title_fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    out = OUT_DIR / "fig_asr_per_model_by_attack.pdf"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"[fig4b] saved → {out.name}")


# ── Figure 4: Overall avg ASR per model — no_def vs PACE ──────────────────

def fig_overall_asr_comparison(min_n: int = 5) -> None:
    """Simple bar chart: avg ASR across all attacks, per model."""
    plt = _plt()
    import numpy as np

    labels, nd_vals, pace_vals, ns = [], [], [], []

    for label, d in VENDOR_DIRS.items():
        path = LOGS / d / "tier2" / "slim5_adaptive.json"
        data = _load_json(path)
        if data is None:
            continue
        n = _get_n(data)
        if n < min_n:
            continue
        nd = data["asr_matrix"].get("no_defense", {})
        pace = data["asr_matrix"].get("PACEDefense", {})
        if not nd:
            continue
        labels.append(label.replace("\n", " "))
        nd_vals.append(sum(nd.values()) / len(nd) * 100)
        pace_vals.append(sum(pace.values()) / len(pace) * 100 if pace else 0)
        ns.append(n)

    if not labels:
        print(f"[fig4] No models with n>={min_n}.")
        return

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 2), 5))
    bars1 = ax.bar(x - width / 2, nd_vals, width, label="No Defense",
                   color=COLORS["No Defense"], alpha=0.85)
    bars2 = ax.bar(x + width / 2, pace_vals, width, label="PACE",
                   color=COLORS["PACE"], alpha=0.85)

    for bars in (bars1, bars2):
        for bar in bars:
            v = bar.get_height()
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5,
                        f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

    for i, (lbl, n) in enumerate(zip(labels, ns)):
        ax.text(x[i], -8, f"n={n}", ha="center", fontsize=8, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Average ASR across attacks (%)")
    ax.set_title("Average ASR: No Defense vs PACE (Tier-2 Slim-5)")
    ax.set_ylim(-12, 110)
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    out = OUT_DIR / "fig_overall_asr_comparison.pdf"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"[fig4] saved → {out.name}")


# ── Figure 5: All defenses ASR heatmap per attack (single model) ──────────

def fig_defense_heatmap_all(model_dir: str, label: str, min_n: int = 5) -> None:
    path = LOGS / model_dir / "tier2" / "slim5_adaptive.json"
    data = _load_json(path)
    if data is None:
        print(f"[fig5] {model_dir}: no data")
        return
    n = _get_n(data)
    if n < min_n:
        print(f"[fig5] {model_dir}: n={n} < {min_n}, skipping")
        return

    plt = _plt()
    import numpy as np

    asr = data["asr_matrix"]
    defenses = list(asr.keys())
    attacks = sorted({a for row in asr.values() for a in row})
    grid = np.array([[asr[d].get(a, 0.0) * 100 for a in attacks] for d in defenses])

    def_labels = [DEFENSE_LABELS.get(d, d) for d in defenses]
    atk_labels = [ATTACK_LABELS.get(a, a) for a in attacks]

    fig, ax = plt.subplots(figsize=(len(attacks) * 1.5 + 2, len(defenses) * 0.6 + 1.5))
    im = ax.imshow(grid, cmap="RdYlGn_r", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(attacks)))
    ax.set_xticklabels(atk_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(defenses)))
    ax.set_yticklabels(def_labels)
    for i in range(len(defenses)):
        for j in range(len(attacks)):
            ax.text(j, i, f"{grid[i,j]:.0f}%", ha="center", va="center",
                    fontsize=9, color="white" if grid[i, j] > 60 else "black")
    plt.colorbar(im, ax=ax, label="ASR (%)")
    ax.set_title(f"ASR Heatmap — {label} (n={n})")
    fig.tight_layout()
    slug = label.lower().replace(" ", "_").replace("\n", "_")
    out = OUT_DIR / f"fig_heatmap_{slug}.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"[fig5] saved → {out.name}")


# ── Figure 6: AgentDojo + InjecAgent PSR — all defenses, per model ─────────

def fig_tier1_benchmarks() -> None:
    """Grouped bar: x=defense, bars=models, for AgentDojo and InjecAgent."""
    plt = _plt()
    import numpy as np

    benchmarks = {
        "AgentDojo IPI": "agentdojo.json",
        "InjecAgent":    "injecagent.json",
    }
    bar_colors = ["#E74C3C", "#3498DB", "#E67E22", "#9B59B6", "#1ABC9C"]

    for bench_label, bench_file in benchmarks.items():
        models_data: Dict[str, Dict[str, float]] = {}
        for label, d in T1_DIRS.items():
            path = LOGS / d / "tier1" / bench_file
            data = _load_json(path)
            if data is None:
                continue
            psr = data.get("psr", {})
            if psr:
                models_data[label.replace("\n", " ")] = psr

        if not models_data:
            print(f"[fig6] {bench_label}: no data")
            continue

        defenses = list(DEFENSE_LABELS.keys())
        def_labels = [DEFENSE_LABELS[d] for d in defenses]
        models = list(models_data.keys())
        x = np.arange(len(defenses))
        width = 0.8 / max(len(models), 1)

        fig, ax = plt.subplots(figsize=(13, 5))
        for i, model in enumerate(models):
            vals = [models_data[model].get(d, 0.0) * 100 for d in defenses]
            offset = (i - len(models) / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width * 0.9, label=model,
                          color=bar_colors[i % len(bar_colors)], alpha=0.85)
            for bar, v in zip(bars, vals):
                if v > 1:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.3,
                            f"{v:.0f}%", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(def_labels, rotation=20, ha="right")
        ax.set_ylabel("Injection Success Rate (%)\n(lower = better)")
        ax.set_title(f"{bench_label} — Injection Success Rate per Defense and Model")
        ax.set_ylim(0, 55)
        ax.legend(loc="upper right", fontsize=9)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

        slug = bench_label.lower().replace(" ", "_").replace("/", "_")
        out = OUT_DIR / f"fig_tier1_{slug}.pdf"
        fig.tight_layout()
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
        plt.close(fig)
        print(f"[fig6] saved → {out.name}")


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prelim", action="store_true",
                    help="Include partial n=3 data (mark as preliminary)")
    ap.add_argument("--min-n", type=int, default=5,
                    help="Minimum n per attack to include (default 5)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    min_n = 1 if args.prelim else args.min_n

    print(f"Generating figures (min_n={min_n}) → {OUT_DIR}")
    print()

    # Always generate memory poisoning (complete data)
    fig_memory_poisoning_psr()

    # Slim5 charts (require n>=min_n)
    fig_asr_per_attack_by_model(min_n=min_n)
    fig_asr_per_model_by_attack(min_n=min_n)
    fig_pace_reduction_by_model(min_n=min_n)
    fig_overall_asr_comparison(min_n=min_n)

    # Tier-1 external benchmarks
    fig_tier1_benchmarks()

    # Per-model heatmaps for defended runs
    defended_map = {
        "Gemini Flash Lite": "vendor_gemini_flash_lite_defended",
        "Llama 3.1 8B":      "vendor_llama_defended",
        "Qwen3 8B":          "vendor_qwen_defended",
        "Mistral Small":     "vendor_mistral_defended",
        "DeepSeek Flash":    "vendor_deepseek_defended",
    }
    for label, d in defended_map.items():
        fig_defense_heatmap_all(d, label, min_n=min_n)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
