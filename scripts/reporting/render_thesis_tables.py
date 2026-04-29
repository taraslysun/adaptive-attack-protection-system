#!/usr/bin/env python3
"""Render thesis-ready LaTeX table fragments from aggregated experiment data.

Each logical table is written under the LaTeX ``\\label{tab:...}`` stem (e.g.\
``tab_tier1_psr.tex``) **and** under thesis placeholder ids (e.g.\
``tier1_psr.tex``) so ``\\input{Generated/tier1_psr}`` matches ``<<TABLE:tier1_psr>>``.

Usage::

    python scripts/reporting/render_thesis_tables.py logs/thesis/<ts>/ --out Overleaf/Generated/
    python scripts/reporting/render_thesis_tables.py logs/thesis/<ts>/ --check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Placeholder id (<<TABLE:…>>) → legacy output stem (tab_*, without .tex)
# ---------------------------------------------------------------------------
PLACEHOLDER_TO_TABLE: Dict[str, str] = {
    "headline_matrix": "tab_tier3_asr",  # legacy alias for the headline tab
    "slim5_asr": "tab_slim5_asr",
    "tier1_psr": "tab_tier1_psr",
    "tier2_asr": "tab_slim5_asr",  # slim-5 matrix
    "tier3_asr": "tab_tier3_asr",
    "spq_telemetry": "tab_spq_telemetry",
    "spq_k_sweep": "tab_spq_k_sweep",
    "cost_of_defense": "tab_cost_of_defense",
    "ablation_layers": "tab_ablation_layers",
    "fpr_per_layer": "tab_fpr_per_layer",
    "latency_per_layer": "tab_latency_per_layer",
    "vendor_comparison": "tab_vendor_comparison",
    "agentharm_asr": "tab_agentharm_asr",
    "tau_bench_utility": "tab_tau_bench_utility",
}


# ---------------------------------------------------------------------------
# Table specifications
# ---------------------------------------------------------------------------

TABLE_SPECS: List[Dict[str, Any]] = [
    {
        "label": "tab:tier1_psr",
        "caption": "Tier-1 PSR (\\%) per defense across injection benchmarks (95\\% bootstrap CI).",
        "columns": ["Defense", "MemPoison", "AgentDojo", "InjecAgent"],
        "source": "aggregate/psr_with_ci.json",
        "extractor": "_extract_tier1_psr",
    },
    {
        "label": "tab:slim5_asr",
        "caption": "Tier-2 judged ASR (\\%) for slim-5 attacks vs.~selected defenses (95\\% CI).",
        "columns": ["Attack", "no\\_defense", "PACE", "Best baseline"],
        "source": "aggregate/asr_with_ci.json",
        "extractor": "_extract_slim5_asr",
    },
    {
        "label": "tab:tier3_asr",
        "caption": "Tier-3 ASR (\\%) on HarmBench / AdvBench / JailbreakBench.",
        "columns": ["Benchmark", "no\\_defense", "PACE", "CI"],
        "source": "aggregate/asr_with_ci.json",
        "extractor": "_extract_tier3_asr",
    },
    {
        "label": "tab:spq_telemetry",
        "caption": "PACE telemetry across victims: CFI violations, abstain, replan, fire rates.",
        "columns": ["Victim", "CFI viol.", "Abstain", "Replan", "Fire", "Quorum margin"],
        "source": "aggregate/spq_with_ci.json",
        "extractor": "_extract_spq_telemetry",
    },
    {
        "label": "tab:spq_k_sweep",
        "caption": "PACE K-sweep ablation: judged ASR and utility delta by (K, q).",
        "columns": ["K", "q", "ASR (\\%)", "Utility $\\Delta$", "p50 latency (s)"],
        "source": "spq_k_sweep_summary.json",
        "extractor": "_extract_k_sweep",
    },
    {
        "label": "tab:cost_of_defense",
        "caption": "Cost of defense: tau-bench utility delta and FPR per defense.",
        "columns": ["Defense", "Utility $\\Delta$ (pp)", "FPR (\\%)", "p50 (ms)"],
        "source": "cost_of_defense.json",
        "extractor": "_extract_cost_of_defense",
    },
    {
        "label": "tab:agentharm_asr",
        "caption": "AgentHarm harmful-tool-call success rate per defense (95\\% bootstrap CI).",
        "columns": ["Defense", "PSR (\\%)", "CI"],
        "source": "tier1/agentharm.json",
        "extractor": "_extract_agentharm",
    },
    {
        "label": "tab:tau_bench_utility",
        "caption": "tau-bench benign task success rate per defense (cost-of-defense signal).",
        "columns": ["Defense", "Task success (\\%)", "CI"],
        "source": "tier1/tau_bench.json",
        "extractor": "_extract_tau_bench",
    },
    {
        "label": "tab:ablation_layers",
        "caption": "Feature-extractor ablation: judged ASR with single hooks disabled.",
        "columns": ["Variant", "ASR (\\%)"],
        "source": "summary.json",
        "extractor": "_extract_ablation_layers",
        "stub_todo": True,
        "todo_body": "ablation\\_layers — wire aggregate or tier2 ablation export when available.",
    },
    {
        "label": "tab:fpr_per_layer",
        "caption": "Per-layer FPR (\\%) on benign probes.",
        "columns": ["Layer", "FPR (\\%)"],
        "source": "summary.json",
        "extractor": "_extract_fpr_per_layer",
        "stub_todo": True,
        "todo_body": "fpr\\_per\\_layer — populate from false\\_positive\\_rate / block\\_attribution export.",
    },
    {
        "label": "tab:latency_per_layer",
        "caption": "Per-layer p50 latency (ms).",
        "columns": ["Layer", "p50 (ms)"],
        "source": "summary.json",
        "extractor": "_extract_latency_per_layer",
        "stub_todo": True,
        "todo_body": "latency\\_per\\_layer — use cost\\_of\\_defense per-layer slices when non-empty.",
    },
    {
        "label": "tab:vendor_comparison",
        "caption": "Vendor-style / secondary baselines: judged ASR (\\%) in the same harness (when run).",
        "columns": ["Defense", "ASR (\\%)"],
        "source": "summary.json",
        "extractor": "_extract_vendor_comparison",
        "stub_todo": True,
        "todo_body": "vendor\\_comparison — requires matrix with --include-secondary baselines.",
    },
]


# ---------------------------------------------------------------------------
# Extractors — each returns list of row-tuples or None if data missing
# ---------------------------------------------------------------------------

def _extract_tier1_psr(data: Dict) -> Optional[List[Tuple]]:
    # New aggregate format: {"psr_with_ci": {"tier1/file.json": {defense: {"psr": {CI}}}}}
    if "psr_with_ci" in data:
        psr_map = data["psr_with_ci"]
        totals: Dict[str, List[float]] = {}
        for file_key, file_data in psr_map.items():
            if not isinstance(file_data, dict) or "tier1" not in file_key:
                continue
            for defense, metrics in file_data.items():
                if defense.startswith("_") or not isinstance(metrics, dict):
                    continue
                psr_info = metrics.get("psr", {})
                if isinstance(psr_info, dict):
                    totals.setdefault(defense, []).append(psr_info.get("point", 0.0))
        rows = []
        for defense, vals in sorted(totals.items()):
            mean_psr = sum(vals) / len(vals) if vals else 0.0
            rows.append((defense, f"{mean_psr*100:.1f}", ""))
        return rows or None
    # Legacy flat format: {defense: {"psr": float, "psr_ci": {...}}}
    rows = []
    for defense, info in data.items():
        if defense.startswith("_"):
            continue
        if isinstance(info, dict) and "psr" in info:
            psr_val = info["psr"]
            ci = info.get("psr_ci", {})
            lo = ci.get("lo", psr_val)
            hi = ci.get("hi", psr_val)
            rows.append((defense, f"{psr_val*100:.1f}", f"[{lo*100:.1f}, {hi*100:.1f}]"))
    return rows or None


def _extract_slim5_asr(data: Dict) -> Optional[List[Tuple]]:
    # New aggregate format: {"asr_with_ci": {"tier2/slim5_adaptive.json": {defense: {attack: {CI}}}}}
    if "asr_with_ci" in data:
        asr_map = data["asr_with_ci"]
        rows = []
        for file_key, file_data in asr_map.items():
            if not isinstance(file_data, dict) or "tier2" not in file_key:
                continue
            for defense, attacks in file_data.items():
                if defense.startswith("_") or not isinstance(attacks, dict):
                    continue
                for attack, ci_info in attacks.items():
                    if not isinstance(ci_info, dict):
                        continue
                    pt = ci_info.get("point", 0.0)
                    lo = ci_info.get("ci95_low", pt)
                    hi = ci_info.get("ci95_high", pt)
                    rows.append((attack, defense, f"{pt*100:.1f}",
                                 f"[{lo*100:.1f}, {hi*100:.1f}]"))
        return rows or None
    # Legacy flat format
    rows = []
    for key, info in data.items():
        if key.startswith("_") or not isinstance(info, dict):
            continue
        for defense, dinfo in info.items():
            if isinstance(dinfo, dict) and "asr" in dinfo:
                asr = dinfo["asr"]
                ci = dinfo.get("ci", {})
                rows.append((key, defense, f"{asr*100:.1f}",
                             f"[{ci.get('lo', asr)*100:.1f}, {ci.get('hi', asr)*100:.1f}]"))
    return rows or None


def _extract_tier3_asr(data: Dict) -> Optional[List[Tuple]]:
    # New aggregate format: {"asr_with_ci": {"tier3/benchmark.json": {defense: {attack: {CI}}}}}
    if "asr_with_ci" in data:
        asr_map = data["asr_with_ci"]
        rows = []
        for file_key, file_data in asr_map.items():
            if not isinstance(file_data, dict) or "tier3" not in file_key:
                continue
            bench = file_key.split("/")[-1].replace(".json", "")
            for defense, attacks in file_data.items():
                if defense.startswith("_") or not isinstance(attacks, dict):
                    continue
                for attack, ci_info in attacks.items():
                    if not isinstance(ci_info, dict):
                        continue
                    pt = ci_info.get("point", 0.0)
                    lo = ci_info.get("ci95_low", pt)
                    hi = ci_info.get("ci95_high", pt)
                    rows.append((bench, defense, attack, f"{pt*100:.1f}",
                                 f"[{lo*100:.1f}, {hi*100:.1f}]"))
        return rows or None
    # Legacy flat format
    rows = []
    for key, info in data.items():
        if key.startswith("_") or "tier3" not in key or not isinstance(info, dict):
            continue
        for defense, dinfo in info.items():
            if isinstance(dinfo, dict) and "asr" in dinfo:
                asr = dinfo["asr"]
                ci = dinfo.get("ci", {})
                rows.append((key, defense, f"{asr*100:.1f}",
                             f"[{ci.get('lo', asr)*100:.1f}, {ci.get('hi', asr)*100:.1f}]"))
    return rows or None


def _extract_spq_telemetry(data: Dict) -> Optional[List[Tuple]]:
    rows = []
    for victim, info in data.items():
        if victim.startswith("_") or not isinstance(info, dict):
            continue
        rows.append((
            victim,
            str(info.get("cfi_violation_count", 0)),
            f"{info.get('abstain_rate', 0)*100:.1f}",
            f"{info.get('replan_rate', 0)*100:.1f}",
            f"{info.get('fire_rate', 0)*100:.1f}",
            f"{info.get('quorum_margin_mean', 0):.2f}",
        ))
    return rows or None


def _extract_k_sweep(data: Dict) -> Optional[List[Tuple]]:
    rows = []
    cells = data.get("cells", data)
    if isinstance(cells, list):
        for cell in cells:
            rows.append((
                str(cell.get("K", "?")),
                str(cell.get("q", "?")),
                f"{cell.get('asr', 0)*100:.1f}",
                f"{cell.get('utility_delta', 0)*100:.1f}",
                f"{cell.get('p50_latency', 0):.2f}",
            ))
    return rows or None


def _fmt_u_fp_p50(name: str, info: Dict[str, Any]) -> Tuple[str, str, str, str]:
    u = info.get("utility_delta")
    util_s = "—" if u is None else f"{u * 100.0:.1f}"
    fp = info.get("fpr")
    fpr_s = "—" if fp is None else f"{float(fp) * 100.0:.1f}"
    p50m = info.get("p50_latency_ms")
    if p50m is None and "p50_latency" in info:
        # legacy: seconds
        p50m = float(info["p50_latency"]) * 1000.0
    lat_s = "—" if p50m is None else f"{float(p50m):.1f}"
    return name, util_s, fpr_s, lat_s


def _extract_cost_of_defense(data: Dict) -> Optional[List[Tuple]]:
    root = data.get("defenses")
    if not isinstance(root, dict):
        return None
    rows: List[Tuple] = []
    for name, info in sorted(root.items(), key=lambda x: str(x[0])):
        if name.startswith("_") or not isinstance(info, dict):
            continue
        rows.append(_fmt_u_fp_p50(name, info))
    if not rows and data.get("_per_tier_legacy") is not None:
        return [
            (
                "(incomplete; re-run after thesis runner emits defenses map)",
                "—",
                "—",
                "—",
            )
        ]
    if not rows:
        return None
    return rows


def _extract_ablation_layers(_data: Dict) -> Optional[List[Tuple]]:
    return None


def _extract_fpr_per_layer(_data: Dict) -> Optional[List[Tuple]]:
    return None


def _extract_latency_per_layer(_data: Dict) -> Optional[List[Tuple]]:
    return None


def _extract_vendor_comparison(_data: Dict) -> Optional[List[Tuple]]:
    return None


EXTRACTORS = {
    "_extract_tier1_psr": _extract_tier1_psr,
    "_extract_slim5_asr": _extract_slim5_asr,
    "_extract_tier3_asr": _extract_tier3_asr,
    "_extract_spq_telemetry": _extract_spq_telemetry,
    "_extract_k_sweep": _extract_k_sweep,
    "_extract_cost_of_defense": _extract_cost_of_defense,
    "_extract_agentharm": lambda d: _extract_psr_generic(d),
    "_extract_tau_bench": lambda d: _extract_psr_generic(d),
    "_extract_ablation_layers": _extract_ablation_layers,
    "_extract_fpr_per_layer": _extract_fpr_per_layer,
    "_extract_latency_per_layer": _extract_latency_per_layer,
    "_extract_vendor_comparison": _extract_vendor_comparison,
}


def _extract_psr_generic(data: Dict) -> Optional[List[Tuple]]:
    """Generic PSR extractor for AgentHarm / tau-bench JSONs."""
    psr_data = data.get("psr", {})
    if not psr_data:
        return None
    rows = []
    for defense, psr_val in psr_data.items():
        details = data.get("details", {}).get(defense, {})
        ci = details.get("psr_ci", {})
        lo = ci.get("lo", psr_val)
        hi = ci.get("hi", psr_val)
        rows.append((defense, f"{psr_val*100:.1f}", f"[{lo*100:.1f}, {hi*100:.1f}]"))
    return rows or None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _render_table(spec: Dict, rows: List[Tuple]) -> str:
    """Render a LaTeX table fragment from spec + row data."""
    cols = spec["columns"]
    ncols = len(cols)
    col_spec = "l" + "r" * (ncols - 1)

    lines = [
        f"% Auto-generated by render_thesis_tables.py — do not edit by hand",
        f"\\begin{{table}}[htbp]",
        f"\\centering",
        f"\\caption{{{spec['caption']}}}",
        f"\\label{{{spec['label']}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        f"\\toprule",
        " & ".join(cols) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        padded = list(row) + [""] * (ncols - len(row))
        lines.append(" & ".join(padded[:ncols]) + " \\\\")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    return "\n".join(lines)


def _render_todo_table(spec: Dict[str, Any], message: str) -> str:
    """Minimal valid table when extractors are not yet wired."""
    safe = spec["label"].replace(":", "_")
    esc = message.replace("%", r"\%")
    return "\n".join([
        f"% Auto-generated placeholder ({safe})",
        f"\\begin{{table}}[htbp]",
        f"\\centering",
        f"\\caption{{{spec['caption']}}}",
        f"\\label{{{spec['label']}}}",
        r"\begin{tabular}{c}",
        r"\small\textit{[Placeholder] " + esc + r"} \\",
        r"\end{tabular}",
        r"\end{table}",
    ])


def _find_source(run_dir: Path, source: str) -> Optional[Path]:
    """Look for a source file in run_dir, model subdirs, or model/seed subdirs."""
    direct = run_dir / source
    if direct.exists():
        return direct
    for model_dir in sorted(run_dir.glob("model_*")):
        candidate = model_dir / source
        if candidate.exists():
            return candidate
        # Also check seed subdirectories within model dirs
        for seed_dir in sorted(model_dir.glob("seed_*")):
            candidate = seed_dir / source
            if candidate.exists():
                return candidate
    return None


def _output_stems_for_label(safe_label: str) -> Set[str]:
    """``tab_tier1_psr`` + every placeholder id that shares this fragment."""
    names: Set[str] = {safe_label}
    for _pid, legacy in PLACEHOLDER_TO_TABLE.items():
        if legacy == safe_label:
            names.add(_pid)
    return names


def _write_tex_dual(out_dir: Path, safe_label: str, tex: str) -> None:
    for stem in _output_stems_for_label(safe_label):
        (out_dir / f"{stem}.tex").write_text(tex, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run_dir", type=Path, help="Root of a thesis run (or aggregate)")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for .tex fragments (default: Overleaf/Generated/)",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="Validate data availability without writing files",
    )
    args = ap.parse_args()

    out_dir = args.out or Path("Overleaf/Generated")
    if not args.check:
        out_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    missing = 0
    written = 0

    for spec in TABLE_SPECS:
        label = spec["label"]
        source_path = _find_source(args.run_dir, spec["source"])

        if source_path is None:
            print(f"  [{label}] MISSING source: {spec['source']}")
            missing += 1
            continue

        try:
            data = json.loads(source_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  [{label}] ERROR reading {source_path}: {e}")
            missing += 1
            continue

        if spec["source"] == "cost_of_defense.json" and "defenses" not in data:
            # Legacy runs: tier-keyed per-layer map only; wrap for the new extractor.
            data = {"defenses": {}, "_per_tier_legacy": dict(data) if isinstance(data, dict) else {}}

        extractor = EXTRACTORS.get(spec["extractor"])
        if extractor is None:
            print(f"  [{label}] ERROR: unknown extractor {spec['extractor']}")
            missing += 1
            continue

        rows = extractor(data)
        if rows is None and spec.get("stub_todo"):
            safe_label = label.replace(":", "_")
            msg = spec.get("todo_body", "pending data")
            if args.check:
                print(f"  [{label}] STUB (placeholder) — {msg}")
                ok += 1
            else:
                tex = _render_todo_table(spec, msg)
                _write_tex_dual(out_dir, safe_label, tex)
                nout = len(_output_stems_for_label(safe_label))
                print(f"  [{label}] -> {out_dir} ({nout} names, TODO placeholder)")
                written += nout
                ok += 1
            continue

        if rows is None:
            print(f"  [{label}] NO DATA in {source_path}")
            missing += 1
            continue

        ok += 1
        if args.check:
            print(f"  [{label}] OK — {len(rows)} rows from {spec['source']}")
        else:
            tex = _render_table(spec, rows)
            safe_label = label.replace(":", "_")
            _write_tex_dual(out_dir, safe_label, tex)
            nout = len(_output_stems_for_label(safe_label))
            print(f"  [{label}] -> {out_dir} ({nout} x .tex, {len(rows)} rows)")
            written += nout

    print(
        f"\n[render_thesis_tables] {ok} OK, {missing} missing, {written} files written (with aliases)"
    )
    if missing > 0 and args.check:
        print("[render_thesis_tables] FAIL — not all tables have data")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
