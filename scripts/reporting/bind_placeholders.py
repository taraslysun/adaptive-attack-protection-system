#!/usr/bin/env python3
"""Map thesis run artefacts to ``<<TYPE:id>>`` placeholders and optionally apply to .tex.

Reads ``summary.json``, ``aggregate/{asr,psr}_with_ci.json`` (when present),
``cost_of_defense.json`` (new ``defenses`` map or legacy tier blobs), and
``docs/captions.yaml`` for <<CAP:...>>.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# <<NUM:asr_baseline_avg>> and LaTeX-escaped <<NUM:asr\_baseline\_avg>>
PLACEHOLDER_RE = re.compile(r"<<([A-Z]+):((?:\\.|[^>])+?)>>", re.DOTALL)


def _unlatex_id(s: str) -> str:
    s = s.replace(r"\\_", "_").replace(r"\_", "_")
    return s


def _latex_escape_id(s: str) -> str:
    return s.replace("_", r"\_")


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    return float(sum(vals)) / float(len(vals))


def _asr_row_means(
    m: Any,
) -> Tuple[Optional[float], Optional[float]]:
    """Return (mean for no_defense, mean for PACEDefense) from asr_matrix rows."""
    if not isinstance(m, dict):
        return None, None
    nd = m.get("no_defense")
    spq = m.get("PACEDefense")
    bvals: List[float] = []
    svals: List[float] = []
    if isinstance(nd, dict):
        bvals = [float(x) for x in nd.values() if isinstance(x, (int, float))]
    if isinstance(spq, dict):
        svals = [float(x) for x in spq.values() if isinstance(x, (int, float))]
    return _mean(bvals), _mean(svals)


def _tier1_psr_means(
    summary: Any,
) -> Tuple[Optional[float], Optional[float]]:
    """Mean PSR (union) for no_defense and PACE across tier-1 PSR sub-tiers."""
    bacc: List[float] = []
    sacc: List[float] = []
    for key, t in (summary or {}).get("tiers", {}).items():
        if "tier1" not in key or not isinstance(t, dict):
            continue
        if "psr" not in t:
            continue
        psr = t["psr"]
        if not isinstance(psr, dict):
            continue
        if "no_defense" in psr and isinstance(psr["no_defense"], (int, float)):
            bacc.append(float(psr["no_defense"]))
        if "PACEDefense" in psr and isinstance(psr["PACEDefense"], (int, float)):
            sacc.append(float(psr["PACEDefense"]))
    return _mean(bacc), _mean(sacc)


def _fpr_spq(summary: Any) -> Optional[float]:
    t = (summary or {}).get("tiers", {}).get("tier2_slim5")
    if not isinstance(t, dict):
        return None
    f = t.get("false_positive_rate")
    if not isinstance(f, dict):
        return None
    v = f.get("PACEDefense")
    return float(v) if isinstance(v, (int, float)) else None


def _load_captions(path: Path) -> Dict[str, str]:
    if not path.is_file():
        return {}
    raw = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        d = yaml.safe_load(raw) or {}
    except Exception:
        d = {}
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            k, _, rest = line.partition(":")
            k, rest = k.strip(), rest.strip()
            if rest.startswith('"'):
                v, _, _ = rest[1:].partition('"')
            else:
                v = rest
            d[k] = v
    return {k: (v or "") for k, v in d.items() if not str(k).startswith("#")}


def _migrate_cost(c: Any) -> Dict[str, Any]:
    if not isinstance(c, dict):
        return {"defenses": {}}
    if "defenses" in c and isinstance(c["defenses"], dict):
        return c
    return {"defenses": {}, "_per_tier_legacy": dict(c)}


def _cost_lookup(cost: Any, dname: str) -> Any:
    c = _migrate_cost(cost) if cost else {"defenses": {}}
    defs = c.get("defenses")
    if not isinstance(defs, dict):
        return None
    return defs.get(dname)


def build_placeholder_values(
    run_dir: Path,
    *,
    caption_file: Path,
) -> Dict[str, str]:
    """Map composite keys ``TYPE:id`` to LaTeX string values."""
    out: Dict[str, str] = {}
    run_dir = run_dir.resolve()
    summary = _read_json(run_dir / "summary.json")
    asr_ci = _read_json(run_dir / "aggregate" / "asr_with_ci.json")
    psr_ci = _read_json(run_dir / "aggregate" / "psr_with_ci.json")
    # Also search seed subdirectories for summary/cost (multi-seed layout)
    if summary is None:
        for seed_dir in sorted(run_dir.glob("seed_*")):
            summary = _read_json(seed_dir / "summary.json")
            if summary is not None:
                break
    cost = _read_json(run_dir / "cost_of_defense.json")
    if cost is None:
        for seed_dir in sorted(run_dir.glob("seed_*")):
            cost = _read_json(seed_dir / "cost_of_defense.json")
            if cost is not None:
                break
    cost = _migrate_cost(cost)
    k_sweep = _read_json(run_dir / "spq_k_sweep_summary.json")
    run_meta = _read_json(run_dir / "run_metadata.json")
    if run_meta is None:
        for seed_dir in sorted(run_dir.glob("seed_*")):
            run_meta = _read_json(seed_dir / "run_metadata.json")
            if run_meta is not None:
                break
    capt = _load_captions(caption_file)

    tiers = (summary or {}).get("tiers", {}) or {}

    # Augment tiers from aggregate format when summary is missing tier data
    if asr_ci and isinstance(asr_ci.get("asr_with_ci"), dict):
        for file_key, file_data in asr_ci["asr_with_ci"].items():
            if not isinstance(file_data, dict):
                continue
            if "tier2" in file_key and "tier2_slim5" not in tiers:
                asr_matrix = {
                    defense: {atk: info.get("point", 0.0)
                               for atk, info in attacks.items()
                               if isinstance(info, dict)}
                    for defense, attacks in file_data.items()
                    if isinstance(attacks, dict)
                }
                tiers["tier2_slim5"] = {"asr_matrix": asr_matrix}
            elif "tier3" in file_key:
                bench = file_key.split("/")[-1].replace(".json", "")
                tkey = f"tier3_{bench}"
                if tkey not in tiers:
                    asr_matrix = {
                        defense: {atk: info.get("point", 0.0)
                                   for atk, info in attacks.items()
                                   if isinstance(info, dict)}
                        for defense, attacks in file_data.items()
                        if isinstance(attacks, dict)
                    }
                    tiers[tkey] = {"asr_matrix": asr_matrix}
    if psr_ci and isinstance(psr_ci.get("psr_with_ci"), dict):
        for file_key, file_data in psr_ci["psr_with_ci"].items():
            if not isinstance(file_data, dict) or "tier1" not in file_key:
                continue
            bench = file_key.split("/")[-1].replace(".json", "")
            tkey = f"tier1_{bench}"
            if tkey not in tiers:
                psr_map = {
                    defense: metrics.get("psr", {}).get("point", 0.0)
                    if isinstance(metrics.get("psr"), dict)
                    else metrics.get("psr", 0.0)
                    for defense, metrics in file_data.items()
                    if isinstance(metrics, dict)
                }
                tiers[tkey] = {"psr": psr_map}

    t2 = tiers.get("tier2_slim5")
    if isinstance(t2, dict) and "asr_matrix" in t2:
        b2, s2 = _asr_row_means(t2["asr_matrix"])
    else:
        b2, s2 = None, None
    if b2 is not None:
        out["NUM:asr_tier2_baseline_avg"] = f"{b2 * 100.0:.1f}\\%"
    if s2 is not None:
        out["NUM:asr_tier2_with_spq_avg"] = f"{s2 * 100.0:.1f}\\%"

    t3keys = ("tier3_harmbench", "tier3_advbench", "tier3_jailbreakbench")
    b3l: List[float] = []
    s3l: List[float] = []
    for k3 in t3keys:
        t3 = tiers.get(k3)
        if not isinstance(t3, dict) or "asr_matrix" not in t3:
            continue
        b, s = _asr_row_means(t3["asr_matrix"])
        if b is not None:
            b3l.append(b)
        if s is not None:
            s3l.append(s)
    b3, s3 = _mean(b3l), _mean(s3l)
    if b3 is not None:
        out["NUM:asr_tier3_baseline_avg"] = f"{b3 * 100.0:.1f}\\%"
    if s3 is not None:
        out["NUM:asr_tier3_with_spq_avg"] = f"{s3 * 100.0:.1f}\\%"

    # Use enriched tiers (from aggregate) for PSR means when summary is missing
    p1b, p1s = _tier1_psr_means({"tiers": tiers})
    if p1b is not None:
        out["NUM:psr_tier1_baseline"] = f"{p1b * 100.0:.1f}\\%"
    if p1s is not None:
        out["NUM:psr_tier1_with_spq"] = f"{p1s * 100.0:.1f}\\%"

    pool_b: List[float] = []
    pool_s: List[float] = []
    for t in tiers.values():
        if not isinstance(t, dict) or "asr_matrix" not in t:
            continue
        b, s = _asr_row_means(t["asr_matrix"])
        if b is not None:
            pool_b.append(b)
        if s is not None:
            pool_s.append(s)
    pool_bm, pool_sm = _mean(pool_b), _mean(pool_s)
    if pool_bm is not None:
        out["NUM:asr_baseline_avg"] = f"{pool_bm * 100.0:.1f}\\%"
    if pool_sm is not None:
        out["NUM:asr_spq_avg"] = f"{pool_sm * 100.0:.1f}\\%"
    if pool_bm is not None and pool_sm is not None and pool_bm > 1e-8:
        red = 100.0 * (1.0 - (pool_sm / pool_bm))
        out["NUM:asr_reduction_pct"] = f"{red:.1f}"

    fp = _fpr_spq(summary)
    if fp is not None:
        out["NUM:fpr_spq_overall"] = f"{fp * 100.0:.1f}\\%"

    t2sc = tiers.get("tier2_slim5")
    if isinstance(t2sc, dict) and t2sc.get("asr_matrix"):
        sm = t2sc["asr_matrix"]
        if isinstance(sm, dict) and "PACEDefense" in sm and "SupplyChainAttack" in sm["PACEDefense"]:
            out["NUM:asr_supply_chain_with_spq"] = (
                f"{float(sm['PACEDefense']['SupplyChainAttack']) * 100.0:.1f}\\%"
            )
            out["NUM:adhoc_asr_supply_chain_with_spq"] = out["NUM:asr_supply_chain_with_spq"]

    tau = tiers.get("tier1_tau_bench")
    if isinstance(tau, dict) and isinstance(tau.get("psr"), dict):
        u0 = tau["psr"].get("no_defense")
        u1 = tau["psr"].get("PACEDefense")
        if u0 is not None:
            out["NUM:utility_baseline"] = f"{float(u0) * 100.0:.1f}\\%"
        if u1 is not None:
            out["NUM:utility_with_spq"] = f"{float(u1) * 100.0:.1f}\\%"

    for dkey, pfx in (("no_defense", "latency_baseline_p50"), ("PACEDefense", "latency_with_spq_p50")):
        dinfo = _cost_lookup(cost, dkey) if cost else None
        if not isinstance(dinfo, dict):
            continue
        ms = dinfo.get("p50_latency_ms")
        if ms is not None and isinstance(ms, (int, float)):
            out[f"NUM:{pfx}_ms"] = f"{float(ms):.1f}"

    for layer, n in (("L1", "1"), ("L6", "6")):
        lms: Optional[float] = None
        if isinstance(t2, dict) and "latency_per_layer" in t2 and isinstance(
            t2.get("latency_per_layer"), dict
        ):
            pl = t2["latency_per_layer"].get("PACEDefense")
            if isinstance(pl, dict) and layer in pl and isinstance(pl[layer], dict):
                p50 = pl[layer].get("p50")
                if p50 is not None:
                    lms = float(p50)
        if lms is not None:
            out[f"NUM:latency_l{n}_p50_ms"] = f"{lms:.1f}"

    if isinstance(k_sweep, dict):
        cells = k_sweep.get("cells", [])
        if isinstance(cells, list) and len(cells) > 0:
            k1 = [c for c in cells if str(c.get("K")) in ("1", "1.0", "1.00")]
            k5 = [c for c in cells if str(c.get("K")) in ("5", "5.0", "5.00")]
            if k1 and k1[0].get("asr") is not None:
                out["NUM:asr_spq_k1"] = f"{float(k1[0]['asr']) * 100.0:.1f}\\%"
            if k5 and k5[0].get("asr") is not None:
                out["NUM:asr_spq_k5"] = f"{float(k5[0]['asr']) * 100.0:.1f}\\%"

    for cap_id, text in capt.items():
        t = (text or "").strip()
        if t:
            out[f"CAP:{cap_id}"] = t

    for tid in (
        "tier1_psr",
        "tier2_asr",
        "tier3_asr",
        "spq_telemetry",
        "ablation_layers",
        "fpr_per_layer",
        "spq_k_sweep",
        "cost_of_defense",
        "vendor_comparison",
        "latency_per_layer",
    ):
        out[f"TABLE:{tid}"] = f"\\input{{Generated/{tid}.tex}}"

    for fid in (
        "asr_bars",
        "asr_heatmap_tier2",
        "asr_heatmap_tier3",
        "k_sweep_heatmap",
        "layer_ablation_bars",
        "latency_bars",
        "fpr_bars",
        "quorum_histogram",
        "latency_breakdown",
    ):
        out[f"FIG:{fid}"] = f"\\includegraphics[width=\\linewidth]{{Figures/Generated/{fid}.pdf}}"

    for pfx in (
        "tier1_readout",
        "tier2_readout",
        "tier3_readout",
        "ablation_readout",
        "fpr_readout",
        "utility_readout",
        "vendor_readout",
    ):
        out[f"PARA:{pfx}"] = (
            f"\\mbox{{\\emph{{[TODO: paragraph { _latex_escape_id(pfx)}]}}}}"
        )
    for sname in (
        "matrix_protocol",
        "judge_pipeline",
        "adaptive_budget",
    ):
        out[f"SCHEMA:{sname}"] = (
            f"\\mbox{{\\emph{{[TODO: schema { _latex_escape_id(sname)}]}}}}"
        )
    for dname in ("eval_pipeline",):
        out[f"DIAGRAM:{dname}"] = (
            f"\\mbox{{\\emph{{[TODO: diagram { _latex_escape_id(dname)}]}}}}"
        )

    if run_meta and isinstance(run_meta, dict) and "seeds" in run_meta:
        seeds = run_meta.get("seeds")
        if isinstance(seeds, list) and len(seeds) > 0:
            out["NUM:adhoc_seed_count"] = str(int(len(seeds)))

    t2a = tiers.get("tier2_slim5")
    keymap = {
        "WildGuardDefense": "asr_wildguard",
        "LlamaFirewall": "asr_llamafirewall",
        "PromptGuard2Defense": "asr_promptguard2",
    }
    if isinstance(t2a, dict) and isinstance(t2a.get("asr_matrix"), dict):
        am = t2a["asr_matrix"]
        for cname, pid in keymap.items():
            row = am.get(cname)
            if isinstance(row, dict) and row:
                mv = _mean(
                    [float(x) for x in row.values() if isinstance(x, (int, float))]
                )
                if mv is not None:
                    out[f"NUM:{pid}"] = f"{mv * 100.0:.1f}\\%"

    if asr_ci and isinstance(asr_ci, dict):
        pass

    return out


def apply_to_files(
    paths: List[Path],
    values: Dict[str, str],
    *,
    dry: bool = False,
) -> Tuple[int, Set[str]]:
    """Return (total replacements, unmapped keys seen in the scanned files)."""
    n = 0
    unmapped: Set[str] = set()

    warned_once: Set[str] = set()

    def one_file(p: Path) -> None:
        nonlocal n
        text = p.read_text(encoding="utf-8")
        for m in PLACEHOLDER_RE.finditer(text):
            kind, raw = m.group(1), m.group(2)
            inner = _unlatex_id(raw)
            k = f"{kind}:{inner}"
            if k not in values:
                unmapped.add(k)
                if k not in warned_once:
                    warned_once.add(k)
                    print(f"  [bind] warning: unmapped {k!r} (first in {p})", file=sys.stderr)
        newt = text
        for key, val in sorted(values.items(), key=str):
            if ":" not in key:
                continue
            kkind, kinner = key.split(":", 1)
            esc = _latex_escape_id(kinner)
            pat = f"<<{kkind}:{esc}>>"
            if pat in newt:
                c = newt.count(pat)
                n += c
                newt = newt.replace(pat, val)
        if not dry and newt != text:
            p.write_text(newt, encoding="utf-8")

    for p in paths:
        p = p.resolve()
        if not p.is_file():
            print(f"  [bind] skip missing {p}", file=sys.stderr)
            continue
        one_file(p)
    return n, unmapped


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run_dir", type=Path, help="Thesis run root (e.g. logs/thesis/0123-.../)")
    ap.add_argument(
        "--out",
        type=Path,
        help="Output JSON (default: Overleaf/Generated/placeholder_values.json)",
    )
    ap.add_argument(
        "--captions",
        type=Path,
        default=PROJECT_ROOT / "docs" / "captions.yaml",
        help="Caption registry (YAML)",
    )
    ap.add_argument(
        "--apply",
        type=Path,
        nargs="+",
        default=None,
        metavar="TEX",
        help="In-place .tex; only keys in the values map are replaced. Repeat files allowed.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="List substitutions only; do not write JSON or .tex",
    )
    args = ap.parse_args()

    run_dir: Path = args.run_dir
    out_path = (args.out or (PROJECT_ROOT / "Overleaf" / "Generated" / "placeholder_values.json")).resolve()

    vals = build_placeholder_values(run_dir, caption_file=Path(args.captions).resolve())
    for k in list(vals.keys()):
        if k.startswith("CAP:") and not vals.get(k, "").strip():
            del vals[k]

    if not args.dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(dict(sorted(vals.items())), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[bind] wrote {out_path} ({len(vals)} keys)")

    if args.dry_run:
        print(f"[bind] dry-run: would write {len(vals)} keys to {out_path}")
        for k, v in list(sorted(vals.items()))[:50]:
            print(f"  {k!r} -> {v!r}")
        if len(vals) > 50:
            print(f"  ... ({len(vals) - 50} more)")

    if args.apply is not None:
        n, um = apply_to_files(list(args.apply), vals, dry=args.dry_run)
        if args.dry_run:
            print(
                f"[bind] DRY: would perform {n} token replacements; "
                f"{len(um)} distinct unmapped id(s) in scan (warnings above)"
            )
        else:
            print(
                f"[bind] applied {n} replacements; "
                f"{len(um)} distinct unmapped id(s) in scan (warnings above)"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
