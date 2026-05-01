"""Polling monitor for the full evaluation run.

Usage:
    python scripts/monitor_eval.py <logroot>

Polls every 5 minutes, checks each group dir for red flags,
prints a status table, and flags anomalies with recommended actions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Expected tool-first models — PACE must fire on these
TOOL_FIRST = {
    "openai_gpt-4o-mini",
    "google_gemini-2.5-flash",
    "meta-llama_llama-3.3-70b-instruct",
    "qwen_qwen-2.5-72b-instruct",
    "deepseek_deepseek-chat",
}

# Known anomalies that are NOT bugs
EXPECTED_ANOMALIES = {
    ("deepseek_deepseek-chat", "mem_poison_spq_blind"): "Attack-path dependency: expected",
    ("meta-llama_llama-3.3-70b-instruct", "utility_cliff"): "Quorum variance cliff: expected thesis finding",
    ("qwen_qwen-2.5-72b-instruct", "poisrag_bypass"): "Structural PACE gap: expected",
}


def _model_slug(path: Path) -> str:
    return path.name.replace("model_", "")


def _load_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _check_metadata(meta: dict, slug: str) -> list[tuple[str, str, str]]:
    """Returns list of (severity, flag_id, message)."""
    flags = []
    fire_rate = meta.get("spq_fire_rate", None)
    utility = meta.get("spq_utility", None)

    if fire_rate is not None and fire_rate == 0.0 and slug in TOOL_FIRST:
        key = (slug, "fire_rate_zero")
        if key not in EXPECTED_ANOMALIES:
            flags.append(("CRITICAL", "fire_rate_zero",
                          f"fire_rate=0 on tool-first model {slug}. "
                          "PACE gate not firing. Check model routing."))

    if utility is not None and utility == 0.0:
        key = (slug, "utility_cliff")
        if key not in EXPECTED_ANOMALIES:
            flags.append(("CRITICAL", "utility_cliff",
                          f"utility=0 on {slug}. PACE quorum collapse. "
                          "Rerun with PACE_K=3."))
        else:
            flags.append(("INFO", "utility_cliff", EXPECTED_ANOMALIES[key]))

    return flags


def _check_tier1(t1_dir: Path, slug: str) -> list[tuple[str, str, str]]:
    flags = []
    mem = _load_json(t1_dir / "memory_poisoning.json")
    if not mem:
        return flags

    nodef = mem.get("no_defense", {}).get("psr")
    spq = mem.get("PACE", {}).get("psr")
    if nodef is None or spq is None:
        return flags

    if spq > nodef:
        key = (slug, "mem_poison_spq_blind")
        if key in EXPECTED_ANOMALIES:
            flags.append(("INFO", "mem_poison_spq_blind",
                          f"PACE PSR ({spq:.3f}) > no_def ({nodef:.3f}): {EXPECTED_ANOMALIES[key]}"))
        else:
            flags.append(("WARN", "mem_poison_spq_blind",
                          f"PACE PSR ({spq:.3f}) > no_def ({nodef:.3f}) on {slug}. "
                          "Possible PACE blind spot. Check fire_rate + unified.log."))

    # Check all defense columns identical (judge failure)
    defense_psrs = [v.get("psr") for v in mem.values() if isinstance(v, dict) and "psr" in v]
    if len(set(defense_psrs)) == 1 and len(defense_psrs) > 2:
        flags.append(("WARN", "constant_judge",
                      f"All {len(defense_psrs)} defenses show identical PSR={defense_psrs[0]} on {slug}. "
                      "Judge may be returning constant output. Check judge availability."))

    return flags


def _check_tier2(t2_dir: Path, slug: str) -> list[tuple[str, str, str]]:
    flags = []
    slim = _load_json(t2_dir / "slim5_adaptive.json")
    if not slim:
        return flags

    spq_results = slim.get("PACE", {})
    hrt_asr = spq_results.get("HumanRedTeam", {}).get("asr")
    pair_asr = spq_results.get("PAIR", {}).get("asr")

    if hrt_asr == 1.0 and pair_asr == 1.0:
        flags.append(("WARN", "spq_asr_ceiling",
                      f"PACE ASR=1.000 on both HRT+PAIR for {slug}. "
                      "Check cfi_real count in unified.log — gate may be inactive."))

    # Check poisrag bypass
    pois_asr = spq_results.get("PoisonedRAG", {}).get("asr")
    if pois_asr == 1.0:
        key = (slug, "poisrag_bypass")
        if key in EXPECTED_ANOMALIES:
            flags.append(("INFO", "poisrag_bypass",
                          f"PoisRAG ASR=1.0 under PACE on {slug}: {EXPECTED_ANOMALIES[key]}"))
        else:
            flags.append(("WARN", "poisrag_bypass",
                          f"PoisRAG ASR=1.0 under PACE on {slug}. PACE does not intercept retrieval context."))

    return flags


def _check_seeds(model_dir: Path) -> list[tuple[str, str, str]]:
    flags = []
    seed_dirs = sorted(model_dir.glob("seed_*"))
    if len(seed_dirs) < 2:
        return flags

    # Compare tier2 slim5 across seeds — if identical, RNG broken
    asr_sets = []
    for sd in seed_dirs[:3]:
        slim = _load_json(sd / "tier2" / "slim5_adaptive.json")
        if slim:
            asr_sets.append(frozenset(
                (k, v.get("asr")) for k, v in slim.get("no_defense", {}).items()
            ))

    if len(asr_sets) >= 2 and len(set(asr_sets)) == 1:
        flags.append(("WARN", "zero_seed_variance",
                      f"Seed results identical across {len(asr_sets)} seeds in {model_dir.name}. "
                      "Check seed passthrough in run_multiseed_matrix.py."))
    return flags


def check_group(group_dir: Path) -> dict:
    """Returns {slug: [flags]} for all completed model dirs in group."""
    results = {}
    if not group_dir.exists():
        return results

    for model_dir in group_dir.glob("model_*"):
        slug = _model_slug(model_dir)
        flags = []

        # Check each seed
        for seed_dir in sorted(model_dir.glob("seed_*")):
            meta = _load_json(seed_dir / "run_metadata.json")
            flags.extend(_check_metadata(meta, slug))

            t1 = seed_dir / "tier1"
            if t1.exists():
                flags.extend(_check_tier1(t1, slug))

            t2 = seed_dir / "tier2"
            if t2.exists():
                flags.extend(_check_tier2(t2, slug))

        flags.extend(_check_seeds(model_dir))
        results[slug] = flags

    return results


def running_pids(logroot: Path) -> list[int]:
    pids_file = logroot / "pids.txt"
    if not pids_file.exists():
        return []
    pids = [int(p.strip()) for p in pids_file.read_text().splitlines() if p.strip()]
    alive = []
    for pid in pids:
        try:
            os.kill(pid, 0)
            alive.append(pid)
        except ProcessLookupError:
            pass
    return alive


def print_status(logroot: Path, iteration: int) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"\n{'='*70}")
    print(f"[monitor] poll #{iteration} at {ts}  root={logroot}")

    alive = running_pids(logroot)
    print(f"[monitor] running PIDs: {alive if alive else 'all done'}")

    groups = sorted(p for p in logroot.iterdir() if p.is_dir() and p.name.startswith("group_"))
    if not groups:
        print("[monitor] No group dirs found yet.")
        return

    any_critical = False
    for g in groups:
        results = check_group(g)
        if not results:
            print(f"\n  {g.name}: (no model dirs yet)")
            continue
        print(f"\n  {g.name}:")
        for slug, flags in results.items():
            criticals = [f for f in flags if f[0] == "CRITICAL"]
            warns = [f for f in flags if f[0] == "WARN"]
            infos = [f for f in flags if f[0] == "INFO"]
            status = "OK" if not criticals and not warns else ("CRITICAL" if criticals else "WARN")
            print(f"    [{status:8s}] {slug}")
            for sev, fid, msg in flags:
                prefix = "  !!" if sev == "CRITICAL" else ("  >" if sev == "WARN" else "  ~")
                print(f"             {prefix} [{fid}] {msg}")
            if criticals:
                any_critical = True

    if any_critical:
        print("\n[monitor] CRITICAL flags present — manual review required.")
    else:
        print("\n[monitor] No critical flags.")


def main() -> None:
    p = argparse.ArgumentParser(description="Poll eval run for anomalies.")
    p.add_argument("logroot", help="Path to fulleval_<ts> log root dir")
    p.add_argument("--interval", type=int, default=300, help="Poll interval seconds (default 300)")
    p.add_argument("--once", action="store_true", help="Run one check then exit")
    args = p.parse_args()

    logroot = Path(args.logroot)
    if not logroot.exists():
        print(f"[monitor] ERROR: {logroot} does not exist")
        sys.exit(1)

    iteration = 0
    while True:
        iteration += 1
        print_status(logroot, iteration)

        if args.once:
            break

        alive = running_pids(logroot)
        if not alive:
            print(f"\n[monitor] All processes finished. Final check complete.")
            break

        print(f"\n[monitor] Next poll in {args.interval}s ...")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
