#!/usr/bin/env python3
"""Verify that every cell in a thesis run has complete call logs.

Usage::

    python scripts/setup/verify_call_logs.py logs/thesis/<ts>/

Exit codes:
    0  GREEN  — all manifests met, every JSONL valid.
    1  AMBER  — some missing roles or manifest mismatch.
    2  RED    — invalid JSONL or zero calls in any cell.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple


REQUIRED_FIELDS = {"ts", "role", "model"}


def _scan_cells(run_dir: Path) -> List[Path]:
    """Return all directories that contain a calls/ subdirectory."""
    cells: List[Path] = []
    for calls_dir in sorted(run_dir.rglob("calls")):
        if calls_dir.is_dir():
            cells.append(calls_dir.parent)
    return cells


def _validate_jsonl(path: Path) -> Tuple[int, List[str]]:
    """Validate a JSONL file. Returns (line_count, list_of_errors)."""
    errors: List[str] = []
    count = 0
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"  line {i}: invalid JSON: {e}")
            continue
        count += 1
        missing = REQUIRED_FIELDS - set(rec.keys())
        if missing:
            errors.append(f"  line {i}: missing fields: {missing}")
        if "response" not in rec and "error" not in rec:
            errors.append(f"  line {i}: neither 'response' nor 'error' present")
    return count, errors


def _check_cell(cell_dir: Path, require_manifest: bool = False) -> Tuple[str, List[str]]:
    """Check one cell. Returns (status, messages)."""
    calls_dir = cell_dir / "calls"
    if not calls_dir.exists():
        return "RED", [f"  {cell_dir}: no calls/ directory"]

    jsonl_files = sorted(calls_dir.glob("*.jsonl"))
    if not jsonl_files:
        return "RED", [f"  {cell_dir}: calls/ exists but no .jsonl files"]

    manifest_path = calls_dir / "_manifest.json"
    manifest = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    elif require_manifest:
        manifest = {}

    msgs: List[str] = []
    status = "GREEN"
    total_calls = 0
    roles_found: List[str] = []

    for jf in jsonl_files:
        role = jf.stem
        roles_found.append(role)
        count, errors = _validate_jsonl(jf)
        total_calls += count
        if errors:
            status = "RED"
            msgs.append(f"  {jf.name}: {len(errors)} errors")
            msgs.extend(errors[:5])
        if count == 0:
            status = "RED"
            msgs.append(f"  {jf.name}: 0 valid records")

    if manifest:
        expected_roles = set(manifest.get("roles_present", []))
        missing_roles = expected_roles - set(roles_found)
        if missing_roles:
            if status != "RED":
                status = "AMBER"
            msgs.append(f"  manifest expects roles {expected_roles}, missing: {missing_roles}")

        lo = manifest.get("expected_calls_min", 1)
        hi = manifest.get("expected_calls_max", 999999)
        if total_calls < lo or total_calls > hi:
            if status != "RED":
                status = "AMBER"
            msgs.append(f"  call count {total_calls} outside manifest range [{lo}, {hi}]")
    elif require_manifest:
        if status != "RED":
            status = "AMBER"
        msgs.append("  _manifest.json missing (required by --require-manifest)")

    if not msgs:
        msgs.append(f"  {cell_dir.name}: {total_calls} calls across {roles_found}")

    return status, msgs


def _print_sample_thread(cell_dir: Path) -> None:
    """Reconstruct and print a sample conversation thread from a cell."""
    calls_dir = cell_dir / "calls"
    records: List[Dict] = []
    for jf in sorted(calls_dir.glob("*.jsonl")):
        for line in jf.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    records.sort(key=lambda r: r.get("ts", 0))

    print(f"\n  --- Sample thread from {cell_dir.name} ({len(records)} calls) ---")
    for rec in records[:10]:
        role = rec.get("role", "?")
        model = rec.get("model", "?")
        latency = rec.get("latency_ms", 0)
        prompt_preview = ""
        if "prompt" in rec:
            p = rec["prompt"]
            if isinstance(p, list):
                last_msg = p[-1].get("content", "") if p else ""
                prompt_preview = last_msg[:80]
            else:
                prompt_preview = str(p)[:80]
        resp_preview = str(rec.get("response", ""))[:80]
        error = rec.get("error", "")
        tag = f"[ERR: {error[:40]}]" if error else ""
        print(f"    {role:12s} ({model:30s}) {latency:7.0f}ms  "
              f"P: {prompt_preview!r}  R: {resp_preview!r} {tag}")
    if len(records) > 10:
        print(f"    ... +{len(records) - 10} more calls")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, help="Root of a thesis run")
    ap.add_argument("--sample", type=int, default=3,
                    help="Number of cells to sample for thread reconstruction")
    ap.add_argument(
        "--require-manifest",
        action="store_true",
        help="Treat missing calls/_manifest.json as AMBER",
    )
    args = ap.parse_args()

    cells = _scan_cells(args.run_dir)
    if not cells:
        # No calls/ dirs at all — check if there is at least a top-level calls/
        top_calls = args.run_dir / "calls"
        if top_calls.exists():
            cells = [args.run_dir]
        else:
            print(f"[verify_call_logs] no calls/ directories found under {args.run_dir}")
            return 1

    print(f"[verify_call_logs] scanning {len(cells)} cell(s) under {args.run_dir}")

    overall = "GREEN"
    for cell in cells:
        status, msgs = _check_cell(cell, require_manifest=args.require_manifest)
        label = cell.relative_to(args.run_dir)
        color = {"GREEN": "GREEN", "AMBER": "AMBER", "RED": "RED  "}[status]
        print(f"\n[{color}] {label}")
        for m in msgs:
            print(m)
        if status == "RED":
            overall = "RED"
        elif status == "AMBER" and overall != "RED":
            overall = "AMBER"

    sample_cells = random.sample(cells, min(args.sample, len(cells)))
    for sc in sample_cells:
        _print_sample_thread(sc)

    exit_code = {"GREEN": 0, "AMBER": 1, "RED": 2}[overall]
    print(f"\n[verify_call_logs] OVERALL: {overall}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
