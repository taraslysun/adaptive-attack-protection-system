"""List currently-routable cheap OpenRouter models for Phase 2 of the
thesis evaluation matrix.

Queries OpenRouter `/models` to verify each candidate is currently exposed
and that prompt+completion price is within the budget. Persists the chosen
list to ``logs/thesis/<out>/models.json`` along with raw price metadata.

Usage::

    python scripts/setup/list_cheap_models.py \
        --max-input-cost 0.20 \
        --max-output-cost 0.40 \
        --out logs/thesis/<ts_cheap>/models.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass


CANDIDATES = [
    # role-analogue, model-id, role-description
    ("openai_family",       "openai/gpt-oss-120b",
        "OpenAI-family stand-in for gpt-4o-2024-11-20"),
    ("frontier_class",      "qwen/qwen3-235b-a22b-2507",
        "Frontier-class stand-in for claude-sonnet-4"),
    ("llama_small",         "meta-llama/llama-3.1-8b-instruct",
        "Llama small stand-in for llama-3.3-70b"),
    ("google_gemma",        "google/gemma-4-26b-a4b-it",
        "Google-family Gemma 4 (MoE)"),
    ("mistral_small",       "mistralai/mistral-small-3.2-24b-instruct",
        "Mistral coverage with proven tool-use"),
    ("deepseek_chat",       "deepseek/deepseek-chat",
        "Reasoning-family cheap stand-in"),
]


def _coerce_float(x):
    try:
        return float(x)
    except Exception:
        return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--max-input-cost", type=float, default=0.20,
                   help="USD per 1M prompt tokens (default 0.20).")
    p.add_argument("--max-output-cost", type=float, default=0.40,
                   help="USD per 1M completion tokens (default 0.40).")
    p.add_argument("--out", default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    api = os.getenv("OPENROUTER_API_KEY")
    if not api:
        print("[cheap-models] OPENROUTER_API_KEY not set", file=sys.stderr)
        return 2
    try:
        import requests  # noqa
    except Exception as e:
        print(f"[cheap-models] requests not installed: {e}", file=sys.stderr)
        return 2

    r = requests.get(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {api}"},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json().get("data", [])
    by_id = {m.get("id"): m for m in data if isinstance(m, dict)}

    chosen = []
    rejected = []
    for role, mid, desc in CANDIDATES:
        m = by_id.get(mid)
        if m is None:
            rejected.append({"role": role, "id": mid, "reason": "not on /models"})
            continue
        pricing = m.get("pricing", {}) or {}
        prompt_cost = _coerce_float(pricing.get("prompt"))
        completion_cost = _coerce_float(pricing.get("completion"))
        # OpenRouter prices are USD per token; multiply by 1e6 for per-Mtok
        prompt_per_m = prompt_cost * 1e6 if prompt_cost is not None else None
        completion_per_m = completion_cost * 1e6 if completion_cost is not None else None
        ok = (
            prompt_per_m is not None
            and completion_per_m is not None
            and prompt_per_m <= args.max_input_cost
            and completion_per_m <= args.max_output_cost
        )
        rec = {
            "role": role,
            "id": mid,
            "description": desc,
            "context_length": m.get("context_length"),
            "prompt_per_Mtok": prompt_per_m,
            "completion_per_Mtok": completion_per_m,
            "ok": ok,
        }
        (chosen if ok else rejected).append(rec)

    summary = {
        "max_input_cost_per_Mtok": args.max_input_cost,
        "max_output_cost_per_Mtok": args.max_output_cost,
        "chosen": chosen,
        "rejected": rejected,
    }

    if args.out and not args.dry_run:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))
        print(f"[cheap-models] wrote {out}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
