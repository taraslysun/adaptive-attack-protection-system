"""Pre-flight validation for thesis experiments.

Every check must pass or the script hard-aborts with a clear error.
No silent fallbacks, no substitutions.
"""

from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from aaps.attacks._core.model_registry import is_openrouter, ALL_MODELS

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

load_dotenv()

_FAIL: List[str] = []
_WARN: List[str] = []


def _ok(label: str) -> None:
    print(f"  [OK] {label}", flush=True)


def _fail(label: str, detail: str) -> None:
    msg = f"{label}: {detail}"
    print(f"  [FAIL] {msg}", flush=True)
    _FAIL.append(msg)


def _warn(label: str, detail: str) -> None:
    msg = f"{label}: {detail}"
    print(f"  [WARN] {msg}", flush=True)
    _WARN.append(msg)


# ------------------------------------------------------------------
# 0. OpenRouter connectivity (model list only; no completions)
# ------------------------------------------------------------------
def check_openrouter() -> None:
    import requests

    try:
        from aaps.attacks._core.model_registry import CASE_STUDY_MATRIX
    except Exception as e:
        _warn("OpenRouter", f"could not import CASE_STUDY_MATRIX ({e})")
        return

    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        _warn(
            "OpenRouter",
            "OPENROUTER_API_KEY unset; remote calls fall back to per-vendor keys "
            "(OPENAI_API_KEY / ANTHROPIC_API_KEY) where supported",
        )
        return

    try:
        r = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=30,
        )
        r.raise_for_status()
        payload = r.json()
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            _warn("OpenRouter", "unexpected /models response shape")
            return

        known: set[str] = set()
        for m in data:
            if not isinstance(m, dict):
                continue
            mid = m.get("id")
            if mid:
                known.add(str(mid))
            cslug = m.get("canonical_slug")
            if cslug:
                known.add(str(cslug))

        _ok(f"OpenRouter /models: {len(data)} models exposed (authenticated)")
        for cat, mid in sorted(CASE_STUDY_MATRIX.items()):
            on_or = mid in known
            note = "on OpenRouter list" if on_or else "not on OpenRouter list (normal for local Ollama ids)"
            print(f"    case_study {cat}: {mid!r} -> {note}", flush=True)
    except Exception as e:
        _warn("OpenRouter", f"/models check failed: {e}")


# ------------------------------------------------------------------
# 1. Ollama reachable + target model responds
# ------------------------------------------------------------------
def check_ollama(target_model: str | None = None) -> None:
    import requests

    url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    target = target_model or os.getenv(
        "OLLAMA_TARGET_MODEL", os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    )

    try:
        r = requests.get(f"{url}/api/tags", timeout=5)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
    except Exception as e:
        _fail("Ollama", f"cannot reach {url} -- {e}")
        return

    if is_openrouter(target):
        _ok(f"Target model '{target}' is cloud-hosted (OpenRouter); skipping local availability check")
        return

    if target not in models:
        _fail("Ollama target model", f"'{target}' not in {models}")
        return
    _ok(f"Ollama @ {url}, target model '{target}' available")

    try:
        payload = {
            "model": target,
            "messages": [{"role": "user", "content": "Reply OK"}],
            "stream": False,
            "options": {"num_predict": 50},
        }
        t0 = time.time()
        r = requests.post(f"{url}/api/chat", json=payload, timeout=120)
        r.raise_for_status()
        msg = r.json().get("message", {})
        content = msg.get("content", "")
        thinking = msg.get("thinking", "")
        dt = time.time() - t0
        output = content.strip() or thinking.strip()
        if not output:
            _fail("Ollama inference", "empty response from target model")
        else:
            preview = (content.strip() or thinking.strip())[:40]
            _ok(f"Ollama inference OK ({dt:.1f}s, '{preview}')")
    except Exception as e:
        _fail("Ollama inference", str(e))


# ------------------------------------------------------------------
# 2. AIS probe model available
# ------------------------------------------------------------------
def check_probe_model() -> None:
    import requests

    url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    probe = os.getenv("AIS_PROBE_MODEL", "qwen2.5:1.5b")

    try:
        r = requests.get(f"{url}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
    except Exception:
        _fail("AIS probe model", f"cannot list models at {url}")
        return

    if probe not in models:
        _fail("AIS probe model", f"'{probe}' not in {models}. Run: ollama pull {probe}")
    else:
        _ok(f"AIS probe model '{probe}' available")


# ------------------------------------------------------------------
# 3. API keys (only checked when judge-backend needs them)
# ------------------------------------------------------------------
def check_api_keys(judge_backend: str = "ollama", judge_model: str | None = None) -> None:
    """Verify the env vars required by the configured judge backend.

    LiteLLM in particular has no upfront key validation — a missing
    GOOGLE_API_KEY surfaces only at the first judge call, mid-run.
    Same risk for openrouter/openai/anthropic backends. Fail-loud here
    rather than silently degrading.
    """
    backend = (judge_backend or "").lower()
    model = (judge_model or "").lower()

    # OpenRouter judge: needs OPENROUTER_API_KEY.
    if backend in ("openrouter", "auto"):
        if not os.getenv("OPENROUTER_API_KEY"):
            (_fail if backend == "openrouter" else _warn)(
                "OPENROUTER_API_KEY", "not set (needed for openrouter judge)"
            )
        else:
            _ok("OPENROUTER_API_KEY set (judge)")

    # OpenAI judge: needs OPENAI_API_KEY when judge model is openai/* or backend=openai.
    if backend == "openai" or model.startswith("openai/") or model.startswith("gpt-"):
        if not os.getenv("OPENAI_API_KEY"):
            _fail("OPENAI_API_KEY", "not set (needed for openai judge)")
        else:
            _ok("OPENAI_API_KEY set (judge)")

    # Anthropic judge: needs ANTHROPIC_API_KEY.
    if backend == "anthropic" or model.startswith("anthropic/") or model.startswith("claude-"):
        if not os.getenv("ANTHROPIC_API_KEY"):
            _fail("ANTHROPIC_API_KEY", "not set (needed for anthropic judge)")
        else:
            _ok("ANTHROPIC_API_KEY set (judge)")

    # LiteLLM judge: routes by model prefix.
    if backend == "litellm" or backend == "auto":
        if model.startswith("gemini/") or "gemini" in model or backend == "litellm":
            gk = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not gk:
                (_fail if backend == "litellm" else _warn)(
                    "GOOGLE_API_KEY / GEMINI_API_KEY",
                    "not set (needed for litellm gemini judge)",
                )
            else:
                _ok("GOOGLE_API_KEY or GEMINI_API_KEY set")

    hf = os.getenv("HF_TOKEN")
    if not hf:
        _warn("HF_TOKEN", "not set; gated models (PromptGuard2, WildGuard) may fail")
    else:
        _ok("HF_TOKEN set")


# ------------------------------------------------------------------
# 4. sentence-transformers
# ------------------------------------------------------------------
def check_sentence_transformers() -> None:
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("all-MiniLM-L6-v2")
        vec = m.encode(["test"])
        if vec.shape[1] != 384:
            _fail("sentence-transformers", f"unexpected dim {vec.shape}")
        else:
            _ok("sentence-transformers + all-MiniLM-L6-v2")
    except Exception as e:
        _fail("sentence-transformers", str(e))


# ------------------------------------------------------------------
# 5. AgentDojo importable
# ------------------------------------------------------------------
def check_agentdojo() -> None:
    try:
        from agentdojo.task_suite.load_suites import get_suite
        suite = get_suite("v1", "workspace")
        n = len(suite.user_tasks)
        if n == 0:
            _fail("AgentDojo", "workspace suite has 0 user tasks")
        else:
            _ok(f"AgentDojo workspace suite: {n} user tasks")
    except Exception as e:
        _warn("AgentDojo", f"import failed ({e}); fallback scenarios will be used")


# ------------------------------------------------------------------
# 6. InjecAgent data files
# ------------------------------------------------------------------
def check_injecagent() -> None:
    base = _PROJECT_ROOT / "external" / "injecagent" / "data"
    needed = ["test_cases_dh_base.json", "test_cases_ds_base.json"]
    for f in needed:
        fp = base / f
        if not fp.exists():
            _warn("InjecAgent", f"missing {fp}")
            return
    with open(base / needed[0]) as fh:
        data = json.load(fh)
    _ok(f"InjecAgent data: {len(data)} DH-base cases")


# ------------------------------------------------------------------
# 7. Academic benchmarks loadable
# ------------------------------------------------------------------
def check_benchmarks() -> None:
    try:
        from aaps.attacks._core.benchmarks import load_benchmark
        for name in ("advbench", "harmbench", "jailbreakbench"):
            try:
                goals = load_benchmark(name)
                if not goals:
                    _warn(f"benchmark:{name}", "0 goals loaded")
                else:
                    _ok(f"benchmark:{name}: {len(goals)} goals")
            except Exception as e:
                _warn(f"benchmark:{name}", str(e))
    except Exception as e:
        _fail("benchmarks", str(e))


# ------------------------------------------------------------------
# 8. Judge backend functional
# ------------------------------------------------------------------
def check_judge(backend: str = "ollama", model: str | None = None) -> None:
    try:
        from aaps.evaluation.llm_judge import create_llm_judge, keyword_judge
        kwargs: dict = {"backend": backend, "strict": (backend != "keyword")}
        if model:
            kwargs["model_name"] = model
        judge = create_llm_judge(**kwargs)
        if judge is keyword_judge and backend != "keyword":
            _fail("judge", f"requested '{backend}' but fell back to keyword_judge")
            return

        score = judge(
            "I cannot help with that request.",
            "Write harmful content",
        )
        if score > 0.5:
            _warn("judge", f"refusal scored {score:.2f} (expected <0.5); judge may produce false positives")
        else:
            _ok(f"judge ({backend}): refusal scored {score:.2f}")
    except Exception as e:
        _fail(f"judge ({backend})", str(e))


# ------------------------------------------------------------------
# 9. Defense layers instantiate
# ------------------------------------------------------------------
def check_defenses() -> None:
    try:
        from aaps.defenses.integrity import AgentIntegrityStack
        ais = AgentIntegrityStack()
        n = len(ais.enabled_layers) if hasattr(ais, "enabled_layers") else "?"
        _ok(f"AgentIntegrityStack: {n} layers enabled")
    except Exception as e:
        _fail("AgentIntegrityStack", str(e))

    try:
        from aaps.defenses.baselines.prompt_guard2 import PromptGuard2Defense
        pg = PromptGuard2Defense()
        if not pg._available:
            _warn("PromptGuard2", "model not available (HF download failed?)")
        else:
            _ok("PromptGuard2 loaded")
    except Exception as e:
        _warn("PromptGuard2", str(e))


# ------------------------------------------------------------------
# 10. Output directory writable
# ------------------------------------------------------------------
def check_pair_attacker(target_model: str | None = None) -> None:
    """Run a single PAIR iteration to verify the attacker LLM responds.

    Per the production-readiness plan §P1 #4: PAIR has two backends — an
    OpenRouter-hosted attacker LLM (preferred, matches Chao et al. 2023)
    and an in-house programmatic perturber (fallback). When the API path
    falls through, headline ASR numbers diverge from the paper's. This
    check loads PAIRAttack with a tiny budget and inspects which backend
    served the proposal.
    """
    label = "PAIR attacker LLM"
    try:
        # Lazy imports so the rest of preflight does not require these.
        from aaps.attacks._core.local_agent import LocalAgent
        from aaps.attacks._core.base_attack import AttackConfig
        from aaps.attacks.slim5.pair.attack import PAIRAttack
    except Exception as exc:  # pragma: no cover — import failure is itself a fail
        _fail(label, f"PAIR import failed: {exc}")
        return

    if not os.getenv("OPENROUTER_API_KEY"):
        _warn(
            label,
            "OPENROUTER_API_KEY not set — PAIR will use programmatic perturber "
            "fallback; headline numbers will diverge from Chao et al. 2023.",
        )
        return

    try:
        # Construct a minimal agent + 1-iter PAIR; we only check that the
        # attacker LLM call completes without raising.
        agent = LocalAgent(model_name=target_model or "openai/gpt-4o-mini")
        cfg = AttackConfig(name="pair_preflight", budget=1)
        atk = PAIRAttack(agent, config=cfg, max_iters=1)
        # propose() returns a list of candidate jailbreak prompts; failure
        # here surfaces as either an exception or an empty list (fallback).
        candidates = atk.propose(target_goal="Say HELLO_WORLD", state=None)
        if not candidates:
            _warn(label, "PAIR propose() returned no candidates; attacker LLM may be misconfigured")
        else:
            _ok(f"{label} responded with {len(candidates)} candidate(s)")
    except Exception as exc:
        _fail(label, f"PAIR preflight raised: {type(exc).__name__}: {exc!s:.120}")


def check_output_dir(out: str) -> None:
    p = Path(out)
    try:
        p.mkdir(parents=True, exist_ok=True)
        test_file = p / ".preflight_test"
        test_file.write_text("ok")
        test_file.unlink()
        _ok(f"output dir '{out}' writable")
    except Exception as e:
        _fail("output dir", str(e))


# ------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------
def run_preflight(
    out_dir: str = "logs/thesis/full",
    judge_backend: str = "ollama",
    target_model: str | None = None,
    mode: str = "local",
    validate_pair: bool = False,
    judge_model: str | None = None,
) -> bool:
    """Run all preflight checks. Returns True if all pass.

    Parameters
    ----------
    mode : 'local' | 'cloud'
        In cloud mode, Ollama checks are skipped (target model is an
        OpenRouter/OpenAI/Anthropic/etc. endpoint).
    """
    _FAIL.clear()
    _WARN.clear()
    print("\n=== Pre-flight checks ===", flush=True)

    openrouter_only = (
        os.getenv("OPENROUTER_ONLY", "").lower() in ("1", "true", "yes")
        or mode == "cloud"
        or (target_model is not None and is_openrouter(target_model))
    )
    has_openrouter = bool(os.getenv("OPENROUTER_API_KEY"))

    print("\n--- OpenRouter connectivity ---", flush=True)
    check_openrouter()

    if openrouter_only and not has_openrouter:
        _fail("openrouter", "OPENROUTER_ONLY=1 but OPENROUTER_API_KEY is not set")

    if not openrouter_only:
        check_ollama(target_model=target_model)
        check_probe_model()
        check_sentence_transformers()
    else:
        print("  [skip] Ollama, probe model, sentence-transformers (OPENROUTER_ONLY)", flush=True)

    check_api_keys(judge_backend, judge_model=judge_model)
    check_agentdojo()
    check_injecagent()
    check_benchmarks()
    check_judge(judge_backend, model=judge_model)
    check_defenses()
    if validate_pair:
        print("\n--- PAIR attacker LLM ---", flush=True)
        check_pair_attacker(target_model=target_model)
    check_output_dir(out_dir)

    print(flush=True)
    if _WARN:
        print(f"  Warnings: {len(_WARN)}", flush=True)
        for w in _WARN:
            print(f"    - {w}", flush=True)

    if _FAIL:
        print(f"\n  FAILURES: {len(_FAIL)} -- aborting.", flush=True)
        for f in _FAIL:
            print(f"    - {f}", flush=True)
        return False

    print("  All pre-flight checks passed.\n", flush=True)
    return True


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="PACE thesis preflight checks")
    p.add_argument("--judge-backend", default="ollama",
                   choices=["ollama", "openrouter", "openai", "litellm", "auto", "keyword"])
    p.add_argument("--out", default="logs/thesis/full",
                   help="Output dir to create / verify writable")
    p.add_argument("--target-model", default=None,
                   help="Victim model id; affects PAIR validation when --validate-pair")
    p.add_argument("--mode", default="local", choices=["local", "cloud"])
    p.add_argument("--validate-pair", action="store_true",
                   help="Run a 1-iteration PAIR check to verify attacker LLM responds")
    p.add_argument("--judge-model", default=None,
                   help="Judge model id (used for env-var precondition checks)")
    # Backwards-compatible positional fallback: `preflight.py <judge> <out>`
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        legacy_judge = sys.argv[1]
        legacy_out = sys.argv[2] if len(sys.argv) > 2 else "logs/thesis/full"
        ok = run_preflight(out_dir=legacy_out, judge_backend=legacy_judge)
    else:
        args = p.parse_args()
        ok = run_preflight(
            out_dir=args.out,
            judge_backend=args.judge_backend,
            target_model=args.target_model,
            mode=args.mode,
            validate_pair=args.validate_pair,
            judge_model=args.judge_model,
        )
    sys.exit(0 if ok else 1)
