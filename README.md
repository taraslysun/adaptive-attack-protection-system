# adaptive-attack-protection-system (`aaps`)

[![smoke](https://github.com/taraslysun/adaptive-attack-protection-system/actions/workflows/smoke.yml/badge.svg)](https://github.com/taraslysun/adaptive-attack-protection-system/actions/workflows/smoke.yml)

**PACE — Plan, Agree, Confirm, Execute** — control-plane defence for tool-using LLM agents under adaptive prompt-injection attack. Companion code for the UCU bachelor thesis *"Development of a Protection System for AI Agents Against Adaptive Attacks"* (Taras Lysun, 2026).

The package ships:

- **PACE** control plane (planner + CFI gate + K-of-q quorum + NLI redundancy filter).
- **All 14 paper-headline defences** (StruQ, SecAlign, MELON, A-MemGuard, SmoothLLM, Spotlighting, RPO, Circuit Breakers, DataSentinel, TrustRAG, Llama Guard, WildGuard, Granite Guardian, Constitutional Classifiers) plus 3 extras (PromptGuard 1+2, LlamaFirewall, PromptSandwiching).
- **All 5 slim-5 adaptive attacks** (PAIR, PoisonedRAG, RL, HumanRedTeam, SupplyChain) and 6 legacy ones (TAP, Crescendo, AdvPrompter, GCG, MAP-Elites, static).
- **Real GRPO training** via `trl.GRPOTrainer` (Shao 2024 DeepSeekMath; group-relative advantage, KL term against frozen reference, PPO-clipped policy ratios).
- **Evaluation harness** with AgentDojo / InjecAgent / AgentHarm / τ-bench adapters, multi-backend LLM judge, bootstrap-CI ASR.
- **CLI:** `aaps system-check`, `aaps run-attack`, `aaps run-bench`, `aaps train-grpo`.
- **12 notebooks**, 10 of which run headlessly with no API keys; 1 paid (AgentDojo cell ≤ $0.50), 1 legacy.

---

## Quickstart

```bash
git clone https://github.com/taraslysun/adaptive-attack-protection-system
cd adaptive-attack-protection-system
python -m venv .venv && source .venv/bin/activate

# Core only — enough for `aaps system-check` + 99% of pytest
pip install -e .

# Optional extras
pip install -e ".[grpo]"        # real GRPO training (transformers + trl + accelerate)
pip install -e ".[guards]"      # Llama Guard / Granite Guardian (transformers)
pip install -e ".[notebooks]"   # jupyter + matplotlib + pandas + nbconvert
pip install -e ".[deepagent]"   # full DeepAgent (Qdrant memory + multimodal RAG + LangChain ToolSuite)
pip install -e ".[all]"         # everything

cp .env.example .env            # fill OPENROUTER_API_KEY or one of the providers

aaps system-check               # GREEN dashboard if everything wired
pytest tests/ -q                # 168 passed
```

Run notebooks: `jupyter lab notebooks/`.

---

## What's in the box

```
aaps/
  agent/                  ReAct DeepAgent + LocalAgent + Qdrant memory + multimodal RAG
  attacks/
    _core/                BaseAttack PSSU loop + scoring + model registry (23 models)
    slim5/                PAIR, PoisonedRAG, RL (incl. real GRPO), HumanRedTeam, SupplyChain
    legacy/               TAP, Crescendo, AdvPrompter, GCG, MAP-Elites, static
    runners/              CLI launchers per family
  defenses/
    pace/                 Planner, K Executors, CFI gate, AgreementVoter, NLI filter, trace logger
    baselines/            17 single-paper defences (14 paper-headline + 3 extras)
    integrity/            Legacy SPQ multi-layer stack (kept for ablation)
    base_defense.py       BaseDefense + DefenseResult contract
  evaluation/
    defense_benchmark.py  DefenseBenchmark.run_matrix → per-cell ASR with bootstrap CI
    external_benchmarks.py AgentDojo + InjecAgent adapters
    benchmarks/           AgentHarm + τ-bench adapters
    llm_judge.py          5 backends (openrouter/openai/ollama/litellm/keyword) + 4-level rubric
    metrics.py            Latency / utility / FPR helpers
    call_logger.py        Optional JSONL audit trail
  scripts/
    check_services.py     Probe Qdrant / Ollama / sandbox / API keys
    system_check.py       One-shot green/yellow/red dashboard
  cli.py                  `aaps` console script
notebooks/                12 runnable notebooks (00..99)
scripts/                  Headline runners + tau-bench utility + L3 single-cell paid run
tests/                    168 default tests + 17 opt-in (ollama, services, grpo)
docker-compose.yml        Qdrant + sandbox local services
```

---

## Architecture

PACE separates the agent's decision into a **control plane** (which tools may fire, agreed across K independent observers) and a **data plane** (the actual tool execution). The control plane is bound at session start, before any untrusted byte enters the loop.

```
User query  ──►  Planner  ──►  Shadow plan π̂ (allowed tools + arg shapes)
                                    │
                                    │  every proposed tool call
                                    ▼
   ┌─────────────────────────  CFI gate  ────────────────────────┐
   │  set-membership: is (tool, arg-shape) ∈ π̂ ?                 │
   └────────┬────────────────────────────────────────────────────┘
            │ pass                                  │ fail
            ▼                                       ▼
   K Executors (one per evidence cluster) ──►   block + log
            │
            ▼  per-cluster proposal
   AgreementVoter (q-of-K on canonical (tool, args))
            │
            ▼ agreement ≥ q
        fire tool
```

Knobs (defaults match thesis Ch. 4 verbatim): `K=5, q=3, embedder='all-MiniLM-L6-v2', τ_NLI=0.70, τ_cos=0.92, planner_temp=0.0, executor_temp=0.0`.

Two falsifiable invariants:
- **CFI:** `cfi_violation_count == 0` end-to-end (any non-zero is a bug, not a tunable FPR).
- **Quorum:** every fired call has `agreement ≥ q` recorded in the trace.

---

## Attack pool

11 attacks, 5 in headline matrix (`slim5`), 6 legacy.

| Family | Tier | Module | Notes |
|---|---|---|---|
| PAIR | slim-5 | `aaps.attacks.slim5.pair.attack:PAIRAttack` | Chao 2023 — black-box iterative refinement |
| PoisonedRAG | slim-5 | `aaps.attacks.slim5.poisoned_rag.attack:PoisonedRAGAttack` | Zou 2024 — corpus poisoning |
| RL | slim-5 | `aaps.attacks.slim5.rl.attack:RLAttack` | two modes: `elite_selection` (default, no GPU) or `grpo_real` (TRL backend) |
| HumanRedTeam | slim-5 | `aaps.attacks.slim5.human_redteam.attack:HumanRedTeamAttack` | Perez 2022 / Ganguli 2022 strategy schedule |
| SupplyChain | slim-5 | `aaps.attacks.slim5.supply_chain.attack:SupplyChainAttack` | Greshake 2023 + OWASP — MCP descriptor mutations |
| TAP | legacy | `aaps.attacks.legacy.tap.attack:TAPAttack` | Mehrotra 2024 — tree of attacks with pruning |
| Crescendo | legacy | `aaps.attacks.legacy.crescendo.attack:CrescendoAttack` | Russinovich 2025 — multi-turn escalation |
| AdvPrompter | legacy | `aaps.attacks.legacy.advprompter.attack:AdvPrompterAttack` | Paulus 2024 — adaptive adversarial prefix |
| GCG | legacy | `aaps.attacks.legacy._legacy.gradient_attack.gcg:GCGAttack` | Zou 2023 — white-box gradient suffix |
| MAP-Elites | legacy | `aaps.attacks.legacy._legacy.search_attack.attack:SearchAttack` | quality-diversity attack archive |
| Static | legacy | `aaps.attacks.legacy.static.static_attacks` | one-shot template injection |

All inherit `BaseAttack` and follow the PSSU (Propose / Score / Select / Update) loop with `AttackConfig.budget` controlling the iteration count.

### Real GRPO training

```bash
aaps train-grpo --policy HuggingFaceTB/SmolLM-135M-Instruct --steps 50 --num-generations 4
```

Wraps `trl.GRPOTrainer` (Shao 2024 *DeepSeekMath*, arXiv:2402.03300). Real implementation: group-relative advantage normalisation, KL term against frozen reference (β=0.04), PPO-clipped policy ratios (ε=0.2). The legacy hand-rolled pairwise loss is retained as `LEGACY mode B` for log reproducibility.

Verified: tiny SmolLM-135M completes 2 GRPO steps with finite loss in ~30 s on CPU (`tests/test_grpo.py`).

---

## Defence pool

All 14 paper-headline defences from thesis §4.4 + 3 extras. 18 classes total.

| # | Paper §4.4 | Class | Backend | Status |
|---:|---|---|---|:---:|
| 1 | StruQ | `StruQDefense` | rule-based | ✅ |
| 2 | SecAlign | `SecAlignDefense` | rule-based | ✅ |
| 3 | MELON | `MELONDefense` | masked re-execution | ✅ |
| 4 | A-MemGuard | `AMemGuard` | memory-write rules | ✅ |
| 5 | SmoothLLM | `SmoothLLMDefense` | input perturbation ensemble | ✅ |
| 6 | Spotlighting | `Spotlighting` | datamarking | ✅ |
| 7 | RPO | `RPODefense` | robust prompt optimisation | ✅ |
| 8 | Circuit Breakers | `CircuitBreakerDefense` | training-time anchor | ✅ |
| 9 | DataSentinel | `DataSentinelDefense` | LLM-judge style | ✅ |
| 10 | TrustRAG | `RAGuard` | retrieval clustering | ✅ |
| 11 | Llama Guard | `LlamaGuardDefense` | Ollama (`llama-guard3:8b`) or HF gated | ⚠️ degrades to allow-all if neither reachable |
| 12 | WildGuard | `WildGuardDefense` | HF (AI2 licence) | ✅ |
| 13 | Granite Guardian | `GraniteGuardianDefense` | HF (`ibm-granite/granite-guardian-3.0-2b`, public) | ✅ MPS-tested |
| 14 | Constitutional Classifiers | `ConstitutionalClassifiersDefense` | regex constitution + `unitary/toxic-bert` | ⚠️ academic proxy of Sharma 2025 (Anthropic weights closed) |
| — | PACE (this thesis) | `PACEDefense` | Planner LLM + K Executor LLMs (Ollama or remote) | ✅ |
| ext | PromptGuard 1 | `PromptGuardFilter` | Meta classifier | ✅ |
| ext | PromptGuard 2 | `PromptGuard2Defense` | Meta classifier (HF public) | ✅ |
| ext | LlamaFirewall | `LlamaFirewall` | Meta 2025 | ✅ |
| ext | PromptSandwiching | `PromptSandwiching` | folk technique | ✅ |

Verifiable: `pytest tests/test_l0_runnability.py -q` exercises all 18 classes against the five hooks.

---

## Notebooks

12 notebooks under `notebooks/`. **10 run headlessly** without any API keys; **2 are documented skips** (paid + legacy).

| # | Notebook | What it shows | Runnable in CI |
|---:|---|---|:---:|
| 00 | `00_setup_and_agent.ipynb` | env probe + DeepAgent build | ✓ |
| 01 | `01_static_attacks.ipynb` | static-template injection × MockAgent | ✓ |
| 02 | `02_pair_attack.ipynb` | PAIR PSSU loop | ✓ |
| 03 | `03_poisoned_rag_attack.ipynb` | PoisonedRAG construction + cycle | ✓ |
| 04 | `04_supply_chain_attack.ipynb` | SupplyChain skill scenarios | ✓ |
| 05 | `05_pace_defense.ipynb` | **PACE deep-dive** — PACEPlan, CFI gate, AgreementVoter | ✓ |
| 06 | `06_benchmark_comparison.ipynb` | DefenseBenchmark with PACE alongside 6 baselines + measured Util Δ table | ✓ |
| 07 | `07_deep_agent_demo.ipynb` | DeepAgent + AgentConfig + ToolSuite surface | ✓ |
| 08 | `08_agentdojo_benchmark.ipynb` | AgentDojo workspace × Mistral via OpenRouter | ⊘ paid (≤ $0.50) |
| 09 | `09_adaptive_budget_sweep.ipynb` | PAIR over budget ∈ {2, 4, 8} | ✓ |
| 10 | `10_deepagent.ipynb` | MemoryManager + MultimodalRetrieval (Qdrant) | ✓ |
| 99 | `99_mim_LEGACY.ipynb` | legacy MIM stack | ⊘ historical |

Run them all: `pytest -m notebooks tests/test_notebooks.py -q` → 10 passed, 2 doc-skip.

---

## CLI

```bash
aaps system-check              # one-shot dashboard (CI-safe)
aaps system-check --json       # machine-readable

aaps train-grpo --policy HuggingFaceTB/SmolLM-135M-Instruct --steps 50

aaps run-attack --family pair --victim mock --n-goals 2

aaps run-bench --benchmark agentdojo --suite workspace --limit 2
aaps run-bench --benchmark injecagent --suite dh_base --limit 4
```

`aaps system-check` expected on a fresh install:

```text
check                                     status  detail
----------------------------------------  ------  ------
pytest_default                            ✓ ok     168 passed in ~75s
services_dashboard                        ·skip   0/10 reachable; ok if running offline
attack:PAIRAttack                         ✓ ok
attack:PoisonedRAGAttack                  ✓ ok
attack:RLAttack                           ✓ ok
attack:HumanRedTeamAttack                 ✓ ok
attack:SupplyChainAttack                  ✓ ok
attack:TAPAttack                          ✓ ok
attack:CrescendoAttack                    ✓ ok
attack:AdvPrompterAttack                  ✓ ok
defence:StruQDefense ··· defence:ConstitutionalClassifiersDefense   ✓ ok ×16
defence:WildGuardDefense                  ·skip   _available=False (HF gated)
defence:LlamaGuardDefense                 ·skip   _available=False (no Ollama tag, no HF licence)

SUMMARY: 25 ok, 2 skipped, 0 failed
OVERALL: ✓ GREEN with documented skips
```

---

## Reproducibility

### Single-cell L3 (cheapest)

```bash
python scripts/l3_single_cell.py --n-goals 10 --budget 4 --seed 0
```

Verified 2026-05-01: PAIR × Mistral-Small-2603 × no_defence × n=10 → **ASR = 0.900** (thesis Table 5.2 reports 0.700). |Δ| = 0.200, **within ±0.30 bootstrap CI tolerance** at n=10. Cost ≈ $0.02. See `docs/reproducing_results.md`.

### Table 5.4 — Utility Δ on τ-bench retail (this repo, n=20, single seed)

```bash
python scripts/run_tau_bench_utility.py
```

Results (Mistral-Small-2603, PACE planner = `llama3.1:8b` on Ollama):

| Defence | Util % | 95% CI | **Util Δ (pp)** |
|---|---:|---:|---:|
| no_defense | 100.0 | [100, 100] | baseline |
| StruQ | 100.0 | [100, 100] | **+0.0** |
| DataSentinel | 100.0 | [100, 100] | **+0.0** |
| MELON | 100.0 | [100, 100] | **+0.0** |
| A-MemGuard | 100.0 | [100, 100] | **+0.0** |
| SmoothLLM | 100.0 | [100, 100] | **+0.0** |
| **PromptGuard2** | **80.0** | **[60, 95]** | **−20.0** (1/20 benign blocked) |
| **PACE** | 100.0 | [100, 100] | **+0.0** |

Headline: **PACE is utility-neutral on benign retail at n=20.** PromptGuard2's −20 pp result is a confirmed FPR (95 % CI excludes 100 %). Other defences indistinguishable from no_defense at this n.

### Logs convention

Every run writes to `logs/thesis/<HHMM-DDMMYYYY>_<tag>/`. Old logs immutable. Per-PACE-decision records under `logs/.../pace_traces.jsonl` with event types `plan_emit, cfi_check, executor_propose, agreement, tool_fire, tool_block`.

---

## External services

`scripts/check_services.py` probes Qdrant, Ollama, sandbox container, OpenRouter, OpenAI, Anthropic, Gemini, HuggingFace, WandB, LiteLLM proxy. Returns 0 unless a *configured* service is failing.

```bash
docker compose up -d qdrant sandbox     # local services
python -m aaps.scripts.check_services
```

| Service | Probe |
|---|---|
| Qdrant | `GET /healthz` |
| Ollama | `GET /api/tags` |
| Sandbox | `docker exec aaps-sandbox echo ok` |
| OpenRouter / OpenAI / Anthropic / Gemini | API `/v1/models` |
| HuggingFace | `whoami-v2` |
| WandB | `api.wandb.ai` |

---

## Configuration

Copy `.env.example` to `.env` and fill the providers you want.

| Variable | Purpose | Default |
|---|---|---|
| `OPENROUTER_API_KEY` | recommended single-key backend | — |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` | direct vendor | — |
| `OLLAMA_URL` | local Ollama HTTP | `http://localhost:11434` |
| `OLLAMA_TARGET_MODEL` | victim default | `llama3.1:8b` |
| `OLLAMA_JUDGE_MODEL` | judge default | `qwen2.5:1.5b` |
| `LLAMA_GUARD_OLLAMA_MODEL` | Llama Guard backend | `llama-guard3:8b` |
| `HF_TOKEN` | HF gated repos (Llama Guard, WildGuard) | — |
| `QDRANT_URL` / `QDRANT_API_KEY` | DeepAgent memory + multimodal RAG | `http://localhost:6333` |
| `WANDB_API_KEY` / `WANDB_PROJECT` | optional run tracking | — |

Full list in `.env.example`.

---

## Tests

```bash
pytest tests/ -q                                 # 168 passed (default suite, no network)
pytest -m grpo tests/test_grpo.py -q             # +4 (real GRPO smoke)
pytest -m notebooks tests/test_notebooks.py -q   # +10 passed, 2 doc-skip
pytest -m ollama tests/test_e2e_ollama.py -q     # +8 (laptop, free)
pytest -m services tests/test_e2e_services.py -q # +5 (docker compose up qdrant sandbox)
```

CI runs the default 168-test suite on Python 3.10 / 3.11 / 3.12 (`.github/workflows/smoke.yml`).

---

## Documentation

- `docs/quickstart.md` — clone → install → smoke → first notebook
- `docs/threat_model.md` — adversary capabilities, hook points, second-mover protocol (matches thesis Ch. 3)
- `docs/reproducing_results.md` — reproducing thesis Table 5.x cells (incl. measured L3 result)
- `docs/runnability.md` — every attack + every defence runnability matrix
- `THESIS_VS_CODE_AUDIT.md` — claim-by-claim mapping from thesis to code paths
- `PRODUCTION_READINESS_AUDIT.md` — pre-sync state of the public repo
- `CHANGELOG.md` — release notes (currently 0.2.0.dev0)

---

## Citation

```bibtex
@thesis{lysun2026pace,
  title  = {Development of a Protection System for AI Agents Against Adaptive Attacks},
  author = {Taras Lysun},
  school = {Ukrainian Catholic University, Faculty of Applied Sciences},
  year   = {2026},
  url    = {https://github.com/taraslysun/adaptive-attack-protection-system}
}
```

---

## License

MIT. See `LICENSE`.
