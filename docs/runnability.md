# Runnability matrix

Source of truth: `tests/test_l0_runnability.py`. Last verified 2026-05-02.

## Attacks (5 slim-5 + 3 legacy class-tests)

| Attack | Class loads | One PSSU cycle | Notes |
|---|:---:|:---:|---|
| PAIR | ✓ | ✓ | budget overridden by `max_iters` ctor kwarg |
| PoisonedRAG | ✓ | ✓ | |
| RL (`mode="elite_selection"`) | ✓ | ✓ | default; no torch state |
| RL (`mode="grpo_real"`) | ✓ | requires `[grpo]` extra | tested in `test_grpo.py` |
| HumanRedTeam | ✓ | ✓ | |
| SupplyChain | ✓ | ✓ | toy mutations; not credible MCP attack |
| TAP (legacy) | ✓ | not exercised | |
| Crescendo (legacy) | ✓ | not exercised | |
| AdvPrompter (legacy) | ✓ | not exercised | |

## Defences (15 baselines + PACE)

| Defence | core only | with extras | Notes |
|---|:---:|:---:|---|
| StruQDefense | ✓ | ✓ | rule-based |
| SecAlignDefense | ✓ | ✓ | |
| MELONDefense | ✓ | ✓ | masked re-execution |
| AMemGuard | ✓ | ✓ | |
| SmoothLLMDefense | ✓ | ✓ | |
| Spotlighting | ✓ | ✓ | |
| PromptSandwiching | ✓ | ✓ | |
| RPODefense | ✓ | ✓ | |
| CircuitBreakerDefense | ✓ | ✓ | |
| DataSentinelDefense | ✓ | ✓ | |
| RAGuard (TrustRAG) | ✓ | ✓ | |
| PromptGuard2Defense | ✓ | ✓ | Meta PG2 (HF public) |
| PromptGuardFilter | ✓ | ✓ | Meta PG1 |
| LlamaFirewall | ✓ | ✓ | |
| WildGuardDefense | needs HF + AI2 licence | ✓ if accepted | |
| **LlamaGuardDefense** *(Phase C)* | needs Ollama or HF gated | ✓ when reachable | passes through if not |
| **GraniteGuardianDefense** *(Phase C)* | ✓ HF public | ✓ | |
| **ConstitutionalClassifiersDefense** *(Phase C)* | ✓ regex | ✓ + toxic-bert | academic proxy |
| **PACEDefense** | needs LLM endpoint | ✓ | full plan + quorum |

## Test totals (target repo)

| Category | Count |
|---|---:|
| `test_smoke.py` | 13 |
| `test_e2e_mock.py` | 9 |
| `test_l0_contracts.py` | 77 |
| `test_l0_runnability.py` | 69 |
| **Default `pytest tests/`** | **168 passed** |
| `-m ollama` | 8 |
| `-m services` | 5 |
| `-m grpo` | 4 |

## External dependencies per layer

- **CI / default** — only core `pyproject.toml` deps; no GPU, no network, no Ollama.
- **`-m ollama`** — Ollama daemon + `llama3.1:8b` + `qwen2.5:1.5b`.
- **`-m services`** — `docker compose up qdrant sandbox`.
- **`-m grpo`** — `pip install -e ".[grpo]"`.
- **`-m remote`** — paid API keys + `RUN_REMOTE_E2E=1`.
