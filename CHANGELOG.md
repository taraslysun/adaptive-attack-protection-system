# Changelog

## 0.1.0 — 2026-05-01 — initial public sync

First fully-installable release. Closes the source-coverage gap between this
public companion repo and the thesis workspace.

### Added
- Full `aaps.defenses.pace` package (planner, executor, agreement voter, NLI
  redundancy filter, plan/CFI gate, trace logger).
- Full `aaps.defenses.baselines` (14 reproduced single-paper defences).
- Legacy `aaps.defenses.integrity` (SPQ-style multi-layer stack, ablation
  column).
- `aaps.attacks._core` PSSU base + benchmarks + model registry.
- `aaps.attacks.slim5` headline attack pool: PAIR, PoisonedRAG, RL,
  HumanRedTeam, SupplyChain.
- `aaps.attacks.legacy` (GCG, TAP, Crescendo, AdvPrompter, MAP-Elites, static).
- `aaps.evaluation` harness (`DefenseBenchmark`, AgentDojo / InjecAgent
  external runners, LLM judge, metrics).
- `aaps.cli` with `smoke`, `run-attack`, `run-bench` subcommands.
- `aaps._compat` legacy-import shim (`from defenses.pace.x` → `from
  aaps.defenses.pace.x`); will be removed in 0.3.x.
- `tests/test_smoke.py` (13 cases) and `tests/test_e2e_mock.py` (9 cases) —
  network-free, must pass on every PR.
- `.github/workflows/smoke.yml` — Python 3.10 / 3.11 / 3.12 matrix on push and
  PR.
- `LICENSE` (MIT, 2026, Taras Lysun).
- `.env.example` enumerating all 21 environment variables.
- 12 runnable notebooks copied from the thesis workspace, renumbered to
  remove the `05_*` collision (`05_spq_defense.ipynb` → `05_pace_defense.ipynb`,
  `05_deepagent.ipynb` → `10_deepagent.ipynb`, `09_mim_LEGACY.ipynb` →
  `99_mim_LEGACY.ipynb`, `10_adaptive_budget_sweep.ipynb` → `09_*`).
- `PRODUCTION_READINESS_AUDIT.md` documenting the pre-sync state.
- `docs/` quickstart + threat-model pointers.

### Changed
- `pyproject.toml` runtime deps: added `anthropic`, `litellm`, `ollama`,
  `scipy`, `datasets`, `pyyaml`. Optional extras unchanged in shape.

### Known gaps (P1 follow-ups)
- `notebooks/05_pace_defense.ipynb` and `notebooks/99_mim_LEGACY.ipynb` carry
  references to removed `feature_extractors` / `shadow_plan` modules. Top
  banners warn testers; cell content is preserved for historical reference
  only. Will be rewritten or removed in 0.2.0.
- `tests/test_e2e_ollama.py` (laptop run, no API cost) and
  `tests/test_e2e_remote.py` (≤ $1.00 cost cap) are out of scope for this
  release; tracked for 0.2.0.
- Thesis Table 5.x cell reproduction (Mistral × PAIR × `n=10`) is deferred
  until a budget for paid runs is approved.
- API keys that resided in `.env` during the sync should be rotated as a
  precaution.
