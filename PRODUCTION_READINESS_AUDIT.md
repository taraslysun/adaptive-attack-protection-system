# Production Readiness Audit — `adaptive-attack-protection-system`

**Audit date:** 2026-05-01
**Target repo:** `github.com/taraslysun/adaptive-attack-protection-system`
**Compared against:** `/Users/tlysu/ucu/diploma` (main thesis workspace)
**Verdict: NOT production-ready, NOT up to date — repo currently does not import.**

---

## Critical findings (P0 — blocker)

### 1. Source code is missing from the published repo

| Location | Main workspace | This repo (tracked) |
|---|---:|---:|
| Total `.py` source files (excluding `__pycache__`) | **135** | **8** |
| Sync ratio | — | **5.9 %** |

Only the following modules have real source files committed:

```
aaps/agent/__init__.py
aaps/agent/config.py
aaps/agent/deep_agent.py
aaps/agent/llm_factory.py
aaps/agent/memory_manager.py
aaps/agent/multimodal_retrieval.py
aaps/agent/tools.py
aaps/evaluation/call_logger.py
```

Everything else listed in `git ls-files` for `aaps/defenses/pace/`, `aaps/defenses/baselines/`, `aaps/attacks/_core/`, `aaps/attacks/slim5/*` is **only `.pyc` bytecode under `__pycache__/`**. The corresponding `.py` source files were never committed (or were deleted before the single `aef7378 code upload` commit).

Affected modules without source:

- `aaps/defenses/pace/`: `pipeline.py`, `planner.py`, `executor.py`, `agreement.py`, `clusters.py`, `plan.py`, `trace_logger.py`, `__init__.py` — **the entire PACE defence (the thesis's headline contribution).**
- `aaps/defenses/baselines/`: `struq.py`, `secalign.py`, `melon.py`, `a_memguard.py`, `smoothllm.py`, `data_sentinel.py`, `circuit_breaker.py`, `prompt_guard2.py`, `prompt_guard_filter.py`, `prompt_guards.py`, `firewall.py`, `rag_guard.py`, `rpo.py`, `wildguard_defense.py`, `__init__.py` — **all 14 reproduced baselines.**
- `aaps/attacks/_core/`: `base_attack.py`, `benchmarks.py`, `config.py`, `__init__.py` — PSSU base.
- `aaps/attacks/slim5/{rl,human_redteam,pair,poisoned_rag,supply_chain}/` — all five headline attacks.
- `aaps/attacks/legacy/{advprompter,tap,crescendo,static,_legacy}/` — legacy attack pool.

### 2. Smoke test fails immediately

```text
$ python -c "import aaps"
# ok (namespace, __file__ is None)

$ python -c "from aaps.defenses.pace import pipeline"
ImportError: cannot import name 'pipeline' from 'aaps.defenses.pace' (unknown location)

$ python -c "from aaps.attacks.slim5.pair import attack"
ImportError: cannot import name 'attack' from 'aaps.attacks.slim5.pair' (unknown location)

$ python -c "from aaps.agent.deep_agent import *"
ModuleNotFoundError: No module named 'aaps.agent.local_agent'
```

`aaps/agent/__init__.py` line 17 references `aaps.agent.local_agent`, which is not committed. Even the agent module that *does* have files is broken on import.

### 3. `pyproject.toml` claims a console script that cannot exist

```toml
[project.scripts]
aaps = "aaps.cli:main"
```

`aaps/cli.py` is not in the repo. Installing the package would expose an `aaps` command that fails on first invocation.

### 4. `scripts/` and `tests/` directories are empty

```text
$ ls scripts/ tests/
# (both empty)
```

The main workspace contains substantial `scripts/` (setup, reporting, runners) and `tests/`. None of it is in the published repo, so testers have nothing runnable beyond notebooks.

---

## Major findings (P1)

### 5. Notebooks are now copied (this audit run) but most will fail

Notebooks copied into `notebooks/` from the main workspace as part of this audit:

```
00_setup_and_agent.ipynb
01_static_attacks.ipynb
02_pair_attack.ipynb
03_poisoned_rag_attack.ipynb
04_supply_chain_attack.ipynb
05_deepagent.ipynb              (already present)
05_spq_defense.ipynb
06_benchmark_comparison.ipynb
07_deep_agent_demo.ipynb
08_agentdojo_benchmark.ipynb
09_mim_LEGACY.ipynb
10_adaptive_budget_sweep.ipynb
```

These notebooks import modules like `aaps.attacks.slim5.pair.attack`, `aaps.defenses.pace.pipeline`, `aaps.evaluation.defense_benchmark` — none of which have source in this repo. Running any notebook past the import cell will raise `ModuleNotFoundError` until the source files are synced.

### 6. `aaps/evaluation/` is missing the headline benchmark code

Main workspace `evaluation/` contains:

```
agentdojo_native.py
defense_benchmark.py
external_benchmarks.py
llm_judge.py
metrics.py
benchmarks/  (directory)
```

Target `aaps/evaluation/` contains only `call_logger.py`. The `DefenseBenchmark` class that produces ASR/PSR numbers in the thesis Tables 5.1–5.5 is not in the published repo at all.

### 7. `.env` file is now copied (this audit run)

Source: `/Users/tlysu/ucu/diploma/.env` → `/Users/tlysu/ucu/diploma/adaptive-attack-protection-system/.env`.

Contains 21 environment variables including:

- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `GEMINI_API_KEY`
- `OPENROUTER_API_KEY`, `HF_TOKEN`
- `QDRANT_URL`, `QDRANT_API_KEY`, `WANDB_API_KEY`
- Default model selections, Ollama configuration

`.gitignore` (`/.env`) covers it, so it will not be tracked accidentally on a normal `git add .`. **Do not** run `git add -f .env`. Rotate keys after defence, especially if any third party has cloned this directory.

---

## Minor findings (P2)

### 8. Documentation gaps

- `README.md` is 11 KB but describes features (PACE, baselines, attacks) whose source code is not present.
- No `INSTALL.md`, no `CONTRIBUTING.md`, no `docs/` directory in the published repo (the main workspace `docs/wiki/` is rich).
- No CI configuration (`.github/workflows/`).
- No license file at the repo root despite `pyproject.toml` declaring MIT.

### 9. Single commit, no history

```text
$ git log --oneline
aef7378 code upload
```

One commit, no tags, no branches. Reviewers cannot trace evolution; bisection is impossible.

### 10. `pyproject.toml` dependency hygiene

- `agentdojo>=0.1` is listed as an optional extra but no version pin or repository link; PyPI publication is uncertain.
- `transformers>=4.40` and `torch>=2.2` listed twice (in `nli` and `multimodal` extras).
- Missing dependencies that the actual code needs (litellm, ollama, anthropic, datasets, pyyaml — based on main workspace imports).

### 11. Notebook duplication

Both `05_deepagent.ipynb` (pre-existing) and `05_spq_defense.ipynb` (newly copied) share the same numeric prefix. Numbering should be re-sequenced or one of them renamed.

---

## What is OK

- `.gitignore` covers `.env`, `__pycache__`, common Python artefacts. Sound.
- `pyproject.toml` build-system declaration is clean, optional extras are well-grouped.
- `aaps/agent/` module files (the ones that exist) are reasonable Python.
- README.md exists and describes the project intent.
- Single root-level `aaps` namespace package layout is appropriate for a Python distribution.

---

## Required actions to reach "production-ready / up to date"

In order of urgency:

1. **Sync the missing 127 source files.** Map main-workspace paths into the `aaps/...` namespace:
   - `attacks/_core/*.py` → `aaps/attacks/_core/`
   - `attacks/{rl_attack,human_redteam,...}` → `aaps/attacks/slim5/{rl,human_redteam,...}`
   - `attacks/_legacy/*` → `aaps/attacks/legacy/`
   - `defenses/pace/*.py` → `aaps/defenses/pace/`
   - `defenses/baselines/*.py` → `aaps/defenses/baselines/`
   - `defenses/integrity/*.py` → `aaps/defenses/integrity/`
   - `evaluation/*.py` → `aaps/evaluation/`
   - `agent/local_agent.py` → `aaps/agent/local_agent.py`
   This is a non-trivial path-rewrite operation. Internal imports must be updated from `from defenses.pace.pipeline import ...` to `from aaps.defenses.pace.pipeline import ...`. Recommend an automated sync script with explicit mapping rather than ad-hoc copies.

2. **Add `aaps/cli.py`** or remove the `[project.scripts]` entry until a CLI exists.

3. **Populate `scripts/`** with at minimum the `setup_environment.sh` equivalent and a smoke-run script that exercises one attack × one defence × one model.

4. **Populate `tests/`** with import smoke tests so future commits cannot regress to a 6 % source-coverage state again.

5. **Add `docs/`** — even a short README pointing to `docs/wiki/` in the main workspace, or copy the wiki in.

6. **Pin or document model endpoints** in `.env.example` (without secrets) so testers know what to fill in.

7. **CI workflow** running `pip install -e .[all]` + `python -c "from aaps.defenses.pace.pipeline import *"` on every PR. Would catch finding 1 instantly.

8. **Add LICENSE** file matching the `pyproject.toml` MIT declaration.

9. **Re-number notebooks** to remove the `05_*` collision.

10. **Document the smoke-run path in README** with one literal command testers can copy:
    ```bash
    cp .env.example .env  # fill keys
    pip install -e ".[all]"
    pytest tests/test_smoke.py
    jupyter lab notebooks/00_setup_and_agent.ipynb
    ```

---

## Smoke run summary (what was attempted and what happened)

| Step | Command | Result |
|---|---|---|
| Copy `.env` | `cp /Users/tlysu/ucu/diploma/.env ./.env` | OK (21 vars) |
| Copy notebooks | `cp ../notebooks/*.ipynb notebooks/` | OK (12 notebooks present) |
| Import root | `python -c "import aaps"` | OK (namespace) |
| Import PACE | `python -c "from aaps.defenses.pace import pipeline"` | **FAIL — ImportError** |
| Import any attack | `python -c "from aaps.attacks.slim5.pair import attack"` | **FAIL — ImportError** |
| Import agent | `python -c "from aaps.agent.deep_agent import *"` | **FAIL — ModuleNotFoundError: aaps.agent.local_agent** |
| Run a notebook | (not attempted; would fail on first import cell) | n/a |

---

## Bottom line

The published thesis repository at `github.com/taraslysun/adaptive-attack-protection-system` currently contains **a `pyproject.toml`, a README, eight Python source files, one notebook, and the bytecode shadows of ≈40 modules whose source was never committed**. It cannot be installed, cannot be imported beyond the agent shim (which itself is broken), and cannot run any of the notebooks the thesis points reviewers to.

This is the single most consequential reproducibility failure for the thesis. Before the defence:

- restore the 127 missing source files,
- run the smoke import test in CI,
- update the README's run instructions to match a passing smoke test,
- rotate the API keys that now live in this directory's `.env`.

If you would like me to perform the source-file sync (with path rewriting), say the word and I will draft the migration script and run it against a fresh branch so you can review the diff before pushing.
