# Quickstart

## 1. Install

```bash
git clone https://github.com/taraslysun/adaptive-attack-protection-system
cd adaptive-attack-protection-system
cp .env.example .env       # fill the providers you want
pip install -e .           # core deps; enough for `aaps smoke`
pip install -e ".[all]"    # add jupyter, transformers, agentdojo, langchain extras
```

## 2. Smoke

```bash
aaps smoke
# {"ok": true, "imports": 15}

pytest tests/test_smoke.py tests/test_e2e_mock.py -q
# 22 passed
```

If any of those fail on a clean clone, open an issue with the full traceback.

## 3. CLI quick-runs

```bash
aaps run-attack --family pair --victim mock --n-goals 2
# wrote logs/cli/cli_pair_mock.json

aaps run-bench --benchmark agentdojo --suite workspace --limit 2
# JSON stub today; will execute the real harness in 0.2.x
```

## 4. Notebooks

```bash
jupyter lab notebooks/00_setup_and_agent.ipynb
```

Recommended path:

1. `00_setup_and_agent.ipynb` — verify your `.env`, build the ReAct agent.
2. `02_pair_attack.ipynb` — single PAIR run on a small victim.
3. `06_benchmark_comparison.ipynb` — small benchmark cell with PACE on / off.
4. `08_agentdojo_benchmark.ipynb` — 4-suite AgentDojo subset against PACE.

`05_pace_defense.ipynb` and `99_mim_LEGACY.ipynb` are tagged legacy (banner at
top); use `06_benchmark_comparison.ipynb` for a runnable PACE example.

## 5. Real-model runs

Either:

- **Local Ollama** — set `OLLAMA_URL` (default `http://localhost:11434`),
  pull `llama3.1:8b-instruct-q4_K_M` and `qwen2.5:1.5b-instruct`. No API
  cost.
- **Remote APIs** — set the API keys in `.env` for the providers you want
  (OpenAI, Anthropic, Google/Gemini, OpenRouter, Mistral, etc.). The thesis
  matrix used Gemini-2.0-Flash-Lite, Mistral-Small-2603, and Llama-3.1-8B
  via OpenRouter as the cheap headline rows.

## 6. Logs

By convention every experiment run writes to a fresh timestamped
directory under `logs/thesis/<HHMM-DDMMYYYY>/`. Old logs are immutable.
See thesis workspace `docs/wiki/operations/reproducibility.md` for the
full convention.
