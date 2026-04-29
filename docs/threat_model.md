# Threat model

A single tool-using agent process. The deployer controls system prompt, tool
registry, and defence code. The frontier model's weights are a black box (no
fine-tuning attacks).

## Untrusted channels (data-flow boundaries)

| Hook | Source | What can be planted |
|---|---|---|
| H1 | end-user input | direct prompt injection |
| H2 | tool call output | tool-return IPI |
| H3 | retrieval corpus | poisoned RAG, EchoLeak-style |
| H4 | persistent memory | long-lived poison across sessions |
| H5 | MCP descriptors / skill manifests | supply-chain abuse |

## Goal classes

- **Integrity of actions.** Get the agent to call a tool the user did not ask
  for, or call a permitted tool with attacker-controlled arguments.
- **Confidentiality.** Exfiltrate user secrets or system-prompt fragments.
- **Persistence.** Plant a policy in memory that survives across sessions.

## Adversary knowledge tiers (per Tramèr 2020 / Nasr 2025)

- **Black box.** Sees only the agent's observable outputs.
- **Grey box.** Knows the defence mechanism and its public hyperparameters
  (PACE's `K`, `q`, clustering rule, NLI threshold). Default tier for the
  headline matrix.
- **Oracle.** Additionally sees PACE-internal text. Reported in its own
  column with explicit consent flag.

## What PACE protects

- Plan-time **CFI gate**: every tool call must match a (tool, arg-shape)
  entry in the planner-emitted shadow plan `π̂`.
- Run-time **agreement gate**: at least `q` of `K` Executors over disjoint
  evidence clusters must propose the same canonical `(tool, args)` before
  the call fires.
- Pre-clustering **NLI redundancy filter**: drops entailment-redundant
  evidence spans so a single semantic message cannot occupy multiple
  clusters at once.

## What PACE does NOT protect

- Schema-conformant supply-chain payloads. If π̂ admits
  `register_tool(name: str, descriptor: dict)`, an attacker-supplied
  descriptor passes the CFI gate.
- Free-text confidentiality leaks where the model speaks but no tool is
  called.
- Multi-modal injection (images, screenshots, PDFs) — out of scope.

See thesis Chapter 3 for the formal threat model and Chapter 4 for the
PACE invariants.
