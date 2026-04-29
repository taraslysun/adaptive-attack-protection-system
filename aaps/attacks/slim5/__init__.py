"""Adaptive (learning) attack families.

* `gradient_attack.gcg`         -- white-box GCG (nanoGCG + builtin fallback)
* `gcg_variants`                -- adaptive GCG variants targeting AIS layers
* `rl_attack.attack`            -- GRPO-style RL attacker
* `search_attack.attack`        -- MAP-Elites evolutionary attacker
* `human_redteam.attack`        -- scripted defense-aware red-team strategies
* `pair.attack`                 -- PAIR (Chao 2023) black-box jailbreak
* `tap.attack`                  -- TAP (Mehrotra 2024) tree-of-attacks
* `poisoned_rag.attack`         -- PoisonedRAG (Zou 2024) RAG corpus poisoning
* `crescendo.attack`            -- Crescendo (Russinovich 2024) multi-turn
* `advprompter.attack`          -- AdvPrompter (Paulus 2024) suffix attack
"""
