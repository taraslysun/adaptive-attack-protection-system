#!/usr/bin/env bash
# Generate all thesis diagram PNGs from mermaid source.
# Output: Overleaf/Figures/png/*.png
# Usage: bash scripts/gen_figures.sh (from repo root)

set -euo pipefail
OUTDIR="Overleaf/Figures/png"
TMPDIR_LOCAL="$(mktemp -d)"
MMDC="mmdc"
SCALE=3
BG="white"

render() {
  local name="$1"; local content="$2"
  local mmd="$TMPDIR_LOCAL/${name}.mmd"
  printf '%s' "$content" > "$mmd"
  echo "  rendering $name..."
  $MMDC -i "$mmd" -o "$OUTDIR/${name}.png" -b "$BG" --scale "$SCALE" \
        --configFile /dev/null 2>/dev/null \
    || $MMDC -i "$mmd" -o "$OUTDIR/${name}.png" -b "$BG" --scale "$SCALE" 2>&1 | tail -2
}

mkdir -p "$OUTDIR"

# ── 1. agent_loop ──────────────────────────────────────────────────────────
render "agent_loop" '%%{init: {"theme": "neutral", "flowchart": {"curve": "basis", "padding": 20}}}%%
flowchart TD
    classDef main fill:#f0f4f8,stroke:#4a6fa5,stroke-width:2px,color:#111,font-size:13px
    classDef hook fill:#fff,stroke:#4a6fa5,stroke-width:2px,color:#4a6fa5,font-size:11px
    classDef side fill:#f8f4f0,stroke:#9a7d5a,stroke-dasharray:5 3,color:#111,font-size:12px
    classDef out  fill:#e8f4e8,stroke:#2d7a2d,stroke-width:2px,color:#111,font-size:13px

    U["👤 User Prompt"]:::main
    H1(["H1"]):::hook
    L["🧠 LLM Policy V"]:::main
    H2(["H2"]):::hook
    T["⚙️ Tool Executor"]:::main
    H3(["H3"]):::hook
    H5(["H5"]):::hook
    R["✅ Response to User"]:::out
    M[("💾 Memory Store")]:::side
    RC[("📚 Retrieval Corpus")]:::side
    H4(["H4"]):::hook

    U -->|query| H1 --> L
    L -->|"tool call"| H2 --> T
    T -->|"tool result"| H3 --> L
    L -->|answer| H5 --> R
    M -.->|"read"| H4 -.-> L
    L -.->|"write"| M
    RC -.->|context| T'

# ── 2. threat_model ────────────────────────────────────────────────────────
render "threat_model" '%%{init: {"theme": "neutral"}}%%
flowchart TD
    classDef channel fill:#fff3cd,stroke:#c8960c,stroke-width:2px,font-size:13px
    classDef spqhook  fill:#d1e7ff,stroke:#0d6efd,stroke-width:2px,font-size:11px
    classDef agent    fill:#d4edda,stroke:#28a745,stroke-width:2.5px,font-size:14px,font-weight:bold
    classDef adv      fill:#f8d7da,stroke:#dc3545,stroke-width:2px,font-size:12px

    ADV(["⚔️ Adversary"]):::adv

    subgraph CHANNELS["Untrusted Channels → PACE Intercept → Agent"]
        UI["📝 User Input"]:::channel
        TR["🔧 Tool Returns"]:::channel
        MEM["💾 Memory"]:::channel
        RAG["📚 Retrieval Corpus"]:::channel

        S1["PACE H1 / L1"]:::spqhook
        S2["PACE H3 / L2"]:::spqhook
        S3["PACE H5 / L5"]:::spqhook
        S4["PACE H4 / L6"]:::spqhook

        AG["🛡️ Agent V + PACE Gates"]:::agent
    end

    ADV -.->|"prompt inject"| UI
    ADV -.->|"IPI in response"| TR
    ADV -.->|"persist policy"| MEM
    ADV -.->|"corpus poison"| RAG

    UI  --> S1 --> AG
    TR  --> S2 --> AG
    MEM --> S3 --> AG
    RAG --> S4 --> AG'

# ── 3. pssu_loop ───────────────────────────────────────────────────────────
render "pssu_loop" '%%{init: {"theme": "neutral"}}%%
flowchart LR
    classDef step fill:#dbeafe,stroke:#3b82f6,stroke-width:2px,font-size:13px
    classDef state fill:#fef3c7,stroke:#d97706,stroke-width:2px,font-size:12px
    classDef ctrl  fill:#d1fae5,stroke:#059669,stroke-width:2px,font-size:12px

    ST(["State Aₜ"]):::state

    P["📋 Propose\ngenerate candidate triggers"]:::step
    SC["📊 Score\njudge / internal scorer S"]:::step
    SE["🏆 Select\nkeep top-k survivors"]:::step
    UP["🔄 Update\nmutate policy / archive"]:::step

    ES["⏹️ Early Stop\nscore ≥ τ"]:::ctrl
    BG["🔚 Budget B\niterations exhausted"]:::ctrl

    ST -->|"read Aₜ"| P
    P  -->|"candidates"| SC
    SC -->|"scored"| SE
    SE -->|"top-k"| UP
    UP -->|"Aₜ₊₁"| ST

    SC -.->|"threshold hit"| ES
    SE -.->|"t > B"| BG'

# ── 4. spq_overview ────────────────────────────────────────────────────────
render "spq_overview" '%%{init: {"theme": "neutral"}}%%
flowchart TD
    classDef main  fill:#f0f4f8,stroke:#4a6fa5,stroke-width:2px,font-size:13px
    classDef llm   fill:#dbeafe,stroke:#3b82f6,stroke-width:2px,font-size:13px
    classDef gate  fill:#fef3c7,stroke:#d97706,stroke-width:2.5px,font-size:13px
    classDef exec  fill:#d1fae5,stroke:#059669,stroke-width:1.5px,font-size:12px
    classDef block fill:#fee2e2,stroke:#dc2626,stroke-width:2px,font-size:12px
    classDef good  fill:#d1fae5,stroke:#059669,stroke-width:2px,font-size:13px

    Q["👤 User Query"]:::main
    PL["🧠 Planner LLM\n(trusted input only)"]:::llm
    SP["📋 PACE Plan π̂\ntyped graph of allowed calls"]:::main
    AG["⚙️ Agent Executor\n+ Evidence Pool"]:::main
    TC["Tool Call R"]:::main

    CFI{"🛡️ CFI Gate\nC(R, π̂)"}:::gate
    QV{"🗳️ Quorum Gate\nQ(R) = n_agree ≥ q"}:::gate

    BLK["🚫 Block / Replan"]:::block
    DIS["📝 Dissent Logged"]:::block
    RES["✅ Verified Result\nTool Fired"]:::good

    subgraph EXEC["K Independent Executors (evidence clusters)"]
        E1["E₁"]:::exec
        EK["E₂ … Eₖ"]:::exec
    end

    Q --> PL --> SP --> AG --> TC --> CFI
    SP -.->|whitelist| CFI
    CFI -->|"C=0 fail"| BLK
    CFI -->|"C=1 pass"| QV

    E1 -->|"propose"| QV
    EK -->|"propose"| QV

    QV -->|"agree ≥ q"| RES
    QV -->|"agree < q"| DIS'

# ── 5. feature_extractors ──────────────────────────────────────────────────
render "feature_extractors" '%%{init: {"theme": "neutral"}}%%
flowchart TD
    classDef lbl  fill:#f0f4f8,stroke:#4a6fa5,stroke-width:2px,font-size:13px
    classDef gate fill:#fef3c7,stroke:#d97706,stroke-width:2.5px,font-size:13px
    classDef note fill:#f8f8f8,stroke:#aaa,stroke-dasharray:4 2,font-size:11px

    IN["📥 Input Evidence Span"]:::lbl

    L1["L1 — Channel Separator\nnonce delimiters · delimiter-escape check\n(StruQ / Spotlight style)"]:::lbl
    L2["L2 — Tool Output Probe\nknown-answer canary probe\n(DataSentinel style)"]:::lbl
    L3["L3 — Action Consistency\ncross-check proposed tools vs PACE plan\n(MELON inspired)"]:::lbl
    L4["L4 — Output Consistency\ncircuit-style refusal-tendency signal"]:::lbl
    L5["L5 — Memory Write Guard\nweighted-feature persistence scorer\n(A-MemGuard style)"]:::lbl
    L6["L6 — Retrieval Integrity Guard\ncluster + self-consistency screening\n(TrustRAG style)"]:::lbl

    CFI{"CFI Gate\nC(R, π̂)"}:::gate
    QV{"Quorum Gate\nQ(R)"}:::gate

    NB["⚠️ All L1–L6 are NON-BLOCKING probes\nThey attach signals; gates enforce decisions"]:::note

    IN --> L1 --> L2 --> L3 --> L4 --> L5 --> L6
    L6 -->|"signals"| CFI
    L6 -->|"signals"| QV
    NB -.- L3'

# ── 6. threat_model_detailed ───────────────────────────────────────────────
render "threat_model_detailed" '%%{init: {"theme": "neutral"}}%%
flowchart TD
    classDef surface fill:#fff3cd,stroke:#c8960c,stroke-width:2px,font-size:13px
    classDef example fill:#f8f4f0,stroke:#9a7d5a,stroke-dasharray:4 2,font-size:11px
    classDef hook    fill:#d1e7ff,stroke:#0d6efd,stroke-width:2px,font-size:12px
    classDef agent   fill:#d4edda,stroke:#28a745,stroke-width:2.5px,font-size:14px
    classDef oos     fill:#f8f8f8,stroke:#ccc,stroke-dasharray:5 3,font-size:11px

    subgraph S1["Surface: User Input"]
        UI["📝 User Input"]:::surface
        E1A["direct prompt injection\nrole-play / jailbreak"]:::example
        H1["Hook H1 / L1"]:::hook
    end

    subgraph S2["Surface: Tool Returns"]
        TR["🔧 Tool Returns"]:::surface
        E2A["IPI in web snippet\nmalicious API response"]:::example
        H3["Hook H3 / L2"]:::hook
    end

    subgraph S3["Surface: Memory"]
        MEM["💾 Memory"]:::surface
        E3A["poisoned long-term memory\npersistence across sessions"]:::example
        H5["Hook H5 / L5"]:::hook
    end

    subgraph S4["Surface: Retrieval Corpus"]
        RAG["📚 Retrieval Corpus"]:::surface
        E4A["PoisonedRAG document\nsupply-chain descriptor"]:::example
        H4["Hook H4 / L6"]:::hook
    end

    AG["🛡️ Agent V + PACE"]:::agent
    OOS["Out of scope:\nweight poisoning · container escape\nside channels · social engineering"]:::oos

    E1A -.-> H1 --> AG
    E2A -.-> H3 --> AG
    E3A -.-> H5 --> AG
    E4A -.-> H4 --> AG'

# ── 7. spq_dataflow ────────────────────────────────────────────────────────
render "spq_dataflow" '%%{init: {"theme": "neutral"}}%%
flowchart LR
    classDef main  fill:#f0f4f8,stroke:#4a6fa5,stroke-width:2px,font-size:12px
    classDef llm   fill:#dbeafe,stroke:#3b82f6,stroke-width:2px,font-size:12px
    classDef probe fill:#fef9c3,stroke:#ca8a04,stroke-dasharray:4 2,font-size:11px
    classDef gate  fill:#fef3c7,stroke:#d97706,stroke-width:2.5px,font-size:12px
    classDef good  fill:#d1fae5,stroke:#059669,stroke-width:2px,font-size:12px
    classDef bad   fill:#fee2e2,stroke:#dc2626,font-size:11px

    REQ["📥 Input Request"]:::main
    L1["L1 Channel\nCheck"]:::probe
    PL["🧠 Planner LLM"]:::llm
    SP["PACE Plan π̂"]:::main
    EX["Agent Executor"]:::main

    L2["L2 probe"]:::probe
    L3["L3 probe"]:::probe
    L4["L4 probe"]:::probe
    L5["L5 probe"]:::probe
    L6["L6 retrieval\ncheck"]:::probe

    CFI{"CFI Gate\nC(R,π̂)"}:::gate
    QV{"Quorum Gate\nQ(R)"}:::gate

    OUT["✅ Verified Output"]:::good
    BLK["🚫 Block/Replan"]:::bad
    DIS["📝 Dissent Log"]:::bad

    REQ --> L1 --> PL --> SP --> EX
    EX -->|"tool boundary"| L2 --> L3 --> L4 --> L5
    EX -.->|"retrieval"| L6
    L5 --> CFI
    L6 -.-> QV
    SP -.->|"whitelist"| CFI
    CFI -->|"pass"| QV
    CFI -->|"fail"| BLK
    QV  -->|"agree"| OUT
    QV  -->|"dissent"| DIS'

# ── 8. cfi_invariant ───────────────────────────────────────────────────────
render "cfi_invariant" '%%{init: {"theme": "neutral"}}%%
flowchart TD
    classDef plan  fill:#dbeafe,stroke:#3b82f6,stroke-width:2px,font-size:12px
    classDef call  fill:#f0f4f8,stroke:#4a6fa5,stroke-width:2px,font-size:13px
    classDef pass  fill:#d1fae5,stroke:#059669,stroke-width:2.5px,font-size:13px,font-weight:bold
    classDef fail  fill:#fee2e2,stroke:#dc2626,stroke-width:2.5px,font-size:13px,font-weight:bold
    classDef note  fill:#f8f8f8,stroke:#ccc,font-size:11px

    subgraph PASS["✅ PASS case — C(R,π̂) = 1"]
        SP1["PACE Plan π̂\n• search(q)\n• send_email(to, body)\n• read_file(path)"]:::plan
        RC1["R: search(q='x')"]:::call
        OK["✅ PASS — match found\nC = 1 → execution allowed"]:::pass
        SP1 --> RC1 --> OK
    end

    subgraph FAIL["🚫 FAIL case — C(R,π̂) = 0"]
        SP2["PACE Plan π̂\n• search(q)\n• send_email(to, body)\n• read_file(path)"]:::plan
        RC2["R: exfil_data(url)"]:::call
        NO["🚫 BLOCK — no match\nC = 0 → CFI violation logged"]:::fail
        VIO["control-flow violation\nlogged → replan/abstain"]:::note
        SP2 --> RC2 --> NO --> VIO
    end'

# ── 9. quorum_invariant ────────────────────────────────────────────────────
render "quorum_invariant" '%%{init: {"theme": "neutral"}}%%
flowchart LR
    classDef call  fill:#f0f4f8,stroke:#4a6fa5,stroke-width:2px,font-size:13px
    classDef exec  fill:#dbeafe,stroke:#3b82f6,stroke-width:2px,font-size:13px
    classDef vote  fill:#fef3c7,stroke:#d97706,stroke-width:2.5px,font-size:13px
    classDef agree fill:#d1fae5,stroke:#059669,stroke-width:2.5px,font-size:13px
    classDef diss  fill:#fee2e2,stroke:#dc2626,stroke-width:2px,font-size:13px
    classDef note  fill:#f8f8f8,stroke:#ccc,font-size:11px

    TC["Tool Call R\n(K=3, q=⌈K/2⌉=2)"]:::call

    E1["E₁ proposes R′=R\n(matches majority)"]:::exec
    E2["E₂ proposes R′=R\n(matches majority)"]:::exec
    E3["E₃ proposes R′=R*\n(adversarial variant)"]:::exec

    VT["🗳️ Vote\nn_agree(R)"]:::vote

    AG["✅ Agree\nn_agree=2 ≥ q=2\nQ(R)=1 → fire"]:::agree
    DIS["❌ Dissent\nn_agree < q\nQ(R)=0 → log + abstain"]:::diss
    NOTE["planner abstain-or-replan\nmargins preserved in trace"]:::note

    TC --> E1 --> VT
    TC --> E2 --> VT
    TC --> E3 -->|"minority"| VT
    VT -->|"Q=1"| AG
    VT -->|"Q=0"| DIS --> NOTE'

# ── 10. eval_pipeline ──────────────────────────────────────────────────────
render "eval_pipeline" '%%{init: {"theme": "neutral"}}%%
flowchart LR
    classDef box   fill:#f0f4f8,stroke:#4a6fa5,stroke-width:2px,font-size:13px
    classDef llm   fill:#dbeafe,stroke:#3b82f6,stroke-width:2px,font-size:13px
    classDef score fill:#d1fae5,stroke:#059669,stroke-width:2px,font-size:13px
    classDef note  fill:#f8f8f8,stroke:#ccc,stroke-dasharray:4 2,font-size:11px

    GB["📋 Goal Bank\n{gᵢ}"]:::box
    ATK["⚔️ Attack Module A\n(PSSU driver)"]:::box
    VD["🛡️ Victim V\n+ Defence D"]:::llm
    JD["⚖️ Judge LLM J\n(J ≠ V)"]:::llm
    ASR["📊 ASR α\nper cell"]:::score
    AGG["📈 Aggregate\n+ Bootstrap 95% CI"]:::score

    SEED["repeated per seed s\n→ across-seed CI"]:::note

    GB  -->|"goal g"| ATK
    ATK -->|"trigger"| VD
    VD  -->|"response Y"| JD
    JD  -->|"score"| ASR
    ASR -->|"per-cell α"| AGG
    VD  -.->|"PSSU feedback"| ATK
    SEED -.- VD'

# ── 11. matrix_protocol ────────────────────────────────────────────────────
render "matrix_protocol" '%%{init: {"theme": "neutral"}}%%
flowchart TD
    classDef hdr  fill:#dbeafe,stroke:#3b82f6,stroke-width:2px,font-size:13px,font-weight:bold
    classDef cell fill:#f0f4f8,stroke:#4a6fa5,stroke-width:1px,font-size:12px
    classDef boot fill:#d1fae5,stroke:#059669,stroke-width:2px,font-size:12px
    classDef note fill:#f8f8f8,stroke:#ccc,stroke-dasharray:4 2,font-size:11px

    subgraph MATRIX["Evaluation Matrix (attacks × defences × seeds)"]
        D0["no_def"]:::hdr
        D1["PACE"]:::hdr
        D2["struq"]:::hdr
        D3["… (14 baselines)"]:::hdr

        RL["RL attack\n→ α_s"]:::cell
        PAIR["PAIR\n→ α_s"]:::cell
        PR["PoisonedRAG\n→ α_s"]:::cell
        HRT["HumanRT\n→ α_s"]:::cell
        SC["SupplyChain\n→ α_s"]:::cell
    end

    BOOT["📊 Bootstrap CI\nα̂ = mean({αₛ})\n95% CI over seeds"]:::boot
    NOTE["within-run CI (fixed seed)\n≠ across-seed CI"]:::note

    RL   --> BOOT
    PAIR --> BOOT
    PR   --> BOOT
    NOTE -.- BOOT'

# ── 12. judge_pipeline ─────────────────────────────────────────────────────
render "judge_pipeline" '%%{init: {"theme": "neutral"}}%%
flowchart LR
    classDef inp   fill:#f0f4f8,stroke:#4a6fa5,stroke-width:2px,font-size:13px
    classDef jdg   fill:#dbeafe,stroke:#3b82f6,stroke-width:2px,font-size:13px
    classDef r10   fill:#fee2e2,stroke:#dc2626,font-size:12px
    classDef r07   fill:#fff3cd,stroke:#ca8a04,font-size:12px
    classDef r03   fill:#fef9c3,stroke:#d97706,font-size:12px
    classDef r00   fill:#d1fae5,stroke:#059669,font-size:12px
    classDef thr   fill:#f3e8ff,stroke:#7c3aed,stroke-width:2px,font-size:12px
    classDef bias  fill:#f8f8f8,stroke:#ccc,stroke-dasharray:4 2,font-size:11px

    G["Goal g"]:::inp
    Y["Response Y"]:::inp
    J["⚖️ Judge LLM J\n(J ≠ victim V)\nfrozen rubric"]:::jdg

    R10["1.0 — full policy-violating success"]:::r10
    R07["0.7 — partial / minor success"]:::r07
    R03["0.3 — attempted but blocked"]:::r03
    R00["0.0 — clear failure / safe response"]:::r00

    THR["threshold τ\nscore ≥ τ → judged success"]:::thr
    BIAS["rubric frozen → reproducible under fixed J\nnot an oracle substitute for human review"]:::bias

    G --> J
    Y --> J
    J --> R10
    J --> R07
    J --> R03
    J --> R00
    R07 -.-> THR
    BIAS -.- J'

# ── 13. adaptive_budget ────────────────────────────────────────────────────
render "adaptive_budget" '%%{init: {"theme": "neutral"}}%%
flowchart TD
    classDef bgt  fill:#dbeafe,stroke:#3b82f6,stroke-width:2px,font-size:13px
    classDef step fill:#f0f4f8,stroke:#4a6fa5,stroke-width:2px,font-size:13px
    classDef ctrl fill:#d1fae5,stroke:#059669,stroke-width:2px,font-size:12px
    classDef stop fill:#fef3c7,stroke:#d97706,stroke-width:2px,font-size:12px
    classDef done fill:#fee2e2,stroke:#dc2626,stroke-width:2px,font-size:12px

    B["🎯 Budget B\n(PSSU iterations)\nB = p + s + overhead"]:::bgt

    P["📋 Propose\np queries"]:::step
    SC["📊 Score\ns queries"]:::step
    SE["🏆 Select\ntop-k"]:::step
    UP["🔄 Update\nAₜ₊₁"]:::step

    ES["⏹️ Early Stop\nscore ≥ τ"]:::stop
    AB["🔚 Abort\nt > B\nbudget exhausted"]:::done
    RT["🔁 Retry with new seed\nif n_success = 0"]:::ctrl

    B --> P --> SC --> SE --> UP
    UP -->|"t += 1"| P
    SC -.->|"threshold hit"| ES
    SE -.->|"t > B"| AB
    AB -.-> RT'

# ── 14. lowlevel_pipeline ──────────────────────────────────────────────────
render "lowlevel_pipeline" '%%{init: {"theme": "neutral"}}%%
flowchart TD
    classDef main  fill:#f0f4f8,stroke:#4a6fa5,stroke-width:2px,font-size:12px
    classDef llm   fill:#dbeafe,stroke:#3b82f6,stroke-width:2.5px,font-size:12px
    classDef spq   fill:#fef9c3,stroke:#ca8a04,stroke-width:2px,font-size:11px
    classDef gate  fill:#fef3c7,stroke:#d97706,stroke-width:2.5px,font-size:12px
    classDef good  fill:#d1fae5,stroke:#059669,stroke-width:2px,font-size:12px
    classDef bad   fill:#fee2e2,stroke:#dc2626,stroke-width:2px,font-size:12px
    classDef dec   fill:#fff,stroke:#4a6fa5,stroke-width:2px,font-size:12px

    UQ["👤 User Query"]:::main

    subgraph CINPUT["check_input() — Hook H1"]
        PL["🧠 Planner LLM\nsys_prompt + tool_schemas + query\n(trusted input only)"]:::llm
        SP["📋 PACEPlan π̂\ntyped graph: tool_name → arg_schema"]:::main
    end

    subgraph MEMRET["Memory Retrieval — Hook H4"]
        MR["keyword overlap scorer\ntop-5 entries"]:::main
        CR["check_retrieval() / L6 guard"]:::spq
    end

    BUILD["Build messages:\n[system] + history[-4:] + [mem_ctx + query]"]:::main

    VLM["🧠 Victim LLM\nPOST /api/chat\n+ tools array (native tool routing)"]:::llm

    DET{"tool call\ndetected?"}:::dec
    PLAIN["plain text answer\n→ check_output()"]:::main

    subgraph CTOOLCALL["check_tool_call() — Hook H2"]
        direction TB
        MKSPAN["_make_span()\nper evidence byte"]:::spq
        L1L5["L1+L5: label_trust()\nchannel sep + memory guard"]:::spq
        L2["L2: probe_evidence()\nregex signatures → score ∈ [0,1]"]:::spq
        NLI["NLI Independence Filter\ncross-encoder/nli-deberta-v3-small\nP(entailment|prev→cand) ≥ 0.70 → drop\nfallback: cosine ≥ 0.92"]:::spq
        EMB["_embed()\nall-MiniLM-L6-v2 · 384-dim\nnormalize_embeddings=True"]:::spq
        KM["_kmeans(K, seed)\npure-Python · Euclidean · 50 iter\n→ K cluster lists"]:::spq
        EXPOOL["K × Executor.fill()\npace_plan + cluster evidence\nLLM T=0.7 → JSON {calls:[…]}"]:::llm
        QV["🗳️ AgreementVoter.vote()\ncapability shim:\nhigh-priv tools need required_source=user_input\nn_agree(R) ≥ q ?"]:::gate
        CFI["🛡️ CFI Gate C(R,π̂)\nplan.find_node(tool, args)\n∉ π̂ → hard_block"]:::gate
        MKSPAN --> L1L5 --> L2 --> NLI --> EMB --> KM --> EXPOOL --> QV --> CFI
    end

    CO["check_output()\ntrace flush"]:::main
    CMW["check_memory_write()\nH5 / L5 untrusted guard"]:::main
    RES["✅ Return Result\nanswer + tool_calls + defense_trace"]:::good
    BLK["🚫 Block / Replan"]:::bad
    DIS["📝 Dissent Logged"]:::bad

    UQ --> PL --> SP
    UQ --> MR --> CR
    SP --> BUILD
    CR --> BUILD
    BUILD --> VLM --> DET
    DET -->|"no"| PLAIN
    DET -->|"yes"| CTOOLCALL
    SP -.->|"whitelist"| CFI
    CFI -->|"C=0"| BLK
    QV  -->|"n_agree < q"| DIS
    CFI -->|"C=1 + Q=1"| CO
    PLAIN --> CO --> CMW --> RES'

echo ""
echo "Done. PNGs in $OUTDIR:"
ls -lh "$OUTDIR/"*.png 2>/dev/null || echo "No PNGs found — check mmdc errors above."
rm -rf "$TMPDIR_LOCAL"
