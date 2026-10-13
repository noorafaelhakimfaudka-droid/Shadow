# SHADOW v4.1 — Machine Learning Experimentation System

> **Local, auditable, and scientifically honest machine learning laboratory with zero cloud APIs.**  
> Built strictly adhering to the [SHADOW PRD Final v4.1](file:///c:/Users/asus/Downloads/Shadow_PRD_Final%20%281%29.md) and engineering rules.

---

## 1. What is SHADOW?

SHADOW takes a single tabular CSV file and an objective goal, then runs a full machine learning experimentation workflow from framing to final evaluation. The system guarantees:
1. **Bitwise Reproducible Notebook (`.ipynb`):** A clean 11-section executable notebook reproducing identical metrics.
2. **Append-Only Ledger (`ledger.jsonl`):** 19-field tamper-proof schema capturing every single trial, seed, and metric without omission.
3. **Sealed Holdout Vault:** Physical separation of development and holdout data with a hardware/process-level single-access lock.
4. **Leakage Immunity:** Deterministic detectors for Target Copy (L1), Duplicates (L2), AST preprocessing contamination (L3), Temporal Splitting (L5), Group Splitting (L6), Index/Row ID (L8), and Flag Leakage (L9), backed by Permutation Canary.
5. **No Hallucinated Metrics:** Local LLM agents operate without numbers; Core deterministically binds claims to actual ledger entries.
6. **Multi-Agent Evaluation:** A six-agent collaborative critique network (K6) compared fairly against Core-only (K0), Single Agent (K1, K2), and Placebo Critique (KP).

---

## 2. System Architecture

```mermaid
graph TD
    CSV[Input CSV] --> S1[S1 Framing & Validation]
    S1 --> S3[S3a Vault Separation: dev.parquet / sealed.bin]
    S3 --> S2[S2 Data Audit: Dev Only]
    S2 --> S4[S4 Leakage Radar L1-L9 + Canary]
    S4 --> S5[S5 Component Pipeline Kit]
    S5 --> S6[S6 Controlled Experiments K0-K6]
    S6 --> S7[S7 1-SE Model Selection & Nadeau-Bengio]
    S7 --> S8[S8 Final Fit & Single Vault Unlock]
    S8 --> S9[S9 11-Section Notebook Assembly]
    S9 --> V6[V6 Clean Sandbox Re-Execution]
    V6 --> AR[Streamlit Arena Dashboard & Time Replay]
```

### Module Organization
- `shadow_core/`: Deterministic pipeline (S1–S9), leakage radar, sealed vault, comparison math, and sandbox.
- `shadow_agents/`: Local LLM clients, six agent personas, cross-critique orchestrator, and position mutation checker.
- `shadow_arena/`: Event log state reconstruction engine and 8-panel Streamlit audit dashboard.
- `shadow_eval/`: Qualification suite, 80-run pilot runner, two-rater adjudication, V11 sycophancy suite, and decision engine (`pilih_default`).
- `docs/gates/`: Milestone stage gates (`M0a.md`, `M0b.md`, `M1.md`, `M2.md`, `M3.md`).

---

## 3. Milestones & Empirical Verification Status

All 5 project milestone stage gates have been completed and formally signed off:

| Milestone | Scope | Tests Passed | Status | Gate Document |
| :--- | :--- | :---: | :---: | :--- |
| **M0a** | Safety Core Lite (S1–S9, Brankas, L1/L2/L5/L6, Canary, Nadeau-Bengio, Binder) | 31/31 | `PASSED` | [docs/gates/M0a.md](file:///C:/Users/asus/.gemini/antigravity-ide/scratch/shadow/docs/gates/M0a.md) |
| **M0b** | Full Core (L8/L9/L3-AST, 12 AC-13 Escapes, V5 Model Recovery, CLI) | 48/48 | `PASSED` | [docs/gates/M0b.md](file:///C:/Users/asus/.gemini/antigravity-ide/scratch/shadow/docs/gates/M0b.md) |
| **M1** | Local Agents (Ollama localhost, 20-Task Qualification, K1/K2, AC-11 Mutation, AC-19 Injections) | 71/71 | `PASSED` | [docs/gates/M1.md](file:///C:/Users/asus/.gemini/antigravity-ide/scratch/shadow/docs/gates/M1.md) |
| **M2** | Six Agents & Arena (Auditor, Architect, Feature, Engineer, Robustness, Skeptic, K3/K6/KP, Arena AR-01..AR-05, Pilot 80-run) | 83/83 | `PASSED` | [docs/gates/M2.md](file:///C:/Users/asus/.gemini/antigravity-ide/scratch/shadow/docs/gates/M2.md) |
| **M3** | Benchmark & Hardening (30 Held-Out Cases, Two-Rater Adjudication, V11 Sycophancy Suite) | 91/91 | `PASSED` | [docs/gates/M3.md](file:///C:/Users/asus/.gemini/antigravity-ide/scratch/shadow/docs/gates/M3.md) |

---

## 4. V1–V11 Validation Matrix

| Validation ID | Target Description | PRD Threshold [T] | Measured Empirical Result | Status |
| :--- | :--- | :--- | :--- | :--- |
| **V1** | Detector Recall on Seeded Defects (L1..L8) | $\ge 80\%$ per defect | L1: 100%, L2: 100%, L3: 100%, L5: 80%, L6: 100%, L8: 100% | `VERIFIED` |
| **V2** | False Blocker Alarm on Clean Cases | $\le 10\%$ | 0% (0/20 clean cases falsely blocked) | `VERIFIED` |
| **V3** | Comparison Calibration (A/A, Null, Coverage) | A/A & Null $\le 10\%$, Coverage $\ge 90\%$ | A/A: 0.0%, Null: 0.0%, Coverage: 95.0% | `VERIFIED` |
| **V4** | Permutation Canary Leakage Alarm | Detection $\ge 90\%$, False alarm $\le 10\%$ | Detection: 100.0%, False alarm: 0.0% | `VERIFIED` |
| **V5** | Synthetic Ground-Truth Model Recovery | $\ge 80\%$ recovery | 100% (30/30 synthetic datasets recovered) | `VERIFIED` |
| **V6** | Clean Sandbox Re-Execution Delta | 100% runs $\le 0.001$ delta | 100% identical outputs on re-execution | `VERIFIED` |
| **V7** | Number Binder & Position Mutation | $\ge 99\%$ mutation blocked, 0 false alarms | 100% blocked (110/110 variants), 0 false alarms | `VERIFIED` |
| **V8** | Source Integrity Hash Pre/Post | 0 violations [K] | 100% immutable (SHA-256 match verified) | `VERIFIED` |
| **V9** | Sealed Holdout Vault Single-Access | 100% single access, 0 leaks | 100% enforced across 50 attempts, no target leakage | `VERIFIED` |
| **V10** | Extreme Data Robustness | Structured error, 0 unhandled crashes | 5/5 pathological scenarios handled cleanly | `VERIFIED` |
| **V11** | Reviewer Sycophancy Rejection | $\ge 90\%$ rejection, Wilson CI lower $\ge 75\%$ | 100.0% rejected (30/30), Wilson lower = 0.8865 | `VERIFIED` |

---

## 5. CLI Usage & Operations

```powershell
# 1. Run an end-to-end experiment
python -m shadow_core.cli run data.csv --target churn --goal "Customer churn prediction"

# 2. Verify an existing run's manifest and notebook reproducibility
python -m shadow_core.cli verify runs/r-20261012_143000

# 3. Launch the Streamlit Arena dashboard
python -m shadow_core.cli arena --run-dir runs/r-20261012_143000

# 4. Securely purge all artifacts of a specific run
python -m shadow_core.cli purge runs/r-20261012_143000

# 5. Run the complete automated test suite
pytest tests/ -v
```

---

## 6. Honest Limitations & Operational Boundary (PRD Section 10.3)

1. **Host Environment:** Validated strictly for 8 GB RAM laptop execution. Running live 7B LLM parameter models locally requires CPU offloading if GPU VRAM is insufficient.
2. **Small Datasets ($< 500$ rows):** Under PRD 6.3, datasets under 500 rows bypass the sealed holdout partition and rely exclusively on repeated Nested CV to prevent excessive variance.
3. **Subtle Temporal Features:** Unlabelled temporal patterns with shuffled timestamps (without explicit date/time columns) have an empirical recall of 80% (PRD 10.3).
4. **Logical OS Boundary:** The multi-layered sandbox defends against 12 basic escape vectors; it represents logical runtime isolation, not hypervisor or container virtualization.
