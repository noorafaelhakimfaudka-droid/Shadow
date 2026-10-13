# STAGE GATE M0b — FULL CORE COMPLEMENT & CLI VERIFICATION

**Date:** 2026-10-12  
**Status:** PASSED  
**Evaluator:** Principal Data Scientist & Independent Experiment Auditor  
**Scope Guardian Enforcement:** PASSED (`tests/test_no_premature_code.py` verified: zero code in `shadow_arena/` before M1 gate)

---

## 1. Executive Summary

Milestone M0b completes the full deterministic core of SHADOW v4.1 (Release R1 "Shadow Core").
All complementary detectors (L8 ID/index, L9 post-outcome flags, L3 static AST code analysis), full S8/S9 diagnostics (calibration curves, ECE, Brier score, error segmentation, generalization gap, learning curves, seed robustness), Model Recovery validation (V5), the 12-test layered sandbox escape defense suite (AC-13), and the CLI commands (`shadow run`, `shadow verify`, `shadow purge`) have been implemented, tested, and empirically verified.

The entire test suite comprising 48 automated tests passes cleanly:
```
======================== 48 passed in 76.02s (0:01:16) ========================
```

---

## 2. Milestone M0b Verification Matrix

| Gate / Requirement | Description | Target / Threshold | Empirical Result | Status |
| :--- | :--- | :--- | :--- | :--- |
| **V1 (L8 Index Leakage)** | Identifier & sequence row leak detection | $\ge 80\%$ recall | 20/20 (100.0%) [95% Wilson CI: 83.9% - 100.0%] | `VERIFIED` |
| **V1 (L9 Flag Leakage)** | Post-outcome & conditional flags | $\ge 80\%$ recall | 10/10 (100.0%) [95% Wilson CI: 72.2% - 100.0%] | `VERIFIED` |
| **V1 (L3 AST Code Analysis)** | Static code AST `fit_transform` before split | $\ge 80\%$ recall, 0 FP clean | 3/3 leaks caught (100%), 0 false alarms on clean code | `VERIFIED` |
| **V5 (Model Recovery)** | Correct recovery of generating model family (Linear vs Non-linear interaction) | $\ge 80\%$ on $\ge 30$ synthetic datasets | 30/30 (100.0%) [95% Wilson CI: 88.6% - 100.0%] | `VERIFIED` |
| **AC-13 (Sandbox Defense)** | 12 basic escape attempts blocked | 12/12 attempts blocked | 12/12 blocked (traversal, env vars, sockets, immutability, fork limits, RAM monitor, timeouts, AST block, pickle block, lock integrity, AES vault isolation, ledger schema) | `VERIFIED` |
| **CLI Validation** | `shadow run`, `shadow verify`, `shadow purge` | Exit code 0, complete cleanup | 100% verified functional in `tests/test_cli.py` | `VERIFIED` |
| **S8/S9 Diagnostics** | Calibration, generalization gap, learning curves, seed robustness | Operational modules & checklist integration | Implemented in `shadow_core/evaluation.py` & `shadow_core/interpret.py` | `VERIFIED` |
| **Full Core Regression** | No regression across M0a & M0b suites | 100% tests pass | 48/48 tests passed | `VERIFIED` |

---

## 3. Detailed Verification Proofs

### 3.1. Verification V1 (L8, L9, L3 AST)
- **Suite:** `tests/test_v1_l8_l9_l3.py`
- **L8 (Index/ID):** Evaluated across 20 synthetic datasets seeded with high-cardinality and monotonic ID columns. 20/20 detected. Distinguished correctly from continuous float measurement features.
- **L9 (Post-outcome flags):** Evaluated across 10 datasets with target-conditional presence. 10/10 detected.
- **L3 (AST Analysis):** Evaluated across AST trees with transformers fitted on whole dataset before `train_test_split`. All 3 defect patterns caught; 0 false alarms on standard clean pipeline code.

### 3.2. Verification V5 (Model Recovery on Known Generating Structures)
- **Suite:** `tests/test_v5_model_recovery.py`
- **Linear Cases:** Evaluated on 15 synthetic linear datasets. In 15/15 cases, Logistic Regression was either the top model or within 1-SE of the ensemble, correctly selected by the 1-SE parsimony rule.
- **Non-linear Interaction Cases:** Evaluated on 15 synthetic datasets with XOR and polynomial boundary interactions. In 15/15 cases, HistGradientBoosting significantly outperformed the linear model by $\ge 0.05$ ROC-AUC, correctly winning model selection.
- **Total Recovery Rate:** 30/30 = 100.0% (surpassing the $\ge 80\%$ acceptance threshold).

### 3.3. Verification AC-13 (Layered Sandbox Defense Suite)
- **Suite:** `tests/test_ac13_sandbox.py`
- All 12 threat scenarios executed:
  1. Outside file access via path traversal: BLOCKED.
  2. Sensitive environment credentials sanitization: BLOCKED.
  3. Outbound non-localhost socket connections (Rule R12): BLOCKED.
  4. Source dataset modification (immutability check): BLOCKED.
  5. Process limit clamp ($n\_jobs \le 2$): ENFORCED.
  6. Memory ceiling limit detection ($90\%$ RAM trigger): DETECTED.
  7. Cell timeout limit ($300$s limit): ENFORCED.
  8. Dangerous OS/subprocess call AST analysis: BLOCKED.
  9. Malicious pickle payload deserialization: BLOCKED.
  10. Dependency file `requirements.lock` tampering: DETECTED.
  11. Direct unencrypted access to `sealed.bin`: BLOCKED (AES-256 Fernet).
  12. Non-conformant ledger writes bypassing append-only schema: BLOCKED.

### 3.4. CLI Functionality
- **Suite:** `tests/test_cli.py`
- `shadow run`: Executed end-to-end dataset framing, splitting, audit, modeling, notebook assembly, and clean replay.
- `shadow verify`: Cleanly replayed notebook and verified binder number consistency against ledger.
- `shadow purge`: Completely removed run artifacts directory without leftover state.

---

## 4. Gate M0b Sign-Off

The requirements for Milestone M0b are fully satisfied and verified with empirical evidence.
Permission is hereby **GRANTED** to proceed to **Milestone M1 (Local LLM Agents & Decision Contracts)**.

**GATE M0b: PASSED**
