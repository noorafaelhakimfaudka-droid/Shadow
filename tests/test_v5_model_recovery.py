# tests/test_v5_model_recovery.py
"""
Verifikasi V5 — Pemulihan Model pada Data Sintetis Berstruktur Diketahui (PRD Bagian 10.1 & T0.10).
- Data linear -> model linear terpilih (atau tidak kalah secara signifikan oleh aturan 1-SE)
- Data interaksi non-linear -> ensemble (RandomForest/HistGB) menang secara signifikan
Target: >= 80% pemulihan benar pada >= 30 dataset sintetis.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from shadow_eval.seed_defects import (
    generate_linear_recovery_dataset,
    generate_nonlinear_recovery_dataset,
    wilson_score_interval,
)
from shadow_core.comparison import select_model_1se_rule


def evaluate_models_on_cv(df: pd.DataFrame, target_col: str = "target", n_splits: int = 5):
    """Menjalankan evaluasi berpasangan antara Linear vs HistGB pada lipatan identik."""
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores_linear = []
    scores_histgb = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        m_lin = LogisticRegression(max_iter=500, random_state=42)
        m_lin.fit(X_tr, y_tr)
        p_lin = m_lin.predict_proba(X_val)[:, 1]
        scores_linear.append(float(roc_auc_score(y_val, p_lin)))

        m_hgb = HistGradientBoostingClassifier(max_iter=40, random_state=42)
        m_hgb.fit(X_tr, y_tr)
        p_hgb = m_hgb.predict_proba(X_val)[:, 1]
        scores_histgb.append(float(roc_auc_score(y_val, p_hgb)))

    cand_scores = {"linear": scores_linear, "histgb": scores_histgb}
    costs = {"linear": 1, "histgb": 2}
    chosen, _ = select_model_1se_rule(cand_scores, model_complexities=costs)
    return chosen, np.mean(scores_linear), np.mean(scores_histgb)



def test_v5_model_recovery_30_datasets():
    """Menguji pemulihan model pada 15 linear + 15 non-linear datasets."""
    n_linear = 15
    n_nonlinear = 15
    correct_recoveries = 0

    # 1. Linear datasets
    for i in range(n_linear):
        df_lin = generate_linear_recovery_dataset(n_samples=350, seed=500 + i)
        chosen, res_lin, res_hgb = evaluate_models_on_cv(df_lin)
        # Pada data linear, linear harus terpilih oleh aturan 1-SE (atau histgb tidak mengungguli secara bermakna)
        if chosen == "linear":
            correct_recoveries += 1
        else:
            # Jika histgb terpilih, cek apakah selisihnya benar-benar besar atau hanya noise
            diff = res_hgb - res_lin
            if diff < 0.02:  # sangat dekat
                correct_recoveries += 1

    # 2. Non-linear datasets
    for i in range(n_nonlinear):
        df_non = generate_nonlinear_recovery_dataset(n_samples=350, seed=600 + i)
        chosen, res_lin, res_hgb = evaluate_models_on_cv(df_non)
        # Pada data non-linear berinteraksi, histgb harus menang
        if chosen == "histgb" and (res_hgb - res_lin) >= 0.05:
            correct_recoveries += 1


    total_runs = n_linear + n_nonlinear
    p, lower, upper = wilson_score_interval(correct_recoveries, total_runs)
    print(f"\n[V5 Model Recovery] Correct: {correct_recoveries}/{total_runs} ({p:.1%}) [95% CI: {lower:.1%} - {upper:.1%}]")

    assert correct_recoveries >= 24, (
        f"Model Recovery di bawah target 80%: {correct_recoveries}/{total_runs}"
    )
