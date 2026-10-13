# tests/test_dynamic_ml_engine.py
"""
Test Suite for Dynamic ML Engine (CSV Input, Analysis, and Debate Generation).
Memastikan:
1. Analisis dataset berhasil mengekstrak baris, kolom, target, dan korelasi.
2. 3 model ML (Logistic Regression, Random Forest, GBDT) dilatih secara deterministik.
3. Canary test dan holdout evaluation menghasilkan metrik yang valid.
4. Langkah debat (14 langkah) dihasilkan dengan referensi kolom dan metrik nyata.
5. Menangani dataset custom dengan target string atau biner.
"""

import pandas as pd
import pytest
from shadow_core.dynamic_ml_engine import analyze_and_train_dataset, generate_dynamic_debate_steps


def test_analyze_and_train_bank_marketing():
    df = pd.read_csv("data/bank_marketing.csv")
    res = analyze_and_train_dataset(df, "deposit")

    assert res["total_rows"] == 600
    assert res["total_cols"] == 8
    assert res["target_col"] == "deposit"
    assert "Logistic Regression" in res["model_metrics"]
    assert "Random Forest" in res["model_metrics"]
    assert "LightGBM / GBDT" in res["model_metrics"]
    assert 0.0 <= res["model_metrics"]["Logistic Regression"]["cv_roc_auc"] <= 1.0
    assert 0.0 <= res["holdout_auc"] <= 1.0
    assert 0.0 <= res["canary_ba"] <= 1.0

    steps = generate_dynamic_debate_steps(res, "Bank Marketing")
    assert len(steps) == 28
    # Memastikan tidak ada field 'time' (timer dihapus)
    for s in steps:
        assert "time" not in s
        assert "speaker" in s
        assert "title" in s
        assert "narration" in s


def test_analyze_and_train_telco_churn():
    df = pd.read_csv("data/telco_churn.csv")
    res = analyze_and_train_dataset(df, "churn")

    assert res["total_rows"] == 600
    assert res["target_col"] == "churn"
    assert res["pos_count"] == 217
    assert res["neg_count"] == 383
    assert res["best_model_name"] in ["Logistic Regression", "Random Forest", "LightGBM / GBDT"]

    steps = generate_dynamic_debate_steps(res, "Telco Churn")
    assert len(steps) == 28
    assert "churn" in steps[0]["auditor_bubble"]


def test_analyze_custom_dataframe():
    # Buat dataframe sintetis kecil untuk simulasi upload CSV oleh user
    custom_df = pd.DataFrame({
        "feature_a": [10, 20, 15, 30, 25, 40, 35, 50, 45, 60],
        "feature_b": ["red", "blue", "red", "blue", "red", "blue", "red", "blue", "red", "blue"],
        "is_fraud": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })
    res = analyze_and_train_dataset(custom_df, "is_fraud")

    assert res["total_rows"] == 10
    assert res["total_cols"] == 3
    assert res["target_col"] == "is_fraud"
    assert res["pos_count"] == 5

    steps = generate_dynamic_debate_steps(res, "Custom Fraud CSV")
    assert len(steps) == 28
    assert "is_fraud" in steps[0]["auditor_bubble"]
