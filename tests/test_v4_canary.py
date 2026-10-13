# tests/test_v4_canary.py
"""
Verifikasi V4: Deteksi Kanari Permutasi (PRD 10.1 & 7.1).
Target [T]: Deteksi pipeline bocor >= 90%; Alarm palsu pada pipeline bersih <= 10%.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin

from shadow_core.leakage.canary import run_permutation_canary
from shadow_eval.seed_defects import generate_clean_dataset, wilson_score_interval


import sys

class ProcedurallyLeakedClassifier(BaseEstimator, ClassifierMixin):
    """Classifier yang sengaja bocor prosedural (mengintip label validasi)."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        frame = sys._getframe(1)
        y_val = frame.f_locals.get("y_val")
        if y_val is not None:
            return np.array(y_val)
        return np.ones(len(X), dtype=int)


def test_v4_canary_clean_pipeline_false_alarm():
    """V4: Alarm palsu pada pipeline bersih <= 10%."""
    df = generate_clean_dataset(n_samples=200, task="binary", seed=42)
    X = df.drop(columns=["target"])
    y = df["target"]

    clean_model = LogisticRegression(max_iter=500)
    rep = run_permutation_canary(
        pipeline_estimator=clean_model,
        X=X,
        y=y,
        is_classification=True,
        n_permutations=5,
        cv_folds=3,
        seed=42,
    )

    print(f"\nV4 Clean Canary Mean Score: {rep.mean_score:.4f} vs Threshold: {rep.alarm_threshold:.4f}")
    assert rep.passed is True, "Alarm palsu pada pipeline bersih!"
    assert rep.is_blocker is False


def test_v4_canary_leaked_pipeline_detection():
    """V4: Deteksi pipeline bocor >= 90%."""
    df = generate_clean_dataset(n_samples=200, task="binary", seed=42)
    X = df.drop(columns=["target"])
    y = df["target"]

    leaked_model = ProcedurallyLeakedClassifier()
    rep = run_permutation_canary(
        pipeline_estimator=leaked_model,
        X=X,
        y=y,
        is_classification=True,
        n_permutations=5,
        cv_folds=3,
        seed=42,
    )

    print(f"\nV4 Leaked Canary Mean Score: {rep.mean_score:.4f} vs Threshold: {rep.alarm_threshold:.4f}")
    assert rep.passed is False, "Kanari gagal mendeteksi pipeline yang bocor!"
    assert rep.is_blocker is True
