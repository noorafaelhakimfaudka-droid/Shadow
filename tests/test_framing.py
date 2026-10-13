# tests/test_framing.py
import pytest
import pandas as pd
import numpy as np
from shadow_core.framing import frame_problem, detect_task_type


def test_framing_balanced_binary():
    df = pd.DataFrame({"feat": range(100), "target": [0] * 50 + [1] * 50})
    res = frame_problem(df, "target", goal="Test balanced")
    assert res.task_type == "binary_classification"
    assert res.is_imbalanced is False
    assert res.primary_metric == "balanced_accuracy"
    assert res.practical_margin_delta == 0.01


def test_framing_imbalanced_binary():
    # 10% class 1 -> minority < 20%
    df = pd.DataFrame({"feat": range(100), "target": [0] * 90 + [1] * 10})
    res = frame_problem(df, "target", goal="Test imbalanced")
    assert res.task_type == "binary_classification"
    assert res.is_imbalanced is True
    assert res.primary_metric == "pr_auc"
    assert "recall_at_precision" in res.companion_metrics


def test_framing_regression():
    np.random.seed(42)
    y = np.random.normal(50, 10, size=200)
    df = pd.DataFrame({"feat": range(200), "val": y})
    res = frame_problem(df, "val", goal="Test regression")
    assert res.task_type == "regression"
    assert res.primary_metric == "mae"
    assert res.practical_margin_delta > 0
