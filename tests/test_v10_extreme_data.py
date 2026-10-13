# tests/test_v10_extreme_data.py
"""
Verifikasi V10: Ketahanan pada Data Ekstrem (PRD 10.1).
Skenario:
- 99% missing
- 1 sampel kelas minoritas
- Korelasi sempurna antar-fitur (collinear)
- Hanya satu kolom (target saja)
- Nol baris (kosong)
Semua skenario harus menghasilkan penanganan kesalahan terstruktur tanpa unhandled crash.
"""

import pytest
from pathlib import Path
import numpy as np
import pandas as pd
from shadow_core.audit import audit_dataset
from shadow_core.framing import frame_problem, detect_task_type


def test_v10_empty_dataset():
    df = pd.DataFrame(columns=["feat", "target"])
    with pytest.raises(ValueError) as excinfo:
        frame_problem(df, "target")
    assert "kosong" in str(excinfo.value).lower()

    rep = audit_dataset(df, "target")
    assert rep.has_blocker is True


def test_v10_single_column_target_only():
    df = pd.DataFrame({"target": [0, 1, 0, 1, 0, 1]})
    framing = frame_problem(df, "target")
    assert framing.task_type == "binary_classification"

    rep = audit_dataset(df, "target")
    # Tidak ada fitur yang bisa dimodelkan
    assert rep.total_columns == 1


def test_v10_extreme_missing_values():
    n = 200
    df = pd.DataFrame({
        "feat_almost_empty": [None] * 198 + [1.0, 2.0],
        "target": [0] * 100 + [1] * 100,
    })
    rep = audit_dataset(df, "target")
    assert any(f.category == "high_missing" for f in rep.warnings)


def test_v10_single_minority_sample():
    # Hanya 1 sampel kelas 1
    df = pd.DataFrame({
        "feat": range(100),
        "target": [0] * 99 + [1],
    })
    framing = frame_problem(df, "target")
    assert framing.is_imbalanced is True
    assert framing.minority_class_ratio == 0.01


def test_v10_perfect_collinearity():
    n = 100
    x = np.random.normal(0, 1, n)
    df = pd.DataFrame({
        "feat1": x,
        "feat2": x,  # Identik persis
        "target": (x > 0).astype(int),
    })
    rep = audit_dataset(df, "target")
    assert rep is not None
