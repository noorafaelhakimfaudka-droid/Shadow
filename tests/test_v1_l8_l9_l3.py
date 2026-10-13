# tests/test_v1_l8_l9_l3.py
"""
Verifikasi V1 lanjutan untuk L8, L9, dan L3 (AST).
PRD Bagian 10.1 & T0.6b:
- L8: Deteksi ID / Indeks baris
- L9: Deteksi flag pasca-kejadian
- L3: Analisis statis AST pola fit_transform sebelum train_test_split
"""

import pytest
import pandas as pd
import numpy as np
from shadow_core.leakage.l8_index import detect_l8_index_leakage
from shadow_core.leakage.l9_flags import detect_l9_flag_leakage
from shadow_core.leakage.l3_static import analyze_code_for_leakage
from shadow_eval.seed_defects import (
    generate_clean_dataset,
    seed_l8_index_leakage,
    wilson_score_interval,
)


def test_v1_l8_index_leakage_recall():
    """Menguji recall detektor L8 pada 20 dataset bersemai ID/Index."""
    n_runs = 20
    detected = 0

    for i in range(n_runs):
        clean = generate_clean_dataset(n_samples=100, seed=100 + i)
        leaked = seed_l8_index_leakage(clean, col_name="patient_uuid", is_monotonic=(i % 2 == 0), seed=i)
        report = detect_l8_index_leakage(leaked, target_col="target")
        if not report.passed and any(f.column == "patient_uuid" for f in report.findings):
            detected += 1

    p, lower, upper = wilson_score_interval(detected, n_runs)
    print(f"\n[V1 - L8] Recall: {detected}/{n_runs} ({p:.1%}) [95% CI: {lower:.1%} - {upper:.1%}]")
    assert detected >= 18, f"Recall L8 di bawah 80%: {detected}/{n_runs}"


def test_v1_l9_flag_leakage_recall():
    """Menguji recall detektor L9 pada pola flag bersyarat dan post-outcome names."""
    n_runs = 10
    detected = 0

    for i in range(n_runs):
        df = generate_clean_dataset(n_samples=120, seed=200 + i)
        # Tanam kolom refund_status yang hanya terisi saat target == 1
        df["refund_status"] = df["target"].apply(lambda t: "Disetujui" if t == 1 else None)

        report = detect_l9_flag_leakage(df, target_col="target")
        if not report.passed and any(f.column == "refund_status" for f in report.findings):
            detected += 1

    p, lower, upper = wilson_score_interval(detected, n_runs)
    print(f"\n[V1 - L9] Recall: {detected}/{n_runs} ({p:.1%}) [95% CI: {lower:.1%} - {upper:.1%}]")
    assert detected >= 9, f"Recall L9 di bawah 80%: {detected}/{n_runs}"


def test_v1_l3_ast_leakage_recall():
    """Menguji detektor AST L3 pada potongan kode bocor vs bersih."""
    leaked_code_snippets = [
        """
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)
""",
        """
imputer = SimpleImputer(strategy='mean')
imputer.fit(data)
X_tr, X_te, y_tr, y_te = train_test_split(X, y)
""",
        """
encoder = OneHotEncoder()
df_enc = encoder.fit_transform(X)
train_test_split(df_enc, y)
""",
    ]

    for snippet in leaked_code_snippets:
        report = analyze_code_for_leakage(snippet)
        assert not report.passed, f"L3 AST gagal mendeteksi kebocoran pada:\n{snippet}"
        assert any(f.pattern == "fit_before_split" for f in report.findings)

    clean_code_snippet = """
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
"""
    clean_report = analyze_code_for_leakage(clean_code_snippet)
    assert clean_report.passed, f"L3 AST menghasilkan alarm palsu pada kode bersih:\n{clean_code_snippet}"
