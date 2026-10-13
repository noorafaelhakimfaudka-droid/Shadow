# tests/test_model_qualification.py
"""
Uji Otomatis Harness Uji Kelayakan Model (PRD 4.3, 12.2 T1.2).
"""

import json
import pytest
from shadow_eval.model_qualification import (
    run_model_qualification,
    STANDARD_20_TASKS,
)


def test_standard_20_tasks_length():
    """Memverifikasi terdapat tepat 20 tugas baku per PRD 4.3."""
    assert len(STANDARD_20_TASKS) == 20


def test_model_qualification_pass_criteria():
    """Memverifikasi model lolos kualifikasi bila memenuhi seluruh ambang PRD 4.3."""
    valid_resp = json.dumps({
        "artifact_id": "a-001",
        "run_id": "r-001",
        "agent": "auditor",
        "seq": 1,
        "jenis": "usulan",
        "klaim": "Model HistGradientBoosting menunjukkan performa terbaik dengan skor {{histgb_score}}.",
        "rujukan_bukti": ["trial:t-001"],
        "keyakinan": "tinggi",
        "alasan_keyakinan": "evaluasi nadeau bengio",
        "risiko_asumsi": "none",
    })

    # Simulasikan 20 respon valid tanpa angka bebas
    mock_responses = [valid_resp] * 20
    report = run_model_qualification(
        model_name="mock_qualified_model",
        mock_responses=mock_responses,
    )

    assert report.total_tasks == 20
    assert report.json_schema_pass_rate >= 0.95
    assert report.no_free_digits_rate >= 0.95
    assert report.no_fabrication_rate == 1.00
    assert report.peak_ram_gb <= 3.5
    assert report.is_qualified is True


def test_model_qualification_rejects_raw_digits():
    """Memverifikasi model ditolak bila menulis angka mentah bebas (melanggar Aturan R16)."""
    invalid_resp = json.dumps({
        "artifact_id": "a-002",
        "run_id": "r-001",
        "agent": "auditor",
        "seq": 1,
        "jenis": "usulan",
        "klaim": "Model HistGB memiliki ROC-AUC 0.892 dan mengalahkan baseline 15 kali.",
        "rujukan_bukti": ["trial:t-001"],
        "keyakinan": "tinggi",
        "alasan_keyakinan": "evaluasi nadeau bengio",
        "risiko_asumsi": "none",
    })

    mock_responses = [invalid_resp] * 20
    report = run_model_qualification(
        model_name="mock_disqualified_model",
        mock_responses=mock_responses,
    )

    assert report.no_free_digits_rate < 0.95
    assert report.is_qualified is False
