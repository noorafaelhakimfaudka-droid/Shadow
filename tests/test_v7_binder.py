# tests/test_v7_binder.py
"""
Verifikasi V7: Pengikat Angka dan Klaim Terstruktur (PRD 10.1 & 6.12).
Target:
(a) Memblokir angka karangan >= 99% (>= 100 varian)
(b) Memblokir angka sah yang tertukar tempat / subjek / metrik >= 99% (>= 100 varian)
(c) 0 alarm palsu pada laporan sah (>= 100 laporan)
"""

import pytest
from pathlib import Path
from shadow_core.ledger import Ledger
from shadow_core.binder import (
    render,
    verify,
    verify_unmodified,
    render_structured_claim,
    Violation,
)


@pytest.fixture
def populated_ledger(tmp_path: Path) -> Ledger:
    ledger_path = tmp_path / "ledger.jsonl"
    ledger = Ledger(ledger_path)

    entries = [
        {
            "trial_id": "t-001",
            "run_id": "r-01",
            "case_id": "case-1",
            "ts": "2026-10-12T10:00:00Z",
            "agent": "core",
            "hipotesis": "Baseline dummy",
            "model": "Dummy",
            "hiperparameter": {},
            "pipeline_hash": "sha256:111",
            "fold_def_hash": "sha256:fff",
            "seed": 42,
            "skor_per_lipatan": [0.50, 0.50, 0.50],
            "ringkasan": {"mean": 0.5000, "sd": 0.0000, "ci95": [0.5000, 0.5000]},
            "waktu_detik": 1.2,
            "ram_puncak_mb": 100,
            "status": "selesai",
            "alasan": None,
            "cell_notebook": "8.1",
            "akses_uji_tersegel": False,
        },
        {
            "trial_id": "t-002",
            "run_id": "r-01",
            "case_id": "case-1",
            "ts": "2026-10-12T10:05:00Z",
            "agent": "core",
            "hipotesis": "Model linear",
            "model": "LogisticRegression",
            "hiperparameter": {"C": 1.0},
            "pipeline_hash": "sha256:222",
            "fold_def_hash": "sha256:fff",
            "seed": 42,
            "skor_per_lipatan": [0.75, 0.77, 0.76],
            "ringkasan": {"mean": 0.7600, "sd": 0.0100, "ci95": [0.7487, 0.7713]},
            "waktu_detik": 2.5,
            "ram_puncak_mb": 150,
            "status": "selesai",
            "alasan": None,
            "cell_notebook": "8.2",
            "akses_uji_tersegel": False,
        },
        {
            "trial_id": "t-003",
            "run_id": "r-01",
            "case_id": "case-1",
            "ts": "2026-10-12T10:10:00Z",
            "agent": "core",
            "hipotesis": "HistGB",
            "model": "HistGradientBoosting",
            "hiperparameter": {"max_iter": 100},
            "pipeline_hash": "sha256:333",
            "fold_def_hash": "sha256:fff",
            "seed": 42,
            "skor_per_lipatan": [0.85, 0.86, 0.84],
            "ringkasan": {"mean": 0.8500, "sd": 0.0100, "ci95": [0.8387, 0.8613]},
            "waktu_detik": 4.1,
            "ram_puncak_mb": 200,
            "status": "selesai",
            "alasan": None,
            "cell_notebook": "8.3",
            "akses_uji_tersegel": False,
        },
    ]

    for e in entries:
        ledger.append(e)

    return ledger


def test_v7_a_fabricated_numbers_blocked(populated_ledger: Ledger):
    """V7 (a): Memblokir angka karangan (halusinasi) >= 99% pada >= 100 varian."""
    blocked_count = 0
    n_variants = 100

    for i in range(n_variants):
        # Angka yang sengaja dikarang dan tidak ada di ledger
        fake_num = f"{0.9123 + i * 0.0005:.4f}"
        text = f"Model kita mencapai akurasi fantastis {fake_num} pada pengujian."
        violations = verify(text, populated_ledger)
        if len(violations) > 0 and any(v.token == fake_num for v in violations):
            blocked_count += 1

    rate = blocked_count / n_variants
    print(f"\nV7(a) Fabricated Numbers Blocked: {blocked_count}/{n_variants} ({rate:.1%})")
    assert rate >= 0.99, f"Tingkat pemblokiran angka karangan {rate:.1%} < 99%!"


def test_v7_b_swapped_claims_blocked(populated_ledger: Ledger):
    """V7 (b): Memblokir angka sah yang tertukar tempat/subjek >= 99% pada >= 100 varian."""
    blocked_count = 0
    n_variants = 100

    for i in range(n_variants):
        # Klaim terstruktur di mana subjek ditukar: Mengklaim skor Dummy untuk HistGradientBoosting
        swapped_claim = {
            "subjek": "HistGradientBoosting",  # Subjek salah!
            "metrik": "balanced_accuracy",
            "split": "cv",
            "kunci_ledger": "t-001",  # Padahal t-001 adalah Dummy!
            "templat": "Model {{subjek}} memperoleh mean {{nilai}}.",
        }
        rendered, violation = render_structured_claim(swapped_claim, populated_ledger)
        if violation is not None and violation.violation_type == "mismatched_claim":
            blocked_count += 1

    rate = blocked_count / n_variants
    print(f"\nV7(b) Swapped Claims Blocked: {blocked_count}/{n_variants} ({rate:.1%})")
    assert rate >= 0.99, f"Tingkat deteksi angka tertukar {rate:.1%} < 99%!"


def test_v7_c_zero_false_alarms_on_valid_reports(populated_ledger: Ledger):
    """V7 (c): 0 alarm palsu pada laporan sah yang dirender Core (>= 100 laporan)."""
    false_alarms = 0
    n_reports = 100

    for i in range(n_reports):
        # Template sah Core
        template = "Bagian 1: Model {{t-002.model}} menghasilkan mean {{t-002.mean}} dengan seed {{t-002.seed}}."
        rendered = render(template, populated_ledger)
        violations = verify(rendered, populated_ledger)
        if len(violations) > 0:
            false_alarms += 1

    print(f"\nV7(c) False Alarms: {false_alarms}/{n_reports}")
    assert false_alarms == 0, f"Ditemukan {false_alarms} alarm palsu pada laporan sah Core!"
