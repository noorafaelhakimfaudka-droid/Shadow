# shadow_core/checklist.py
"""
Daftar Periksa Validitas dan Nilai Penyerahan (PRD 6.13 & Aturan R11).
Pemeriksaan deterministik: Blocker / Mayor / Minor.
Nilai penyerahan:
- 'Lengkap' : Tanpa Blocker dan tanpa Mayor
- 'Dengan peringatan' : Tanpa Blocker, ada Mayor/Minor
- 'Parsial' : Ada Blocker pada tahap akhir
- 'Ditolak' : Blocker kebocoran/brankas atau akses tersegel > 1
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


Grade = Literal["Lengkap", "Dengan peringatan", "Parsial", "Ditolak"]


@dataclass
class CheckItem:
    check_id: str
    name: str
    level: Literal["Blocker", "Mayor", "Minor"]
    passed: bool
    details: str


@dataclass
class ChecklistResult:
    grade: Grade
    passed: bool
    total_checks: int
    passed_checks: int
    blockers: List[CheckItem]
    mayors: List[CheckItem]
    minors: List[CheckItem]
    all_items: List[CheckItem]
    summary_text: str


def evaluate_checklist(checks: Dict[str, Dict[str, Any]]) -> ChecklistResult:
    """Mengevaluasi seluruh daftar periksa validitas deterministik sesuai Tabel 6.13."""
    items: List[CheckItem] = []

    # Daftar aturan pemeriksaan baku
    rules = [
        ("sealed_split_access", "Uji tersegel dipisah sebelum pemodelan; akses = 1", "Blocker"),
        ("split_strategy_structure", "Strategi split sesuai struktur data (grup/waktu)", "Blocker"),
        ("permutation_canary", "Kanari permutasi lolos (tidak ada alarm prosedural)", "Blocker"),
        ("leakage_detectors", "Tidak ada temuan L1/L2/L8 tanpa keputusan operator", "Blocker"),
        ("baseline_evaluated", "Baseline dummy dan baseline sederhana dievaluasi", "Blocker"),
        ("identical_folds", "Lipatan identik untuk semua kandidat", "Blocker"),
        ("reproducible_clean_run", "Notebook lolos eksekusi ulang; metrik cocok dalam toleransi", "Blocker"),
        ("zero_unbound_numbers", "Nol angka tanpa sumber di teks laporan", "Blocker"),
        ("source_hash_unmodified", "Hash file sumber utuh (0 modifikasi)", "Blocker"),
        ("metric_rules_compliant", "Metrik sesuai aturan 6.1 (bukan akurasi saja pada data imbalanced)", "Blocker"),
        ("pairwise_comparison", "Perbandingan memakai selisih berpasangan + rentang Nadeau-Bengio", "Mayor"),
        ("error_analysis_available", "Kalibrasi, ambang, analisis galat tersedia", "Mayor"),
        ("robustness_reported", "Variasi seed dan ketahanan dilaporkan", "Mayor"),
        ("generalization_gap", "Selisih generalisasi dalam batas (CV - test <= 0.05)", "Mayor"),
        ("limitations_stated", "Batasan dan detektor yang tidak dijalankan dituliskan", "Mayor"),
        ("selection_bias_handled", "Peringatan bias seleksi ditangani bila percobaan > 50", "Minor"),
        ("high_cardinality_handled", "Kolom kardinalitas tinggi ditangani dengan benar", "Minor"),
    ]

    for check_id, name, level in rules:
        c_data = checks.get(check_id, {})
        passed = c_data.get("passed", False)
        details = c_data.get("details", "Belum diverifikasi" if not passed else "Lolos verifikasi")
        items.append(CheckItem(
            check_id=check_id,
            name=name,
            level=level,  # type: ignore
            passed=passed,
            details=details,
        ))

    blockers = [item for item in items if item.level == "Blocker" and not item.passed]
    mayors = [item for item in items if item.level == "Mayor" and not item.passed]
    minors = [item for item in items if item.level == "Minor" and not item.passed]

    # Penentuan Nilai Penyerahan (PRD 6.13)
    if any(item.check_id in ("sealed_split_access", "leakage_detectors", "source_hash_unmodified") for item in blockers):
        grade: Grade = "Ditolak"
    elif len(blockers) > 0:
        grade = "Parsial"
    elif len(mayors) > 0 or len(minors) > 0:
        grade = "Dengan peringatan"
    else:
        grade = "Lengkap"

    summary = (
        f"Nilai Penyerahan: {grade} | "
        f"Total Pemeriksaan: {len(items)} | Lolos: {len(items) - len(blockers) - len(mayors) - len(minors)} | "
        f"Blocker: {len(blockers)} | Mayor: {len(mayors)} | Minor: {len(minors)}"
    )

    return ChecklistResult(
        grade=grade,
        passed=(grade in ("Lengkap", "Dengan peringatan")),
        total_checks=len(items),
        passed_checks=len(items) - len(blockers) - len(mayors) - len(minors),
        blockers=blockers,
        mayors=mayors,
        minors=minors,
        all_items=items,
        summary_text=summary,
    )
