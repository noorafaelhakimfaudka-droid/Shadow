# shadow_eval/decide.py
"""
Aturan Keputusan Go/No-Go & Pemilihan Default (PRD Bagian 11.5).
Fungsi kode deterministik untuk memilih konfigurasi default secara adil
dan objektif berdasarkan bukti terukur (K0, K1, K2, K3, K6, KP).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class KonfigurasiHasil:
    recall_mean: float
    recall_ci_lower: float
    precision_mean: float
    median_time_minutes: float
    failure_rate: float
    false_positives_per_run: float
    sealed_perf_diff_vs_baseline: float = 0.0
    sealed_perf_ci_lower: float = 0.0


@dataclass
class EvaluasiHasil:
    configs: Dict[str, KonfigurasiHasil]
    n_kasus_simpanan: int
    protokol_penuh: bool = False
    k6_vs_kp_recall_ci_lower: float = 0.0
    delta: float = 0.01


@dataclass
class Keputusan:
    default_config: str
    klaim_penalaran: bool
    status: str  # 'tervalidasi' | 'indikatif'


def lolos(
    x_name: str,
    pembanding: str,
    hasil: EvaluasiHasil,
) -> bool:
    """Memeriksa apakah konfigurasi X memenuhi 5 syarat nilai tambah atas pembanding R."""
    if x_name not in hasil.configs or pembanding not in hasil.configs:
        return False

    x = hasil.configs[x_name]
    r = hasil.configs[pembanding]

    # K2 referensi waktu jika ada, jika tidak gunakan pembanding
    k2 = hasil.configs.get("K2", r)
    ref_time = k2.median_time_minutes

    # 1. Nilai Tambah
    diff_recall = x.recall_mean - r.recall_mean
    diff_recall_lower = x.recall_ci_lower - r.recall_mean
    has_recall_gain = (diff_recall >= 0.15) and (diff_recall_lower > 0.0)

    has_sealed_gain = (x.sealed_perf_diff_vs_baseline >= hasil.delta) and (x.sealed_perf_ci_lower > 0.0)

    if not (has_recall_gain or has_sealed_gain):
        return False

    # 2. Presisi >= 0.80
    if x.precision_mean < 0.80:
        return False

    # 3. Waktu <= 3x K2 dan <= 45 menit
    if x.median_time_minutes > (3.0 * ref_time) or x.median_time_minutes > 45.0:
        return False

    # 4. Kegagalan <= R + 0.05
    if x.failure_rate > (r.failure_rate + 0.05):
        return False

    # 5. Temuan salah (FP) <= R + 1.0
    if x.false_positives_per_run > (r.false_positives_per_run + 1.0):
        return False

    return True


def lolos_vs_plasebo(
    k6_name: str,
    kp_name: str,
    hasil: EvaluasiHasil,
) -> bool:
    """Memeriksa apakah K6 mengungguli KP (plasebo) dengan batas bawah CI > 0."""
    return hasil.k6_vs_kp_recall_ci_lower > 0.0


def pilih_default(hasil: EvaluasiHasil) -> Keputusan:
    """
    Algoritma pemilihan konfigurasi default PRD 11.5.
    Mengembalikan default_config, klaim_penalaran, dan status.
    """
    if hasil.n_kasus_simpanan < 5 or not hasil.configs:
        raise ValueError("Kasus evaluasi kurang dari 5 atau input konfigurasi kosong!")

    default = "K0"
    for candidate in ["K1", "K2", "K3", "K6"]:
        if candidate in hasil.configs and lolos(candidate, pembanding=default, hasil=hasil):
            default = candidate

    klaim_penalaran = (default == "K6") and lolos_vs_plasebo("K6", "KP", hasil)
    status = "tervalidasi" if (hasil.n_kasus_simpanan >= 30 and hasil.protokol_penuh) else "indikatif"

    return Keputusan(
        default_config=default,
        klaim_penalaran=klaim_penalaran,
        status=status,
    )
