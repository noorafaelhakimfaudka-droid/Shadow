# tests/test_adjudication.py
"""
Uji Unit untuk Protokol Dua Penilai & Adjudikasi (PRD Bagian 11.4).
- Penilaian buta dua penilai A & B
- Resolusi perbedaan pendapat dengan penalti konservatif (tidak terselesaikan -> dihitung tidak ditemukan)
- Perhitungan kesepakatan per kategori (target >= 85%)
"""

import pytest
from shadow_eval.adjudication import ReviewItem, adjudicate_reviews, AgreementReport


def test_adjudicate_reviews_perfect_agreement():
    """Memverifikasi kesepakatan 100% jika semua label A dan B identik."""
    items = [
        ReviewItem("i-01", "L1", "Klaim fitur bocor target", "ledger:t-01", "valid", "valid"),
        ReviewItem("i-02", "L1", "Klaim duplikat fitur", "ledger:t-02", "duplikat", "duplikat"),
        ReviewItem("i-03", "L5", "Klaim waktu diacak", "ledger:t-03", "valid", "valid"),
        ReviewItem("i-04", "L6", "Klaim grup bocor", "ledger:t-04", "valid", "valid"),
        ReviewItem("i-05", "L8", "Klaim ID berkorelasi target", "ledger:t-05", "salah", "salah"),
    ]

    report = adjudicate_reviews(items)
    assert report.total_items == 5
    assert report.overall_agreement_pct == 1.0
    assert report.unresolved_count == 0
    assert report.category_agreements["L1"] == 1.0
    assert report.category_agreements["L5"] == 1.0
    assert report.category_agreements["L6"] == 1.0
    assert report.category_agreements["L8"] == 1.0

    # Pastikan label valid dianggap TP
    tp_items = [adj for adj in report.adjudications if adj.is_considered_tp]
    assert len(tp_items) == 3


def test_adjudicate_reviews_with_disagreement_and_unresolved():
    """
    Memverifikasi penanganan perbedaan pendapat:
    - Jika salah satu 'tidak_jelas', menjadi 'tidak_terselesaikan' dan BUKAN TP (PRD 11.4.8).
    - Jika A valid dan B salah, konservatif memilih 'salah'.
    """
    items = [
        # Setuju valid
        ReviewItem("i-01", "L1", "Klaim valid", "ledger:t-01", "valid", "valid"),
        # Beda pendapat: valid vs salah -> konservatif salah
        ReviewItem("i-02", "L1", "Klaim diragukan", "ledger:t-02", "valid", "salah"),
        # Beda pendapat: valid vs tidak_jelas -> unresolved
        ReviewItem("i-03", "L5", "Klaim ambigu", "ledger:t-03", "valid", "tidak_jelas"),
        # Setuju valid
        ReviewItem("i-04", "L5", "Klaim valid waktu", "ledger:t-04", "valid", "valid"),
    ]

    report = adjudicate_reviews(items)
    assert report.total_items == 4
    assert report.overall_agreement_pct == 0.50  # 2 dari 4 setuju langsung
    assert report.unresolved_count == 1

    adj_map = {adj.item_id: adj for adj in report.adjudications}
    assert adj_map["i-01"].final_label == "valid"
    assert adj_map["i-01"].is_considered_tp is True

    assert adj_map["i-02"].final_label == "salah"
    assert adj_map["i-02"].is_considered_tp is False

    assert adj_map["i-03"].final_label == "tidak_terselesaikan"
    assert adj_map["i-03"].is_unresolved is True
    assert adj_map["i-03"].is_considered_tp is False  # PRD 11.4.8


def test_adjudicate_empty_input():
    """Memverifikasi penanganan daftar kosong."""
    report = adjudicate_reviews([])
    assert report.total_items == 0
    assert report.overall_agreement_pct == 1.0
    assert report.unresolved_count == 0
    assert len(report.adjudications) == 0
