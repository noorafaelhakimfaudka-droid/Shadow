# tests/test_t1_5_position_mutation.py
"""
Suite Mutasi Posisi & Klaim Terstruktur (PRD 6.12, 12.2 T1.5 & AC-11).
Memverifikasi bahwa >= 100 varian mutasi posisi:
- Tertukar model / subjek
- Tertukar split (CV vs Sealed)
- Angka digit bebas dalam narasi
- Kata-angka verbal dalam narasi
terdeteksi >= 99% oleh Pemeriksa Posisi Core, dengan 0 alarm palsu pada klaim sah.
"""

import pytest
from pathlib import Path
from shadow_core.ledger import Ledger
from shadow_agents.contracts import AgentArtifact, StructuredClaim
from shadow_agents.position_checker import verify_agent_artifact_position
from shadow_eval.seed_defects import wilson_score_interval


@pytest.fixture
def test_ledger(tmp_path):
    """Menyiapkan ledger berisi entri percobaan baku."""
    ledger_p = tmp_path / "ledger.jsonl"
    ledger = Ledger(ledger_p)

    # t-001: Dummy classifier on CV
    ledger.append({
        "trial_id": "t-001", "run_id": "r-01", "case_id": "c-01",
        "agent": "core", "hipotesis": "Baseline dummy",
        "model": "DummyClassifier", "hiperparameter": {},
        "pipeline_hash": "h1", "fold_def_hash": "f1", "seed": 42,
        "skor_per_lipatan": [0.5, 0.5, 0.5],
        "ringkasan": {"mean": 0.5000, "sd": 0.0},
        "waktu_detik": 1.0, "ram_puncak_mb": 100,
        "status": "selesai", "alasan": None, "cell_notebook": "8.1",
        "akses_uji_tersegel": False,
    })

    # t-002: Logistic Regression on CV
    ledger.append({
        "trial_id": "t-002", "run_id": "r-01", "case_id": "c-01",
        "agent": "core", "hipotesis": "Linear model",
        "model": "LogisticRegression", "hiperparameter": {"C": 1.0},
        "pipeline_hash": "h2", "fold_def_hash": "f1", "seed": 42,
        "skor_per_lipatan": [0.81, 0.82, 0.80],
        "ringkasan": {"mean": 0.8100, "sd": 0.01},
        "waktu_detik": 2.5, "ram_puncak_mb": 150,
        "status": "selesai", "alasan": None, "cell_notebook": "8.2",
        "akses_uji_tersegel": False,
    })

    # t-003: HistGB on CV
    ledger.append({
        "trial_id": "t-003", "run_id": "r-01", "case_id": "c-01",
        "agent": "core", "hipotesis": "GBDT model",
        "model": "HistGradientBoostingClassifier", "hiperparameter": {},
        "pipeline_hash": "h3", "fold_def_hash": "f1", "seed": 42,
        "skor_per_lipatan": [0.86, 0.88, 0.87],
        "ringkasan": {"mean": 0.8700, "sd": 0.01},
        "waktu_detik": 5.0, "ram_puncak_mb": 200,
        "status": "selesai", "alasan": None, "cell_notebook": "8.3",
        "akses_uji_tersegel": False,
    })

    # t-004: HistGB on Sealed Holdout
    ledger.append({
        "trial_id": "t-004", "run_id": "r-01", "case_id": "c-01",
        "agent": "core", "hipotesis": "Final sealed evaluation",
        "model": "HistGradientBoostingClassifier", "hiperparameter": {},
        "pipeline_hash": "h4", "fold_def_hash": "f1", "seed": 42,
        "skor_per_lipatan": [0.8650],
        "ringkasan": {"mean": 0.8650, "sd": 0.0},
        "waktu_detik": 1.5, "ram_puncak_mb": 220,
        "status": "selesai", "alasan": None, "cell_notebook": "9.1",
        "akses_uji_tersegel": True,
    })

    return ledger


def test_position_mutation_suite_110_variants(test_ledger):
    """Menguji 110 varian mutasi posisi dan klaim terstruktur."""
    total_mutations = 0
    detected_mutations = 0

    # 1. 30 varian: Model/Subjek Tertukar (Swap Model)
    # Menghubungkan trial t-002 (LogisticRegression) tetapi klaim menyebut HistGB atau sebaliknya
    for i in range(30):
        total_mutations += 1
        claim = StructuredClaim(
            subject="HistGradientBoosting" if i % 2 == 0 else "RandomForest",
            metric="roc_auc",
            split="cv",
            trial_key="t-002",  # sebenarnya LogisticRegression
            template="Model {{subjek}} memperoleh performa {{nilai}}.",
        )
        art = AgentArtifact(
            artifact_id=f"a-swap-model-{i}",
            run_id="r-01",
            agent="model_architect",
            seq=i,
            jenis="temuan",
            klaim="Hasil evaluasi model pada validasi silang.",
            rujukan_bukti=["trial:t-002"],
            keyakinan="tinggi",
            alasan_keyakinan="validasi CV",
            risiko_asumsi="asumsi umum",
            structured_claim=claim,
        )
        res = verify_agent_artifact_position(art, test_ledger)
        if not res.is_valid and any("Mutasi Subjek" in v for v in res.violations):
            detected_mutations += 1

    # 2. 30 varian: Split Tertukar (Swap Split)
    # Menghubungkan trial CV t-003 tetapi mengklaim sebagai uji tersegel (atau sebaliknya)
    for i in range(30):
        total_mutations += 1
        claim = StructuredClaim(
            subject="HistGradientBoostingClassifier",
            metric="roc_auc",
            split="sealed_test" if i % 2 == 0 else "uji_tersegel",
            trial_key="t-003",  # sebenarnya CV, bukan sealed!
            template="Model {{subjek}} pada uji tersegel memperoleh {{nilai}}.",
        )
        art = AgentArtifact(
            artifact_id=f"a-swap-split-{i}",
            run_id="r-01",
            agent="experiment_engineer",
            seq=30 + i,
            jenis="hasil_eksperimen",
            klaim="Hasil performa model akhir.",
            rujukan_bukti=["trial:t-003"],
            keyakinan="tinggi",
            alasan_keyakinan="uji brankas",
            risiko_asumsi="asumsi umum",
            structured_claim=claim,
        )
        res = verify_agent_artifact_position(art, test_ledger)
        if not res.is_valid and any("Mutasi Split" in v for v in res.violations):
            detected_mutations += 1

    # 3. 30 varian: Angka Digit Bebas dalam Narasi Agen (Aturan R16)
    for i in range(30):
        total_mutations += 1
        fraud_val = 0.85 + (i * 0.003)
        art = AgentArtifact(
            artifact_id=f"a-raw-digit-{i}",
            run_id="r-01",
            agent="experiment_engineer",
            seq=60 + i,
            jenis="temuan",
            klaim=f"Model mencapai akurasi sebesar {fraud_val:.4f} yang sangat memuaskan.",
            rujukan_bukti=["trial:t-003"],
            keyakinan="tinggi",
            alasan_keyakinan="evaluasi",
            risiko_asumsi="asumsi umum",
        )
        res = verify_agent_artifact_position(art, test_ledger)
        if not res.is_valid and any("Aturan R16" in v and "angka digit" in v for v in res.violations):
            detected_mutations += 1

    # 4. 20 varian: Kata-Angka Verbal dalam Narasi Agen (Aturan R16)
    number_words = ["delapan puluh persen", "sembilan puluh", "delapan belas", "nol koma delapan puluh tujuh", "eighty percent"]
    for i in range(20):
        total_mutations += 1
        w = number_words[i % len(number_words)]
        art = AgentArtifact(
            artifact_id=f"a-word-num-{i}",
            run_id="r-01",
            agent="skeptic",
            seq=90 + i,
            jenis="kritik",
            klaim=f"Model kami memiliki skor sekitar {w} pada data validasi.",
            rujukan_bukti=["trial:t-003"],
            keyakinan="sedang",
            alasan_keyakinan="ulasan",
            risiko_asumsi="asumsi verbal",
        )
        res = verify_agent_artifact_position(art, test_ledger)
        if not res.is_valid and any("kata-angka verbal" in v for v in res.violations):
            detected_mutations += 1

    p, lower, upper = wilson_score_interval(detected_mutations, total_mutations)
    print(f"\n[Position Mutation Suite] Detected: {detected_mutations}/{total_mutations} ({p:.1%}) [95% CI: {lower:.1%} - {upper:.1%}]")

    assert detected_mutations >= int(0.99 * total_mutations), (
        f"Deteksi mutasi di bawah 99%: {detected_mutations}/{total_mutations}"
    )

    # 5. Uji Kontrol: 20 Klaim Sah Bebas Angka Tidak Boleh Ada Alarm Palsu
    false_alarms = 0
    for i in range(20):
        claim = StructuredClaim(
            subject="HistGradientBoostingClassifier",
            metric="roc_auc",
            split="sealed_test",
            trial_key="t-004",  # benar-benar sealed test HistGB
            template="Model {{subjek}} pada uji tersegel memperoleh {{nilai}}.",
        )
        honest_art = AgentArtifact(
            artifact_id=f"a-honest-{i}",
            run_id="r-01",
            agent="experiment_engineer",
            seq=120 + i,
            jenis="hasil_eksperimen",
            klaim="Hasil model final pada evaluasi uji tersegel independen.",
            rujukan_bukti=["trial:t-004"],
            keyakinan="tinggi",
            alasan_keyakinan="evaluasi brankas",
            risiko_asumsi="tidak ada",
            structured_claim=claim,
        )
        res = verify_agent_artifact_position(honest_art, test_ledger)
        if not res.is_valid:
            false_alarms += 1

    assert false_alarms == 0, f"Alarm palsu terdeteksi pada klaim sah: {false_alarms}/20"
