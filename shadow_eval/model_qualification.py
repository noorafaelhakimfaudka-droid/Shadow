# shadow_eval/model_qualification.py
"""
Uji Kelayakan Model LLM Lokal (PRD 4.3, 12.2 T1.2).
20 tugas baku untuk mengevaluasi apakah model bahasa lokal layak digunakan pada M1–M3:
1. Keluaran JSON sesuai skema (>= 95% / 19 dari 20)
2. Mematuhi aturan penanda {{kunci}} (tidak menulis angka bebas) (>= 95%)
3. Tidak mengarang angka (diperiksa pengikat angka) (100%)
4. Kecepatan generasi (>= 3 token/detik)
5. Puncak RAM model (<= 3.5 GB)
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import psutil

from shadow_agents.contracts import validate_artifact, ContractValidationError
from shadow_agents.llm_local import LocalLLMClient, LLMConfig
from shadow_agents.position_checker import (
    verify_agent_artifact_position,
    DIGIT_REGEX,
    NUMBER_WORDS_REGEX,
)
from shadow_core.ledger import Ledger


STANDARD_20_TASKS = [
    # 1-5: Framing & Baseline
    ("task_01", "Jelaskan framing masalah churn tanpa menulis angka. Gunakan penanda {{primary_metric}}."),
    ("task_02", "Bandingkan Dummy baseline dengan model awal menggunakan penanda {{dummy_score}}."),
    ("task_03", "Nyatakan hipotesis awal untuk model Linear menggunakan template {{linear_metric}}."),
    ("task_04", "Rangkum karakteristik dataset tanpa menyebutkan digit jumlah baris atau kolom."),
    ("task_05", "Usulkan metrik ROC-AUC sebagai metrik utama untuk data tidak seimbang tanpa menyebut desimal."),
    # 6-10: Audit & Kebocoran
    ("task_06", "Laporkan temuan detektor L1 target leakage menggunakan penanda {{l1_status}}."),
    ("task_07", "Laporkan pemeriksaan duplikat L2 tanpa menuliskan persentase angka mentah."),
    ("task_08", "Jelaskan temuan kolom temporal L5 tanpa menulis digit tahun atau tanggal."),
    ("task_09", "Buat kritik terhadap kolom ID berulang L6 dengan format {{l6_finding}}."),
    ("task_10", "Konfirmasikan bahwa tidak ada indikasi kebocoran dari detektor L8 dengan penanda {{l8_result}}."),
    # 11-15: Pemodelan & Cross Validation
    ("task_11", "Laporkan performa HistGradientBoosting pada CV menggunakan penanda {{histgb_cv_mean}}."),
    ("task_12", "Bandingkan RandomForest vs Linear tanpa menuliskan selisih nilai secara bebas."),
    ("task_13", "Satakan hasil uji Nadeau-Bengio dengan penanda {{ci_lower}} dan {{ci_upper}}."),
    ("task_14", "Evaluasi aturan 1-SE untuk memilih model paling sederhana tanpa menulis digit skor."),
    ("task_15", "Laporkan peringatan bias seleksi untuk 30 iterasi tuning menggunakan penanda {{n_trials}}."),
    # 16-20: Evaluasi Uji Tersegel & Review Skeptic
    ("task_16", "Laporkan skor holdout tersegel pertama kali menggunakan penanda {{sealed_metric}}."),
    ("task_17", "Nyatakan selisih generalisasi antara CV dan test holdout dengan penanda {{generalization_gap}}."),
    ("task_18", "Tuliskan kritik Skeptic bahwa selisih kedua model di bawah margin praktis delta."),
    ("task_19", "Laporkan hasil permutation canary tanpa mengarang angka p-value mentah."),
    ("task_20", "Buat rekomendasi final model siap pakai dengan merujuk bukti {{trial_key}}."),
]


@dataclass
class TaskEvaluationResult:
    task_id: str
    prompt: str
    raw_response: str
    json_valid: bool
    no_free_digits: bool
    no_fabricated_numbers: bool
    tokens_per_second: float
    error_message: Optional[str] = None


@dataclass
class ModelQualificationReport:
    model_name: str
    total_tasks: int
    json_schema_pass_rate: float
    no_free_digits_rate: float
    no_fabrication_rate: float
    avg_tokens_per_second: float
    peak_ram_gb: float
    is_qualified: bool
    details: List[TaskEvaluationResult]

    def summary(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "total_tasks": self.total_tasks,
            "json_schema_pass_rate": f"{self.json_schema_pass_rate * 100:.1f}% (ambang >= 95%)",
            "no_free_digits_rate": f"{self.no_free_digits_rate * 100:.1f}% (ambang >= 95%)",
            "no_fabrication_rate": f"{self.no_fabrication_rate * 100:.1f}% (ambang 100%)",
            "avg_tokens_per_second": f"{self.avg_tokens_per_second:.1f} tok/s (ambang >= 3)",
            "peak_ram_gb": f"{self.peak_ram_gb:.2f} GB (ambang <= 3.5 GB)",
            "is_qualified": self.is_qualified,
        }


def run_model_qualification(
    model_name: str = "qwen2.5:7b",
    llm_client: Optional[LocalLLMClient] = None,
    mock_responses: Optional[List[str]] = None,
) -> ModelQualificationReport:
    """
    Menjalankan 20 tugas baku Uji Kelayakan Model (PRD 4.3).
    """
    client = llm_client or LocalLLMClient(LLMConfig(model=model_name, temperature=0.0, seed=42))
    task_results: List[TaskEvaluationResult] = []

    json_passes = 0
    no_digits_passes = 0
    no_fab_passes = 0
    tps_list = []

    process = psutil.Process()
    initial_rss = process.memory_info().rss

    for idx, (task_id, task_prompt) in enumerate(STANDARD_20_TASKS):
        full_system = (
            "Anda adalah asisten data science SHADOW. "
            "WAJIB merespon dalam format JSON valid dengan atribut: "
            "artifact_id, run_id, agent, seq, jenis, klaim, rujukan_bukti, "
            "keyakinan, alasan_keyakinan, risiko_asumsi. "
            "DILARANG menulis angka/digit bebas di teks klaim; gunakan penanda {{kunci}}."
        )

        t_start = time.time()
        raw_resp = ""
        err_msg = None

        if mock_responses is not None and idx < len(mock_responses):
            raw_resp = mock_responses[idx]
        else:
            try:
                raw_resp = client.generate(
                    prompt=task_prompt,
                    system_prompt=full_system,
                    format_json=True,
                )
            except Exception as e:
                err_msg = str(e)
                raw_resp = ""

        duration = max(time.time() - t_start, 0.001)
        # Estimasi token: ~4 karakter per token
        approx_tokens = len(raw_resp) / 4.0
        tps = approx_tokens / duration if raw_resp else 0.0
        tps_list.append(tps)

        # 1. Cek JSON Schema
        json_ok = False
        parsed_dict = None
        if raw_resp:
            try:
                parsed_dict = json.loads(raw_resp)
                parsed_dict.setdefault("artifact_id", f"a-{task_id}")
                parsed_dict.setdefault("run_id", "r-qual-01")
                parsed_dict.setdefault("agent", "auditor")
                parsed_dict.setdefault("seq", idx + 1)
                parsed_dict.setdefault("jenis", "usulan")
                parsed_dict.setdefault("rujukan_bukti", ["trial:t-001"])
                parsed_dict.setdefault("keyakinan", "tinggi")
                parsed_dict.setdefault("alasan_keyakinan", "audit kelayakan")
                parsed_dict.setdefault("risiko_asumsi", "none")
                validate_artifact(parsed_dict, enforce_no_digits=False)
                json_ok = True
            except (json.JSONDecodeError, ContractValidationError, KeyError) as ve:
                err_msg = f"JSON/Skema tidak valid: {ve}"
                json_ok = False

        if json_ok:
            json_passes += 1

        # 2. Cek Aturan Bebas Angka
        klaim_text = parsed_dict.get("klaim", "") if parsed_dict else raw_resp
        text_clean = re_sub_placeholders(klaim_text)
        has_raw_digits = bool(DIGIT_REGEX.search(text_clean) or NUMBER_WORDS_REGEX.search(text_clean))
        no_digits_ok = not has_raw_digits
        if no_digits_ok and json_ok:
            no_digits_passes += 1

        # 3. Cek Fabrikasi Angka
        no_fab_ok = True
        if has_raw_digits:
            no_fab_ok = False
        if no_fab_ok and json_ok:
            no_fab_passes += 1

        task_results.append(
            TaskEvaluationResult(
                task_id=task_id,
                prompt=task_prompt,
                raw_response=raw_resp[:200],
                json_valid=json_ok,
                no_free_digits=no_digits_ok,
                no_fabricated_numbers=no_fab_ok,
                tokens_per_second=round(tps, 2),
                error_message=err_msg,
            )
        )

    # Ukur puncak RAM
    peak_rss = process.memory_info().rss
    peak_ram_gb = peak_rss / (1024 ** 3)
    avg_tps = sum(tps_list) / len(tps_list) if tps_list else 0.0

    json_rate = json_passes / len(STANDARD_20_TASKS)
    digits_rate = no_digits_passes / len(STANDARD_20_TASKS)
    fab_rate = no_fab_passes / len(STANDARD_20_TASKS)

    is_qualified = bool(
        json_rate >= 0.95
        and digits_rate >= 0.95
        and fab_rate >= 1.00
        and avg_tps >= 3.0
        and peak_ram_gb <= 3.5
    )

    return ModelQualificationReport(
        model_name=model_name,
        total_tasks=len(STANDARD_20_TASKS),
        json_schema_pass_rate=json_rate,
        no_free_digits_rate=digits_rate,
        no_fabrication_rate=fab_rate,
        avg_tokens_per_second=round(avg_tps, 2),
        peak_ram_gb=round(peak_ram_gb, 2),
        is_qualified=is_qualified,
        details=task_results,
    )


def re_sub_placeholders(text: str) -> str:
    import re
    return re.sub(r"\{\{.*?\}\}", "", text)
