# shadow_agents/position_checker.py
"""
Pemeriksa Posisi & Validasi Klaim Terstruktur (PRD 6.12, Lampiran D #4 & T1.5).
Aturan v4.1 [K]: Agen tidak menulis angka bebas. Core merender angka dari klaim terstruktur
dan memverifikasi kecocokan subjek, metrik, split, dan entri ledger.
Mencegah kasus 'angka sah tetapi tertukar posisi/subjek/metrik/split' (AC-11 & T1.5).
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from shadow_core.ledger import Ledger
from shadow_agents.contracts import AgentArtifact, StructuredClaim, DIGIT_REGEX

# Kata-angka dalam bahasa Indonesia & Inggris yang dilarang ditulis bebas oleh agen
NUMBER_WORDS_REGEX = re.compile(
    r"\b(nol|satu|dua|tiga|empat|lima|enam|tujuh|delapan|sembilan|sepuluh|"
    r"sebelas|belasan|puluh|ratus|ribu|persen|point|"
    r"zero|one|two|three|four|five|six|seven|eight|nine|ten|percent)\b",
    re.IGNORECASE
)


@dataclass
class PositionCheckResult:
    is_valid: bool
    violations: List[str]
    rendered_text: Optional[str] = None


def check_structured_claim_alignment(
    claim: StructuredClaim,
    ledger: Ledger,
) -> Tuple[bool, List[str], Optional[float]]:
    """Memeriksa apakah subjek, metrik, dan split cocok 100% dengan entri ledger yang ditunjuk."""
    violations: List[str] = []
    entries = {e["trial_id"]: e for e in ledger.read()}

    if claim.trial_key not in entries:
        violations.append(f"Trial key '{claim.trial_key}' tidak ditemukan di ledger!")
        return False, violations, None

    entry = entries[claim.trial_key]

    # 1. Verifikasi Subjek (Model)
    expected_model = entry.get("model", "")
    if claim.subject.lower() not in expected_model.lower() and expected_model.lower() not in claim.subject.lower():
        violations.append(
            f"Mutasi Subjek Terdeteksi: Klaim menyebut '{claim.subject}', "
            f"tetapi trial '{claim.trial_key}' adalah model '{expected_model}'!"
        )

    # 2. Verifikasi Split
    is_sealed = bool(entry.get("akses_uji_tersegel", False))
    is_claiming_sealed = any(s in claim.split.lower() for s in ("test", "sealed", "uji_tersegel", "holdout"))
    if is_claiming_sealed and not is_sealed:
        violations.append(
            f"Mutasi Split Terdeteksi: Klaim menyebut split '{claim.split}', "
            f"tetapi trial '{claim.trial_key}' adalah evaluasi CV (bukan uji tersegel)!"
        )
    elif not is_claiming_sealed and is_sealed:
        violations.append(
            f"Mutasi Split Terdeteksi: Klaim menyebut split '{claim.split}', "
            f"tetapi trial '{claim.trial_key}' adalah uji tersegel!"
        )


    # 3. Ambil Nilai Metrik dari Ledger
    metric_val = None
    summary = entry.get("ringkasan", {})
    if "mean" in summary:
        metric_val = float(summary["mean"])
    elif "skor_per_lipatan" in entry and entry["skor_per_lipatan"]:
        metric_val = float(entry["skor_per_lipatan"][0])

    if metric_val is None:
        violations.append(f"Trial '{claim.trial_key}' tidak memiliki nilai metrik ringkasan yang sah!")

    is_valid = len(violations) == 0
    return is_valid, violations, metric_val


def verify_agent_artifact_position(
    artifact: AgentArtifact,
    ledger: Ledger,
) -> PositionCheckResult:
    """
    Pemeriksaan posisi komprehensif terhadap artefak agen.
    Memastikan tidak ada angka bebas di narasi dan semua klaim terstruktur cocok dengan ledger.
    """
    violations: List[str] = []

    # 1. Cek angka mentah di narasi agen (Aturan R16)
    text_clean = re.sub(r"\{\{.*?\}\}", "", artifact.klaim)
    if DIGIT_REGEX.search(text_clean):
        violations.append(f"Aturan R16: Teks agen memuat angka digit bebas: '{artifact.klaim}'")

    if NUMBER_WORDS_REGEX.search(text_clean):
        violations.append(f"Aturan R16: Teks agen memuat kata-angka verbal: '{artifact.klaim}'")

    rendered_text = None

    # 2. Jika ada klaim terstruktur, verifikasi keselarasan posisi dan render nilai Core
    if artifact.structured_claim:
        claim_ok, claim_viols, metric_val = check_structured_claim_alignment(
            artifact.structured_claim,
            ledger=ledger,
        )
        violations.extend(claim_viols)

        if claim_ok and metric_val is not None:
            # Core merender template secara deterministik
            rendered_text = artifact.structured_claim.template.replace(
                "{{nilai}}", f"{metric_val:.4f}"
            ).replace(
                "{{subjek}}", artifact.structured_claim.subject
            ).replace(
                "{{metrik}}", artifact.structured_claim.metric
            )

    return PositionCheckResult(
        is_valid=len(violations) == 0,
        violations=violations,
        rendered_text=rendered_text,
    )
