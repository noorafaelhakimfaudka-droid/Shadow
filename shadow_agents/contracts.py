# shadow_agents/contracts.py
"""
Kontrak Artefak & Skema JSON Agen (PRD 8.4, 6.12, 12.2 T1.3 & Aturan R16).
Aturan R16: Agen tidak boleh menulis angka bebas; angka hanya dirender Core dari klaim terstruktur.
Rujukan bukti wajib untuk temuan, hasil_eksperimen, dan keputusan.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

JenisArtefak = Literal[
    "usulan",
    "temuan",
    "hasil_eksperimen",
    "kritik",
    "keputusan",
    "pertanyaan",
]

StatusArtefak = Literal[
    "diusulkan",
    "terverifikasi",
    "dibantah",
    "belum_terselesaikan",
]


class ContractValidationError(ValueError):
    """Artefak melanggar kontrak skema atau aturan bukti PRD."""
    pass


DIGIT_REGEX = re.compile(r"\b\d+(\.\d+)?\b")


@dataclass
class StructuredClaim:
    """Klaim terstruktur bebas angka (Aturan R16). Core yang mengisi angkanya."""
    subject: str  # misal 'Linear' atau 'HistGB'
    metric: str  # misal 'roc_auc'
    split: str  # misal 'cv' atau 'sealed_test'
    trial_key: str  # misal 't-002'
    template: str  # templat dengan penanda {{nilai}}


@dataclass
class AgentArtifact:
    artifact_id: str
    run_id: str
    agent: str
    seq: int
    jenis: JenisArtefak
    klaim: str
    rujukan_bukti: List[str]
    keyakinan: str
    alasan_keyakinan: str
    risiko_asumsi: str
    status: StatusArtefak = "diusulkan"
    menanggapi: Optional[str] = None
    structured_claim: Optional[StructuredClaim] = None
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_artifact(artifact_dict: Dict[str, Any], enforce_no_digits: bool = True) -> AgentArtifact:
    """Memvalidasi dictionary artefak terhadap kontrak PRD 8.4 dan Aturan R16."""
    required_fields = {
        "artifact_id", "run_id", "agent", "seq", "jenis",
        "klaim", "rujukan_bukti", "keyakinan", "alasan_keyakinan",
        "risiko_asumsi",
    }
    missing = required_fields - set(artifact_dict.keys())
    if missing:
        raise ContractValidationError(f"Skema artefak tidak lengkap, atribut kurang: {missing}")

    jenis = artifact_dict["jenis"]
    valid_jenis = {"usulan", "temuan", "hasil_eksperimen", "kritik", "keputusan", "pertanyaan"}
    if jenis not in valid_jenis:
        raise ContractValidationError(f"Jenis artefak '{jenis}' tidak valid! Harus salah satu dari {valid_jenis}")

    # Rujukan bukti WAJIB untuk temuan, hasil_eksperimen, dan keputusan
    rujukan = artifact_dict.get("rujukan_bukti", [])
    if jenis in {"temuan", "hasil_eksperimen", "keputusan"}:
        if not rujukan or len(rujukan) == 0:
            raise ContractValidationError(
                f"PRD 8.4: Artefak jenis '{jenis}' WAJIB memiliki 'rujukan_bukti' pendukung!"
            )

    # Aturan R16: Cek angka bebas di teks klaim
    klaim_text = artifact_dict.get("klaim", "")
    if enforce_no_digits:
        # Pengecualian: {{kunci}} placeholder tidak dianggap angka bebas
        text_without_placeholders = re.sub(r"\{\{.*?\}\}", "", klaim_text)
        if DIGIT_REGEX.search(text_without_placeholders):
            raise ContractValidationError(
                f"Aturan R16 Dilanggar: Agen dilarang menulis angka bebas ('{klaim_text}')! "
                "Gunakan penanda {{kunci}} atau klaim terstruktur."
            )

    sc_dict = artifact_dict.get("structured_claim")
    sc_obj = StructuredClaim(**sc_dict) if sc_dict else None

    return AgentArtifact(
        artifact_id=artifact_dict["artifact_id"],
        run_id=artifact_dict["run_id"],
        agent=artifact_dict["agent"],
        seq=int(artifact_dict["seq"]),
        jenis=jenis,
        klaim=klaim_text,
        rujukan_bukti=rujukan,
        keyakinan=artifact_dict["keyakinan"],
        alasan_keyakinan=artifact_dict["alasan_keyakinan"],
        risiko_asumsi=artifact_dict["risiko_asumsi"],
        status=artifact_dict.get("status", "diusulkan"),
        menanggapi=artifact_dict.get("menanggapi"),
        structured_claim=sc_obj,
        ts=artifact_dict.get("ts", datetime.now(timezone.utc).isoformat()),
    )
