# shadow_agents/agents/robustness.py
"""
Robustness Investigator (PRD 8.2).
Peran: Menilai ketahanan model: kanari permutasi, variasi multi-seed, kalibrasi, stres data.
Artefak: Laporan investigasi ketahanan.
Batasan: Boleh mengajukan blocker (misal bila kanari permutasi gagal alarm).
"""

from __future__ import annotations
import json
from typing import Any, Dict, List, Optional
from shadow_agents.contracts import AgentArtifact, validate_artifact
from shadow_agents.llm_local import LocalLLMClient


class RobustnessInvestigatorAgent:
    def __init__(self, llm_client: Optional[LocalLLMClient] = None):
        self.llm = llm_client or LocalLLMClient()

    def run(
        self,
        run_id: str,
        case_id: str,
        canary_passed: bool,
        canary_description: str,
        seed_variance_cv: float,
    ) -> AgentArtifact:
        """Menghasilkan artefak evaluasi ketahanan model dan kanari permutasi."""
        prompt = (
            f"Anda adalah Robustness Investigator untuk kasus '{case_id}'.\n"
            f"Status kanari permutasi: {'Lolos' if canary_passed else 'GAGAL'}.\n"
            f"Keterangan kanari: {canary_description}.\n"
            "Buat temuan ketahanan model. Bila kanari gagal, tandai sebagai blocker."
        )

        sys_prompt = (
            "Keluarkan JSON dengan field: artifact_id, run_id, agent, seq, jenis, "
            "klaim, rujukan_bukti, keyakinan, alasan_keyakinan, risiko_asumsi."
        )

        try:
            resp_str = self.llm.generate(prompt=prompt, system_prompt=sys_prompt)
            data = json.loads(resp_str)
            data["agent"] = "robustness"
            data.setdefault("artifact_id", f"a-robustness-{run_id[:8]}")
            data.setdefault("run_id", run_id)
            data.setdefault("agent", "robustness")
            data.setdefault("seq", 5)
            data.setdefault("jenis", "temuan")
            data.setdefault("rujukan_bukti", ["canary:permutation_test"])
            data.setdefault("keyakinan", "tinggi")
            data.setdefault("alasan_keyakinan", "evaluasi kanari Core")
            data.setdefault("risiko_asumsi", "asumsi normal")
            return validate_artifact(data, enforce_no_digits=False)
        except Exception:
            klaim = "Kanari permutasi lolos, membuktikan prosedur bebas kebocoran label." if canary_passed else "Kanari permutasi GAGAL, kebocoran prosedural terdeteksi."
            fallback_dict = {
                "artifact_id": f"a-robustness-{run_id[:8]}",
                "run_id": run_id,
                "agent": "robustness",
                "seq": 5,
                "jenis": "temuan",
                "klaim": klaim,
                "rujukan_bukti": ["canary:permutation_test"],
                "keyakinan": "tinggi",
                "alasan_keyakinan": "evaluasi kanari permutasi S4",
                "risiko_asumsi": "tidak ada",
            }
            return validate_artifact(fallback_dict, enforce_no_digits=False)
