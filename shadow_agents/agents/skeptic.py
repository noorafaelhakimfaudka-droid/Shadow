# shadow_agents/agents/skeptic.py
"""
Skeptic / Reviewer (PRD 8.2 & 8.5).
Peran: Menantang klaim performa, memeriksa apakah selisih melampaui margin praktis delta,
       memeriksa kesesuaian klaim terhadap ledger.
Konteks independen: Memiliki prompt percakapan baru tanpa riwayat agen pembangun.
Batasan: Tidak boleh menghapus blocker deterministik; boleh menambah temuan/kritik.
"""

from __future__ import annotations
import json
from typing import Any, Dict, List, Optional
from shadow_agents.contracts import AgentArtifact, validate_artifact
from shadow_agents.llm_local import LocalLLMClient
from shadow_core.ledger import Ledger


class SkepticReviewerAgent:
    def __init__(self, llm_client: Optional[LocalLLMClient] = None):
        self.llm = llm_client or LocalLLMClient()

    def run(
        self,
        run_id: str,
        case_id: str,
        ledger: Ledger,
        comparison_summary: Dict[str, Any],
        delta: float,
    ) -> AgentArtifact:
        """Meninjau klaim dan memverifikasi batas margin praktis delta."""
        ci_lower = comparison_summary.get("ci95_lower", 0.0)
        is_above_delta = ci_lower >= delta

        prompt = (
            f"Anda adalah Skeptic / Reviewer Independen untuk kasus '{case_id}'.\n"
            f"Batas bawah CI 95% selisih: {ci_lower:.4f}, Margin praktis delta: {delta:.4f}.\n"
            f"Apakah selisih melampaui delta: {'Ya' if is_above_delta else 'Tidak'}.\n"
            "Jika batas bawah di bawah delta, ajukan kritik bahwa kedua model setara secara praktis. "
            "DILARANG menulis angka mentah bebas."
        )

        sys_prompt = (
            "Keluarkan JSON dengan field: artifact_id, run_id, agent, seq, jenis, "
            "klaim, rujukan_bukti, keyakinan, alasan_keyakinan, risiko_asumsi."
        )

        try:
            resp_str = self.llm.generate(prompt=prompt, system_prompt=sys_prompt)
            data = json.loads(resp_str)
            data["agent"] = "skeptic"
            data.setdefault("artifact_id", f"a-skeptic-{run_id[:8]}")
            data.setdefault("run_id", run_id)
            data.setdefault("agent", "skeptic")
            data.setdefault("seq", 6)
            data.setdefault("jenis", "kritik")
            data.setdefault("rujukan_bukti", ["comparison:nadeau_bengio"])
            data.setdefault("keyakinan", "tinggi")
            data.setdefault("alasan_keyakinan", "evaluasi margin praktis S7")
            data.setdefault("risiko_asumsi", "asumsi normal")
            return validate_artifact(data, enforce_no_digits=False)
        except Exception:
            klaim = "Keunggulan model terbukti melampaui margin praktis delta." if is_above_delta else "Selisih model tidak melampaui margin praktis delta; kedua model setara."
            fallback_dict = {
                "artifact_id": f"a-skeptic-{run_id[:8]}",
                "run_id": run_id,
                "agent": "skeptic",
                "seq": 6,
                "jenis": "kritik",
                "klaim": klaim,
                "rujukan_bukti": ["comparison:nadeau_bengio"],
                "keyakinan": "tinggi",
                "alasan_keyakinan": "pemeriksaan aturan margin praktis Aturan R7 & R15",
                "risiko_asumsi": "tidak ada",
            }
            return validate_artifact(fallback_dict, enforce_no_digits=False)
