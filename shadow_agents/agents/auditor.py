# shadow_agents/agents/auditor.py
"""
Data Auditor (PRD 8.2).
Peran: Meninjau laporan audit S2 dan hasil detektor L1/L2/L8.
Artefak: Laporan audit, daftar kandidat kebocoran, rekomendasi split.
Batasan: Boleh mengajukan blocker; TIDAK boleh mengubah data asli (Aturan R9).
"""

from __future__ import annotations
import json
from typing import Any, Dict, List, Optional
from shadow_agents.contracts import AgentArtifact, validate_artifact
from shadow_agents.llm_local import LocalLLMClient


class DataAuditorAgent:
    def __init__(self, llm_client: Optional[LocalLLMClient] = None):
        self.llm = llm_client or LocalLLMClient()

    def run(
        self,
        run_id: str,
        case_id: str,
        audit_summary: Dict[str, Any],
        detector_results: Dict[str, Any],
    ) -> AgentArtifact:
        """Menghasilkan artefak audit kebocoran dan rekomendasi strategi split."""
        prompt = (
            f"Anda adalah Data Auditor untuk kasus '{case_id}' (run '{run_id}').\n"
            f"Ringkasan audit: {json.dumps(audit_summary)}\n"
            f"Hasil detektor kebocoran: {json.dumps(detector_results)}\n"
            "Evaluasi temuan audit dan detektor. "
            "DILARANG menulis angka/digit bebas di klaim; gunakan penanda {{kunci}} jika perlu. "
            "Keluarkan JSON artefak jenis 'temuan' dengan rujukan_bukti."
        )

        sys_prompt = (
            "Keluarkan JSON dengan field: artifact_id, run_id, agent, seq, jenis, "
            "klaim, rujukan_bukti, keyakinan, alasan_keyakinan, risiko_asumsi."
        )

        try:
            resp_str = self.llm.generate(prompt=prompt, system_prompt=sys_prompt)
            data = json.loads(resp_str)
            data["agent"] = "auditor"
            data.setdefault("artifact_id", f"a-auditor-{run_id[:8]}")
            data.setdefault("run_id", run_id)
            data.setdefault("agent", "auditor")
            data.setdefault("seq", 1)
            data.setdefault("jenis", "temuan")
            data.setdefault("rujukan_bukti", ["detector:S2_audit"])
            data.setdefault("keyakinan", "tinggi")
            data.setdefault("alasan_keyakinan", "audit deterministik Core")
            data.setdefault("risiko_asumsi", "asumsi normal")
            return validate_artifact(data, enforce_no_digits=False)
        except Exception:
            # Fallback deterministik aman (AC-16)
            has_leak = any(not v.get("passed", True) for v in detector_results.values() if isinstance(v, dict))
            klaim = "Detektor kebocoran menandai potensi anomali fitur." if has_leak else "Tidak ada indikasi kebocoran dari detektor data."
            fallback_dict = {
                "artifact_id": f"a-auditor-{run_id[:8]}",
                "run_id": run_id,
                "agent": "auditor",
                "seq": 1,
                "jenis": "temuan",
                "klaim": klaim,
                "rujukan_bukti": ["detector:S2_audit"],
                "keyakinan": "tinggi",
                "alasan_keyakinan": "evaluasi profil Core",
                "risiko_asumsi": "tidak ada",
            }
            return validate_artifact(fallback_dict, enforce_no_digits=False)
