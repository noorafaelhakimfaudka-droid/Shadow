# shadow_agents/agents/architect.py
"""
Model Architect (PRD 8.2).
Peran: Menyusun rencana eksperimen: hipotesis, metrik, delta, kandidat, anggaran.
Artefak: Rencana eksperimen yang dikunci dengan hash SHA-256 sebelum eksekusi.
Batasan: Rencana terkunci tidak boleh diubah diam-diam; perubahan = persetujuan operator.
"""

from __future__ import annotations
import hashlib
import json
from typing import Any, Dict, List, Optional
from shadow_agents.contracts import AgentArtifact, validate_artifact
from shadow_agents.llm_local import LocalLLMClient


class ModelArchitectAgent:
    def __init__(self, llm_client: Optional[LocalLLMClient] = None):
        self.llm = llm_client or LocalLLMClient()

    def run(
        self,
        run_id: str,
        case_id: str,
        task_type: str,
        primary_metric: str,
        candidates: List[str],
        delta: float,
    ) -> AgentArtifact:
        """Menyusun rencana eksperimen terkunci dengan hash."""
        plan_content = {
            "case_id": case_id,
            "task_type": task_type,
            "primary_metric": primary_metric,
            "candidates": candidates,
            "delta": delta,
            "validation_strategy": "RepeatedKFold",
        }
        plan_hash = hashlib.sha256(json.dumps(plan_content, sort_keys=True).encode()).hexdigest()

        prompt = (
            f"Anda adalah Model Architect untuk kasus '{case_id}'.\n"
            f"Jenis tugas: {task_type}, Metrik utama: {primary_metric}, Kandidat: {candidates}.\n"
            f"Rencana eksperimen terkunci (hash: {plan_hash[:12]}).\n"
            "Rumuskan hipotesis eksperimen tanpa angka bebas."
        )

        sys_prompt = (
            "Keluarkan JSON dengan field: artifact_id, run_id, agent, seq, jenis, "
            "klaim, rujukan_bukti, keyakinan, alasan_keyakinan, risiko_asumsi."
        )

        try:
            resp_str = self.llm.generate(prompt=prompt, system_prompt=sys_prompt)
            data = json.loads(resp_str)
            data["agent"] = "architect"
            data.setdefault("artifact_id", f"a-architect-{run_id[:8]}")
            data.setdefault("run_id", run_id)
            data.setdefault("agent", "architect")
            data.setdefault("seq", 2)
            data.setdefault("jenis", "usulan")
            data.setdefault("rujukan_bukti", [f"plan_hash:{plan_hash[:16]}"])
            data.setdefault("keyakinan", "tinggi")
            data.setdefault("alasan_keyakinan", "analisis framing S1")
            data.setdefault("risiko_asumsi", "asumsi normal")
            return validate_artifact(data, enforce_no_digits=False)
        except Exception:
            fallback_dict = {
                "artifact_id": f"a-architect-{run_id[:8]}",
                "run_id": run_id,
                "agent": "architect",
                "seq": 2,
                "jenis": "usulan",
                "klaim": f"Rencana terkunci: membandingkan {len(candidates)} model kandidat terhadap baseline.",
                "rujukan_bukti": [f"plan_hash:{plan_hash[:16]}"],
                "keyakinan": "tinggi",
                "alasan_keyakinan": "framing aturan metrik S1",
                "risiko_asumsi": "tidak ada",
            }
            return validate_artifact(fallback_dict, enforce_no_digits=False)
