# shadow_agents/agents/feature.py
"""
Feature Strategist (PRD 8.2).
Peran: Spesifikasi fitur/pipeline dari kit komponen; ablasi grup fitur.
Artefak: Konfigurasi pipeline fit-di-dalam-lipatan.
Batasan: Dilarang menyentuh uji tersegel (Aturan R2, PRD 8.2).
"""

from __future__ import annotations
import json
from typing import Any, Dict, List, Optional
from shadow_agents.contracts import AgentArtifact, validate_artifact
from shadow_agents.llm_local import LocalLLMClient


class FeatureStrategistAgent:
    def __init__(self, llm_client: Optional[LocalLLMClient] = None):
        self.llm = llm_client or LocalLLMClient()

    def run(
        self,
        run_id: str,
        case_id: str,
        num_features: int,
        cat_features: int,
    ) -> AgentArtifact:
        """Mengusulkan strategi rekayasa fitur dari kit komponen tervalidasi."""
        prompt = (
            f"Anda adalah Feature Strategist untuk kasus '{case_id}'.\n"
            f"Fitur numerik: {num_features}, Fitur kategorikal: {cat_features}.\n"
            "Pilih konfigurasi pra-pemrosesan (scaling, encoding, imputasi) dari kit komponen Core. "
            "Ingat: seluruh transformasi wajib fit-di-dalam-lipatan."
        )

        sys_prompt = (
            "Keluarkan JSON dengan field: artifact_id, run_id, agent, seq, jenis, "
            "klaim, rujukan_bukti, keyakinan, alasan_keyakinan, risiko_asumsi."
        )

        try:
            resp_str = self.llm.generate(prompt=prompt, system_prompt=sys_prompt)
            data = json.loads(resp_str)
            data["agent"] = "feature_strategist"
            data.setdefault("artifact_id", f"a-feature-{run_id[:8]}")
            data.setdefault("run_id", run_id)
            data.setdefault("agent", "feature_strategist")
            data.setdefault("seq", 3)
            data.setdefault("jenis", "usulan")
            data.setdefault("rujukan_bukti", ["component_kit:ColumnTransformer"])
            data.setdefault("keyakinan", "tinggi")
            data.setdefault("alasan_keyakinan", "kit komponen tervalidasi S5")
            data.setdefault("risiko_asumsi", "asumsi normal")
            return validate_artifact(data, enforce_no_digits=False)
        except Exception:
            fallback_dict = {
                "artifact_id": f"a-feature-{run_id[:8]}",
                "run_id": run_id,
                "agent": "feature_strategist",
                "seq": 3,
                "jenis": "usulan",
                "klaim": "Menggunakan RobustScaler untuk numerik dan OneHotEncoder fit-di-dalam-lipatan.",
                "rujukan_bukti": ["component_kit:ColumnTransformer"],
                "keyakinan": "tinggi",
                "alasan_keyakinan": "konstruksi fit-di-dalam-lipatan",
                "risiko_asumsi": "tidak ada",
            }
            return validate_artifact(fallback_dict, enforce_no_digits=False)
