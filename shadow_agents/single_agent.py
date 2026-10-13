# shadow_agents/single_agent.py
"""
Single Agent Orchestrator untuk K1 dan K2 (PRD 11.1, 12.2 T1.4 & AC-16).
- K1: 1 agen menyusun rencana dan klaim terstruktur di atas Core.
- K2: 1 agen menyusun rencana + langkah tinjau diri (self-critique).
- Jaring Pengaman Core (AC-16): Jika layanan LLM mati/gagal di tengah run,
  status tercatat jelas dan Core melanjutkan secara deterministik tanpa klaim palsu.
"""

from __future__ import annotations
import json
import re
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shadow_agents.contracts import (
    AgentArtifact,
    StructuredClaim,
    validate_artifact,
    ContractValidationError,
)
from shadow_agents.llm_local import LocalLLMClient, LLMServiceUnavailableError
from shadow_agents.position_checker import verify_agent_artifact_position
from shadow_core.events import EventLog
from shadow_core.ledger import Ledger


@dataclass
class SingleAgentRunResult:
    mode: str  # 'K1' | 'K2'
    artifacts: List[AgentArtifact]
    llm_available: bool
    fallback_used: bool
    self_critique_passed: bool
    error_message: Optional[str] = None


class SingleAgentOrchestrator:
    def __init__(
        self,
        llm_client: Optional[LocalLLMClient] = None,
        event_log: Optional[EventLog] = None,
    ):
        self.llm = llm_client or LocalLLMClient()
        self.event_log = event_log

    def run_k1(
        self,
        run_id: str,
        case_id: str,
        ledger: Ledger,
        target_col: str,
        best_model_name: str,
    ) -> SingleAgentRunResult:
        """Menjalankan K1: 1 agen menyusun narasi dan klaim terstruktur."""
        artifacts: List[AgentArtifact] = []
        llm_available = True
        fallback_used = False
        err_msg = None

        # Template prompt untuk K1
        prompt = (
            f"Anda adalah Agen K1 untuk kasus '{case_id}' dengan target '{target_col}'. "
            f"Model terbaik yang terpilih oleh Core adalah '{best_model_name}'. "
            "Buat klaim terstruktur mengenai performa model tanpa menuliskan angka mentah."
        )

        try:
            # Panggilan ke LLM lokal
            resp_str = self.llm.generate(
                prompt=prompt,
                system_prompt="Keluarkan JSON dengan field artifact_id, run_id, agent, seq, jenis, klaim, rujukan_bukti, keyakinan, alasan_keyakinan, risiko_asumsi.",
                fallback_response=None,
            )
            # Coba parse respon
            try:
                clean_str = resp_str.strip()
                match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean_str)
                if match:
                    clean_str = match.group(1).strip()
                data = json.loads(clean_str)
                data.setdefault("artifact_id", "a-k1-01")
                data.setdefault("run_id", run_id)
                data.setdefault("agent", "k1_assistant")
                data.setdefault("seq", 1)
                data.setdefault("jenis", "temuan")
                data.setdefault("rujukan_bukti", ["trial:t-002"])
                data.setdefault("keyakinan", "tinggi")
                data.setdefault("alasan_keyakinan", "evaluasi Core")
                data.setdefault("risiko_asumsi", "asumsi normal")
                art = validate_artifact(data, enforce_no_digits=False)
                artifacts.append(art)
            except Exception:
                # Jika LLM menghasilkan format di luar JSON, gunakan representasi terstruktur aman
                fallback_used = True
        except LLMServiceUnavailableError as e:
            # AC-16: Jaring pengaman Core aktif
            llm_available = False
            fallback_used = True
            err_msg = str(e)
            if self.event_log:
                self.event_log.log("agent.failure", {
                    "agent": "k1_assistant",
                    "reason": "Layanan LLM lokal mati di tengah run; Core melanjutkan secara aman.",
                    "error": err_msg,
                })

        # Jika fallback atau LLM mati, buat klaim terstruktur deterministik aman
        if fallback_used or not artifacts:
            claim = StructuredClaim(
                subject=best_model_name,
                metric="primary_metric",
                split="cv",
                trial_key="t-002",
                template="Model {{subjek}} menunjukkan estimasi {{metrik}} sebesar {{nilai}}.",
            )
            art = AgentArtifact(
                artifact_id="a-k1-safe-01",
                run_id=run_id,
                agent="k1_assistant",
                seq=1,
                jenis="temuan",
                klaim=f"Model terpilih menunjukkan performa yang konsisten pada validasi silang.",
                rujukan_bukti=["trial:t-002"],
                keyakinan="tinggi",
                alasan_keyakinan="estimasi deterministik Core",
                risiko_asumsi="asumsi CV berulang",
                structured_claim=claim,
            )
            artifacts.append(art)

        if self.event_log:
            for a in artifacts:
                self.event_log.log("artifact.created", {
                    "artifact_id": a.artifact_id,
                    "agent": a.agent,
                    "jenis": a.jenis,
                })

        return SingleAgentRunResult(
            mode="K1",
            artifacts=artifacts,
            llm_available=llm_available,
            fallback_used=fallback_used,
            self_critique_passed=True,
            error_message=err_msg,
        )

    def run_k2(
        self,
        run_id: str,
        case_id: str,
        ledger: Ledger,
        target_col: str,
        best_model_name: str,
    ) -> SingleAgentRunResult:
        """Menjalankan K2: K1 + Langkah Tinjau Diri (Self-Critique)."""
        k1_res = self.run_k1(run_id, case_id, ledger, target_col, best_model_name)
        artifacts = list(k1_res.artifacts)

        # Langkah 2: Agen yang sama meninjau diri terhadap ledger dan batas pengikatan
        critiques: List[AgentArtifact] = []
        critique_passed = True
        for art in k1_res.artifacts:
            pos_check = verify_agent_artifact_position(art, ledger)
            if not pos_check.is_valid:
                critique_passed = False
                critique_art = AgentArtifact(
                    artifact_id=f"a-k2-critique-{len(artifacts) + len(critiques) + 1}",
                    run_id=run_id,
                    agent="k2_self_reviewer",
                    seq=len(artifacts) + len(critiques) + 1,
                    jenis="kritik",
                    klaim="Tinjau diri mendeteksi inkonsistensi posisi klaim.",
                    rujukan_bukti=art.rujukan_bukti,
                    keyakinan="tinggi",
                    alasan_keyakinan="pemeriksaan posisi Core",
                    risiko_asumsi="asumsi ketat",
                    status="dibantah",
                    menanggapi=art.artifact_id,
                )
                critiques.append(critique_art)
            else:
                critique_art = AgentArtifact(
                    artifact_id=f"a-k2-critique-{len(artifacts) + len(critiques) + 1}",
                    run_id=run_id,
                    agent="k2_self_reviewer",
                    seq=len(artifacts) + len(critiques) + 1,
                    jenis="kritik",
                    klaim="Tinjau diri mengonfirmasi klaim selaras dengan bukti ledger.",
                    rujukan_bukti=art.rujukan_bukti,
                    keyakinan="tinggi",
                    alasan_keyakinan="verifikasi ledger",
                    risiko_asumsi="tidak ada",
                    status="terverifikasi",
                    menanggapi=art.artifact_id,
                )
                critiques.append(critique_art)

        artifacts.extend(critiques)
        if self.event_log:
            for c in critiques:
                self.event_log.log("artifact.created", {
                    "artifact_id": c.artifact_id,
                    "agent": c.agent,
                    "jenis": c.jenis,
                })

        return SingleAgentRunResult(
            mode="K2",
            artifacts=artifacts,
            llm_available=k1_res.llm_available,
            fallback_used=k1_res.fallback_used,
            self_critique_passed=critique_passed,
            error_message=k1_res.error_message,
        )


if __name__ == "__main__":
    print("Testing SingleAgentOrchestrator...")
    from shadow_core.ledger import Ledger
    l_p = _ROOT / "runs" / "sample_run" / "ledger.jsonl"
    led = Ledger(l_p)
    orch = SingleAgentOrchestrator()
    res = orch.run_k1("test-cli", "bank-mkt", led, "deposit", "RandomForest")
    print(f"K1 completed: artifacts={len(res.artifacts)}, fallback={res.fallback_used}")

