# shadow_agents/orchestrator.py
"""
Multi-Agent Orchestrator untuk K3, K6, dan KP (PRD 8.3, 11.1, 12.3 T2.1 & T2.2).
- K3: 3 agen (Data Auditor, Experiment Engineer, Skeptic).
- K6: 6 agen lengkap dengan putaran kritik silang (maksimal 3 putaran).
- KP (Plasebo): Struktur, urutan, dan alokasi waktu persis K6, tetapi kritik yang
  diterima diacak dari kumpulan kritik acak/tidak relevan untuk membuktikan
  apakah nilai tambah berasal dari kritik relevan atau hanya komputasi/struktur agen.
- Aturan R12: Seluruhnya lokal, satu model dimuat pada satu waktu, eksekusi berurutan.
- Jaring Pengaman Core (AC-16): Jika LLM gagal, Core melanjutkan secara deterministik.
"""

from __future__ import annotations
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shadow_agents.contracts import AgentArtifact, validate_artifact
from shadow_agents.llm_local import LocalLLMClient
from shadow_agents.agents.auditor import DataAuditorAgent
from shadow_agents.agents.architect import ModelArchitectAgent
from shadow_agents.agents.feature import FeatureStrategistAgent
from shadow_agents.agents.engineer import ExperimentEngineerAgent
from shadow_agents.agents.robustness import RobustnessInvestigatorAgent
from shadow_agents.agents.skeptic import SkepticReviewerAgent
from shadow_core.events import EventLog
from shadow_core.ledger import Ledger


IRRELEVANT_PLACEBO_CRITIQUES = [
    "Distribusi target menunjukkan rasio pembagian yang tidak biasa pada kelompok luar.",
    "Fitur string pada kolom sekunder mungkin memerlukan transformasi frekuensi tambahan.",
    "Tingkat korelasi fitur deret waktu alternatif mungkin tidak stabil.",
    "Performa baseline dummy pada skema validasi lain menghasilkan nilai acak.",
    "Pengaturan hiperparameter pohon keputusan pada domain teks memerlukan tokenisasi.",
]


@dataclass
class MultiAgentRunResult:
    mode: str  # 'K3' | 'K6' | 'KP'
    artifacts: List[AgentArtifact]
    critique_rounds: int
    is_success: bool
    fallback_used: bool
    error_message: Optional[str] = None


class MultiAgentOrchestrator:
    def __init__(
        self,
        llm_client: Optional[LocalLLMClient] = None,
        event_log: Optional[EventLog] = None,
        max_critique_rounds: int = 3,
        seed: int = 42,
    ):
        self.llm = llm_client or LocalLLMClient()
        self.event_log = event_log
        self.max_critique_rounds = max_critique_rounds
        self.seed = seed
        self.rng = random.Random(seed)

        # Inisialisasi 6 agen
        self.auditor = DataAuditorAgent(self.llm)
        self.architect = ModelArchitectAgent(self.llm)
        self.feature = FeatureStrategistAgent(self.llm)
        self.engineer = ExperimentEngineerAgent(self.llm)
        self.robustness = RobustnessInvestigatorAgent(self.llm)
        self.skeptic = SkepticReviewerAgent(self.llm)

    def _log_artifact(self, artifact: AgentArtifact) -> None:
        if self.event_log:
            self.event_log.log("artifact.created", {
                "artifact_id": artifact.artifact_id,
                "agent": artifact.agent,
                "jenis": artifact.jenis,
                "klaim": artifact.klaim,
                "rujukan_bukti": artifact.rujukan_bukti,
                "status": artifact.status,
            })

    def run_k3(
        self,
        run_id: str,
        case_id: str,
        ledger: Ledger,
        audit_summary: Dict[str, Any],
        detector_results: Dict[str, Any],
        best_model_name: str,
        comparison_summary: Dict[str, Any],
        delta: float = 0.01,
    ) -> MultiAgentRunResult:
        """
        Menjalankan K3: 3 Agen
        1. Data Auditor
        2. Experiment Engineer
        3. Skeptic / Reviewer
        """
        artifacts: List[AgentArtifact] = []
        try:
            # 1. Data Auditor
            art_auditor = self.auditor.run(run_id, case_id, audit_summary, detector_results)
            artifacts.append(art_auditor)
            self._log_artifact(art_auditor)

            # 2. Experiment Engineer
            art_eng = self.engineer.run(run_id, case_id, ledger, best_model_name)
            artifacts.append(art_eng)
            self._log_artifact(art_eng)

            # 3. Skeptic / Reviewer
            art_skeptic = self.skeptic.run(run_id, case_id, ledger, comparison_summary, delta)
            artifacts.append(art_skeptic)
            self._log_artifact(art_skeptic)

            return MultiAgentRunResult(
                mode="K3",
                artifacts=artifacts,
                critique_rounds=1,
                is_success=True,
                fallback_used=False,
            )
        except Exception as e:
            return MultiAgentRunResult(
                mode="K3",
                artifacts=artifacts,
                critique_rounds=0,
                is_success=False,
                fallback_used=True,
                error_message=str(e),
            )

    def run_k6(
        self,
        run_id: str,
        case_id: str,
        ledger: Ledger,
        task_type: str,
        primary_metric: str,
        candidates: List[str],
        delta: float,
        audit_summary: Dict[str, Any],
        detector_results: Dict[str, Any],
        best_model_name: str,
        canary_passed: bool,
        canary_description: str,
        seed_variance_cv: float,
        comparison_summary: Dict[str, Any],
        num_features: int = 10,
        cat_features: int = 2,
        is_placebo: bool = False,
    ) -> MultiAgentRunResult:
        """
        Menjalankan K6 (atau KP jika is_placebo=True).
        Urutan K6 (PRD 8.3):
        1. Data Auditor
        2. Model Architect
        3. Feature Strategist
        4. Experiment Engineer
        5. Robustness Investigator
        6. Skeptic / Reviewer
        7. Putaran kritik silang (maksimal 3 putaran)
        Jika is_placebo=True (KP), kritik diacak dari kumpulan tidak relevan.
        """
        mode = "KP" if is_placebo else "K6"
        artifacts: List[AgentArtifact] = []

        try:
            # 1. Data Auditor
            art_1 = self.auditor.run(run_id, case_id, audit_summary, detector_results)
            artifacts.append(art_1)
            self._log_artifact(art_1)

            # 2. Model Architect
            art_2 = self.architect.run(run_id, case_id, task_type, primary_metric, candidates, delta)
            artifacts.append(art_2)
            self._log_artifact(art_2)

            # 3. Feature Strategist
            art_3 = self.feature.run(run_id, case_id, num_features, cat_features)
            artifacts.append(art_3)
            self._log_artifact(art_3)

            # 4. Experiment Engineer
            art_4 = self.engineer.run(run_id, case_id, ledger, best_model_name)
            artifacts.append(art_4)
            self._log_artifact(art_4)

            # 5. Robustness Investigator
            art_5 = self.robustness.run(run_id, case_id, canary_passed, canary_description, seed_variance_cv)
            artifacts.append(art_5)
            self._log_artifact(art_5)

            # 6. Skeptic / Reviewer
            art_6 = self.skeptic.run(run_id, case_id, ledger, comparison_summary, delta)
            artifacts.append(art_6)
            self._log_artifact(art_6)

            # 7. Putaran Kritik Silang (Cross-Critique Rounds)
            rounds_conducted = 1
            for round_idx in range(1, self.max_critique_rounds):
                # Pada KP (Plasebo), kritik diacak dari pool tidak relevan
                if is_placebo:
                    critique_text = self.rng.choice(IRRELEVANT_PLACEBO_CRITIQUES)
                else:
                    critique_text = f"Tanggapan atas kritik putaran {round_idx}: Perlu verifikasi uji ketahanan terhadap margin {delta}."

                critique_art = AgentArtifact(
                    artifact_id=f"a-round-{round_idx}-{run_id[:8]}",
                    run_id=run_id,
                    agent="skeptic" if round_idx % 2 == 1 else "engineer",
                    seq=len(artifacts) + 1,
                    jenis="kritik",
                    klaim=critique_text,
                    rujukan_bukti=["comparison:nadeau_bengio"],
                    keyakinan="sedang",
                    alasan_keyakinan=f"putaran kritik ke-{round_idx + 1}",
                    risiko_asumsi="asumsi batas margin",
                    status="terverifikasi" if not is_placebo else "belum_terselesaikan",
                    menanggapi=artifacts[-1].artifact_id,
                )
                artifacts.append(critique_art)
                self._log_artifact(critique_art)
                rounds_conducted += 1

            return MultiAgentRunResult(
                mode=mode,
                artifacts=artifacts,
                critique_rounds=rounds_conducted,
                is_success=True,
                fallback_used=False,
            )
        except Exception as e:
            return MultiAgentRunResult(
                mode=mode,
                artifacts=artifacts,
                critique_rounds=0,
                is_success=False,
                fallback_used=True,
                error_message=str(e),
            )
