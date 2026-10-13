# shadow_eval/matcher.py
"""
Pencocok Temuan Otomatis & Penilaian Bukti Deterministik (PRD 11.4).
Mencocokkan temuan agen dengan katalog defect (defects.yaml).
- Cocok penuh: jenis sama DAN kolom beririsan (atau katalog tanpa kolom).
- Cocok sebagian: jenis sama ATAU kolom sama.
- Tidak cocok: selain itu.
Verifikasi bukti: temuan valid HANYA jika rujukan_bukti menunjuk entri yang ada dan mendukung.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from shadow_agents.contracts import AgentArtifact
from shadow_core.ledger import Ledger


@dataclass
class CatalogDefect:
    defect_id: str
    jenis: str  # misal L1, L2, L5, L6, L8
    kolom: List[str]
    lokasi: str
    deskripsi: str
    aturan_cocok: str = "jenis sama dan kolom beririsan"


@dataclass
class MatchResult:
    artifact_id: str
    matched_defect_id: Optional[str]
    match_type: str  # 'cocok_penuh' | 'cocok_sebagian' | 'tidak_cocok'
    is_valid_evidence: bool
    is_duplicate: bool
    is_false_positive: bool
    is_outside_catalog: bool
    description: str


@dataclass
class EvaluationSummary:
    true_positives: int
    false_positives: int
    duplicates: int
    outside_catalog: int
    total_catalog_defects: int
    recall: float
    precision: float
    matches: List[MatchResult]


def verify_evidence_reference(
    artifact: AgentArtifact,
    ledger_entries: Dict[str, Dict[str, Any]],
    detector_outputs: Dict[str, Any],
) -> bool:
    """Memverifikasi bahwa rujukan_bukti benar-benar ada dan mendukung klaim."""
    if not artifact.rujukan_bukti:
        return False

    for ref in artifact.rujukan_bukti:
        # Format ref: 'trial:t-001' atau 'detector:L1'
        if ref.startswith("trial:"):
            t_id = ref.split(":", 1)[1]
            if t_id in ledger_entries:
                return True
        elif ref.startswith("detector:"):
            d_name = ref.split(":", 1)[1]
            if d_name in detector_outputs:
                return True
        elif ref in ledger_entries or ref in detector_outputs:
            return True

    return False


def match_findings_to_catalog(
    findings: List[AgentArtifact],
    catalog: List[CatalogDefect],
    ledger: Optional[Ledger] = None,
    detector_outputs: Optional[Dict[str, Any]] = None,
) -> EvaluationSummary:
    """Mencocokkan temuan agen dengan katalog defect secara deterministik."""
    ledger_entries = {e["trial_id"]: e for e in (ledger.read() if ledger else [])}
    det_outs = detector_outputs or {}

    matched_defect_ids: Set[str] = set()
    matches: List[MatchResult] = []

    tp = 0
    fp = 0
    dup_count = 0
    outside_cat_count = 0

    for art in findings:
        has_valid_evidence = verify_evidence_reference(art, ledger_entries, det_outs)

        # Cari kecocokan di katalog
        best_match: Optional[CatalogDefect] = None
        match_type = "tidak_cocok"

        for cat in catalog:
            same_type = bool(cat.jenis.lower() in art.klaim.lower() or cat.jenis.lower() in art.rujukan_bukti)
            # Kolom beririsan
            cols_cat = set(c.lower() for c in cat.kolom)
            cols_art = set(w.lower() for w in art.klaim.replace(",", " ").split() if w.lower() in cols_cat)
            has_col_overlap = len(cols_art) > 0 if cols_cat else True

            if same_type and has_col_overlap:
                best_match = cat
                match_type = "cocok_penuh"
                break
            elif (same_type or has_col_overlap) and match_type != "cocok_penuh":
                best_match = cat
                match_type = "cocok_sebagian"

        if not has_valid_evidence:
            # Temuan tanpa rujukan bukti yang valid dihitung sebagai FP (PRD 11.4.3)
            fp += 1
            matches.append(MatchResult(
                artifact_id=art.artifact_id,
                matched_defect_id=best_match.defect_id if best_match else None,
                match_type=match_type,
                is_valid_evidence=False,
                is_duplicate=False,
                is_false_positive=True,
                is_outside_catalog=False,
                description=f"Temuan '{art.artifact_id}' ditolak: Rujukan bukti tidak valid.",
            ))
            continue

        if match_type == "cocok_penuh" and best_match:
            if best_match.defect_id in matched_defect_ids:
                dup_count += 1
                matches.append(MatchResult(
                    artifact_id=art.artifact_id,
                    matched_defect_id=best_match.defect_id,
                    match_type=match_type,
                    is_valid_evidence=True,
                    is_duplicate=True,
                    is_false_positive=False,
                    is_outside_catalog=False,
                    description=f"Duplikat untuk defect '{best_match.defect_id}'.",
                ))
            else:
                tp += 1
                matched_defect_ids.add(best_match.defect_id)
                matches.append(MatchResult(
                    artifact_id=art.artifact_id,
                    matched_defect_id=best_match.defect_id,
                    match_type=match_type,
                    is_valid_evidence=True,
                    is_duplicate=False,
                    is_false_positive=False,
                    is_outside_catalog=False,
                    description=f"Cocok penuh dengan defect '{best_match.defect_id}'.",
                ))
        elif match_type == "tidak_cocok" and has_valid_evidence:
            outside_cat_count += 1
            matches.append(MatchResult(
                artifact_id=art.artifact_id,
                matched_defect_id=None,
                match_type=match_type,
                is_valid_evidence=True,
                is_duplicate=False,
                is_false_positive=False,
                is_outside_catalog=True,
                description="Temuan sah dengan bukti valid di luar katalog defect.",
            ))
        else:
            fp += 1
            matches.append(MatchResult(
                artifact_id=art.artifact_id,
                matched_defect_id=best_match.defect_id if best_match else None,
                match_type=match_type,
                is_valid_evidence=True,
                is_duplicate=False,
                is_false_positive=True,
                is_outside_catalog=False,
                description="Hanya cocok sebagian atau bukti tidak mendukung defect katalog.",
            ))

    total_cat = len(catalog)
    recall = tp / total_cat if total_cat > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    return EvaluationSummary(
        true_positives=tp,
        false_positives=fp,
        duplicates=dup_count,
        outside_catalog=outside_cat_count,
        total_catalog_defects=total_cat,
        recall=round(recall, 4),
        precision=round(precision, 4),
        matches=matches,
    )
