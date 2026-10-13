# shadow_eval/adjudication.py
"""
Protokol Evaluator & Adjudikasi Dua Penilai (PRD 11.4).
- Dua penilai (A & B) menilai temuan secara buta: 'valid' / 'duplikat' / 'salah' / 'tidak_jelas'.
- Bila A dan B berbeda: adjudikasi menggunakan aturan_cocok.
- Jika tetap tidak terselesaikan: dihitung 'tidak ditemukan' (konservatif).
- Menghitung persentase kesepakatan per kategori (target >= 85%).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ReviewItem:
    item_id: str
    category: str  # misal L1, L2, L5, L6, L8, metric
    finding_text: str
    evidence_ref: str
    label_a: str  # 'valid' | 'duplikat' | 'salah' | 'tidak_jelas'
    label_b: str


@dataclass
class AdjudicationResult:
    item_id: str
    final_label: str
    is_unresolved: bool
    is_considered_tp: bool


@dataclass
class AgreementReport:
    overall_agreement_pct: float
    category_agreements: Dict[str, float]
    total_items: int
    unresolved_count: int
    adjudications: List[AdjudicationResult]


def adjudicate_reviews(
    items: List[ReviewItem],
    unresolved_penalty_conservative: bool = True,
) -> AgreementReport:
    """Melakukan adjudikasi terhadap hasil tinjauan dua penilai."""
    if not items:
        return AgreementReport(1.0, {}, 0, 0, [])

    total = len(items)
    agreed_count = 0
    cat_counts: Dict[str, int] = {}
    cat_agreed: Dict[str, int] = {}
    adjudications: List[AdjudicationResult] = []
    unresolved_count = 0

    for item in items:
        cat = item.category
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

        if item.label_a == item.label_b:
            agreed_count += 1
            cat_agreed[cat] = cat_agreed.get(cat, 0) + 1
            final_label = item.label_a
            is_unres = False
        else:
            # Perbedaan pendapat: jika salah satu melabel 'tidak_jelas', anggap unresolved
            if "tidak_jelas" in (item.label_a, item.label_b):
                final_label = "tidak_terselesaikan"
                is_unres = True
                unresolved_count += 1
            else:
                # Diskusi aturan cocok (jika salah satu 'salah', konservatif memilih 'salah')
                final_label = "salah"
                is_unres = False

        # PRD 11.4.8: Butir tidak terselesaikan dihitung tidak ditemukan (bukan TP)
        is_tp = (final_label == "valid") if not is_unres else False

        adjudications.append(AdjudicationResult(
            item_id=item.item_id,
            final_label=final_label,
            is_unresolved=is_unres,
            is_considered_tp=is_tp,
        ))

    overall_pct = agreed_count / total
    cat_pcts = {cat: cat_agreed.get(cat, 0) / count for cat, count in cat_counts.items()}

    return AgreementReport(
        overall_agreement_pct=round(overall_pct, 4),
        category_agreements={k: round(v, 4) for k, v in cat_pcts.items()},
        total_items=total,
        unresolved_count=unresolved_count,
        adjudications=adjudications,
    )
