# shadow_core/comparison.py
"""
S7 — Aturan Perbandingan Model [K] (PRD 6.7).
Rumus Terkunci [K]: Uji-t Resampling Terkoreksi Nadeau–Bengio.
Bootstrap atas lipatan DILARANG (R7 & C.2 #6).
Klaim 'A lebih baik dari B' HANYA diizinkan jika batas bawah rentang selisih >= delta.
Jika tidak lolos, model dipilih dengan Aturan Satu Galat Baku (1-SE).
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy import stats


@dataclass
class ComparisonResult:
    model_a_name: str
    model_b_name: str
    mean_a: float
    mean_b: float
    mean_diff: float  # A - B
    se_diff: float
    ci95_lower: float
    ci95_upper: float
    practical_margin_delta: float
    is_statistically_better: bool  # ci95_lower >= delta
    is_equivalent: bool  # ci95 mencakup 0 atau selisih < delta
    verdict_text: str
    holms_adjusted_pvalue: Optional[float] = None


def nadeau_bengio_ci(
    diffs: List[float] | np.ndarray,
    n_train_ratio: float = 0.8,
    n_test_ratio: float = 0.2,
    confidence: float = 0.95,
) -> Tuple[float, float, float, float]:
    """Menghitung selisih rata-rata, SE terkoreksi Nadeau-Bengio, dan rentang kepercayaan 95%."""
    d = np.array(diffs, dtype=float)
    J = len(d)
    if J < 2:
        val = float(d[0]) if J == 1 else 0.0
        return val, 0.0, val, val

    mean_d = float(np.mean(d))
    var_d = float(np.var(d, ddof=1))

    # Rasio ukuran uji terhadap latih dalam 1 fold (misal 5-fold: 0.2 / 0.8 = 0.25)
    split_ratio = n_test_ratio / n_train_ratio if n_train_ratio > 0 else 0.25

    # Rumus PRD 6.7: SE = sqrt((1/J + n_test/n_train) * s^2)
    se_factor = (1.0 / J) + split_ratio
    se_d = math.sqrt(max(0.0, se_factor * var_d))

    # Derajat kebebasan J - 1
    t_crit = float(stats.t.ppf((1.0 + confidence) / 2.0, df=J - 1))
    margin = t_crit * se_d

    ci_lower = mean_d - margin
    ci_upper = mean_d + margin

    return mean_d, se_d, ci_lower, ci_upper


def compare_models_pairwise(
    scores_a: List[float] | np.ndarray,
    scores_b: List[float] | np.ndarray,
    name_a: str,
    name_b: str,
    delta: float = 0.01,
    n_train_ratio: float = 0.8,
    n_test_ratio: float = 0.2,
    higher_is_better: bool = True,
) -> ComparisonResult:
    """Melakukan perbandingan berpasangan terkunci antara Model A dan Model B."""
    arr_a = np.array(scores_a, dtype=float)
    arr_b = np.array(scores_b, dtype=float)

    if len(arr_a) != len(arr_b):
        raise ValueError(f"Jumlah fold tidak sama: {len(arr_a)} vs {len(arr_b)}")

    # Selisih fold demi fold
    diffs = (arr_a - arr_b) if higher_is_better else (arr_b - arr_a)

    mean_a = float(np.mean(arr_a))
    mean_b = float(np.mean(arr_b))

    mean_diff, se_diff, ci_lower, ci_upper = nadeau_bengio_ci(
        diffs=diffs,
        n_train_ratio=n_train_ratio,
        n_test_ratio=n_test_ratio,
    )

    # Aturan 6.7: Klaim 'A lebih baik dari B' HANYA jika batas bawah rentang selisih >= delta
    is_better = bool(ci_lower >= delta)
    is_equivalent = not is_better

    if is_better:
        verdict = (
            f"{name_a} mengungguli {name_b} secara bermakna (selisih rata-rata: {mean_diff:+.4f}, "
            f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}] >= margin praktis δ = {delta})."
        )
    elif ci_lower > 0 and ci_lower < delta:
        verdict = (
            f"{name_a} memiliki estimasi lebih tinggi dari {name_b} ({mean_diff:+.4f}), "
            f"namun batas bawah rentang [{ci_lower:.4f}] di bawah margin praktis δ = {delta}. "
            f"Keduanya dianggap tidak dapat dibedakan secara bermakna."
        )
    else:
        verdict = (
            f"{name_a} dan {name_b} tidak dapat dibedakan secara bermakna "
            f"(selisih rata-rata: {mean_diff:+.4f}, 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}] mencakup nol)."
        )

    return ComparisonResult(
        model_a_name=name_a,
        model_b_name=name_b,
        mean_a=round(mean_a, 4),
        mean_b=round(mean_b, 4),
        mean_diff=round(mean_diff, 4),
        se_diff=round(se_diff, 4),
        ci95_lower=round(ci_lower, 4),
        ci95_upper=round(ci_upper, 4),
        practical_margin_delta=delta,
        is_statistically_better=is_better,
        is_equivalent=is_equivalent,
        verdict_text=verdict,
    )


def select_model_1se_rule(
    candidate_scores: Dict[str, List[float]],
    model_complexities: Optional[Dict[str, int]] = None,
    higher_is_better: bool = True,
) -> Tuple[str, str]:
    """Aturan Satu Galat Baku (1-SE Rule):
    Memilih model paling sederhana yang skornya masih berada dalam 1 SE dari model terbaik."""
    if not candidate_scores:
        raise ValueError("Tidak ada model kandidat untuk diseleksi!")

    # Default complexity: dummy=0, linear=1, rf=2, histgb=3
    default_complexity = {
        "Dummy": 0,
        "Linear": 1,
        "RandomForest": 2,
        "HistGradientBoosting": 3,
    }
    comp_map = model_complexities or default_complexity

    means = {}
    ses = {}
    for name, scs in candidate_scores.items():
        arr = np.array(scs, dtype=float)
        means[name] = float(np.mean(arr))
        ses[name] = float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0

    # Model dengan performa numerik tertinggi
    best_name = max(means, key=lambda k: means[k]) if higher_is_better else min(means, key=lambda k: means[k])
    best_mean = means[best_name]
    best_se = ses[best_name]

    threshold = (best_mean - best_se) if higher_is_better else (best_mean + best_se)

    # Cari semua model yang berada dalam toleransi 1 SE
    eligible = []
    for name, m in means.items():
        within_1se = (m >= threshold) if higher_is_better else (m <= threshold)
        if within_1se:
            # Ambil skor kompleksitas
            c_score = comp_map.get(name, 5)
            eligible.append((c_score, name))

    # Pilih model dengan kompleksitas terendah
    eligible.sort(key=lambda x: (x[0], -means[x[1]] if higher_is_better else means[x[1]]))
    chosen_name = eligible[0][1]

    rationale = (
        f"Model '{chosen_name}' dipilih menggunakan aturan 1-SE "
        f"(skor: {means[chosen_name]:.4f} berada dalam 1 SE dari model terbaik '{best_name}': {best_mean:.4f} ± {best_se:.4f})."
    )

    return chosen_name, rationale
