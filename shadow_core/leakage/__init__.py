"""
SHADOW Leakage Defense Suite (L1–L9 + Permutation Canary).
Aturan R8: Tidak ada kalimat 'data bebas kebocoran'; hanya 'tidak ada indikasi dari detektor X, Y, Z'.
"""

from shadow_core.leakage.l1_target import detect_l1_target_leakage, L1Report, L1Finding
from shadow_core.leakage.l2_duplicates import detect_l2_cross_split_duplicates, L2Report
from shadow_core.leakage.l5_temporal import detect_l5_temporal_leakage, L5Report
from shadow_core.leakage.l6_group import detect_l6_group_leakage, L6Report
from shadow_core.leakage.l8_index import detect_l8_index_leakage, L8Report, L8Finding
from shadow_core.leakage.l9_flags import detect_l9_flag_leakage, L9Report, L9Finding
from shadow_core.leakage.l3_static import analyze_code_for_leakage, L3Report, L3Finding
from shadow_core.leakage.canary import run_permutation_canary, CanaryReport

# Alias
detect_l2_duplicates = detect_l2_cross_split_duplicates

__all__ = [
    "detect_l1_target_leakage",
    "detect_l2_duplicates",
    "detect_l2_cross_split_duplicates",
    "detect_l5_temporal_leakage",
    "detect_l6_group_leakage",
    "detect_l8_index_leakage",
    "detect_l9_flag_leakage",
    "analyze_code_for_leakage",
    "run_permutation_canary",
    "L1Report",
    "L1Finding",
    "L2Report",
    "L5Report",
    "L6Report",
    "L8Report",
    "L8Finding",
    "L9Report",
    "L9Finding",
    "L3Report",
    "L3Finding",
    "CanaryReport",
]
