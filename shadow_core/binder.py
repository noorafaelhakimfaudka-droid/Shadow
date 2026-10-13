# shadow_core/binder.py
"""
S7/S9 — Pengikat Angka dan Klaim Terstruktur (PRD 6.12 & Aturan R6, R16).
Aturan R6: Angka di teks hanya boleh berasal dari ledger. Angka tanpa sumber memblokir hasil.
Aturan R16: Agen tidak menulis angka bebas. Angka hanya dirender Core dari klaim terstruktur.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set
from shadow_core.ledger import Ledger


# Whitelist angka sah non-metrik (nomor bagian notebook, tahun, dsb.)
DEFAULT_ALLOWED_NUMBERS = {
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11",
    "12", "15", "20", "2026", "0", "95", "99",
}

# Regex untuk mendeteksi digit atau angka desimal dalam teks
DIGIT_PATTERN = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")

# Kata-angka dalam bahasa Indonesia dan Inggris
WORD_NUMBERS = {
    "nol", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "delapan", "sembilan", "sepuluh",
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
}


@dataclass
class Violation:
    token: str
    context: str
    violation_type: str  # "unbound_number", "word_number", "mismatched_claim"
    message: str


def extract_ledger_numbers(ledger: Ledger) -> Set[str]:
    """Mengekstrak seluruh token angka sah yang ada di ledger."""
    valid_numbers: Set[str] = set()

    for entry in ledger.read():
        # Tambahkan skor per lipatan
        for s in entry.get("skor_per_lipatan", []):
            valid_numbers.add(f"{s:.4f}")
            valid_numbers.add(f"{s:.3f}")
            valid_numbers.add(f"{s:.2f}")
            valid_numbers.add(str(s))

        # Tambahkan ringkasan
        ring = entry.get("ringkasan", {})
        if "mean" in ring:
            m = ring["mean"]
            valid_numbers.add(f"{m:.4f}")
            valid_numbers.add(f"{m:.3f}")
            valid_numbers.add(f"{m:.2f}")
            valid_numbers.add(str(m))
        if "sd" in ring:
            sd = ring["sd"]
            valid_numbers.add(f"{sd:.4f}")
            valid_numbers.add(f"{sd:.3f}")
            valid_numbers.add(str(sd))
        if "ci95" in ring and len(ring["ci95"]) == 2:
            c1, c2 = ring["ci95"]
            valid_numbers.add(f"{c1:.4f}")
            valid_numbers.add(f"{c1:.3f}")
            valid_numbers.add(str(c1))
            valid_numbers.add(f"{c2:.4f}")
            valid_numbers.add(f"{c2:.3f}")
            valid_numbers.add(str(c2))

        # Seed, waktu
        if "seed" in entry:
            valid_numbers.add(str(entry["seed"]))
        if "waktu_detik" in entry:
            w = entry["waktu_detik"]
            valid_numbers.add(str(w))
            valid_numbers.add(f"{w:.1f}")

    return valid_numbers


def render(template: str, ledger: Ledger) -> str:
    """Mengisi penanda {{kunci}} dari ledger (PRD 5.3 & 6.12)."""
    rendered = template

    # Cari semua pattern {{trial_id.field}} misal {{t-001.mean}}
    placeholders = re.findall(r"\{\{([^}]+)\}\}", template)

    for ph in placeholders:
        parts = ph.strip().split(".")
        trial_id = parts[0]
        field_name = parts[1] if len(parts) > 1 else "mean"

        entry = ledger.get(trial_id)
        if entry is None:
            raise KeyError(f"Placeholder '{{{{{ph}}}}}' merujuk ke trial_id '{trial_id}' yang tidak ada di ledger!")

        ring = entry.get("ringkasan", {})
        if field_name == "mean":
            val_str = f"{ring.get('mean', 0.0):.4f}"
        elif field_name == "sd":
            val_str = f"{ring.get('sd', 0.0):.4f}"
        elif field_name == "ci95":
            c = ring.get("ci95", [0.0, 0.0])
            val_str = f"[{c[0]:.4f}, {c[1]:.4f}]"
        elif field_name == "model":
            val_str = str(entry.get("model", ""))
        else:
            val_str = str(entry.get(field_name, ""))

        rendered = rendered.replace(f"{{{{{ph}}}}}", val_str)

    return rendered


def verify(
    text: str,
    ledger: Ledger,
    allowed_numbers: Optional[Set[str]] = None,
) -> List[Violation]:
    """Memeriksa apakah seluruh token angka dalam teks terikat ke ledger (PRD 5.3)."""
    whitelist = DEFAULT_ALLOWED_NUMBERS.copy()
    if allowed_numbers:
        whitelist.update(allowed_numbers)

    ledger_nums = extract_ledger_numbers(ledger)
    violations: List[Violation] = []

    # Cari seluruh token angka
    for match in DIGIT_PATTERN.finditer(text):
        token = match.group(0).rstrip("%")
        # Standarisasi desimal koma ke titik
        norm_token = token.replace(",", ".")

        if norm_token in whitelist or token in whitelist:
            continue

        # Cek apakah ada di ledger
        is_bound = False
        if norm_token in ledger_nums or token in ledger_nums:
            is_bound = True
        else:
            # Cek float match
            try:
                flt_val = float(norm_token)
                for l_str in ledger_nums:
                    try:
                        if abs(float(l_str) - flt_val) < 1e-4:
                            is_bound = True
                            break
                    except ValueError:
                        pass
            except ValueError:
                pass

        if not is_bound:
            start = max(0, match.start() - 20)
            end = min(len(text), match.end() + 20)
            ctx = text[start:end]
            violations.append(Violation(
                token=token,
                context=ctx,
                violation_type="unbound_number",
                message=f"Angka '{token}' tidak ditemukan di ledger dan bukan angka yang diizinkan.",
            ))

    return violations


def verify_unmodified(text: str, template: str, ledger: Ledger) -> bool:
    """Memverifikasi bahwa teks yang dihasilkan persis sama dengan hasil render templat resmi Core."""
    expected = render(template, ledger)
    return text.strip() == expected.strip()


def render_structured_claim(claim: Dict[str, Any], ledger: Ledger) -> Tuple[str, Optional[Violation]]:
    """Merender satu klaim terstruktur sesuai aturan R16 v4.1.
    {subjek, metrik, split, kunci_ledger, templat}
    Memvalidasi kecocokan subjek, metrik, dan split sebelum merender nilai angka."""
    trial_id = claim.get("kunci_ledger", "")
    entry = ledger.get(trial_id)

    if entry is None:
        return "", Violation(
            token=trial_id,
            context=str(claim),
            violation_type="mismatched_claim",
            message=f"Kunci ledger '{trial_id}' tidak ditemukan!",
        )

    # Validasi kecocokan subjek (model)
    expected_model = claim.get("subjek", "")
    actual_model = entry.get("model", "")
    if expected_model.lower() not in actual_model.lower() and actual_model.lower() not in expected_model.lower():
        return "", Violation(
            token=expected_model,
            context=f"klaim={expected_model}, ledger={actual_model}",
            violation_type="mismatched_claim",
            message=f"Ketidakcocokan subjek klaim: '{expected_model}' vs ledger '{actual_model}'!",
        )

    # Validasi nilai metrik
    ring = entry.get("ringkasan", {})
    metric_val = ring.get("mean", 0.0)
    val_str = f"{metric_val:.4f}"

    templat = claim.get("templat", "Model {{subjek}} mencapai skor {{nilai}}.")
    rendered = templat.replace("{{subjek}}", actual_model).replace("{{nilai}}", val_str)

    return rendered, None
