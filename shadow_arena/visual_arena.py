# shadow_arena/visual_arena.py
"""
Komponen Rendering Visual Arena untuk SHADOW Streamlit.
Menghasilkan kanvas visual esports 1280x656 responsive yang menampilkan 6 kartu agen lengkap:
Data Auditor, Model Architect, Statistician, Feature Strategist, Robustness Investigator, Skeptic.
Dilengkapi Center Stage Graphics 8-Fase, Radar Telemetry, Status Pill Milestone, dan Equalizer Audio Bubble.
"""

from __future__ import annotations


def get_avatar_svg(agent_idx: int, color: str) -> str:
    """Menghasilkan ilustrasi SVG avatar unik untuk masing-masing agen."""
    if agent_idx == 0:  # DATA AUDITOR (Detektif Topi & Kaca Pembesar)
        return f"""
        <svg viewBox="0 0 64 64" width="64" height="64">
          <ellipse cx="32" cy="60" rx="22" ry="13" fill="{color}"/>
          <circle cx="32" cy="36" r="13" fill="#f3d2b3"/>
          <path d="M20 28 Q32 6 44 28Z" fill="#0f172a"/>
          <rect x="13" y="27" width="38" height="4" rx="2" fill="#0f172a"/>
          <rect x="20" y="22" width="24" height="3" fill="{color}"/>
          <circle cx="27" cy="37" r="2" fill="#0f172a"/>
          <circle cx="37" cy="37" r="2" fill="#0f172a"/>
          <path d="M28 44 Q32 46 36 44" stroke="#7c4a2d" stroke-width="1.6" fill="none"/>
          <circle cx="47" cy="46" r="7" fill="rgba(45,212,191,.35)" stroke="#e2e8f0" stroke-width="2.5"/>
          <line x1="52" y1="51" x2="58" y2="58" stroke="#e2e8f0" stroke-width="3" stroke-linecap="round"/>
        </svg>
        """
    elif agent_idx == 1:  # MODEL ARCHITECT (Helm Arsitek & Jangka Drafting)
        return f"""
        <svg viewBox="0 0 64 64" width="64" height="64">
          <ellipse cx="32" cy="60" rx="22" ry="13" fill="{color}"/>
          <circle cx="32" cy="38" r="13" fill="#e8c39e"/>
          <path d="M17 31 Q32 4 47 31Z" fill="{color}"/>
          <rect x="13" y="29" width="38" height="5" rx="2" fill="#fde68a"/>
          <rect x="29" y="10" width="6" height="16" rx="2" fill="#fde68a"/>
          <circle cx="26" cy="39" r="5" fill="#0f172a" stroke="{color}" stroke-width="2"/>
          <circle cx="38" cy="39" r="5" fill="#0f172a" stroke="{color}" stroke-width="2"/>
          <rect x="30" y="38" width="4" height="2" fill="{color}"/>
          <path d="M28 47 Q32 50 36 47" stroke="#7c4a2d" stroke-width="1.6" fill="none"/>
          <path d="M50 58 L54 42 L58 58" stroke="#e2e8f0" stroke-width="2.5" fill="none" stroke-linecap="round"/>
        </svg>
        """
    elif agent_idx == 2:  # STATISTICIAN (Topi Cendekiawan & Monokel)
        return f"""
        <svg viewBox="0 0 64 64" width="64" height="64">
          <ellipse cx="32" cy="60" rx="22" ry="13" fill="{color}"/>
          <path d="M20 46 Q32 66 44 46Z" fill="#e2e8f0"/>
          <circle cx="32" cy="38" r="12" fill="#f1d9c0"/>
          <path d="M17 30 L32 1 L47 30Z" fill="#6b21a8"/>
          <ellipse cx="32" cy="30" rx="20" ry="4" fill="#7e22ce"/>
          <path d="M24 26 Q32 8 40 26" stroke="#fde68a" stroke-width="2" fill="none"/>
          <circle cx="26" cy="38" r="2" fill="#0f172a"/>
          <circle cx="38" cy="38" r="5" fill="rgba(253,230,138,.25)" stroke="#fde68a" stroke-width="2"/>
          <circle cx="38" cy="38" r="1.8" fill="#0f172a"/>
          <path d="M43 41 L48 52" stroke="#fde68a" stroke-width="1.5"/>
        </svg>
        """
    elif agent_idx == 3:  # FEATURE STRATEGIST (Kuncir Kembar & Labu Alkimia)
        return f"""
        <svg viewBox="0 0 64 64" width="64" height="64">
          <circle cx="15" cy="18" r="8" fill="#be185d"/>
          <circle cx="49" cy="18" r="8" fill="#be185d"/>
          <ellipse cx="32" cy="60" rx="22" ry="13" fill="{color}"/>
          <circle cx="32" cy="32" r="17" fill="#be185d"/>
          <circle cx="32" cy="37" r="13" fill="#f3d2b3"/>
          <circle cx="26" cy="37" r="3" fill="#0f172a"/>
          <circle cx="38" cy="37" r="3" fill="#0f172a"/>
          <circle cx="27" cy="36" r="1" fill="#fff"/>
          <circle cx="39" cy="36" r="1" fill="#fff"/>
          <path d="M28 44 Q32 48 36 44" stroke="#9d174d" stroke-width="1.8" fill="none"/>
          <path d="M49 40 h6 v7 l5 11 h-16 l5 -11z" fill="#fce7f3" stroke="#f9a8d4" stroke-width="1.5"/>
          <circle cx="52" cy="52" r="2" fill="{color}"/>
        </svg>
        """
    elif agent_idx == 4:  # ROBUSTNESS INVESTIGATOR (Helm Astronot & Perisai)
        return f"""
        <svg viewBox="0 0 64 64" width="64" height="64">
          <path d="M32 66 Q8 58 11 46 L53 46 Q56 58 32 66Z" fill="{color}"/>
          <line x1="32" y1="18" x2="32" y2="8" stroke="{color}" stroke-width="2.5"/>
          <circle cx="32" cy="7" r="3.5" fill="#fde68a"/>
          <rect x="15" y="18" width="34" height="29" rx="11" fill="#1e3a5f" stroke="{color}" stroke-width="2"/>
          <rect x="19" y="28" width="26" height="10" rx="5" fill="#0ea5e9"/>
          <circle cx="27" cy="33" r="2.2" fill="#e0f2fe"/>
          <circle cx="37" cy="33" r="2.2" fill="#e0f2fe"/>
          <path d="M27 42 h10" stroke="#7dd3fc" stroke-width="2" stroke-linecap="round"/>
          <path d="M46 46 h13 v8 q0 7 -6.5 9 q-6.5 -2 -6.5 -9z" fill="#0f172a" stroke="#e0f2fe" stroke-width="2"/>
          <path d="M49 53 l3 3 l5 -6" stroke="#4ade80" stroke-width="2" fill="none"/>
        </svg>
        """
    else:  # SKEPTIC (Topeng Rubah & Simbol Larangan)
        return f"""
        <svg viewBox="0 0 64 64" width="64" height="64">
          <path d="M18 20 L11 3 L27 14Z" fill="#fecaca"/>
          <path d="M46 20 L53 3 L37 14Z" fill="#fecaca"/>
          <path d="M13 64 Q8 14 32 12 Q56 14 51 64Z" fill="#7f1d1d"/>
          <ellipse cx="32" cy="37" rx="12" ry="14" fill="#f3d2b3"/>
          <path d="M23 35 h8" stroke="#0f172a" stroke-width="3" stroke-linecap="round"/>
          <circle cx="39" cy="35" r="2.2" fill="#0f172a"/>
          <path d="M34 29 L44 25" stroke="#0f172a" stroke-width="2.6" stroke-linecap="round"/>
          <path d="M26 46 h12" stroke="#7f1d1d" stroke-width="2" stroke-linecap="round"/>
          <circle cx="50" cy="52" r="9" fill="#0f172a" stroke="{color}" stroke-width="2"/>
          <path d="M46 48 l8 8 M54 48 l-8 8" stroke="{color}" stroke-width="2.5" stroke-linecap="round"/>
        </svg>
        """


def get_center_phase_svg(ph: int, step: int) -> str:
    """Menghasilkan SVG visual interaktif panggung tengah untuk masing-masing dari 8 fase."""
    if ph == 0:  # FASE 1: INGESTI & BRANKAS TERSEGEL
        return """
        <svg viewBox="0 0 280 170" width="280" height="170">
          <text x="10" y="16" fill="#2dd4bf" font-size="10" font-weight="700">KELAS TIMPANG + BRANKAS TERSEGEL</text>
          <rect x="24" y="40" width="44" height="100" fill="#2dd4bf"/>
          <rect x="84" y="124" width="44" height="16" fill="#f87171"/>
          <text x="22" y="156" fill="#94a3b8" font-size="9">Mayoritas</text>
          <text x="82" y="156" fill="#94a3b8" font-size="9">Minoritas</text>
          <path d="M188 84 v-14 a18 18 0 0 1 36 0 v14" fill="none" stroke="#2dd4bf" stroke-width="5"/>
          <rect x="176" y="84" width="60" height="46" rx="8" fill="#0d2a2e" stroke="#2dd4bf" stroke-width="2"/>
          <circle cx="206" cy="103" r="5" fill="#2dd4bf"/>
          <rect x="204" y="106" width="4" height="12" fill="#2dd4bf"/>
          <text x="176" y="148" fill="#94a3b8" font-size="9">20% disegel SHA-256</text>
        </svg>
        """
    elif ph == 1:  # FASE 2: EDA & PENCILAN IQR
        return """
        <svg viewBox="0 0 280 170" width="280" height="170">
          <text x="10" y="16" fill="#c084fc" font-size="10" font-weight="700">SEBARAN, PENCILAN IQR, KORELASI</text>
          <g fill="#c084fc">
            <rect x="20" y="50" width="20" height="90"/>
            <rect x="44" y="70" width="20" height="70"/>
            <rect x="68" y="90" width="20" height="50"/>
            <rect x="92" y="106" width="20" height="34"/>
            <rect x="116" y="118" width="20" height="22"/>
            <rect x="140" y="126" width="20" height="14"/>
            <rect x="164" y="132" width="20" height="8"/>
          </g>
          <circle cx="250" cy="134" r="6" fill="#f87171"/>
          <circle cx="250" cy="134" r="11" fill="none" stroke="#f87171" stroke-dasharray="3 3"/>
          <text x="214" y="118" fill="#fca5a5" font-size="9">pencilan</text>
          <rect x="40" y="150" width="70" height="12" fill="none" stroke="#e9d5ff" stroke-width="1.5"/>
          <line x1="62" y1="150" x2="62" y2="162" stroke="#e9d5ff" stroke-width="2"/>
          <line x1="20" y1="156" x2="40" y2="156" stroke="#e9d5ff"/>
          <line x1="110" y1="156" x2="190" y2="156" stroke="#e9d5ff"/>
        </svg>
        """
    elif ph == 2:  # FASE 3: TARGET LEAKAGE AUDIT & VETO
        return """
        <svg viewBox="0 0 280 170" width="280" height="170">
          <text x="10" y="16" fill="#f87171" font-size="10" font-weight="700">FITUR JUJUR VS FITUR BOCOR</text>
          <line x1="30" y1="145" x2="260" y2="145" stroke="#475569"/>
          <line x1="30" y1="145" x2="30" y2="26" stroke="#475569"/>
          <g fill="#2dd4bf" opacity=".85">
            <circle cx="60" cy="120" r="4"/><circle cx="90" cy="90" r="4"/>
            <circle cx="80" cy="128" r="4"/><circle cx="120" cy="104" r="4"/>
            <circle cx="150" cy="70" r="4"/><circle cx="130" cy="128" r="4"/>
            <circle cx="170" cy="110" r="4"/><circle cx="110" cy="60" r="4"/>
          </g>
          <line x1="40" y1="140" x2="255" y2="34" stroke="#f87171" stroke-width="2.5" stroke-dasharray="6 4"/>
          <g fill="#f87171">
            <circle cx="110" cy="106" r="4"/><circle cx="160" cy="81" r="4"/><circle cx="210" cy="57" r="4"/>
          </g>
          <text x="176" y="92" fill="#fca5a5" font-size="10" font-weight="700">BOCOR</text>
          <path d="M236 118 l16 16 m0 -16 l-16 16" stroke="#f87171" stroke-width="3" stroke-linecap="round"/>
        </svg>
        """
    elif ph == 3:  # FASE 4: NLP & TF-IDF TOKENISASI
        return """
        <svg viewBox="0 0 280 170" width="280" height="170">
          <text x="10" y="16" fill="#f472b6" font-size="10" font-weight="700">TOKEN MENJADI BOBOT TF-IDF</text>
          <g fill="rgba(244,114,182,.18)" stroke="#f472b6">
            <rect x="14" y="30" width="58" height="24" rx="12"/>
            <rect x="76" y="30" width="58" height="24" rx="12"/>
            <rect x="138" y="30" width="58" height="24" rx="12"/>
            <rect x="200" y="30" width="66" height="24" rx="12"/>
          </g>
          <g fill="#fce7f3" font-size="10" text-anchor="middle">
            <text x="43" y="46">promo</text><text x="105" y="46">gratis</text>
            <text x="167" y="46">batal</text><text x="233" y="46">puas</text>
          </g>
          <rect x="28" y="66" width="30" height="74" fill="#f472b6"/>
          <rect x="90" y="100" width="30" height="40" fill="#f472b6" opacity=".7"/>
          <rect x="152" y="60" width="30" height="80" fill="#f472b6"/>
          <rect x="218" y="90" width="30" height="50" fill="#f472b6" opacity=".7"/>
          <text x="14" y="158" fill="#94a3b8" font-size="9">tinggi batang = bobot keunikan kata</text>
        </svg>
        """
    elif ph == 4:  # FASE 5: TURNAMEN ML & KURVA ROC
        o1 = "1" if step >= 16 else "0.2"
        o2 = "1" if step >= 17 else "0.2"
        o3 = "1" if step >= 18 else "0.2"
        return f"""
        <svg viewBox="0 0 280 170" width="280" height="170">
          <text x="10" y="16" fill="#facc15" font-size="10" font-weight="700">KURVA ROC TURNAMEN</text>
          <rect x="30" y="24" width="230" height="116" fill="none" stroke="#334155"/>
          <line x1="30" y1="140" x2="260" y2="24" stroke="#64748b" stroke-dasharray="4 4"/>
          <path d="M30 140 C70 95 150 62 260 36" fill="none" stroke="#c084fc" stroke-width="3" opacity="{o1}"/>
          <path d="M30 140 C55 80 120 50 260 30" fill="none" stroke="#38bdf8" stroke-width="3" opacity="{o2}"/>
          <path d="M30 140 C45 70 95 40 260 26" fill="none" stroke="#facc15" stroke-width="3" opacity="{o3}"/>
          <g font-size="9">
            <text x="34" y="158" fill="#c084fc">Logistic</text>
            <text x="100" y="158" fill="#38bdf8">Random Forest</text>
            <text x="190" y="158" fill="#facc15">LightGBM</text>
          </g>
        </svg>
        """
    elif ph == 5:  # FASE 6: VALIDASI STATISTIK & KALIBRASI
        return """
        <svg viewBox="0 0 280 170" width="280" height="170">
          <text x="10" y="16" fill="#c084fc" font-size="10" font-weight="700">KALIBRASI: KEYAKINAN VS KENYATAAN</text>
          <rect x="30" y="24" width="150" height="116" fill="none" stroke="#334155"/>
          <line x1="30" y1="140" x2="180" y2="24" stroke="#64748b" stroke-dasharray="4 4"/>
          <path d="M30 138 C70 112 120 70 180 28" fill="none" stroke="#facc15" stroke-width="3"/>
          <g fill="#c084fc">
            <circle cx="58" cy="114" r="4"/><circle cx="90" cy="96" r="4"/>
            <circle cx="120" cy="70" r="4"/><circle cx="152" cy="46" r="4"/>
          </g>
          <text x="198" y="60" fill="#e9d5ff" font-size="11" font-weight="700">Brier</text>
          <text x="198" y="92" fill="#facc15" font-size="26" font-weight="800">0.135</text>
          <text x="198" y="112" fill="#94a3b8" font-size="9">makin kecil makin jujur</text>
        </svg>
        """
    elif ph == 6:  # FASE 7: UJI KANARI & ATRIBUSI SHAP
        return """
        <svg viewBox="0 0 280 170" width="280" height="170">
          <text x="10" y="16" fill="#38bdf8" font-size="10" font-weight="700">UJI KANARI + ATRIBUSI SHAP</text>
          <rect x="24" y="40" width="40" height="100" fill="#38bdf8"/>
          <rect x="74" y="84" width="40" height="56" fill="#f87171"/>
          <line x1="18" y1="84" x2="124" y2="84" stroke="#fde68a" stroke-dasharray="4 3"/>
          <text x="24" y="156" fill="#94a3b8" font-size="9">Asli</text>
          <text x="74" y="156" fill="#94a3b8" font-size="9">Label acak</text>
          <g fill="#38bdf8">
            <rect x="160" y="36" width="100" height="14"/>
            <rect x="160" y="60" width="76" height="14"/>
            <rect x="160" y="84" width="54" height="14"/>
            <rect x="160" y="108" width="32" height="14"/>
          </g>
          <g fill="#94a3b8" font-size="9">
            <text x="134" y="47">A</text><text x="134" y="71">B</text>
            <text x="134" y="95">C</text><text x="134" y="119">D</text>
          </g>
        </svg>
        """
    else:  # FASE 8: BRANKAS TERBUKA & MAHKOTA JUARA
        return """
        <svg viewBox="0 0 280 170" width="280" height="170">
          <text x="10" y="16" fill="#facc15" font-size="10" font-weight="700">BRANKAS TERBUKA + MAHKOTA JUARA</text>
          <rect x="20" y="36" width="100" height="110" rx="8" fill="#1a1218" stroke="#f87171" stroke-width="2"/>
          <path d="M120 36 L156 48 V136 L120 146Z" fill="#3b1d1d" stroke="#f87171"/>
          <path d="M46 92 l16 16 l34 -34" fill="none" stroke="#4ade80" stroke-width="7" stroke-linecap="round"/>
          <path d="M170 130 L176 82 L198 108 L220 70 L242 108 L264 82 L270 130Z" fill="#facc15" stroke="#fde68a" stroke-width="2"/>
          <circle cx="220" cy="122" r="5" fill="#f87171"/>
          <circle cx="196" cy="122" r="4" fill="#38bdf8"/>
          <circle cx="244" cy="122" r="4" fill="#c084fc"/>
        </svg>
        """


from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
STANDALONE_HTML_PATH = ROOT_DIR / "standalone_arena.html"


def render_svg_arena(
    auditor_bubble: str = "",
    architect_bubble: str = "",
    statistician_bubble: str = "",
    feature_bubble: str = "",
    robustness_bubble: str = "",
    skeptic_bubble: str = "",
    auditor_status: str = "Siaga",
    architect_status: str = "Siaga",
    statistician_status: str = "Siaga",
    feature_status: str = "Siaga",
    robustness_status: str = "Siaga",
    skeptic_status: str = "Siaga",
    center_pill_text: str = "Pusat Debat Data Science",
    mode: str = "normal",
    selected_agent: str = "auditor",
    step: int = 1,
    total_steps: int = 28,
) -> str:
    """
    Merender seluruh arena interaktif React 28-babak dari standalone_arena.html
    ke dalam komponen Streamlit.
    """
    if STANDALONE_HTML_PATH.exists():
        return STANDALONE_HTML_PATH.read_text(encoding="utf-8")

    agents = [
        {"name": "DATA AUDITOR", "role": "Detektif Kejujuran Data", "c": "#2dd4bf", "b": "rgba(13,42,46,0.92)", "x": 200, "y": 170, "key": "auditor"},
        {"name": "MODEL ARCHITECT", "role": "Arsitek Taktik AI", "c": "#facc15", "b": "rgba(44,38,12,0.92)", "x": 640, "y": 110, "key": "architect"},
        {"name": "STATISTICIAN", "role": "Wasit Penilai Skor Ilmiah", "c": "#c084fc", "b": "rgba(36,18,54,0.92)", "x": 640, "y": 565, "key": "statistician"},
        {"name": "FEATURE STRATEGIST", "role": "Peracik Fitur & NLP", "c": "#f472b6", "b": "rgba(44,16,34,0.92)", "x": 200, "y": 500, "key": "feature"},
        {"name": "ROBUSTNESS INVESTIGATOR", "role": "Penguji Ketahanan & Kanari", "c": "#38bdf8", "b": "rgba(14,34,58,0.92)", "x": 1080, "y": 170, "key": "robustness"},
        {"name": "SKEPTIC", "role": "Si Paling Kritis & Veto", "c": "#f87171", "b": "rgba(48,16,18,0.92)", "x": 1080, "y": 500, "key": "skeptic"},
    ]

    # Identifikasi pembicara aktif
    sel_lower = (selected_agent or "auditor").lower()
    active_idx = 0
    for idx, ag in enumerate(agents):
        if ag["key"] in sel_lower:
            active_idx = idx
            break

    active_ag = agents[active_idx]
    ph = 0 if step <= 3 else 1 if step <= 6 else 2 if step <= 9 else 3 if step <= 14 else 4 if step <= 18 else 5 if step <= 21 else 6 if step <= 25 else 7

    # Kalimat ucapan aktif
    bubbles = [auditor_bubble, architect_bubble, statistician_bubble, feature_bubble, robustness_bubble, skeptic_bubble]
    active_quote = (
        bubbles[active_idx]
        or auditor_bubble
        or architect_bubble
        or statistician_bubble
        or feature_bubble
        or robustness_bubble
        or skeptic_bubble
        or "Menganalisis pipeline..."
    ).strip()

    # Hitung posisi ekor balon ucapan (pointing tail)
    if active_idx in [0, 3]:
        side_style = f"left:-6px; top:28px; border-left:1.5px solid {active_ag['c']}; border-bottom:1.5px solid {active_ag['c']};"
    elif active_idx in [4, 5]:
        side_style = f"right:-6px; top:28px; border-top:1.5px solid {active_ag['c']}; border-right:1.5px solid {active_ag['c']};"
    elif active_idx == 1:
        side_style = f"left:284px; top:-6px; border-top:1.5px solid {active_ag['c']}; border-left:1.5px solid {active_ag['c']};"
    else:
        side_style = f"left:284px; bottom:-6px; border-bottom:1.5px solid {active_ag['c']}; border-right:1.5px solid {active_ag['c']};"

    # Bangun kartu-kartu agen
    agent_cards_html = ""
    for idx, ag in enumerate(agents):
        is_on = (idx == active_idx)
        card_border = f"2.5px solid {ag['c']}" if is_on else f"1px solid {ag['c']}"
        card_scale = "scale(1.06)" if is_on else "scale(1)"
        card_opacity = "1" if is_on else "0.65"
        card_shadow = f"box-shadow:0 0 35px {ag['c']};" if is_on else ""
        card_anim = "animation:bob 2s ease-in-out infinite;" if is_on else ""
        badge_bicara = f'<div style="position:absolute; top:-11px; right:14px; padding:2px 9px; border-radius:999px; background:{ag["c"]}; color:#070c14; font-size:10px; font-weight:800; letter-spacing:.08em;">SEDANG BICARA</div>' if is_on else ""

        agent_cards_html += f"""
        <div style="position:absolute; left:{ag['x'] - 115}px; top:{ag['y'] - 52}px; width:230px; height:104px; box-sizing:border-box; padding:0 14px; display:flex; align-items:center; border-radius:16px; border:{card_border}; background:{ag['b']}; opacity:{card_opacity}; transform:{card_scale}; {card_shadow} {card_anim} transition:all 0.2s ease;">
          {badge_bicara}
          <div style="display:flex; align-items:center; gap:12px;">
            <div style="flex:none; width:66px; height:66px; border-radius:50%; border:2px solid {ag['c']}; background:radial-gradient(circle at 50% 35%,#16233a,#070c14); overflow:hidden; display:flex; align-items:flex-end; justify-content:center;">
              {get_avatar_svg(idx, ag['c'])}
            </div>
            <div>
              <div style="font-size:13px; font-weight:800; color:{ag['c']}; font-family:'Chakra Petch', sans-serif;">{ag['name']}</div>
              <div style="font-size:11px; color:#94a3b8; font-family:'Chakra Petch', sans-serif;">{ag['role']}</div>
            </div>
          </div>
        </div>
        """

    center_stage_svg = get_center_phase_svg(ph, step)

    full_html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@500;700;800&display=swap" rel="stylesheet">
<style>
* {{ box-sizing: border-box; margin:0; padding:0; }}
body {{
  background: #070c14;
  overflow: hidden;
  font-family: 'Chakra Petch', ui-sans-serif, 'Segoe UI', sans-serif;
  color: #e2e8f0;
  display: flex;
  justify-content: center;
  align-items: flex-start;
  width: 100vw;
  height: 100vh;
}}
@keyframes spin {{ to {{ transform: rotate(360deg); }} }}
@keyframes pulse {{ 0%, 100% {{ opacity: 0.45; }} 50% {{ opacity: 1; }} }}
@keyframes bob {{ 0%, 100% {{ transform: scale(1.06) translateY(0); }} 50% {{ transform: scale(1.06) translateY(-6px); }} }}
#arena-wrapper {{
  position: relative;
  width: 1280px;
  height: 656px;
  flex-shrink: 0;
  transform-origin: top center;
  background: #070c14;
  overflow: hidden;
}}
</style>
</head>
<body>
<div id="arena-wrapper">
  <!-- Radial Ambient Glow -->
  <div style="position:absolute; left:0; top:0; width:1280px; height:656px; background:radial-gradient(circle at 50% 45%, {active_ag['c']}24, transparent 65%);"></div>

  <!-- SVG Network Grid & Tactical Links -->
  <svg viewBox="0 0 1280 656" width="1280" height="656" style="position:absolute; left:0; top:0;">
    <defs>
      <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
        <path d="M40 0H0V40" fill="none" stroke="#1e293b" stroke-width="1" opacity=".4"/>
      </pattern>
    </defs>
    <rect width="1280" height="656" fill="url(#grid)"/>
    <polygon points="640,44 920,186 920,470 640,612 360,470 360,186" fill="none" stroke="#1e3a4a" stroke-width="1.5" stroke-dasharray="6 8"/>
    <g stroke="#1e3a4a" stroke-width="1">
      <line x1="200" y1="170" x2="640" y2="328"/>
      <line x1="200" y1="500" x2="640" y2="328"/>
      <line x1="640" y1="110" x2="640" y2="328"/>
      <line x1="640" y1="565" x2="640" y2="328"/>
      <line x1="1080" y1="170" x2="640" y2="328"/>
      <line x1="1080" y1="500" x2="640" y2="328"/>
    </g>
    <line x1="{active_ag['x']}" y1="{active_ag['y']}" x2="640" y2="328" stroke="{active_ag['c']}" stroke-width="2.5" opacity=".85" stroke-dasharray="10 6"/>
  </svg>

  <!-- Center Radar Beacon -->
  <div style="position:absolute; left:640px; top:250px; width:200px; height:200px; margin:-100px 0 0 -100px; border-radius:50%; border:1px solid #1e3a4a; background:radial-gradient(circle,rgba(45,212,191,.08),transparent 70%);"></div>
  <div style="position:absolute; left:640px; top:250px; width:200px; height:200px; margin:-100px 0 0 -100px; border-radius:50%; background:conic-gradient(from 0deg,rgba(45,212,191,.35),transparent 25%); animation:spin 5s linear infinite;"></div>
  <div style="position:absolute; left:640px; top:250px; width:250px; height:250px; margin:-125px 0 0 -125px; border-radius:50%; border:1px dashed #334155; animation:spin 30s linear infinite;"></div>

  <!-- CENTER STAGE CARD (8 PHASES DYNAMIC) -->
  <div style="position:absolute; left:500px; top:160px; width:280px; height:160px; border-radius:14px; background:rgba(7,12,20,.72); border:1.5px solid {active_ag['c']}; overflow:hidden; box-shadow:0 0 24px {active_ag['c']}44;">
    {center_stage_svg}
  </div>

  <!-- MILESTONE STATUS PILL -->
  <div style="position:absolute; left:640px; top:338px; transform:translate(-50%,-50%); padding:5px 16px; border-radius:999px; border:1.5px solid {active_ag['c']}; background:rgba(7,12,20,.95); box-shadow:0 0 24px {active_ag['c']}66; font-size:12px; font-weight:800; color:{active_ag['c']}; white-space:nowrap;">
    {center_pill_text}
  </div>

  <!-- 6 AGENT CARDS -->
  {agent_cards_html}

  <!-- SPEECH BUBBLE WITH AUDIO EQUALIZER (COLLISION-FREE) -->
  <div style="position:absolute; left:310px; top:352px; width:660px; max-height:142px; box-sizing:border-box; padding:12px 18px; border-radius:14px; border:1.5px solid {active_ag['c']}; background:{active_ag['b']}; box-shadow:0 0 30px rgba(0,0,0,.75); overflow-y:auto; z-index:4;">
    <div style="position:absolute; width:12px; height:12px; transform:rotate(45deg); background:{active_ag['b']}; {side_style}"></div>
    <div style="display:flex; align-items:center; gap:8px; font-size:11px; font-weight:800; letter-spacing:.08em; color:{active_ag['c']};">
      <span>{active_ag['name']}</span>
      <span style="display:flex; align-items:flex-end; gap:2px; height:12px;">
        <span style="width:3px; height:12px; background:{active_ag['c']}; animation:pulse .8s infinite;"></span>
        <span style="width:3px; height:7px; background:{active_ag['c']}; animation:pulse .8s .2s infinite;"></span>
        <span style="width:3px; height:10px; background:{active_ag['c']}; animation:pulse .8s .4s infinite;"></span>
      </span>
    </div>
    <div style="font-size:13px; font-weight:600; line-height:1.48; margin-top:6px; color:#ffffff;">“{active_quote}”</div>
  </div>
</div>
<script>
function resizeArena() {{
  const w = window.innerWidth;
  const wrap = document.getElementById('arena-wrapper');
  if (wrap) {{
    const scale = Math.min(1, w / 1280);
    wrap.style.transform = 'scale(' + scale + ')';
  }}
}}
window.addEventListener('resize', resizeArena);
resizeArena();
</script>
</body>
</html>
"""
    return full_html


def render_avatar_row_svg() -> str:
    """Merender baris avatar untuk screen setup."""
    return ""
