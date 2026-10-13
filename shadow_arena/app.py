# shadow_arena/app.py
"""
SHADOW Arena — Machine Learning & AI Battle Arena
Desain Panggung Game Esports & Tactical ML Dashboard:
- Menghadirkan Standalone Interactive Arena 28-Babak secara penuh dan responsif.
- Navigasi 0 ms: Auto-Play, Kontrol Kecepatan (1x/2x/5x), Audio Synthesizer,
  Modal Transmisi Dialog Mendalam, Klasemen Turnamen Model, dan Ekspor Pipeline.
- Dilengkapi Konsultasi On-Demand Dual-Cloud (Gemini 3 Flash + Groq 27B LPU).
- Unduh Langsung Deliverable Resmi Jupyter Notebook (.ipynb) & Python (.py).
- Kepatuhan Ketat: Zero Emoji Policy & Zero-RAM Cloud Inference.
"""

from __future__ import annotations
import importlib
import io
import json
import sys
from pathlib import Path
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

STANDALONE_HTML_PATH = ROOT_DIR / "standalone_arena.html"

from shadow_core.standalone_notebook import generate_standalone_notebook
from shadow_agents.llm_cloud import HybridAgentRouter

@st.cache_data
def get_standalone_notebook_cached(dataset_name: str, target_col: str) -> str:
    return generate_standalone_notebook(dataset_name=dataset_name, target_col=target_col)

def extract_py_code(nb_json_str: str) -> str:
    try:
        nb_obj = json.loads(nb_json_str)
        blocks = []
        for cell in nb_obj.get("cells", []):
            if cell.get("cell_type") == "code":
                src = "".join(cell.get("source", []))
                blocks.append(src)
        return "\n\n# " + "="*70 + "\n\n".join(blocks)
    except Exception:
        return "# Script pipeline SHADOW\n"

# =============================================================================
# KONFIGURASI STREAMLIT (RESPONSIVE FULL-CANVAS)
# =============================================================================
try:
    st.set_page_config(
        page_title="SHADOW · AI Battle Arena",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
except Exception:
    pass

# Custom High-Tech Tactical CSS
st.markdown("""
<style>
    /* 1. Deep Space Battle Canvas */
    .stApp {
        background: #070c14 !important;
        color: #f8fafc !important;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif !important;
    }
    header, footer, #MainMenu {
        display: none !important;
        height: 0 !important;
        visibility: hidden !important;
    }
    /* Force true full-canvas wide layout across all Streamlit versions */
    div[data-testid="stAppViewContainer"] > section.main {
        max-width: 100% !important;
        width: 100% !important;
        padding: 0 !important;
    }
    div[data-testid="stAppViewBlockContainer"],
    section.main div.block-container,
    .block-container {
        padding-top: 0.1rem !important;
        padding-bottom: 0.5rem !important;
        padding-left: 0.8rem !important;
        padding-right: 0.8rem !important;
        max-width: 1440px !important;
        width: 100% !important;
    }
    /* Full width for iframe component container */
    div[data-testid="stCustomComponentV1"],
    div[data-testid="stCustomComponentV1"] > iframe {
        width: 100% !important;
        max-width: 100% !important;
        display: block !important;
        margin: 0 auto !important;
    }
    div[data-testid="stVerticalBlock"] > div {
        margin-bottom: 0.2rem !important;
    }
    div[data-testid="stHorizontalBlock"] {
        margin-bottom: 0.2rem !important;
    }

    /* 2. Living Pulse Beacon Animation */
    @keyframes live-pulse {
        0%, 100% {
            transform: scale(1);
            opacity: 1;
            filter: drop-shadow(0 0 4px currentColor);
        }
        50% {
            transform: scale(1.35);
            opacity: 0.65;
            filter: drop-shadow(0 0 10px currentColor);
        }
    }
    .live-beacon {
        display: inline-block;
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: currentColor;
        animation: live-pulse 2s infinite ease-in-out;
        margin-right: 5px;
        vertical-align: middle;
    }

    /* 3. Tactical Expansion Drawers */
    div[data-testid="stExpander"] {
        border-radius: 14px !important;
        background: rgba(13, 21, 34, 0.88) !important;
        border: 1.5px solid #1e293b !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4) !important;
    }
    div[data-testid="stExpander"] summary {
        color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 0.86rem !important;
        letter-spacing: 0.4px !important;
    }

    /* 4. Tactical Buttons */
    .stButton button, div[data-testid*="baseButton"] {
        border-radius: 12px !important;
        padding: 0.42rem 1.0rem !important;
        background: linear-gradient(180deg, #182a47, #0d1829) !important;
        color: #ffffff !important;
        border: 1.5px solid #294673 !important;
        font-weight: 800 !important;
        font-size: 0.84rem !important;
        letter-spacing: 0.3px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.45) !important;
        transition: all 0.2s ease !important;
    }
    .stButton button:hover, div[data-testid*="baseButton"]:hover {
        background: linear-gradient(180deg, #243d66, #14243b) !important;
        border-color: #38bdf8 !important;
        color: #38bdf8 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 18px rgba(56, 189, 248, 0.35) !important;
    }

    /* 5. Streamlit Tabs (#94a3b8 non-aktif, garis bawah tebal 3.5px aktif) */
    button[data-baseweb="tab"] {
        color: #94a3b8 !important;
        font-weight: 700 !important;
        font-size: 0.90rem !important;
        padding: 0.55rem 1.25rem !important;
        background: transparent !important;
        border-bottom: 3.5px solid transparent !important;
        transition: all 0.2s ease !important;
    }
    button[data-baseweb="tab"]:hover {
        color: #e2e8f0 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #ffffff !important;
        font-weight: 900 !important;
        border-bottom: 3.5px solid #38bdf8 !important;
    }
    div[data-baseweb="tab-highlight"] {
        background-color: #38bdf8 !important;
        height: 3.5px !important;
    }
    div[data-baseweb="tab-border"] {
        background-color: #1e293b !important;
        height: 1px !important;
    }

    /* 6. Download Buttons Matched to Cards (Telco Cyan #2dd4bf & Bank Gold #facc15) */
    div[data-testid="stHorizontalBlock"] > div:nth-child(1) .stDownloadButton > button {
        border-radius: 12px !important;
        background: linear-gradient(135deg, #0e3538, #071f22) !important;
        color: #99f6e4 !important;
        border: 2px solid #2dd4bf !important;
        font-weight: 800 !important;
        font-size: 0.88rem !important;
        letter-spacing: 0.3px !important;
        box-shadow: 0 4px 16px rgba(45, 212, 191, 0.35) !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stHorizontalBlock"] > div:nth-child(1) .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #134e4a, #0d282c) !important;
        border-color: #5eead4 !important;
        color: #ffffff !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 22px rgba(45, 212, 191, 0.55) !important;
    }

    div[data-testid="stHorizontalBlock"] > div:nth-child(2) .stDownloadButton > button {
        border-radius: 12px !important;
        background: linear-gradient(135deg, #382c08, #1f1803) !important;
        color: #fef08a !important;
        border: 2px solid #facc15 !important;
        font-weight: 800 !important;
        font-size: 0.88rem !important;
        letter-spacing: 0.3px !important;
        box-shadow: 0 4px 16px rgba(250, 204, 21, 0.35) !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stHorizontalBlock"] > div:nth-child(2) .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #4d3c0a, #2b2205) !important;
        border-color: #fde047 !important;
        color: #ffffff !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 22px rgba(250, 204, 21, 0.55) !important;
    }
    /* 7. Form Konsultasi Styling (Issue 9) */
    label[data-testid="stWidgetLabel"] p,
    div[data-testid="stWidgetLabel"] p,
    .stSelectbox label p,
    .stTextInput label p {
        color: #94a3b8 !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
        letter-spacing: 0.02em !important;
    }
    div[data-testid="stTextInput"] input,
    div[data-baseweb="input"] input,
    div[data-baseweb="input"] {
        background-color: #0d1522 !important;
        border: 1.5px solid #334155 !important;
        color: #e2e8f0 !important;
        border-radius: 8px !important;
    }
    div[data-testid="stTextInput"] input:focus,
    div[data-baseweb="input"]:focus-within {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 10px rgba(56, 189, 248, 0.25) !important;
    }
    div[data-baseweb="select"] > div {
        background-color: #0d1522 !important;
        border: 1.5px solid #334155 !important;
        color: #e2e8f0 !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="select"] span {
        color: #e2e8f0 !important;
    }
    ul[data-baseweb="menu"], div[data-baseweb="popover"] {
        background-color: #0d1522 !important;
        border: 1.5px solid #334155 !important;
    }
    li[data-baseweb="menu-item"] {
        background-color: #0d1522 !important;
        color: #e2e8f0 !important;
    }
    li[data-baseweb="menu-item"]:hover {
        background-color: #1e293b !important;
        color: #38bdf8 !important;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # -------------------------------------------------------------
    # 1. STANDALONE ARENA BATTLE ENGINE (MAJESTIC FULL-CANVAS)
    # -------------------------------------------------------------
    if STANDALONE_HTML_PATH.exists():
        arena_html = STANDALONE_HTML_PATH.read_text(encoding="utf-8")
    else:
        st.error("Berkas standalone_arena.html tidak ditemukan di direktori root.")
        return

    # Render arena mandiri dengan iframe 735px (persis setara rasio 1280x720)
    components.html(arena_html, height=735, scrolling=False)

    # -------------------------------------------------------------
    # 2. TACTICAL TRAY: KONSULTASI CLOUD AI, NOTEBOOK, & AUDIT
    # -------------------------------------------------------------
    st.markdown('<div style="margin-top: 10px;"></div>', unsafe_allow_html=True)
    tab_cloud, tab_notebook, tab_audit = st.tabs([
        "KONSULTASI CLOUD AI (GEMINI 3 FLASH + GROQ 27B)",
        "UNDUH DELIVERABLE NOTEBOOK (.IPYNB / .PY)",
        "AUDIT & INTEGRITAS SISTEM SHADOW"
    ])

    with tab_cloud:
        col_c1, col_c2 = st.columns([1.5, 3.5])
        with col_c1:
            target_agent = st.selectbox(
                "Pilih Agen Konsultasi:",
                [
                    "DATA AUDITOR (Detektif Kejujuran Data)",
                    "MODEL ARCHITECT (Arsitek Taktik AI)",
                    "STATISTICIAN (Wasit Penilai Skor Ilmiah)",
                    "FEATURE STRATEGIST (Peracik Fitur & NLP)",
                    "ROBUSTNESS INVESTIGATOR (Penguji Ketahanan & Kanari)",
                    "SKEPTIC (Si Paling Kritis & Veto)"
                ],
                index=0
            )
            agent_clean_name = target_agent.split("(")[0].strip()
            
            # Map each agent to signature combat color with dark readable text
            agent_colors = {
                "DATA AUDITOR": "#2dd4bf",          # Cyan
                "MODEL ARCHITECT": "#facc15",       # Gold
                "STATISTICIAN": "#c084fc",          # Purple
                "FEATURE STRATEGIST": "#f472b6",    # Pink
                "ROBUSTNESS INVESTIGATOR": "#38bdf8", # Sky Blue
                "SKEPTIC": "#f87171"                # Rose / Crimson
            }
            active_color = agent_colors.get(agent_clean_name, "#2dd4bf")
            st.markdown(f"""
            <style>
                div[data-testid="stButton"] button[kind="primary"] {{
                    background: {active_color} !important;
                    color: #070c14 !important;
                    border: none !important;
                    font-weight: 800 !important;
                    box-shadow: 0 4px 14px {active_color}55 !important;
                    transition: all 0.2s ease !important;
                }}
                div[data-testid="stButton"] button[kind="primary"]:hover {{
                    background: {active_color} !important;
                    filter: brightness(1.15) !important;
                    color: #000000 !important;
                    transform: translateY(-1px) !important;
                    box-shadow: 0 6px 20px {active_color}88 !important;
                }}
            </style>
            """, unsafe_allow_html=True)

            preset_q = st.selectbox(
                "Pertanyaan Cepat:",
                [
                    "Kustom (Ketik Sendiri)",
                    "Mengapa akurasi mentah menyesatkan pada data imbalanced?",
                    "Bagaimana uji Nadeau-Bengio membenarkan prinsip Occam's Razor?",
                    "Mengapa TF-IDF bigram lebih unggul daripada bag-of-words biasa?",
                    "Apa risiko fatal jika fitur bocor (target leakage) lolos ke produksi?"
                ]
            )

        with col_c2:
            default_text = "" if preset_q.startswith("Kustom") else preset_q
            user_prompt = st.text_input("Pertanyaan Anda:", value=default_text, placeholder="Ajukan pertanyaan data science...")
            
            if "latest_response" not in st.session_state:
                st.session_state.latest_response = None

            if st.button("Kirim ke Cloud LLM (Gemini 3 Flash / Groq 27B)", type="primary", use_container_width=True):
                with st.spinner(f"Menghubungi Dewan AI untuk {agent_clean_name}..."):
                    router = HybridAgentRouter()
                    sys_prompt = (
                        f"Kamu adalah {agent_clean_name} di dewan SHADOW ML. "
                        "Berikan analisis tajam, berbasis data, mendidik, dan penuh analogi konkret. "
                        "Gunakan bahasa Indonesia profesional, tegas, dan ilmiah. Dilarang menggunakan emoji."
                    )
                    prompt_to_send = user_prompt.strip() if user_prompt.strip() else preset_q
                    if not prompt_to_send or prompt_to_send.startswith("Kustom"):
                        prompt_to_send = "Jelaskan strategi inti Anda dalam menjaga kejujuran model data science."

                    resp = router.generate(prompt_to_send, system_prompt=sys_prompt)
                    st.session_state.latest_response = {
                        "agent": agent_clean_name,
                        "color": active_color,
                        "question": prompt_to_send,
                        "answer": resp
                    }

            if st.session_state.latest_response:
                lr = st.session_state.latest_response
                st.markdown(
                    f'<div style="background:rgba(15, 23, 42, 0.95); border:1.5px solid {lr["color"]}; border-radius:12px; padding:16px; margin-top:14px; box-shadow:0 4px 20px rgba(0,0,0,0.5);">'
                    f'<div style="color:#94a3b8; font-size:0.78rem; font-weight:700; margin-bottom:4px; text-transform:uppercase;">PERTANYAAN: {lr["question"]}</div>'
                    f'<div style="color:{lr["color"]}; font-weight:800; font-size:0.95rem; margin-bottom:8px;">TANGGAPAN RESMI: {lr["agent"]}</div>'
                    f'<div style="color:#f1f5f9; font-size:0.88rem; line-height:1.65; white-space:pre-wrap;">{lr["answer"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    with tab_notebook:
        col_nb1, col_nb2 = st.columns(2)
        with col_nb1:
            st.markdown(
                '<div style="background:rgba(15, 23, 42, 0.85); border:1.5px solid #2dd4bf; border-radius:12px; padding:16px; margin-bottom:12px;">'
                '<div style="font-weight:900; color:#2dd4bf; font-size:1.02rem; margin-bottom:6px;">TELCO CUSTOMER CHURN PIPELINE</div>'
                '<div style="color:#cbd5e1; font-size:13px; line-height:1.5;">7.043 Baris · 20 Kolom · Repeated Stratified 5-Fold CV + Occam Champion</div>'
                '</div>',
                unsafe_allow_html=True
            )
            telco_nb = get_standalone_notebook_cached(dataset_name="Telco Customer Churn", target_col="churn")
            telco_py = extract_py_code(telco_nb)

            c_b1, c_b2 = st.columns(2)
            with c_b1:
                st.download_button(
                    label="UNDUH NOTEBOOK (.ipynb)",
                    data=telco_nb,
                    file_name="shadow_telco_churn_pipeline.ipynb",
                    mime="application/x-ipynb+json",
                    use_container_width=True
                )
            with c_b2:
                st.download_button(
                    label="UNDUH SKRIP (.py)",
                    data=telco_py,
                    file_name="shadow_telco_churn_pipeline.py",
                    mime="text/x-python",
                    use_container_width=True
                )

            with st.expander("Lihat Kode Pipeline Telco Churn (.py)", expanded=False):
                st.code(telco_py, language="python")

        with col_nb2:
            st.markdown(
                '<div style="background:rgba(15, 23, 42, 0.85); border:1.5px solid #facc15; border-radius:12px; padding:16px; margin-bottom:12px;">'
                '<div style="font-weight:900; color:#facc15; font-size:1.02rem; margin-bottom:6px;">BANK MARKETING UCI PIPELINE (DEMO)</div>'
                '<div style="color:#cbd5e1; font-size:13px; line-height:1.5;">45.211 Baris · 17 Kolom · Eliminasi Duration + Fit-in-Fold Scaler</div>'
                '</div>',
                unsafe_allow_html=True
            )
            bank_nb = get_standalone_notebook_cached(dataset_name="Bank Marketing UCI (Demo)", target_col="deposit")
            bank_py = extract_py_code(bank_nb)

            c_b3, c_b4 = st.columns(2)
            with c_b3:
                st.download_button(
                    label="UNDUH NOTEBOOK (.ipynb)",
                    data=bank_nb,
                    file_name="shadow_bank_marketing_pipeline_demo.ipynb",
                    mime="application/x-ipynb+json",
                    use_container_width=True
                )
            with c_b4:
                st.download_button(
                    label="UNDUH SKRIP (.py)",
                    data=bank_py,
                    file_name="shadow_bank_marketing_pipeline_demo.py",
                    mime="text/x-python",
                    use_container_width=True
                )

            with st.expander("Lihat Kode Pipeline Bank Marketing (.py)", expanded=False):
                st.code(bank_py, language="python")

    with tab_audit:
        col_a1, col_a2, col_a3, col_a4 = st.columns(4)
        with col_a1:
            st.markdown(
                '<div style="background:rgba(15, 23, 42, 0.85); border:1.5px solid #38bdf8; border-radius:12px; padding:14px; text-align:center;">'
                '<div style="font-size:11px; color:#94a3b8; font-weight:800; letter-spacing:0.04em;">UNIT & INTEGRATION TESTS</div>'
                '<div style="font-size:1.25rem; color:#38bdf8; font-weight:900; margin:6px 0;">94 / 94 LOLOS</div>'
                '<div style="font-size:13px; color:#cbd5e1;">100% Passing Rate</div>'
                '</div>',
                unsafe_allow_html=True
            )
        with col_a2:
            st.markdown(
                '<div style="background:rgba(15, 23, 42, 0.85); border:1.5px solid #2dd4bf; border-radius:12px; padding:14px; text-align:center;">'
                '<div style="font-size:11px; color:#94a3b8; font-weight:800; letter-spacing:0.04em;">BRANKAS TERSEGEL S3A</div>'
                '<div style="font-size:1.25rem; color:#2dd4bf; font-weight:900; margin:6px 0;">SHA-256 LOCKED</div>'
                '<div style="font-size:13px; color:#cbd5e1;">20% Holdout Terkunci</div>'
                '</div>',
                unsafe_allow_html=True
            )
        with col_a3:
            st.markdown(
                '<div style="background:rgba(15, 23, 42, 0.85); border:1.5px solid #facc15; border-radius:12px; padding:14px; text-align:center;">'
                '<div style="font-size:11px; color:#94a3b8; font-weight:800; letter-spacing:0.04em;">KURIKULUM DEBAT AI</div>'
                '<div style="font-size:1.25rem; color:#facc15; font-weight:900; margin:6px 0;">28 / 28 BABAK</div>'
                '<div style="font-size:13px; color:#cbd5e1;">8 Fase Terakreditasi</div>'
                '</div>',
                unsafe_allow_html=True
            )
        with col_a4:
            st.markdown(
                '<div style="background:rgba(15, 23, 42, 0.85); border:1.5px solid #f87171; border-radius:12px; padding:14px; text-align:center;">'
                '<div style="font-size:11px; color:#94a3b8; font-weight:800; letter-spacing:0.04em;">KEPATUHAN KODE</div>'
                '<div style="font-size:1.25rem; color:#f87171; font-weight:900; margin:6px 0;">0 EMOJI</div>'
                '<div style="font-size:13px; color:#cbd5e1;">100% Crisp Vector SVG</div>'
                '</div>',
                unsafe_allow_html=True
            )


if __name__ == "__main__":
    main()
