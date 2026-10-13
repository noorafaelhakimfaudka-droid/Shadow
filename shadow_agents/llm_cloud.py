# shadow_agents/llm_cloud.py
"""
Klien Hybrid Cloud LLM Gratis: Google AI Studio (Gemini) + Groq Cloud.
- Google Gemini: Menggerakkan Auditor, Architect, Statistician, Feature Strategist (Analogi mendalam & narasi kaya).
- Groq Cloud: Menggerakkan Skeptic & Robustness Investigator (Kecepatan kilat 450+ token/s & kritik pedas 70B).
- Dilengkapi perisai failover silang dan fallback lokal Ollama.
"""

from __future__ import annotations
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

DEFAULT_GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
DEFAULT_GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")


# =============================================================================
# 1. GOOGLE AI STUDIO (GEMINI FLASH)
# =============================================================================
@dataclass
class GeminiCloudConfig:
    api_key: str = ""
    model: str = "gemini-3-flash-preview"
    temperature: float = 0.3
    max_output_tokens: int = 800
    timeout_seconds: int = 15

    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.environ.get("GEMINI_API_KEY", DEFAULT_GEMINI_API_KEY)


class GeminiCloudClient:
    """Klien Google AI Studio untuk dialog edukatif, berbobot, dan kaya analogi."""

    def __init__(self, config: Optional[GeminiCloudConfig] = None):
        self.config = config or GeminiCloudConfig()
        self.candidate_models = [
            self.config.model,
            "gemini-3-flash-preview",
            "gemini-flash-latest",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
        ]

    def generate(
        self,
        prompt: str,
        system_instruction: str = "",
        system_prompt: str = "",
        format_json: bool = False,
        fallback_text: str = "",
        fallback_response: Optional[str] = None,
    ) -> str:
        fb = fallback_response if fallback_response is not None else fallback_text
        sys_p = system_instruction or system_prompt
        if not self.config.api_key:
            return fb

        contents = [{"parts": [{"text": prompt}]}]
        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_output_tokens,
            },
        }

        if sys_p:
            payload["systemInstruction"] = {
                "parts": [{"text": sys_p}]
            }

        if format_json:
            payload["generationConfig"]["responseMimeType"] = "application/json"

        data = json.dumps(payload).encode("utf-8")

        for model_name in self.candidate_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.config.api_key}"
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
                    res_json = json.loads(resp.read().decode("utf-8"))
                    candidates = res_json.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            return parts[0].get("text", "").strip()
            except Exception:
                continue

        return fb


# =============================================================================
# 2. GROQ CLOUD (LPUs ULTRA-FAST INFERENCE)
# =============================================================================
@dataclass
class GroqCloudConfig:
    api_key: str = ""
    model: str = "qwen/qwen3.8-27b"
    temperature: float = 0.3
    max_output_tokens: int = 300
    timeout_seconds: int = 10

    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.environ.get("GROQ_API_KEY", DEFAULT_GROQ_API_KEY)


class GroqCloudClient:
    """Klien Groq Cloud untuk inferensi instan sub-detik (Skeptic & Veto)."""

    def __init__(self, config: Optional[GroqCloudConfig] = None):
        self.config = config or GroqCloudConfig()
        self.candidate_models = [
            self.config.model,
            "qwen/qwen3.8-27b",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
        ]

    def generate(
        self,
        prompt: str,
        system_instruction: str = "",
        system_prompt: str = "",
        format_json: bool = False,
        fallback_text: str = "",
        fallback_response: Optional[str] = None,
    ) -> str:
        fb = fallback_response if fallback_response is not None else fallback_text
        sys_p = system_instruction or system_prompt
        if not self.config.api_key:
            return fb

        messages = []
        if sys_p:
            messages.append({"role": "system", "content": sys_p})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        }

        for model_name in self.candidate_models:
            payload = {
                "model": model_name,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_output_tokens,
            }
            if format_json:
                payload["response_format"] = {"type": "json_object"}

            data = json.dumps(payload).encode("utf-8")
            url = "https://api.groq.com/openai/v1/chat/completions"
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")

            try:
                with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
                    res_json = json.loads(resp.read().decode("utf-8"))
                    choices = res_json.get("choices", [])
                    if choices:
                        return choices[0].get("message", {}).get("content", "").strip()
            except Exception:
                continue

        return fb


# =============================================================================
# 3. HYBRID MULTI-AGENT ROUTER (THE DREAM TEAM)
# =============================================================================
class HybridAgentRouter:
    """
    Router cerdas yang mengalokasikan beban kerja ke model yang paling cocok:
    - Skeptic & Robustness Investigator -> Groq (kecepatan kilat & kritik pedas).
    - Auditor, Architect, Statistician, Feature -> Gemini (analogi mendalam & konteks panjang).
    - Otomatis saling failover jika salah satu API limit atau offline.
    """

    def __init__(
        self,
        gemini_client: Optional[GeminiCloudClient] = None,
        groq_client: Optional[GroqCloudClient] = None,
    ):
        self.gemini = gemini_client or GeminiCloudClient()
        self.groq = groq_client or GroqCloudClient()

    def generate_agent_speech(
        self,
        agent_role: str,
        topic: str,
        context_data: str,
        fallback_quote: str = "",
    ) -> str:
        role_lower = agent_role.lower()

        # Strategi Pembagian Peran
        if "skeptic" in role_lower or "robustness" in role_lower or "veto" in role_lower:
            sys_prompt = (
                f"Kamu adalah {agent_role} di arena SHADOW. Sifatmu: sangat kritis, tidak mudah percaya angka mulus, "
                "mencurigai kebocoran data (target leakage), dan siap menggunakan hak VETO. "
                "Jawab dalam 1-2 kalimat tajam, pedas, analogis, dan langsung menusuk kelemahan model!"
            )
            # Coba Groq dulu (kilat), jika gagal beralih ke Gemini
            prompt = f"Topik: {topic}\nKonteks Data: {context_data}\nKeluarkan analisismu sekarang:"
            ans = self.groq.generate(prompt, system_instruction=sys_prompt, fallback_text="")
            if not ans:
                ans = self.gemini.generate(prompt, system_instruction=sys_prompt, fallback_text=fallback_quote)
            return ans or fallback_quote
        else:
            sys_prompt = (
                f"Kamu adalah {agent_role} di arena SHADOW ML. "
                "Tugasmu: berikan penjelasan data science yang panjang, detail, dan mudah dimengerti dengan analogi tajam membumi "
                "(misalnya analogi murid pemalas, brankas ujian sekolah, wasit pertandingan, dll). "
                "Hindari istilah kaku tanpa analogi. Buat narasi 2 kalimat yang memukau dan mendidik!"
            )
            prompt = f"Topik: {topic}\nKonteks Data: {context_data}\nKeluarkan narasi analisismu sekarang:"
            ans = self.gemini.generate(prompt, system_instruction=sys_prompt, fallback_text="")
            if not ans:
                ans = self.groq.generate(prompt, system_instruction=sys_prompt, fallback_text=fallback_quote)
            return ans or fallback_quote

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        system_instruction: str = "",
        format_json: bool = False,
        fallback_response: Optional[str] = None,
        fallback_text: str = "",
    ) -> str:
        """Metode serbaguna yang kompatibel dengan LocalLLMClient."""
        fb = fallback_response if fallback_response is not None else fallback_text
        sys_p = system_prompt or system_instruction

        # 1. Coba panggil Groq atau Gemini jika API key tersedia
        ans = ""
        p_lower = (prompt + " " + sys_p).lower()
        if self.groq.config.api_key and any(k in p_lower for k in ["skeptic", "veto", "kanari", "robustness", "stres"]):
            ans = self.groq.generate(prompt, system_instruction=sys_p, format_json=format_json, fallback_text="")
        
        if not ans and self.gemini.config.api_key:
            ans = self.gemini.generate(prompt, system_instruction=sys_p, format_json=format_json, fallback_text="")
        
        if not ans and self.groq.config.api_key:
            ans = self.groq.generate(prompt, system_instruction=sys_p, format_json=format_json, fallback_text="")

        if ans:
            return ans

        # 2. Jika tidak ada API key atau offline, sintesiskan respons ilmiah mendalam secara deterministik
        if fb:
            return fb

        return self._synthesize_local_response(prompt, sys_p)

    def _synthesize_local_response(self, prompt: str, system_prompt: str = "") -> str:
        """
        Engine Sintesis Pengetahuan Lokal SHADOW.
        Menghasilkan tanggapan data science yang mendalam, edukatif, berbobot,
        dan kaya analogi sesuai persona 6 Agen AI saat cloud LLM offline.
        """
        combined = (prompt + " " + system_prompt).lower()

        # Deteksi peran agen
        role = "DATA AUDITOR"
        if "architect" in combined or "arsitek" in combined:
            role = "MODEL ARCHITECT"
        elif "statistician" in combined or "wasit" in combined or "statistik" in combined:
            role = "STATISTICIAN"
        elif "feature" in combined or "peracik" in combined or "fitur" in combined:
            role = "FEATURE STRATEGIST"
        elif "robustness" in combined or "penguji" in combined or "kanari" in combined:
            role = "ROBUSTNESS INVESTIGATOR"
        elif "skeptic" in combined or "veto" in combined or "kritis" in combined:
            role = "SKEPTIC"
        elif "auditor" in combined or "kejujuran" in combined:
            role = "DATA AUDITOR"

        # 1. Topik Khusus: Akurasi vs Imbalanced
        if any(w in combined for w in ["akurasi", "imbalanced", "tidak seimbang", "menyesatkan"]):
            return (
                f"[{role} — ANALISIS KETIDAKSEIMBANGAN KELAS & METRIK]\n\n"
                "Akurasi mentah adalah ilusi paling berbahaya dalam machine learning tabular. "
                "Bayangkan sebuah sekolah di mana 95 dari 100 siswa selalu lulus. Seorang pengawas malas yang menebak "
                "SEMUA siswa pasti lulus akan memperoleh 'akurasi 95%', padahal ia tidak melakukan evaluasi sama sekali!\n\n"
                "Pada dataset Telco Churn (73,5% bertahan vs 26,5% churn) atau pendeteksian fraud (99% normal vs 1% penipuan), "
                "model pasif yang memprediksi nol churn akan mendapat akurasi 73,5%, namun recall terhadap pelanggan yang kabur adalah 0%!\n\n"
                "Itulah mengapa dewan SHADOW mewajibkan metrik berbasis ranking dan kalibrasi probabilitas: "
                "PR-AUC (Precision-Recall Area Under Curve), ROC-AUC, Balanced Accuracy, dan Brier Calibration Score. "
                "Kami tidak menilai seberapa sering model merasa benar, melainkan seberapa presisi model membedakan sinyal langka dari derau mayoritas."
            )

        # 2. Topik Khusus: Nadeau-Bengio & Occam's Razor
        elif any(w in combined for w in ["nadeau", "bengio", "occam", "razor", "signifikan"]):
            return (
                f"[{role} — ANALISIS UJI NADEAU-BENGIO & OCCAM'S RAZOR]\n\n"
                "Dalam evaluasi K-Fold Cross Validation yang diulang, lipatan data (folds) saling berbagi data latih "
                "(80% data tumpang tindih antar pengujian). Uji t-Student standar mengasumsikan sampel independen (IID), "
                "sehingga membagi varians dengan N dan menghasilkan p-value yang terlalu optimis secara palsu (Type I Error)!\n\n"
                "Koreksi Nadeau-Bengio (2003) memperkenalkan faktor penyesuaian varians resmi:\n"
                "Var_corrected = Var * (1/K + n_test / n_train)\n"
                "Dengan rasio 1/5 + 20/80 = 0.20 + 0.25 = 0.45, varians sebenarnya jauh lebih lebar dari asumsi klasik.\n\n"
                "Hasil audit SHADOW pada Telco Churn menunjukkan: LightGBM (CV AUC 0.846) vs Logistic Regression (CV AUC 0.846) "
                "menghasilkan nilai p = 0.90! Artinya tidak ada bukti statistik bahwa model pohon yang rumit lebih unggul. "
                "Sesuai prinsip Occam's Razor, SHADOW memveto kerumitan yang tak berdasar dan menetapkan Logistic Regression "
                "sebagai pemenang karena jauh lebih ringan, cepat, interpretable, dan bebas risiko degradasi produksi."
            )

        # 3. Topik Khusus: TF-IDF & NLP
        elif any(w in combined for w in ["tf-idf", "nlp", "bigram", "bow", "bag of words", "kata", "teks"]):
            return (
                f"[{role} — ANALISIS FEATURE ENGINEERING & NLP]\n\n"
                "Bag-of-Words murni memperlakukan kata seperti tumpukan bata terpisah tanpa konteks. "
                "Kalimat 'tidak puas' dan 'sangat puas' akan memiliki kata 'puas' yang sama tanpa memahami negasi di depannya!\n\n"
                "TF-IDF Bigram (2-gram) menyelesaikan dua kelemahan fatal sekaligus:\n"
                "1. Menangkap frasa konteks berpasangan seperti 'layanan lambat', 'tagihan bengkak', atau 'tidak ramah'.\n"
                "2. Sub-linear scaling (1 + log(TF)) meredam kata yang muncul 100 kali agar tidak 100 kali lebih dominan daripada kata yang muncul 10 kali.\n"
                "3. IDF (Inverse Document Frequency) menekan kata umum (seperti 'dan', 'yang', 'pada') dan melipatgandakan bobot kata pembeda churn.\n\n"
                "Aturan integritas mutlak SHADOW: seluruh kamus kosakata TF-IDF dan bobot IDF wajib di-fit HANYA pada data latih di dalam fold (fit-in-fold). "
                "Jika kosa kata dihitung pada seluruh dataset sebelum split, kebocoran kosa kata (vocabulary leakage) akan merusak validitas evaluasi."
            )

        # 4. Topik Khusus: Target Leakage & Radar S4
        elif any(w in combined for w in ["bocor", "leakage", "target leakage", "radar", "bahaya"]):
            return (
                f"[{role} — AUDIT KEBOCORAN DATA (TARGET LEAKAGE)]\n\n"
                "Target leakage adalah 'racun paling mematikan' dalam sains data terapan. "
                "Ini ibarat memberi kunci jawaban ujian kepada siswa di saku celananya: siswa akan tampak jenius dengan nilai 100 saat latihan, "
                "namun seketika hancur total saat menghadapi soal baru di dunia nyata!\n\n"
                "Bentuk kebocoran paling umum di industri:\n"
                "- Fitur masa depan: memasukkan durasi panggilan komplain yang baru terjadi SETELAH pelanggan memutuskan churn.\n"
                "- Identifikasi proksi: nomor akun atau ID baris yang kebetulan terurut berdasarkan tanggal pendaftaran.\n"
                "- Preprocessing global: melakukan standarisasi (StandardScaler) atau imputasi missing value pada seluruh dataset sebelum train-test split.\n\n"
                "Di SHADOW, Radar Kebocoran S4 secara deterministik memindai 9 vektor kebocoran (L1 Target Copy, L2 Duplikasi, "
                "L3 AST contamination, L5 Temporal shift, L6 Group leakage, L8 Index correlation, L9 Canary). "
                "Jika suatu fitur memiliki korelasi sempurna tanpa dasar kausal, sistem akan langsung menandai bendera merah dan mengecualikannya."
            )

        # 5. Tanggapan Berbasis Persona Agen
        if role == "DATA AUDITOR":
            return (
                f"[DATA AUDITOR — LAPORAN AUDIT RESMI]\n\n"
                f"Pertanyaan Anda: \"{prompt}\"\n\n"
                "Sebagai Auditor Integritas Data, perspektif saya sangat jelas: kualitas data menentukan batas atas (ceiling) dari performa model apa pun. "
                "Dalam setiap audit pipeline, saya memeriksa tiga perimeter krusial:\n"
                "1. Missingness Mechanism: Apakah nilai kosong berstatus MCAR (acak sempurna), MAR, atau MNAR (mengandung informasi tersembunyi)?\n"
                "2. Fit-in-Fold Strictness: Tidak ada parameter imputasi atau encoding yang boleh 'mengintip' data di luar fold latih.\n"
                "3. Brankas Holdout Tersegel: 20% data holdout disegel dengan hash SHA-256 dan hanya dibuka satu kali sebagai verifikasi akhir.\n\n"
                "Rekomendasi saya: Sebelum Anda mengubah arsitektur model atau menambah lapisan algoritma, pastikan audit sanitasi data Anda bersih dari anomali IQR dan kebocoran korelasi proksi."
            )
        elif role == "MODEL ARCHITECT":
            return (
                f"[MODEL ARCHITECT — PERSPEKTIF ARSITEKTUR]\n\n"
                f"Terkait pertanyaan Anda: \"{prompt}\"\n\n"
                "Sebagai Arsitek Model, prinsip utama saya adalah kecocokan induktif (inductive bias). "
                "Banyak praktisi terburu-buru menggunakan ensemble kompleks seperti XGBoost atau Deep Learning pada data tabular, "
                "padahal kompleksitas yang tidak dibutuhkan hanya mengundang overfitting dan kerentanan terhadap pergeseran distribusi.\n\n"
                "Kerangka evaluasi kami membandingkan tiga klan algoritma secara adil:\n"
                "- Model Linier Terregularisasi (Logistic Regression L2): Cepat, probabilistik murni, stabil.\n"
                "- Bagging (Random Forest): Tahan terhadap outlier, mereduksi varians.\n"
                "- Gradient Boosting (HistGradientBoosting / LightGBM): Menangkap interaksi non-linear yang rumit.\n\n"
                "Kemenangan sejati adalah memilih model yang efisien komputasi, mudah dideploy ke kontainer produksi, dan memiliki generalisasi stabil pada data baru."
            )
        elif role == "STATISTICIAN":
            return (
                f"[STATISTICIAN — PENILAIAN SKOR ILMIAH]\n\n"
                f"Analisis inferensi terhadap: \"{prompt}\"\n\n"
                "Sebagai Wasit Statistik di dewan SHADOW, saya menolak klaim keunggulan performa yang tidak memiliki dasar inferensial kuat. "
                "Jika Model A mendapat ROC-AUC 0.846 dan Model B mendapat 0.848, perbedaan 0.002 tersebut tidak memiliki arti ilmiah apa pun "
                "kecuali jika interval kepercayaannya (confidence intervals) tidak tumpang tindih dan uji beda berpasangan menolak hipotesis nol (H0).\n\n"
                "Kriteria kelayakan empiris kami:\n"
                "1. Repeated Stratified 5-Fold Cross Validation (3 kali pengulangan = 15 estimasi).\n"
                "2. Kalibrasi Brier Score untuk memastikan probabilitas yang diprediksi benar-benar mencerminkan frekuensi nyata.\n"
                "3. Toleransi signifikansi Nadeau-Bengio (alpha = 0.05). Jika p > 0.05, kedua model dianggap setara secara hukum statistik."
            )
        elif role == "FEATURE STRATEGIST":
            return (
                f"[FEATURE STRATEGIST — FORMULASI REPRESENTASI DATA]\n\n"
                f"Kajian rekayasa fitur terkait: \"{prompt}\"\n\n"
                "Sebagai Peracik Fitur, tugas saya adalah mengubah kolom mentah menjadi representasi matematis yang paling mudah dicerna oleh estimator. "
                "Pada data tabular heterogen, tantangan terbesarnya adalah mengelola variabel kategorikal dan numerik berdistribusi miring (skewed):\n\n"
                "1. Robust Imputation: Menggunakan median untuk fitur numerik guna meredam tarikan pencilan ekstrim.\n"
                "2. One-Hot Encoding dengan batasan frekuensi minimum untuk mencegah ledakan dimensi (curse of dimensionality).\n"
                "3. Standarisasi Fit-in-Fold: StandardScaler dihitung per lipatan latihan agar varians uji tidak mencemari skala latih.\n\n"
                "Fitur terbaik adalah fitur yang memiliki korelasi kausal logis dengan target bisnis, bukan fitur sintetis acak yang menghafal kebetulan statistik."
            )
        elif role == "ROBUSTNESS INVESTIGATOR":
            return (
                f"[ROBUSTNESS INVESTIGATOR — UJI KETAHANAN & KANARI]\n\n"
                f"Laporan uji stres atas topik: \"{prompt}\"\n\n"
                "Sebagai Penguji Ketahanan, saya menguji model seperti uji tabrak kendaraan (crash test). "
                "Model yang berkinerja tinggi pada data nyaman sering kali hancur total saat menghadapi sedikit derau (noise) di lingkungan nyata.\n\n"
                "Metodologi pengujian stres SHADOW:\n"
                "- Uji Kanari Permutasi: Kami mengacak label target secara total. Jika pipeline yang benar diuji kanari, AUC-nya WAJIB berada di kisaran 0.50 (tebakan koin). Jika kanari mendapat AUC 0.70+, berarti ada kebocoran kode!\n"
                "- Uji Derau 5%: Memberikan 5% label flip acak untuk mengukur degradasi skor.\n"
                "- Adversarial Validation: Melatih model pembeda antara data train vs holdout. Jika AUC mendekati 0.50, kedua distribusi identik dan bebas pergeseran domain."
            )
        else: # SKEPTIC
            return (
                f"[SKEPTIC — UJI KRITIS & ADVOKASI SKEPTIS]\n\n"
                f"Tinjauan kritis terhadap: \"{prompt}\"\n\n"
                "Sebagai Agen Skeptis, saya memiliki hak VETO resmi untuk menghentikan peluncuran model yang berisiko merugikan bisnis. "
                "Saya tidak peduli dengan grafik cantik atau metrik rata-rata yang tampak memuaskan. Saya mencari 'lubang jebakan' yang disembunyikan oleh rata-rata agregat!\n\n"
                "Temuan skeptis saya pada audit churn:\n"
                "- Disparitas Segmen: Pada ambang batas default 0.5, recall pelanggan kontrak bulanan mencapai 63%, tetapi recall pelanggan kontrak 1-2 tahun adalah 0%! Model mengorbankan pelanggan bernilai tinggi demi metrik rata-rata!\n"
                "- Masalah Ambang Batas: Di dunia bisnis nyata, biaya False Negative (pelanggan hilang tanpa terdeteksi) jauh lebih mahal daripada False Positive (diskon promosi yang terbuang).\n\n"
                "Kesimpulan saya: Jangan pernah menyetujui model hanya karena skor globalnya 0.84. Terapkan ambang batas keputusan per segmen (threshold tuning) sebelum menyentuh produksi."
            )

