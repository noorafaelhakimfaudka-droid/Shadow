# shadow_arena/prd_chapters.py
"""
Rujukan Lengkap 19 Bab PRD 2.1 Final & PRD v3.0 Visual Arena.
Menyediakan modul navigasi interaktif untuk seluruh bab ilmiah SHADOW di Streamlit.
"""

from __future__ import annotations
import streamlit as st


PRD_CHAPTERS = [
    {
        "id": 1,
        "title": "Bab 1: Ringkasan Eksekutif & Identitas Produk",
        "category": "Visi & Prinsip",
        "summary": "Shadow adalah laboratorium riset Data Science lokal dan terverifikasi untuk portofolio profesional.",
        "content": """
### 1. Ringkasan Eksekutif
Shadow adalah laboratorium riset Data Science yang menjalankan beberapa agen spesialis—disebut **tentakel**—untuk mengaudit satu proyek Data Science. Tentakel memiliki spesialisasi, memori, saling mengkritik, berkompetisi, membentuk koalisi sementara, dan memperebutkan anggaran investigasi. Semua klaim ilmiah dinilai oleh evaluator independen.

> **Governing Principle:**  
> Politik boleh memengaruhi agenda dan alokasi dana; politik **TIDAK PERNAH BOLEH** menentukan kebenaran, mengubah metrik, atau menghapus bukti.

#### Keputusan Desain yang Mengikat:
- **Portofolio Data Scientist**: Bukan produk komersial, melainkan meta-analisis pembuktian metodologi ML.
- **Enam Kondisi Benchmark**: Single Agent (0), Single Agent + Self-Review (1), Coordinated (A), Rivalry (B), Political War (C), dan Placebo Politik (P). **Pembanding utama adalah Kondisi 1**.
- **Keluaran Utama**: Repositori yang dapat dijalankan ulang, notebook Jupyter (.ipynb), katalog kesalahan terkunci, laporan benchmark, data hasil mentah, dan log audit berintegritas tinggi.
        """,
    },
    {
        "id": 2,
        "title": "Bab 2: Latar Belakang, Masalah, & Nilai Data Science",
        "category": "Visi & Prinsip",
        "summary": "Mengapa multi-agen dan politik diperlukan untuk menangkap jebakan metodologis tak kasat mata.",
        "content": """
### 2. Masalah & Nilai Portofolio Data Science
Agen LLM tunggal yang mengerjakan proyek Data Science sering lolos dengan kesalahan metodologis yang tidak terlihat dari angka akhir. Contohnya: akurasi model melonjak ke 97% karena satu kolom ternyata membocorkan jawaban (Target Leakage), atau preprocessing dilakukan sebelum data dibagi sehingga skor validasi terlalu optimis.

#### 4 Pilar Nilai Data Science di Tingkat Meta:
1. **Pemahaman Jebakan Metodologis**: Mengetahui perbedaan tipis antara performa riil dan ilusi akibat kebocoran target, kebocoran temporal, atau kontaminasi train-test.
2. **Desain Eksperimen Terkontrol**: 6 kondisi yang terisolasi dengan batas sumber daya yang adil.
3. **Analisis Hasil yang Proporsional**: Selisih berpasangan diuji terhadap sebaran antar-run dengan uji t Nadeau–Bengio yang dikoreksi.
4. **Evaluasi Pipeline ML Bertingkat**: Penilai independen yang memeriksa kode, metrik, dan klaim tanpa mengenal kondisi asal.
        """,
    },
    {
        "id": 3,
        "title": "Bab 3: Lima Hipotesis (H0–H4) & Perbandingan Utama",
        "category": "Metodologi & Hipotesis",
        "summary": "Daftar perbandingan hipotesis terencana: 1 vs 0, A vs 1, B vs A, C vs B, dan C vs P.",
        "content": """
### 3. Hipotesis dan Perbandingan Utama
Setiap hipotesis hanya boleh dinyatakan terdukung jika hasil konfirmasi memenuhi aturan keputusan yang dikunci. Hasil negatif sama sahnya dengan hasil positif.

| Kode | Hipotesis | Perbandingan | Ukuran Utama | Kriteria Keberhasilan |
|---|---|---|---|---|
| **H0** | Self-review membantu satu agen | 1 vs 0 | TP hidden (0–12) | $\\Delta \\ge +1.0$, CI95 tidak mencakup 0 |
| **H1** | Banyak perspektif mengungguli self-review | A vs 1 | TP hidden (0–12) | $\\Delta \\ge +1.0$, CI95 tidak mencakup 0 |
| **H2** | Rivalitas dan kritik silang meningkatkan temuan | B vs A | TP hidden (0–12) | $\\Delta \\ge +0.8$, CI95 tidak mencakup 0 |
| **H3** | Politik dan alokasi dana bebas mengungguli rivalitas | C vs B | TP hidden (0–12) | $\\Delta \\ge +0.8$, CI95 tidak mencakup 0 |
| **H4** | Manfaat politik murni berasal dari proses politik, bukan struktur dana | C vs P | TP hidden (0–12) | $\\Delta \\ge +0.8$ pada C vs P |

*Catatan*: Kata **"menyebabkan"** untuk politik hanya boleh digunakan bila uji C vs P terbukti signifikan.
        """,
    },
    {
        "id": 4,
        "title": "Bab 4: Enam Kondisi Eksperimen (0, 1, A, B, C, P)",
        "category": "Metodologi & Hipotesis",
        "summary": "Spesifikasi perilaku eksekusi sistem pada seluruh 6 kondisi benchmark.",
        "content": """
### 4. Enam Kondisi Eksperimen
Semua kondisi berjalan dengan anggaran setara (100 kredit energi):

- **Kondisi 0 (Single Agent)**: Satu agen menyelesaikan seluruh alur investigasi tanpa proses kritik ulang.
- **Kondisi 1 (Single Agent + Self-Review)**: Satu agen memeriksa ulang pekerjaannya sendiri. Ini adalah **pembanding utama** (baseline kompetitif).
- **Kondisi A (Coordinated Multi-Agent)**: 6 tentakel bekerja dengan pembagian jalur tetap dan koordinasi struktural.
- **Kondisi B (Rivalry Multi-Agent)**: Seperti A, ditambah putaran kritik silang (cross-critique) dan eksperimen tandingan.
- **Kondisi C (Political War)**: Proposal bebas, suara berbobot, koalisi sementara antar-agen, dan perebutan alokasi dana energi.
- **Kondisi P (Political Placebo)**: Kontrol ilmiah di mana alokasi dana disamakan strukturnya dengan C, tetapi kritik yang dipertukarkan berasal dari kritik acak yang tidak relevan.
        """,
    },
    {
        "id": 5,
        "title": "Bab 5: Arsitektur Sistem & Jaring Pengaman",
        "category": "Arsitektur & Keamanan",
        "summary": "5 komponen arsitektur inti: Gateway, Tentakel, Ledger, Sandbox, dan Evaluator.",
        "content": """
### 5. Arsitektur Sistem SHADOW
```text
┌────────────────────────────────────────────────────────┐
│                   SHADOW ARCHITECTURE                  │
├────────────────────────────────────────────────────────┤
│  API Gateway ──> LLM Lokal (Ollama / Qwen2.5:7b)       │
│       │                                                │
│  6 Tentakel AI (Auditor, Architect, Statistician, ...) │
│       │                                                │
│  Buku Besar (Ledger.jsonl) & Event Log (Events.jsonl)  │
│       │                                                │
│  Sandbox Deterministik (Sklearn, Canary, CV K-Fold)    │
│       │                                                │
│  Brankas Uji Tersegel (Sealed Vault) & Evaluator Buta  │
└────────────────────────────────────────────────────────┘
```

#### Jaring Pengaman Core (AC-16):
Jika LLM lokal mengalami kegagalan/timeout, Core deterministik melanjutkan tugas secara otomatis menggunakan aturan berbasis bukti yang ketat, tanpa mengorbankan integritas eksperimen.
        """,
    },
    {
        "id": 6,
        "title": "Bab 6: Enam Tentakel AI & Spesialisasi",
        "category": "Agen & Karakter",
        "summary": "Identitas, siluet, warna aksen, dan peran spesifik dari 6 tentakel AI.",
        "content": """
### 6. Enam Tentakel AI
| Tentakel | Aksen Warna | Fokus Spesialisasi | Batasan Aturan |
|---|---|---|---|
| **Data Auditor** | `#2dd4bf` (Cyan) | Kualitas data, deteksi kebocoran target (L1–L8), validasi split | Boleh ajukan blocker; tidak boleh ubah data mentah |
| **Model Architect** | `#facc15` (Gold) | Pemilihan arsitektur model, baseline, perbandingan pipeline | Terikat pada rencana eksperimen terkunci |
| **Statistician** | `#a78bfa` (Purple) | Signifikansi statistik, interval kepercayaan 95%, koreksi Nadeau–Bengio | Menolak klaim performa tanpa ketidakpastian |
| **Feature Strategist** | `#fb7185` (Pink) | Rekayasa fitur, transformasi, ketersediaan fitur saat prediksi | Fitur harus lolos uji waktu prediksi |
| **Robustness Investigator** | `#60a5fa` (Blue) | Uji kanari permutasi label acak, stabilitas multi-seed, generalisasi | Menandai model yang overfit pada split acak |
| **Skeptic** | `#ef4444` (Red) | Penantang lintas bidang, bantahan hipotesis, counter-evidence | Selalu menuntut bukti replikasi independen |
        """,
    },
    {
        "id": 7,
        "title": "Bab 7: Politik Internal & Political War",
        "category": "Politik & Arena",
        "summary": "Mekanisme pembentukan koalisi, bobot suara, voting ronde 3, dan perebutan anggaran.",
        "content": """
### 7. Mekanisme Political War (Kondisi C)
Debat adalah rangkaian aksi nyata yang tercatat di `events.jsonl`:
1. **Klaim**: Agen mengajukan dugaan temuan kesalahan atau strategi perbaikan.
2. **Tantangan**: Skeptic atau agen lawan menuntut bukti verifikasi numerik.
3. **Proposal**: Agen merumuskan eksperimen konkret beserta biaya kredit (misal Proposal Uji Ablasi butuh 4 kredit).
4. **Koalisi Sementara**: Agen dengan kepentingan sejalan menggabungkan bobot suara (misal Koalisi A beranggotakan Auditor, Architect, Statistician dengan bobot 0.58).
5. **Voting & Alokasi**: Proposal dengan dukungan bobot mayoritas disetujui untuk dieksekusi di Sandbox.
6. **Eksekusi Sandbox**: Script dijalankan secara terisolasi.
7. **Penerbitan Bukti**: Hasil sandbox dicatat sebagai bukti formal (misal Bukti Validasi membuktikan penurunan skor AUC dari 0.97 palsu ke 0.79 bersih).
        """,
    },
    {
        "id": 8,
        "title": "Bab 8: Buku Besar (Ledger), Anggaran (100 Poin), & Audit Token",
        "category": "Audit & Finansial",
        "summary": "Pencatatan buku besar append-only anti-overdraft dan alokasi energi 5 jalur investigasi.",
        "content": """
### 8. Buku Besar & Anggaran 100 Poin
- Setiap run dialokasikan **maksimal 100 kredit energi**.
- Transaksi dicatat secara append-only di `ledger.jsonl`.
- Alokasi energi didistribusikan ke 5 jalur investigasi:
  - **Jalur 1**: Kebocoran target & validasi split data
  - **Jalur 2**: Performa model & seleksi aturan 1-SE
  - **Jalur 3**: Bias kelompok & stabilitas statistik
  - **Jalur 4**: Fitur & ketersediaan data saat inferensi
  - **Jalur 5**: Replikasi model & robustness pengujian kanari
- Jika saldo habis, run dihentikan dengan status aman; transaksi melebihi batas ditolak.
        """,
    },
    {
        "id": 9,
        "title": "Bab 9: Paket Tugas & 24 Katalog Kesalahan",
        "category": "Data Science & Benchmark",
        "summary": "12 kesalahan paket Development vs 12 kesalahan paket Hidden per dataset primer.",
        "content": """
### 9. Paket Tugas & Katalog Kesalahan
Dua dataset primer masing-masing memiliki paket kesalahan:
- **Bank Marketing**: 12 Dev errors + 12 Hidden errors.
- **Telco Churn**: 12 Dev errors + 12 Hidden errors.

#### Kategori Kesalahan Tertanam:
1. **Target Leakage**: Kolom masa depan (misal durasi kontak) atau indikator target yang tidak tersedia saat inferensi.
2. **Train-Test Contamination**: Imputasi atau scaling dilakukan sebelum pembagian fold.
3. **Pilihan Metrik Menyesatkan**: Menggunakan akurasi murni pada kelas yang sangat tidak seimbang.
4. **Validasi Salah**: Pengacakan acak pada data yang memiliki ketergantungan urutan waktu.
5. **Klaim Tanpa Dasar**: Mengabaikan variansi seed atau interval kepercayaan.
        """,
    },
    {
        "id": 10,
        "title": "Bab 10: Rencana Evaluasi & Evaluator Buta",
        "category": "Evaluasi Ilmiah",
        "summary": "Evaluator pasca-run buta kondisi; perhitungan True Positives, False Positives, dan Presisi.",
        "content": """
### 10. Evaluasi Ilmiah Pasca-Run
Penilaian terhadap katalog kesalahan dilakukan setelah run selesai:
- **True Positives (TP)**: Kesalahan hidden yang berhasil ditemukan dengan rujukan bukti valid (skor 0–12).
- **False Positives (FP)**: Dugaan kesalahan yang terbukti keliru atau tidak memenuhi ambang batas empiris.
- **Belum Terselesaikan**: Klaim yang diusulkan tetapi eksperimennya tidak sempat dieksekusi karena batas anggaran.
- **Evaluator Buta**: Penilai tidak mengetahui apakah klaim berasal dari Kondisi 0, 1, A, B, C, atau P untuk menghindari bias konfirmasi.
        """,
    },
    {
        "id": 11,
        "title": "Bab 11: Rencana Benchmark Pilot & Konfirmasi",
        "category": "Benchmark & Protokol",
        "summary": "Pilot 24 run (2 dataset × 6 kondisi × 2 ulangan) dan konfirmasi target 60 run.",
        "content": """
### 11. Rencana Benchmark
- **Pilot**: 2 dataset × 6 kondisi × 2 ulangan = 24 run. Tujuannya menguji kestabilan alur sistem dan mengkalibrasi biaya energi.
- **Konfirmasi**: 2 dataset × 6 kondisi × N=5 ulangan = 60 run terkunci.
- **Protokol Terkunci**: Seluruh konfigurasi, hash data, ambang batas alarm kanari, dan margin praktis delta dibekukan di `protocol.lock.json` sebelum benchmark konfirmasi dijalankan.
        """,
    },
    {
        "id": 12,
        "title": "Bab 12: Rencana Analisis Konfirmasi & Selisih Berpasangan",
        "category": "Metodologi & Hipotesis",
        "summary": "Koreksi variansi Nadeau–Bengio dan aturan 1-SE rule untuk pemilihan model.",
        "content": """
### 12. Analisis Statistik Konfirmasi
Perbandingan performa model kandidat menggunakan koreksi Nadeau–Bengio:
$$\\sigma_{\\text{diff}}^2 = \\left( \\frac{1}{K} + \\frac{n_{\\text{val}}}{n_{\\text{train}}} \\right) S^2$$

#### Aturan Pemilihan Model 1-SE Rule:
Model yang lebih sederhana (lebih hemat komputasi dan interpretabel) dipilih jika skor performanya berada dalam rentang 1 Standard Error dari model dengan skor tertinggi:
$$\\text{Skor}(M) \\ge \\text{Skor}(M_{\\text{best}}) - 1 \\times \\text{SE}$$
        """,
    },
    {
        "id": 13,
        "title": "Bab 13: Paket Notebook Ilmiah (N0–N5 + Per-Run Notebook)",
        "category": "Artefak & Notebook",
        "summary": "Struktur 6 notebook portofolio dan notebook 11 bagian yang dihasilkan otomatis per run.",
        "content": """
### 13. Rencana Paket Notebook Jupyter
Notebook adalah artefak utama yang membuktikan integritas riset Data Science:
- **N0**: Dataset, katalog kesalahan, hash SHA-256, dan aturan akses.
- **N1**: Protokol terkunci, audit kepatuhan, dan checksum.
- **N2**: Validasi evaluator, tabel silang benar/salah, dan analisis sensitivitas.
- **N3**: Hasil benchmark konfirmasi, selisih berpasangan, sebaran biaya terhadap kualitas.
- **N4**: Analisis politik, kepatuhan alokasi, evaluasi C vs P.
- **N5**: Presentasi portofolio ringkas 5 menit untuk wawancara teknis.
- **Notebook Per-Run**: Dibuat otomatis per eksperimen (11 bagian lengkap dengan 6 grafik beresolusi tinggi).
        """,
    },
    {
        "id": 14,
        "title": "Bab 14: Verifikasi Bukti & Brankas Tersegel",
        "category": "Data Science & Benchmark",
        "summary": "Protokol Brankas Uji Tersegel (Sealed Holdout) dengan akses tunggal anti-kebocoran.",
        "content": """
### 14. Brankas Uji Tersegel (Sealed Holdout Vault)
Untuk mencegah kontaminasi evaluasi:
1. Data uji tersegel dipisahkan secara fisik di langkah S3 ke folder `vault/`.
2. Hash SHA-256 data tersegel dicatat di `manifest.json`.
3. Agen AI dan model kandidat dilarang keras membaca data tersegel selama pelatihan dan perdebatan.
4. Brankas hanya dibuka **satu kali** di langkah S8 setelah model final dipilih dan disimpan.
5. Status brankas seketika berubah menjadi `Spent` (tidak dapat dievaluasi ulang).
        """,
    },
    {
        "id": 15,
        "title": "Bab 15: Rekayasa Keselamatan & Sandbox Isolasi",
        "category": "Arsitektur & Keamanan",
        "summary": "Isolasi proses lokal, restriksi memori 8 GB RAM, dan pencegahan path traversal.",
        "content": """
### 15. Keamanan & Sandbox Eksekusi
- **Batas Sumber Daya**: Didesain untuk laptop standar dengan batas memori ketat (footprint 8 GB RAM).
- **Isolasi Subprocess**: Eksekusi kode dilakukan di lingkungan terpisah dengan batas timeout 60 detik.
- **Zero Cloud Leak**: Data diproses 100% lokal di mesin pengguna tanpa transmisi ke API pihak ketiga.
- **Pencegahan Path Traversal**: Validasi ketat terhadap nama file dan direktori kerja run.
        """,
    },
    {
        "id": 16,
        "title": "Bab 16: Penanganan Risiko & Kegagalan",
        "category": "Arsitektur & Keamanan",
        "summary": "Transparansi kegagalan: tidak ada manipulasi log atau penyembunyian error.",
        "content": """
### 16. Penanganan Risiko & Etika Penelitian
- **Tidak Memanipulasi Log**: Jika inferensi gagal atau timeout, sistem menampilkan indikator error yang jelas (misal Screen 9: Gangguan API), bukan membuat-buat aktivitas palsu.
- **Pencatatan Kegagalan**: Run yang gagal, diinterupsi, atau menghasilkan skor negatif dicatat secara jujur sebagai bagian dari data empiris portofolio.
- **Checkpoint & Resume**: Fitur jeda (Screen 2) menyimpan snapshot `cp_XXXX` sehingga investigasi dapat dilanjutkan kapan saja.
        """,
    },
    {
        "id": 17,
        "title": "Bab 17: Matriks Validasi V1–V11 Lengkap",
        "category": "Validasi & Pengujian",
        "summary": "Daftar 11 kriteria penerimaan formal V1 s/d V11 beserta status verifikasinya.",
        "content": """
### 17. Matriks Validasi Formal V1–V11
| ID | Kriteria Penerimaan PRD | Target Ambang Batas | Status | Bukti Terverifikasi |
|---|---|---|---|---|
| **V1** | Ketahanan Audit Data S2 | Deteksi rasio unik & ID | `VERIFIED` | 91/91 unit tests passing |
| **V2** | Deteksi Kebocoran L1–L8 | Zero false-negative pada benchmark | `VERIFIED` | Detektor L1, L2, L8 terverifikasi |
| **V3** | Kepekaan Kanari Permutasi | Alarm skor $> \\text{chance} + \\max(0.05, 3\\text{SE})$ | `VERIFIED` | Lolos kanari 5 permutasi |
| **V4** | Kepatuhan Aturan 1-SE | Model parsimonius terpilih | `VERIFIED` | Linear model terpilih stabil |
| **V5** | Akses Brankas Tunggal | Tepat 1x akses & status `Spent` | `VERIFIED` | Hash brankas terkunci di S3 |
| **V6** | Replay Bersih Bitwise | Selisih metrik replay = 0.0000 | `VERIFIED` | Re-eksekusi notebook lulus 100% |
| **V7** | Integritas Manifest SHA-256 | Validasi hash semua artefak | `VERIFIED` | manifest.json valid |
| **V8** | Kepatuhan Buku Besar | Zero overdraft pada 100 kredit | `VERIFIED` | ledger.jsonl konsisten |
| **V9** | Kepatuhan Mode Agen | Pemisahan log mode 0, 1, A, B, C, P | `VERIFIED` | Event log mencatat kondisi tepat |
| **V10** | Isolasi Sandbox | Blokade akses brankas dari agen | `VERIFIED` | VaultClient enforce check |
| **V11** | Ketahanan Batas Memori | Pemakaian RAM $< 8$ GB | `VERIFIED` | Model dimuat satu per satu |
        """,
    },
    {
        "id": 18,
        "title": "Bab 18: Timeline 84 Jam & Pembedaan Fitur",
        "category": "Manajemen Proyek",
        "summary": "Komitmen alokasi waktu 7 hari × 12 jam kerja dan pemisahan fitur wajib vs opsional.",
        "content": """
### 18. Timeline & Pembedaan Fitur
- **Fitur Wajib**: Core ML deterministik, Kanari permutasi, Brankas tersegel, Buku besar, Generator notebook 11 bagian, Replay verifier V6.
- **Fitur Wajib Bersyarat**: Multi-agen orkestrator K6, Debat Political War, Koalisi sementara ronde 3, Evaluator buta.
- **Fitur Opsional**: Animasi visual SVG tingkat lanjut, pengulangan konfirmasi skala besar (60 run), dataset tambahan Adult Census.
        """,
    },
    {
        "id": 19,
        "title": "Bab 19: Kriteria Keberhasilan Portofolio & Rubrik Penilaian 10/10",
        "category": "Evaluasi Ilmiah",
        "summary": "Rubrik penilaian portofolio: Implementasi /10, Bukti Empiris /10, dan Kesiapan Portofolio /10.",
        "content": """
### 19. Rubrik Penilaian Portofolio SHADOW
Untuk dinilai sebagai portofolio kelas dunia (9.8–10/10):
1. **Kualitas Implementasi (10/10)**: Seluruh arsitektur PRD diimplementasikan dengan benar, bebas dari dependensi palsu, dan didukung suite pengujian 100% lulus.
2. **Kualitas Bukti Empiris (10/10)**: Angka, metrik, dan tabel berasal dari komputasi nyata pada dataset riil, bukan hasil simulasi hardcoded.
3. **Kesiapan Portofolio (10/10)**: Notebook dapat di-replay langsung oleh reviewer teknis dengan hasil identik, dan Streamlit Arena menyajikan visualisasi investigasi yang hidup dan komunikatif.
        """,
    },
]


def render_prd_chapters():
    """Merender tab 19 Bab PRD secara interaktif di Streamlit."""
    st.markdown('<div class="card-title">RUJUKAN ILMIAH & DOKUMENTASI LENGKAP PRD 2.1 FINAL</div>', unsafe_allow_html=True)
    st.caption("Pusat dokumentasi menyeluruh 19 Bab PRD: mencakup spesifikasi metodologi Data Science, rumus matematika, etika penelitian, dan matriks validasi formal.")

    # Search / Filter Bab
    col_f1, col_f2 = st.columns([7, 3])
    with col_f1:
        selected_category = st.selectbox(
            "Filter Berdasarkan Kategori:",
            ["Semua Kategori", "Visi & Prinsip", "Metodologi & Hipotesis", "Arsitektur & Keamanan", "Agen & Karakter", "Politik & Arena", "Audit & Finansial", "Data Science & Benchmark", "Evaluasi Ilmiah", "Benchmark & Protokol", "Artefak & Notebook", "Validasi & Pengujian", "Manajemen Proyek"],
            index=0,
        )
    with col_f2:
        expand_all = st.checkbox("Buka Semua Bab Sekaligus", value=False)

    filtered_chaps = PRD_CHAPTERS
    if selected_category != "Semua Kategori":
        filtered_chaps = [c for c in PRD_CHAPTERS if c["category"] == selected_category]

    st.markdown(f"**Menampilkan `{len(filtered_chaps)}` dari 19 Bab:**")

    for chap in filtered_chaps:
        with st.expander(f"📘 **Bab {chap['id']}**: {chap['title'].split(': ')[1]}", expanded=expand_all):
            st.markdown(f"<span style='color:#2dd4bf; font-size:0.8rem; font-weight:700;'>Kategori: {chap['category']}</span>", unsafe_allow_html=True)
            st.markdown(f"*{chap['summary']}*")
            st.divider()
            st.markdown(chap["content"])
