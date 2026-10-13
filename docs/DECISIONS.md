# SHADOW — Keputusan Arsitektur dan Metodologi (Decisions & Log)

Dokumen ini mencatat seluruh keputusan tetap [K], resolusi pertanyaan terbuka (OQ), resolusi konflik dokumen terdahulu (Lampiran C.2), dan log keputusan formal untuk **SHADOW v4.1**.

---

## 1. Keputusan Tetap [K] (PRD Bagian 15.2)

1. **Identitas Produk**: Proyek Data Science dan Machine Learning murni yang memprioritaskan validitas metodologi eksperimen, bukan showcase software engineering atau framework orkestrasi AI.
2. **Lingkungan**: 100% lokal pada laptop tunggal (RAM 8 GB), tanpa dependensi API cloud atau gateway jaringan eksternal.
3. **Format Data**: Input berupa CSV tabular (dan notebook `.ipynb` untuk audit statis). Output utama berupa notebook baru 11-bagian, ledger JSONL, dan manifest reproduksibilitas.
4. **Perlindungan Berkas Sumber**: File data sumber bersifat *read-only*. Hash SHA-256 diperiksa sebelum dan sesudah eksekusi (0 toleransi mutasi).
5. **Uji Tersegel Satu Sentuhan**: Data uji tersegel dipisahkan secara fisik di awal (`sealed.bin` terenkripsi) dan hanya dievaluasi tepat satu kali untuk model final.
6. **Integritas Ledger & Angka**: Seluruh percobaan dicatat secara append-only di ledger. Teks laporan dilarang memuat angka tanpa rujukan ledger; agen LLM dilarang menulis angka langsung (R16).
7. **Perlakuan Input**: Seluruh isi dataset dan notebook diperlakukan sebagai data mentah tak tepercaya, bukan instruksi pengendali sistem.
8. **Visualisasi Jujur**: Tampilan antarmuka hanya memproyeksikan kejadian riil dari event log dan ledger.
9. **Prinsip Bukti Ilmiah**: Keunggulan sistem multi-agen tidak diasumsikan, melainkan harus dibuktikan lewat uji empiris dan algoritma keputusan `pilih_default`.
10. **Urutan Tahapan Ketat**: Gerbang M0a wajib lulus sebelum implementasi agen (M1), dan gerbang M0b wajib lulus sebelum enam agen dan Arena (M2).

---

## 2. Resolusi Pertanyaan Terbuka (Open Questions OQ-01 – OQ-08)

| ID | Pertanyaan | Status / Keputusan | Dasar Pertimbangan |
|---|---|---|---|
| **OQ-01** | Model bahasa lokal mana yang dipilih? | **Qwen 2.5 7B terpasang di host Ollama, dengan fallback teruji ke Qwen 2.5 3B / Llama 3.2 3B** | Model lokal kelas 3–4B Q4 dievaluasi pada Uji Kelayakan Model (4.3). Core memiliki failover deterministik mandiri. |
| **OQ-02** | Pustaka ML yang diizinkan? | **Scikit-Learn 1.9+, SciPy 1.18+, NumPy 2.5+, Pandas 3.0+** | LightGBM/XGBoost bersifat opsional jika terpasang dan dicatat di lockfile. |
| **OQ-03** | Pemilihan kasus emas dan defect tersemai? | **Dataset publik berlisensi terbuka (misal UCI Cleveland Heart Disease) + dataset sintetis berstruktur diketahui** | Mencegah kontaminasi pre-training LLM dan menjamin ground-truth 100% terkontrol. |
| **OQ-04** | Metode rentang perbandingan model? | **Terkunci [K]: Koreksi Nadeau-Bengio** | Telah diputuskan permanen di PRD 6.7; V3 berfungsi sebagai gerbang aktifasi klaim atau mode deskriptif. |
| **OQ-05** | Siapa penilai kedua (evaluator)? | **Protokol uji-ulang diri dengan jeda (self-retest) + pencocok otomatis berbasis aturan** | Sesuai PRD 11.4: status tetap Terukur-Pilot jika hanya ada satu penilai independen. |
| **OQ-06** | Anggaran waktu run pilot & M3? | **40 jam (pilot) dan 120 jam (M3) [A]** | Waktu median diukur secara empiris dari Pilot-0, bukan ditebak. |
| **OQ-07** | GPU / Deep Learning / Deret Waktu Penuh? | **Tidak didukung (di luar cakupan)** | Menjaga kepatuhan pada batasan laptop 8 GB RAM. |
| **OQ-08** | Peninjau independen arsitektur? | **Audit mandiri berbasis aturan ketat (Shadow Project Rules)** | Traceability matriks dan check deterministik memverifikasi setiap kriteria. |

---

## 3. Resolusi Konflik Dokumen Terdahulu (Lampiran C.2)

1. **Taksonomi Kebocoran**: Menggunakan taksonomi v2.1 (L1: Target derivation, L2: Duplicates, L3: Leakage outside folds, L5: Temporal, L6: Group, L8: ID/Index, L9: Post-event flags).
2. **Ambang Ketidakseimbangan Kelas**: Kelas minoritas $< 20\%$ dianggap tidak seimbang (menggunakan PR-AUC sebagai metrik utama).
3. **Metrik Utama Default**: Balanced Accuracy / Macro-F1 untuk klasifikasi seimbang; PR-AUC untuk tidak seimbang; MAE/RMSE untuk regresi.
4. **Toleransi Reproduksibilitas**: Identik 100% untuk model deterministik; selisih absolut $\le 0{,}001$ untuk model stokastik ber-seed.
5. **Klaim Keunggulan Model**: Batas bawah selisih Nadeau-Bengio wajib $\ge \delta$ (lebih konservatif daripada sekadar $> 0$).
6. **Pelarangan Bootstrap atas Lipatan**: Bootstrap atas lipatan dilarang keras karena lipatan saling berbagi data; Nadeau-Bengio terkunci permanen.
7. **Biaya Eksperimen**: Diukur dalam jam CPU lokal laptop, bukan USD API.
8. **Penyederhanaan Multi-Agen**: Peran Ahli Statistik dan sistem skor kredit reputasi dihapus; digantikan dengan pengujian empiris plasebo (KP).

---

## 4. Log Keputusan Formal (`docs/decision_log.md`)
Setiap amandemen atau modifikasi terhadap aturan baku [K] wajib dicatat di `docs/decision_log.md` mencakup: tanggal, aturan terkait, perubahan yang disetujui, dan alasan teknis.
