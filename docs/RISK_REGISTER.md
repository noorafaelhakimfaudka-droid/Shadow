# SHADOW — Daftar Risiko dan Mitigasi (Risk Register)

Dokumen ini memetakan seluruh risiko teknis, metodologis, dan operasional (RK-01 s/d RK-21) dari **SHADOW PRD Final v4.1**, beserta tingkat keparahan, probabilitas, mekanisme mitigasi, dan rencana verifikasinya.

---

## Matriks Risiko Utama (RK-01 – RK-21)

| ID | Deskripsi Risiko | Keparahan | Probabilitas | Mekanisme Mitigasi | Uji Verifikasi |
|---|---|:---:|:---:|---|---|
| **RK-01** | Kebocoran data lolos dari detektor menyebabkan metrik terlalu optimis | Kritis | Sedang | Pertahanan berlapis L1–L9, kanari permutasi label acak, brankas uji tersegel, pelaporan batas detektor secara transparan | V1 (recall ≥ 80%), V4 (kanari deteksi ≥ 90%) |
| **RK-02** | Bias seleksi akibat mencoba terlalu banyak konfigurasi (data dredging) | Tinggi | Tinggi | Menghitung & mencatat jumlah percobaan di ledger; peringatan otomatis bila percobaan > 50; estimasi performa akhir HANYA dari uji tersegel | Uji unit checklist (peringatan seleksi), V8 |
| **RK-03** | Uji tersegel berukuran kecil (< 500 baris / minoritas < 50) menghasilkan estimasi tidak stabil | Sedang | Sedang | Penanganan khusus data kecil: gunakan nested CV (luar estimasi, dalam tuning), tampilkan peringatan "tidak kuat secara statistik" | Uji kasus data kecil di `tests/test_splitting.py` |
| **RK-04** | Aturan perbandingan model 6.7 terlalu ketat atau longgar | Tinggi | Rendah | Mengunci rumus Nadeau-Bengio; validasi empiris V3 (A/A test, null test, coverage test); jika gagal masuk ke mode deskriptif terkunci | V3 (a, b, c) kalibrasi rumus |
| **RK-05** | LLM lokal mengarang angka atau klaim palsu (halusinasi) | Kritis | Tinggi | Agen dilarang menulis angka (R16); agen hanya memilih klaim terstruktur; Core memvalidasi subjek/metrik/split dan mengisi angka riil dari ledger | V7 mutasi angka ≥ 99% diblokir |
| **RK-06** | JSON dari model LLM kecil (3–4B) rusak atau tidak sesuai skema | Tinggi | Tinggi | Skema JSON ketat, mekanisme retry maksimal 2x, failover otomatis ke Core deterministik tanpa menghentikan eksperimen | Uji `tests/test_contracts.py` & failover |
| **RK-07** | Kehabisan RAM pada laptop 8 GB menyebabkan crash atau thrashing swap | Kritis | Sedang | Batasan 1 model dimuat dalam satu waktu; pemantau `psutil` (peringatan jika RAM > 90% selama 10 detik); `HistGradientBoosting` hemat memori; limit data 100k baris / 50MB | Uji Anggaran Memori 4.5 (AC-14) |
| **RK-08** | Waktu eksekusi CPU terlalu lama | Tinggi | Sedang | Batas waktu 10 menit untuk Core, 45 menit untuk agen; tuning dibatasi 30 iterasi (mode cepat 8); `n_jobs=2` | Uji benchmarking waktu di `shadow_eval/run_plan.py` |
| **RK-09** | Arsitektur multi-agen menambah waktu tanpa nilai tambah nyata | Sedang | Tinggi | Konfigurasi pembanding: K0 (Core-only), K1, K2, K3, K6, dan KP (plasebo kritik acak); fungsi keputusan formal `pilih_default` | Uji unit `pilih_default` (10 skenario tabel 11.5) |
| **RK-10** | Agen Skeptic/Reviewer menyetujui klaim salah (sikofansi LLM) | Tinggi | Sedang | Reviewer dievaluasi dengan prompt independen tanpa riwayat builder; checklist deterministik tidak dapat dihapus LLM; uji injeksi sikofansi V11 | V11 (penolakan klaim palsu ≥ 90%) |
| **RK-11** | Prompt injection melalui dataset atau teks notebook | Tinggi | Sedang | Seluruh input diperlakukan sebagai data (bukan instruksi); pembersihan tag; tindakan sistem memerlukan otorisasi Core | Uji injeksi AC-19 (≥ 10 payload dasar di M1, ≥ 50 di M2) |
| **RK-12** | Penilaian evaluator terhadap agen bersifat subjektif | Sedang | Sedang | Pencocok otomatis (`matcher.py`), tinjauan ganda tersamar (blind), perhitungan persetujuan inter-rater per kategori defect | `tests/test_matcher.py` & protokol 11.4 |
| **RK-13** | Kontaminasi data pada pre-training LLM lokal | Sedang | Sedang | Menggunakan dataset sintetis buatan baru dan modifikasi skema untuk evaluasi golden cases | Kasus sintetis di `shadow_eval/cases/` |
| **RK-14** | Ketidakpastian nondeterministik merusak reproduksibilitas | Kritis | Sedang | Mengunci seed global (Python, NumPy, Scikit-Learn); mengunci versi paket di `requirements.lock`; re-eksekusi bersih diverifikasi otomatis | V6 (100% run ulang cocok ≤ 0.001) |
| **RK-15** | Ruang lingkup terlalu besar untuk kapasitas waktu pengerjaan | Tinggi | Tinggi | Pembagian bertahap M0a (Lite/Safety) dan M0b (Full); uji `test_no_premature_code` mencegah loncat tahapan; rilis M0a sudah merupakan produk utuh | Gerbang M0a, M0b, M1, M2 |
| **RK-16** | Analisis statis AST pada notebook luar tidak menangkap semua kebocoran | Sedang | Rendah | Mengakui batasan secara eksplisit di laporan bahwa analisis AST bersifat best-effort | Dokumentasi batasan di `docs/LIMITATIONS.md` |
| **RK-17** | Brankas bocor melalui pemilih split atau inspeksi memori | Kritis | Rendah | S3a: pemilih split tidak membaca target; pemisahan fisik `dev.parquet` & `sealed.bin` terenkripsi; proses terpisah `vault_process.py`; verifikasi V9 (a-d) | V9 (100% uji isolasi brankas) |
| **RK-18** | Rumus Nadeau-Bengio bias pada skema CV berulang | Sedang | Rendah | Pengujian kalibrasi V3; jika coverage < 90% atau false positive > 10%, sistem masuk mode deskriptif tanpa mengklaim "lebih baik" | V3 kalibrasi Nadeau-Bengio |
| **RK-19** | Agen menarik kesimpulan keliru dari angka yang benar | Tinggi | Sedang | Klaim terstruktur dengan template terkunci; Skeptic memvalidasi konsistensi kesimpulan terhadap tabel perbandingan | Evaluasi Skeptic & laporan keterbatasan |
| **RK-20** | Pemakaian memori di laptop riil melebihi estimasi teoritis | Kritis | Sedang | Eksekusi protokol Uji Anggaran Memori 4.5 dengan pemantauan psutil; implementasi tangga penyelamat berjenjang | Pengukuran empiris psutil saat run skenario terburuk |
| **RK-21** | Ruang lingkup membengkak melampaui PRD | Sedang | Sedang | Penjaga ruang lingkup 12.0; aturan gerbang dokumen formal; penolakan fitur tanpa kaitan ke V1–V11 | Uji `test_no_premature_code` |

---

## Protokol Uji Anggaran Memori & Tangga Penyelamat (PRD 4.5)

### Batasan Memori Target (Laptop 8 GB RAM):
- OS & aplikasi latar: ~2.5 GB
- Model LLM lokal (Q4, 3–4B): ≤ 3.5 GB
- Python, Scikit-Learn, Streamlit: ≤ 2.0 GB
- **Batas Aman Puncak Sistem**: ≤ 7.0 GB RAM (menghindari thrashing paging file / swap).

### Tangga Penyelamat (Rescue Ladder) jika Memori > 7.0 GB:
1. Kurangi panjang konteks LLM: 4096 token → 2048 token.
2. Turunkan paralelisme Scikit-Learn: `n_jobs = 2` → `n_jobs = 1`.
3. Kurangi jumlah estimator Random Forest: 200 pohon → 100 pohon.
4. Turunkan batas baris data: 100.000 baris → 50.000 baris.
5. Ganti model LLM ke parameter lebih kecil: kelas 3-4B → ≤ 2B (atau jalankan Core-only).
6. Tunda pembukaan Streamlit Arena hingga proses eksperimen selesai penuh.
Semua aktivasi tangga penyelamat wajib dicatat secara transparan di `docs/decision_log.md` dan `manifest.json`.
