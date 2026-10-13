# SHADOW — Arsitektur Sistem v4.1

## 1. Filosofi dan Prinsip Arsitektur
SHADOW adalah sistem laboratorium eksperimen Data Science dan Machine Learning lokal yang mengutamakan **kejujuran ilmiah, auditabilitas penuh, dan pencegahan kebocoran data**.

### Prinsip Inti:
1. **Pemisahan Deterministik vs AI**: Fondasi eksekusi, metrik, validasi, dan brankas dijalankan secara 100% deterministik oleh Python (Shadow Core). Agen LLM hanya bertindak sebagai pengusul konfigurasi dan peninjau kritis di dalam koridor kit komponen yang valid.
2. **Kedaulatan Bukti Ledger**: Teks narasi, grafik, dan tabel tidak boleh memuat angka yang tidak tercatat di ledger. Angka tidak dikarang oleh LLM.
3. **Isolasi Fisik Data Tersegel (Sealed Test Vault)**: Data uji tersegel dipisahkan secara fisik dan dienkripsi ke `sealed.bin`. Proses pemodelan utama tidak pernah memuat data tersegel ke memori. Pembacaan hanya dilakukan satu kali oleh proses terpisah (`vault_process.py`), yang mencatat tanda habis pakai permanen di disk (`vault.spent`).

---

## 2. Tiga Lapisan Arsitektur (3-Tier Layered Design)

```
┌─────────────────────────────────────────────────────────────┐
│                    L3 — Streamlit Arena                     │
│  (UI Reaktif: Membaca events.jsonl & ledger.jsonl saja)    │
└──────────────────────────────▲──────────────────────────────┘
                               │
┌──────────────────────────────┴──────────────────────────────┐
│                    L2 — Enam Agen LLM                       │
│  (Auditor, Architect, Feature, Engineer, Robustness, Skeptic│
│   Hanya mengeluarkan JSON terstruktur & klaim tanpa digit)  │
└──────────────────────────────▲──────────────────────────────┘
                               │
┌──────────────────────────────┴──────────────────────────────┐
│                    L1 — Shadow Core (M0)                    │
│  S1 Framing → S3a Vault Split → S2 Audit → S3b Validation   │
│  → S4 Leakage Defense → S5 Pipeline Kit → S6 Experiment     │
│  → S7 Nadeau-Bengio Comparison → S8 Vault Eval → S9 Report  │
│  [Ledger JSONL] [Events JSONL] [Binder] [Checklist Engine] │
└─────────────────────────────────────────────────────────────┘
```

### L1 — Shadow Core (Deterministik)
- **Komponen**: S1–S9, detektor kebocoran L1–L9, kanari permutasi, pengikat angka (binder), ledger JSONL, manifest, runner sandbox, checklist validitas.
- **Tanggung Jawab**: Menjamin validitas metodologis, mencegah leakage, mencatat segala aktivitas secara append-only, merakit notebook 11-bagian, dan melakukan verifikasi re-eksekusi bersih.
- **Independensi**: Mampu berjalan penuh secara mandiri lewat CLI (`shadow run --mode core`) tanpa memerlukan LLM maupun UI.

### L2 — Enam Agen LLM (Lokal Penuh)
- **Komponen**:
  1. *Data Auditor*: Meninjau audit dan temuan detektor.
  2. *Model Architect*: Menyusun rencana eksperimen berpasangan (dikunci hash).
  3. *Feature Strategist*: Mengonfigurasi pra-pemrosesan dari kit komponen.
  4. *Experiment Engineer*: Memandu tuning dan eksplorasi kandidat.
  5. *Robustness Investigator*: Memeriksa stres data, kanari, dan stabilitas seed.
  6. *Skeptic / Reviewer*: Menantang klaim berdasar ledger dan daftar periksa independen.
- **Batasan Ketat (R16)**: Agen LLM **dilarang menulis angka langsung**. Agen hanya menghasilkan klaim terstruktur `{subjek, metrik, split, kunci_ledger, templat}`. Core memvalidasi kecocokan subjek/metrik/split sebelum merender angka ke laporan.
- **Failover**: Jika LLM offline atau output rusak > 2x, Core melanjutkan sendiri secara deterministik.

### L3 — Streamlit Arena (Visualisasi Jujur)
- **Komponen**: Forest plot perbandingan model, radar kebocoran L1–L9, peta validasi split/vault, drawer bukti klik-ke-ledger, replay kontrol, indikator segel reproduksibilitas.
- **Integritas**: State UI hanya dibaca dari `events.jsonl` dan `ledger.jsonl`. Tidak ada gerak atau angka dummy yang tidak didukung data riil.

---

## 3. Alur Eksekusi End-to-End (S1–S9)

```
[CSV Sumber] (Read-Only, SHA-256 dicatat)
     │
     ▼
[S1: Framing] ───────────► Deteksi tugas (klasifikasi/regresi), metrik utama & pendamping, margin praktis δ
     │
     ▼
[S3a: Pemisahan Brankas] ──► Deteksi kolom grup/waktu TANPA membaca target.
     │                    Pisahkan: dev.parquet (80%) & sealed.bin (20%, terenkripsi).
     │
     ▼
[S2: Audit Data] ────────► Hanya membaca dev.parquet. Deteksi missing, anomali, duplikat, adversarial validation.
     │
     ▼
[S3b: Strategi Validasi] ─► RepeatedStratifiedKFold (5x3) / TimeSeriesSplit / GroupKFold pada dev.parquet.
     │
     ▼
[S4: Pertahanan Leakage] ─► Uji L1, L2, L5, L6, L8, L9 + Kanari permutasi (label diacak).
     │
     ▼
[S5: Pipeline Fit-in-Fold]► Kit komponen ColumnTransformer: Imputer, Scaler, Encoder, Selector fit HANYA di train fold.
     │
     ▼
[S6: Eksperimen Terkontrol]► Baseline Dummy + Linear + RF + HistGB. Evaluasi pada lipatan identik.
     │
     ▼
[S7: Perbandingan Model] ─► Nadeau-Bengio Corrected Resampled t-test pada selisih berpasangan. Aturan 1-SE.
     │
     ▼
[S8: Evaluasi Akhir] ────► Model final dikirim ke VaultProcess (1x akses). Hitung metrik uji tersegel.
     │                    Tulis marker disk vault.spent.
     │
     ▼
[S9: Interpretasi & Doc] ─► Permutation importance (held-out), perakitan notebook 11-bagian, checklist grading.
     │
     ▼
[Verifikasi Re-eksekusi] ─► Eksekusi ulang notebook di sandbox bersih. Cocokkan metrik terhadap ledger.
```

---

## 4. Arsitektur Isolasi Brankas Uji Tersegel (Sealed Test Vault)

```
┌────────────────────────────────────────────────────────┐
│                   Proses Utama SHADOW                  │
│  - Memegang dev.parquet (hanya data latihan/validasi) │
│  - Memegang sealed_indices_sha256 (hanya hash)        │
│  - TIDAK MEMILIKI KUNCI atau DATA sealed.bin           │
│  - Mengirim serialisasi model ke VaultClient           │
└───────────────────────────┬────────────────────────────┘
                            │ (Subprocess IPC terisolasi)
┌───────────────────────────▼────────────────────────────┐
│              Proses Brankas (vault_process.py)         │
│  - Memeriksa file disk: runs/<id>/vault.spent          │
│    (Jika ada -> Lempar SealedAccessError, skor Ditolak)│
│  - Memuat vault.key dan membuka sealed.bin             │
│  - Menghitung metrik evaluasi + bootstrap 95% CI       │
│  - Menulis flag disk permanen: vault.spent             │
│  - Mengembalikan HANYA ringkasan numerik ke proses     │
│    utama (tidak ada baris data tersegel yang bocor)   │
└────────────────────────────────────────────────────────┘
```

- **Penyimpanan**: `dev.parquet` disimpan terbuka untuk pemodelan; `sealed.bin` dienkripsi dengan kunci AES-256 / Fernet lokal yang disimpan terpisah.
- **Single-Use Enforcement**: File `vault.spent` ditulis di disk sistem. Meskipun proses utama di-restart atau crash, brankas tetap tidak bisa dibuka untuk kedua kalinya dalam run yang sama.
- **Model Ancaman**: Mencegah kebocoran tidak sengaja dari pipeline, bug kode, atau prompt agen LLM.

---

## 5. Arsitektur Pengikat Angka & Klaim Terstruktur (Binder Engine)

```
[Agen LLM / Narasi] ──► Menghasilkan Klaim Terstruktur:
                        {
                          "subjek": "HistGB",
                          "metrik": "pr_auc",
                          "split": "cv_test",
                          "kunci_ledger": "trial:t-004",
                          "templat": "Model {{subjek}} mencapai PR-AUC {{nilai}}."
                        }
                             │
                             ▼
[Binder Validator] ────► 1. Cek apakah trial:t-004 ada di ledger?
                         2. Cek apakah subjek == HistGB?
                         3. Cek apakah metrik == pr_auc?
                         4. Render {{nilai}} dari ledger.summary.mean
                             │
                             ▼
[Pemeriksa Bebas Angka] ─► Verifikasi bahwa teks akhir TIDAK memuat angka bebas (digit atau kata-angka)
                         yang tidak terdaftar dalam whitelist atau klaim ledger.
```

---

## 6. Arsitektur Sandbox Berlapis (Layered Sandboxing)
1. **Subprocess Execution**: Kode notebook dan script dijalankan di subproses mandiri dengan direktori kerja terisolasi di dalam `runs/<run_id>/`.
2. **Resource Throttling**:
   - `psutil` pemantau memori latar belakang: Jika penggunaan RAM > 90% selama 10 detik, proses dihentikan dengan status `gagal: memori`.
   - Timeout ketat: 300 detik per cell notebook, maksimal 10 menit total untuk Shadow Core.
3. **No-Network Guard**: Soket keluar dinonaktifkan atau diverifikasi gagal; request HTTP luar ditolak. Panggilan LLM dibatasi secara eksplisit ke `127.0.0.1:11434`.
4. **AST Static Checker**: Kode notebook dianalisis sebelum eksekusi untuk mendeteksi pemanggilan modul berbahaya (`os.system`, `subprocess`, socket, token injection).
