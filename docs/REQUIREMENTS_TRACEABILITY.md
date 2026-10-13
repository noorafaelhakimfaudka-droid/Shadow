# SHADOW — Matriks Penelusuran Persyaratan (Requirements Traceability Matrix)

Dokumen ini melacak seluruh persyaratan dari **SHADOW PRD Final v4.1** (`Shadow_PRD_Final.md`) ke implementasi kode, pengujian, dan bukti verifikasi.

## Disiplin Status Bukti
- `IMPLEMENTED`: Kode telah dibuat dan terstruktur.
- `TESTED`: Pengujian otomatis telah dijalankan dan lulus.
- `VERIFIED`: Kriteria penerimaan telah dipenuhi dengan bukti empiris (log, metrik, hash, artefak).
- `FAILED`: Gagal memenuhi kriteria penerimaan.
- `BLOCKED`: Terhalang oleh dependensi atau prasyarat yang belum terpenuhi.

---

## 1. Aturan Baku [K] (R1–R16)

| ID | Deskripsi Singkat | Status | Modul Implementasi | Uji Otomatis | Bukti / Catatan |
|---|---|:---:|---|---|---|
| R1 | Data nyata mengalahkan teori/AI | TERCATAT | `shadow_core/runner.py` | `tests/test_rules.py` | Eksekusi kode deterministik selalu diutamakan |
| R2 | Uji tersegel dipisah awal & disentuh 1x | TERCATAT | `shadow_core/splitting.py`, `vault_process.py` | `tests/test_vault.py` | File `sealed.bin` terenkripsi + flag `vault.spent` |
| R3 | Data berwaktu dilarang di-split acak | TERCATAT | `shadow_core/splitting.py` | `tests/test_splitting.py` | Deteksi kolom waktu memblokir k-fold acak |
| R4 | Fit-di-dalam-lipatan untuk semua transformasi | TERCATAT | `shadow_core/pipeline.py` | `tests/test_pipeline.py` | `ColumnTransformer` + fit hanya pada data train fold |
| R5 | Semua percobaan masuk ledger (termasuk gagal) | TERCATAT | `shadow_core/ledger.py` | `tests/test_ledger.py` | JSONL append-only |
| R6 | Angka di teks wajib berasal dari ledger | TERCATAT | `shadow_core/binder.py` | `tests/test_binder.py` | Token angka diverifikasi terhadap entri ledger |
| R7 | Perbandingan model wajib berpasangan + Nadeau-Bengio + δ | TERCATAT | `shadow_core/comparison.py` | `tests/test_comparison.py` | Koreksi resampling CV Nadeau-Bengio terkunci |
| R8 | Tidak ada klaim "bebas kebocoran" | TERCATAT | `shadow_core/checklist.py`, `notebook.py` | `tests/test_checklist.py` | Hanya menyatakan "tidak ada indikasi dari detektor X, Y, Z" |
| R9 | Sumber read-only, hash dicek sebelum & sesudah | TERCATAT | `shadow_core/manifest.py`, `runner.py` | `tests/test_manifest.py` | SHA-256 dicocokkan sebelum dan sesudah run |
| R10 | Isi data & notebook adalah data, bukan perintah | TERCATAT | `shadow_core/runner.py` | `tests/test_security.py` | Sanitasi input, mitigasi prompt injection |
| R11 | Pemeriksaan validitas hidup di Core (kode), bukan prompt | TERCATAT | `shadow_core/checklist.py` | `tests/test_checklist.py` | Check deterministik tidak dapat dilewati agen |
| R12 | Semuanya lokal (localhost saja) | TERCATAT | `shadow_agents/llm_local.py` | `tests/test_llm_local.py` | Request keluar localhost diblokir keras |
| R13 | Tampilan hanya menunjukkan kejadian nyata dari ledger/event | TERCATAT | `shadow_arena/app.py` | `tests/test_arena_backend.py` | Tidak ada animasi/angka palsu |
| R14 | Tidak ada klaim sebab-akibat dari korelasi | TERCATAT | `shadow_core/interpret.py` | `tests/test_interpret.py` | Penjelasan berlabel "penjelasan model, bukan sebab-akibat" |
| R15 | Rumus perbandingan model terkunci | TERCATAT | `shadow_core/comparison.py` | `tests/test_comparison.py` | Nadeau-Bengio permanen, V3 menentukan mode klaim |
| R16 | Agen tidak menulis angka bebas; klaim terstruktur | TERCATAT | `shadow_core/binder.py` | `tests/test_binder_structured.py` | Agen hanya memilih klaim, Core mengisi & memvalidasi angka |

---

## 2. Kebutuhan Fungsional (FR-01 – FR-24)

| ID | Deskripsi | Prio | Tahap | Modul Target | Status |
|---|---|:---:|:---:|---|:---:|
| FR-01 | Input CSV + validasi (encoding, delimiter, max 100k rows/50MB) | P0 | M0a | `shadow_core/audit.py`, `cli.py` | TERCATAT |
| FR-02 | Pelestarian sumber, salinan kerja, verifikasi SHA-256 | P0 | M0a | `shadow_core/manifest.py`, `runner.py` | TERCATAT |
| FR-03 | Framing S1: jenis tugas, aturan metrik 6.1, baseline wajib | P0 | M0a | `shadow_core/framing.py` | TERCATAT |
| FR-04 | Audit S2: struktur data, kualitas, validasi adversarial | P0 | M0a | `shadow_core/audit.py` | TERCATAT |
| FR-05 | Split S3a: pemisahan brankas tanpa target + counter disk 1x | P0 | M0a | `shadow_core/splitting.py`, `vault_process.py` | TERCATAT |
| FR-06 | Detektor kebocoran L1-L9 + kanari permutasi | P0 | M0a/b| `shadow_core/leakage/*`, `canary.py` | TERCATAT |
| FR-07 | Kit komponen pipeline fit-di-dalam-lipatan | P0 | M0a | `shadow_core/pipeline.py` | TERCATAT |
| FR-08 | Pemodelan S6: baseline + 3 kandidat pada fold identik | P0 | M0a | `shadow_core/modeling.py` | TERCATAT |
| FR-09 | Perbandingan berpasangan S7: Nadeau-Bengio + δ + 1-SE | P0 | M0a | `shadow_core/comparison.py` | TERCATAT |
| FR-10 | Evaluasi S8: brankas 1x, kalibrasi, error analysis, ketahanan | P0 | M0a/b| `shadow_core/evaluation.py` | TERCATAT |
| FR-11 | Ledger S5-S9: append-only JSONL | P0 | M0a | `shadow_core/ledger.py` | TERCATAT |
| FR-12 | Notebook 11 bagian + pengikat angka + re-eksekusi bersih | P0 | M0a | `shadow_core/notebook.py`, `binder.py` | TERCATAT |
| FR-13 | Manifest reproducibility + verifikasi run-ulang | P0 | M0a | `shadow_core/manifest.py`, `runner.py` | TERCATAT |
| FR-14 | Checklist validitas 6.13 + nilai penyerahan | P0 | M0a | `shadow_core/checklist.py` | TERCATAT |
| FR-15 | Sandbox berlapis: batas memori/waktu, no-network, monitor RAM | P0 | M0a | `shadow_core/runner.py` | TERCATAT |
| FR-16 | Harness evaluasi: golden cases, seed defects, V1-V10 | P0 | M0a/b| `shadow_eval/*` | TERCATAT |
| FR-17 | Klien LLM lokal (localhost) + Uji Kelayakan Model | P0 | M1 | `shadow_agents/llm_local.py` | TERCATAT |
| FR-18 | K1/K2 agen tunggal + failover ke Core deterministik | P0 | M1 | `shadow_agents/single_agent.py` | TERCATAT |
| FR-19 | Evaluator harness: matcher, adjudikasi, decide.py, run_plan.py | P0 | M1 | `shadow_eval/decide.py`, `matcher.py` | TERCATAT |
| FR-20 | Enam agen: K3, K6, KP (plasebo) + orchestrator | P0 | M2 | `shadow_agents/orchestrator.py` | TERCATAT |
| FR-21 | Streamlit Arena sederhana (9.3) + event reader + replay | P0 | M2 | `shadow_arena/app.py` | TERCATAT |
| FR-22 | Hapus data kasus (`shadow purge <run_id>`) | P0 | M1 | `shadow_core/cli.py` | TERCATAT |
| FR-23 | Audit notebook sumber (statis AST) + diff | P1 | M3 | `shadow_core/leakage/l3_static.py` | TERCATAT |
| FR-24 | Visual polish Arena, tema terang, UI bilingual | P2 | M3 | `shadow_arena/*` | TERCATAT |

---

## 3. Kriteria Penerimaan (AC-01 – AC-25)

| AC | Deskripsi Kriteria | Target Batas | Status |
|---|---|---|:---:|
| AC-01 | Sealed test split sebelum pemodelan; akses = 1x; akses ke-2 ditolak & status Ditolak | 100% kepatuhan | TERCATAT |
| AC-02 | Split sesuai struktur; split acak pada data waktu diblokir | 100% kepatuhan | TERCATAT |
| AC-03 | Lipatan identik untuk semua model (hash fold sama di ledger) | 100% identik | TERCATAT |
| AC-04 | Semua percobaan di ledger; 0 angka tanpa sumber di notebook | 100% terlacak | TERCATAT |
| AC-05 | Klaim "lebih baik" hanya jika CI lower ≥ δ; selain itu "setara" | 100% kepatuhan | TERCATAT |
| AC-06 | Kanari permutasi diuji pada final pipeline | Deteksi ≥ 90%, FP ≤ 10% | TERCATAT |
| AC-07 | Status & batas detektor dilaporkan; dilarang "bebas kebocoran" | 100% kepatuhan | TERCATAT |
| AC-08 | Baseline dummy + linear sederhana + ≥ 3 kandidat dievaluasi | 100% kepatuhan | TERCATAT |
| AC-09 | Hash file sumber sebelum vs sesudah identik 100% | 0 pelanggaran | TERCATAT |
| AC-10 | Eksekusi ulang notebook bersih cocok dalam toleransi (≤ 0.001) | 100% run lulus | TERCATAT |
| AC-11 | Pengikat angka: uji mutasi ≥ 100 varian diblokir | ≥ 99% diblokir | TERCATAT |
| AC-12 | Manifest lengkap mencatat hash, seed, lockfile, model | 100% valid skema | TERCATAT |
| AC-13 | Sandbox berlapis: ≥ 12 percobaan escape gagal | 12/12 gagal | TERCATAT |
| AC-14 | Uji memori 4.5 lulus laptop 8 GB (puncak ≤ 7.0 GB, Core ≤ 10m) | Puncak ≤ 7.0 GB | TERCATAT |
| AC-15 | Klien LLM menolak host selain localhost | 100% ditolak | TERCATAT |
| AC-16 | LLM mati di tengah run -> Core lanjut otomatis tanpa crash | 100% failover | TERCATAT |
| AC-17 | Ctrl+C berhenti ≤ 30 detik, status Parsial, tanpa notebook palsu | 100% graceful | TERCATAT |
| AC-18 | `shadow purge` membersihkan semua file kasus terverifikasi | 100% terhapus | TERCATAT |
| AC-19 | Prompt injection mitigasi: 0 tindakan berbahaya tanpa konfirmasi | 0 tindakan tak sah | TERCATAT |
| AC-20 | Enam agen memiliki artefak terstruktur; kritik via eksperimen | 100% terstruktur | TERCATAT |
| AC-21 | Streamlit Arena AR-01 s/d AR-05 lulus (sinkron ledger/event) | 100% sinkron | TERCATAT |
| AC-22 | `pilih_default` lulus 10 uji unit tabel 11.5 | 10/10 skenario | TERCATAT |
| AC-23 | Pilot 80 run selesai dengan protokol evaluator 11.4 | Lengkap terukur | TERCATAT |
| AC-24 | Sikofansi V11 diuji (menolak klaim tanpa bukti) | ≥ 90% ditolak | TERCATAT |
| AC-25 | Keputusan go/no-go pada set simpan ≥ 30 kasus | Tervalidasi | TERCATAT |

---

## 4. Rangkaian Validasi V1–V11

| Kode | Sasaran Uji | Ambang Target [T] | Status Verifikasi |
|---|---|---|:---:|
| V1 | Recall detektor defect tersemai (L1, L2, L3, L5, L6, L8) | ≥ 80% per jenis | TERCATAT |
| V2 | False Positive / blocker salah pada kasus bersih | ≤ 10% | TERCATAT |
| V3 | Kalibrasi rumus perbandingan: (a) A/A ≤ 10%, (b) Null ≤ 10%, (c) Coverage ≥ 90% | Memenuhi (a), (b), (c) | TERCATAT |
| V4 | Kanari permutasi (deteksi kebocoran prosedural & FP) | Deteksi ≥ 90%, FP ≤ 10% | TERCATAT |
| V5 | Pemulihan model benar pada data sintetis diketahui (linear vs nonlinear) | ≥ 80% pemulihan | TERCATAT |
| V6 | Re-eksekusi notebook bersih identik / Δ ≤ 0.001 | 100% cocok | TERCATAT |
| V7 | Pengikatan angka: (a) angka liar, (b) angka tertukar posisi, (c) FP | (a),(b) ≥ 99%, (c) 0 FP | TERCATAT |
| V8 | Integritas file sumber (hash identik) | 0 perubahan (100%) | TERCATAT |
| V9 | Keamanan brankas: akses ke-2 ditolak, tidak ada jalur intip memori, split tanpa target | 100% aman | TERCATAT |
| V10 | Data ekstrem: 99% missing, 1 sampel minoritas, collinear sempurna, 0 baris | Error tertangani, no crash | TERCATAT |
| V11 | Uji sikofansi agen reviewer (menolak klaim halusinasi buatan) | ≥ 90% penolakan | TERCATAT |
