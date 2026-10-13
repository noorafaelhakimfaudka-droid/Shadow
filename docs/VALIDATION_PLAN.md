# SHADOW — Rencana Validasi Komprehensif (Validation Plan V1–V11)

Dokumen ini mendefinisikan metodologi verifikasi empiris, rumus statistik, ambang penerimaan [T], dan protokol pengujian untuk seluruh kriteria **V1 s/d V11** sesuai **SHADOW PRD Final v4.1**.

---

## 1. Rumus Interval Kepercayaan Wilson 95%
Untuk seluruh pengujian proporsi binomial (recall, false positive rate, coverage, deteksi mutasi), interval ketidakpastian wajib dihitung menggunakan rumus **Wilson Score Interval 95%**:

$$CI_{Wilson} = \frac{p + \frac{z^2}{2n} \pm z \sqrt{\frac{p(1-p)}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$

di mana:
- $p = \frac{k}{n}$ (proporsi keberhasilan dari $n$ sampel pengujian)
- $z = 1{,}95996$ (kuantil normal baku untuk tingkat kepercayaan 95%)

**Aturan Pelaporan:**
Suatu target proporsi (misalnya $\ge 80\%$) hanya dinyatakan **Tervalidasi [V]** apabila titik estimasi DAN batas bawah (*lower bound*) memenuhi kriteria penerimaan yang ditetapkan. Jika ukuran sampel kecil ($n < 30$), status dibatasi menjadi **Terukur-Pilot**.

---

## 2. Rincian Metodologi Pengujian V1–V11

### V1 — Recall Detektor Kebocoran Data (L1, L2, L3, L5, L6, L8)
- **Tujuan**: Memastikan detektor kebocoran mendeteksi defect yang sengaja ditanam pada dataset sintetis dan realistis.
- **Katalog Uji**: Minimal 8 kasus uji tersemai per jenis kebocoran (L1, L2, L3, L5, L6, L8).
  - *L1*: Fitur salinan target dengan derau sangat kecil ($R^2 > 0.99$ atau $AUC > 0.99$).
  - *L2*: Duplikasi baris lintas split train dan test (persis & fuzzy kuantisasi).
  - *L3*: Analisis AST mendeteksi pemanggilan `.fit()` atau `.fit_transform()` sebelum pembagian data.
  - *L5*: Data berwaktu (kolom tanggal/timestamp) yang di-split secara acak tanpa embargo.
  - *L6*: Data ber-kelompok (ID entitas berulang) yang bocor ke fold uji.
  - *L8*: Kolom ID/indeks atau urutan baris yang memiliki korelasi buatan dengan target.
- **Target [T]**: Recall $\ge 80\%$ per jenis kebocoran.
- **Toleransi**: Detektor yang memiliki recall rendah wajib dilaporkan secara terbuka di bagian Keterbatasan Sistem (`docs/LIMITATIONS.md`).

### V2 — False Positive Rate pada Kasus Bersih
- **Tujuan**: Memastikan detektor tidak menghasilkan alarm palsu (blocker yang salah) pada dataset bersih tanpa kebocoran.
- **Dataset**: Minimal 20 dataset kontrol bersih (dari berbagai domain: tabular biner, multikelas, dan regresi).
- **Target [T]**: Blocker palsu $\le 10\%$ (batas atas Wilson 95%).

### V3 — Kalibrasi Rumus Perbandingan Model (Nadeau-Bengio)
- **Tujuan**: Membuktikan bahwa rumus selisih berpasangan terkoreksi Nadeau-Bengio valid pada skema CV 5x3 ($J=15$ lipatan) dan tidak menghasilkan klaim palsu.
- **Tiga Skenario Pengujian**:
  1. **(a) Uji A/A**: Dua model identik (algoritma sama, arsitektur sama) hanya berbeda seed pelatihan. Target: Klaim salah "A lebih baik dari B" $\le 10\%$ (batas atas Wilson 95% dari $\ge 200$ simulasi).
  2. **(b) Uji Null**: Label target diacak sepenuhnya ($y \sim \text{Permute}(y)$) dengan dua algoritma berbeda (misal Ridge vs Random Forest). Target: Klaim keunggulan palsu $\le 10\%$.
  3. **(c) Uji Cakupan (Coverage)**: Data sintetis di mana selisih performa sejati $\Delta^*$ diketahui. Hitung seberapa sering interval 95% memuat nilai $\Delta^*$. Target: Cakupan $\ge 90\%$ dari $\ge 200$ simulasi lintas 3 ukuran data (300, 2.000, 20.000 baris).
- **Penanganan Kegagalan**: Jika V3 gagal, Shadow otomatis mengunci sistem ke **Mode Deskriptif**: melaporkan selisih dan rentang tanpa kata "lebih baik", dan pemilihan model mengandalkan aturan 1-SE.

### V4 — Deteksi Kanari Permutasi
- **Tujuan**: Memverifikasi kemampuan kanari mendeteksi kebocoran prosedural (label leakage di dalam pipeline transformasi).
- **Metode**: Eksekusi pipeline penuh dengan label yang diacak ($\ge 5$ permutasi). Ambang alarm: skor kanari $> \text{chance} + \max(0.05, 3 \times SE)$.
- **Dataset**: $\ge 30$ kasus pipeline bocor prosedural + $\ge 30$ pipeline bersih.
- **Target [T]**: Deteksi pipeline bocor $\ge 90\%$; False Alarm $\le 10\%$.

### V5 — Pemulihan Model yang Benar (Model Recovery)
- **Tujuan**: Menguji apakah sistem memilih model yang sesuai dengan struktur data yang diketahui.
  - Data yang dibangkitkan dari relasi linear murni: model linear tidak kalah dari model kompleks.
  - Data dengan interaksi non-linear kuat: model ensemble/tree secara konsisten mengungguli model linear.
- **Dataset**: $\ge 30$ dataset sintetis berstruktur terkontrol.
- **Target [T]**: Pemulihan keputusan tepat $\ge 80\%$.

### V6 — Reproduksibilitas Run-Ulang Bersih
- **Tujuan**: Memastikan notebook hasil Shadow dapat dieksekusi ulang dari nol di lingkungan bersih dan menghasilkan metrik yang sama.
- **Kriteria Toleransi**:
  - Model deterministik: Nilai metrik **identik 100%**.
  - Model stokastik (dengan seed tetap): Selisih absolut metrik $\le 0{,}001$.
- **Target [T]**: $100\%$ run Core berhasil diverifikasi ulang.

### V7 — Pengikat Angka dan Klaim Terstruktur (Binder)
- **Tujuan**: Memastikan teks narasi bebas dari angka karangan (halusinasi) dan angka tertukar posisi.
- **Suite Uji Mutasi**:
  1. *Sub-tes (a)*: $\ge 100$ varian teks dengan angka liar yang tidak ada di ledger. Target: $\ge 99\%$ diblokir.
  2. *Sub-tes (b)*: $\ge 100$ varian teks dengan angka sah tetapi tertukar subjek, metrik, split, atau fold. Target: $\ge 99\%$ diblokir via template & structured claim checker.
  3. *Sub-tes (c)*: $\ge 100$ laporan sah yang dihasilkan Core. Target: $0$ alarm palsu ($0\%$).

### V8 — Integritas Berkas Sumber
- **Tujuan**: Memastikan file CSV sumber tidak pernah dimodifikasi selama dan setelah eksekusi.
- **Metode**: Hash SHA-256 dihitung sebelum pipeline berjalan dan sesudah pipeline selesai (pada kondisi sukses, gagal, maupun dihentikan).
- **Target [K]**: $100\%$ kecocokan hash ($0$ mutasi).

### V9 — Keamanan Brankas Uji Tersegel (Vault Isolation)
- **Empat Pilar Uji**:
  1. *(a) Single-Use Enforcement*: Pemanggilan kedua ke brankas dalam run yang sama wajib gagal (`SealedAccessError`), bahkan jika proses Python di-restart (tanda `runs/<run_id>/vault.spent` ada di disk). Nilai run langsung diturunkan ke `Ditolak`.
  2. *(b) Memory & Process Isolation*: Proses utama, agen, dan pipeline diverifikasi tidak memiliki pointer atau variabel data tersegel di RAM (pemindaian fingerprint baris tersegel di heap/locals proses utama menghasilkan 0 temuan).
  3. *(c) Target-Blind Splitting*: Pemilih strategi split tidak membaca target (mengacak urutan label target menghasilkan partisi split dan pemilihan strategi yang identik).
  4. *(d) Dev-Only Audit*: Modul audit S2 hanya menerima `dev.parquet` dan menolak jika diberikan akses ke brankas.
- **Target [K]**: $100\%$ kepatuhan pada keempat pilar ($\ge 50$ percobaan akses ganda + $\ge 20$ skenario jalur baca).

### V10 — Ketahanan pada Kondisi Data Ekstrem
- **Tujuan**: Memastikan Shadow menangani kasus tepi ekstrem secara elegan tanpa crash yang tidak terkontrol.
- **Skenario Uji**:
  1. Data dengan 99% nilai hilang (*missing values*).
  2. Data dengan hanya 1 sampel kelas minoritas.
  3. Data dengan multikolinearitas sempurna ($r = 1.0$).
  4. Data dengan hanya 1 kolom fitur.
  5. Dataset kosong (0 baris).
  6. Dataset dengan semua nilai konstan.
- **Target [T]**: Menghasilkan error terstruktur dengan pesan diagnostik yang jelas; tidak ada *unhandled exception* atau *process hang*.

### V11 — Uji Sikofansi Reviewer LLM
- **Tujuan**: Menguji apakah agen Skeptic/Reviewer mampu menolak klaim palsu yang disuntikkan secara sengaja.
- **Metode**: $\ge 30$ klaim palsu tanpa bukti di ledger disuntikkan ke dokumen peninjauan.
- **Target [T]**: Agen menolak klaim tidak berdasar $\ge 90\%$ (batas bawah Wilson $\ge 75\%$).

---

## 3. Rencana Gerbang Kelulusan Tahap (Stage Gates)

```
[M0a: Inti Keselamatan] ──► Gate M0a: V1(L1,L2,L5,L6), V2, V3, V4, V6, V7, V8, V9, V10
         │ (Lulus & Dokumen docs/gates/M0a.md lengkap)
         ▼
[M0b: Pelengkap Core]   ──► Gate M0b: V1(L8,L9,L3-AST), V5, Full Resilience
         │ (Lulus & Dokumen docs/gates/M0b.md lengkap)
         ▼
[M1: Agen Tunggal]      ──► Gate M1: T1.1-T1.9, LLM Local Host Guard, Mutasi Posisi V7 >= 99%,
         │                  Unit test pilih_default (10 skenario), Pilot-0a
         ▼
[M2: Enam Agen & Arena] ──► Gate M2: K3, K6, KP (plasebo), Streamlit Arena AR-01..05,
         │                  Pilot 80 run, Keputusan pilih_default indikatif
         ▼
[M3: Validasi Empiris]  ──► Gate M3: Benchmark Set Simpanan (>= 30 kasus), Sikofansi V11
```
Semua tahapan wajib melewati gerbang masing-masing sebelum melangkah ke tahapan berikutnya.
