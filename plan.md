# nu-paddle — Plan & Dokumentasi

> **Arsitektur:** NuExtract3-GGUF sebagai **model utama** (ekstraksi JSON
> terstruktur) dibantu **PaddleOCR** sebagai **OCR pendukung** (sumber teks +
> confidence), dengan **preprocessing** yang diambil dari project
> `AI-Document/preprocessing` agar setiap dokumen melewati perbaikan gambar
> yang layak (quality check → deskew [off] → denoise → contrast → sharpen →
> resize).
>
> Project ini adalah **gabungan** dari tiga project lama:
> `doc-validation` (NuExtract3-GGUF + validasi), `paddleocr`
> (PaddleOCR detect-then-recognize), dan `AI-Document` (preprocessing +
> pola arsitektur app/registry/selection) — hanya bagian yang benar-benar
> dipakai yang dibawa ke sini.

---

## 1. Tujuan

1. Menjaga **NuExtract3-GGUF sebagai ekstraktor utama** (hasil akhir JSON
   terstruktur, sudah terbukti akurat pada field numerik/tanggal).
2. **PaddleOCR membantu** memberi teks konteks yang **bersih** agar NuExtract
   tidak meleset pada field nama/perusahaan.
3. **Semua dokumen wajib lewat preprocessing yang layak** (dari AI-Document)
   supaya tidak ada dokumen yang terlewat/miss.
4. Fase-1: **batch extraction semua dokumen contoh** untuk melihat hasil
   sebelum membangun UI.

---

## 2. Struktur Folder

```
nu-paddle/
├── .gitignore
├── plan.md                     # dokumen ini
├── AGENT.md                    # aturan kerja untuk agent/coder
├── requirements.txt
├── config.py                   # path model, threshold, pemetaan jenis dokumen
├── run_batch.py                # ENTRY POINT fase-1 (batch)
├── preprocessing/              # SALINAN dari AI-Document/preprocessing
│   └── pipeline.py             #   preprocess() wajib untuk semua dokumen
├── models/                     # pola AI-Document/models/
│   ├── base.py                 #   ModelResult + BaseExtractionModel
│   ├── paddleocr_model.py      #   PaddleOCR (role="support") + run_region/run_header
│   ├── layout_model.py         #   PP-DocLayoutV3 (deteksi zona layout, offline)
│   ├── nuextract_gguf_model.py #   NuExtract3-GGUF (role="main")
│   └── registry.py             #   AVAILABLE_MODELS
├── extraction/
│   ├── nuextract3_chat_template.jinja   # template wajib NuExtract
│   └── schemas.py              #   schema + instruksi per jenis dokumen
├── selection/
│   └── selector.py             #   kapan teks OCR ikut dikirim ke NuExtract
├── validation.py               # normalisasi + validasi PASSED/FAILED
├── data/models/nuextract3/     # GGUF Q4_K_M (2.6G) + mmproj (645M) [disalin]
├── contoh invoice/             # 8 dokumen contoh (2 vendor)
└── results/                    # output JSON per dokumen [tidak di-commit]
```

---

## 3. Alur Per Dokumen (fase-1, `run_batch.py`)

```text
dokumen (PDF/gambar)
   │
   ▼
1. preprocess() [AI-Document pipeline]
   pdf_to_image -> quality_check -> [deskew OFF] -> denoise
   -> contrast (CLAHE) -> sharpen -> resize      -> gambar PNG bersih
   │
   ▼
2. PaddleOCR (support) — detect-then-recognize + CLAHE + spatial sort
   -> baris teks + confidence + bbox
   +
   OCR per-layout (PP-DocLayoutV3, model lokal/offline):
   - band header/logo: run_header() crop 22% atas + upscale 2x (nama perusahaan
     di dalam logo — sengaja TIDAK pakai bbox zona karena teks logo melewati
     batas zona header_image/header, wide-band terbukti lebih akurat)
   - zona table: PP-DocLayoutV3 -> crop bbox zona + upscale 1.8x (cap 2400px)
     untuk angka kecil di tabel
   Model layout dimuat per kebutuhan lalu langsung di-unload (hemat RAM).
   │
   ▼
3. selector.evaluate() — TANPA aturan pemblokiran:
   teks kosong  -> teks OCR dilewati
   selebihnya   -> teks OCR disertakan
   │
   ▼
4. NuExtract3-GGUF (main)
   gambar (di-fit <=900k px) + teks OCR gabungan ([ZONA: header], [ZONA: table]) + schema
   -> JSON terstruktur (field)
   │
   ▼
5. koreksi dari OCR: `correct_codes_from_ocr()` (field kode/nomor + NPWP dari
   baris ber-conf tinggi) -> normalisasi (tanggal ISO, uang angka) -> validasi
   (field wajib, angka positif, tanggal valid) -> PASSED/FAILED
   │
   ▼
6. simpan results/<jenis>_<nama>.json + tabel
```

---

## 4. Keputusan Desain & Temuan

| # | Keputusan | Alasan |
|---|-----------|--------|
| 1 | **Deskew DINONAKTIFKAN** (`USE_DESKEW=False`) | Temuan kritis fase-1: `minAreaRect` mengembalikan sudut **90°** pada halaman yang sebenarnya lurus → deskewer meng-rotasi **-90°** → baris teks kecil (header: No. Invoice, Tanggal, nama) **hilang dari deteksi PaddleOCR** (51 → 34 baris, header `0685/ISA/INV/VI/26` tidak terbaca → NuExtract mengarang `PO-578`/`2020-08-28`). Konsisten dengan temuan project `paddleocr` (agent_guide.md: deskew merusak → dimatikan). Sudut di normalisasi ulang (±90→0) + toggle. |
| 2 | **NuExtract3-GGUF = main, PaddleOCR = support** | Sesuai keinginan user & rencana `doc-validation/PLAN.md`. NuExtract output JSON; PaddleOCR hanya teks konteks. |
| 3 | **Teks OCR hanya konteks** | Gambar tetap sumber utama (bukti: notebook RapidOCR mengirim OCR garbled → nama jadi salah `PT. INTISOLUSINDO GABADI`). Selector filter confidence >= 0.6. |
| 4 | **Guard gambar <= 900k px** | llama.cpp CLIP/mmproj **segfault** di atas ~1 MP (temuan project doc-validation). Wajib `fit_image_for_vision`. |
| 5 | **Preprocessing pakai PyMuPDF** (bukan poppler) | Render PDF→gambar self-contained, zoom agar lebar >= 1800 px. |
| 6 | **PaddleX di-patch offline** | Memakai model lokal `~/.paddlex/official_models` (PP-OCRv6), tidak download. |
| 7 | **Quality gate non-strict** (`STRICT_QUALITY_GATE=False`) | Blur/resolusi tetap dicatat di hasil, tapi dokumen tetap diproses (anti miss). |
| 8 | **Venv baru khusus** + **model disalin** | Folder standalone (tidak bergantung path lama). |
| 9 | **`kwitansi_number` tidak wajib** | Dokumen kwitansi ekadata memang tidak mencetak No. Kwitansi → kalau wajib, hasil jujur (`null`) salah ditandai FAILED. |
| 10 | **Prioritas instruksi: OCR = sumber nilai persis** | Sebelumnya "gambar yang paling menentukan" → NuExtract mengalahkan OCR yang benar (`POBSP`→`POISSP`). Dibalik: teks OCR paling akurat untuk kode/angka/tanggal/nama; gambar hanya untuk layout bila OCR tak jelas. |
| 11 | **Koreksi field kode/nomor dari OCR** (`correct_codes_from_ocr`) | PaddleOCR lebih akurat untuk kode: field kode/nomor diperbaiki dari baris OCR ber-conf ≥0.95 yang hampir identik; NPWP diambil dari baris `NPWP:`. Nilai model yang sudah benar tidak dirusak (edit-distance kecil + conf tinggi). |
| 12 | **OCR per-layout (PP-DocLayoutV3)** | Deteksi zona `header_image`/`header`/`table` (offline). Band header/logo memakai wide-band `run_header` (bukan bbox zona — teks logo melewati batas zona), zona `table` dicrop + upscale 1.8x (cap 2400px). Model layout dimuat per dokumen lalu di-unload (hemat RAM). |
| 10 | **`results/`, model, venv di-gitignore** | Output + biner besar tidak di-commit. |

---

## 5. Skema Per Jenis Dokumen

Deteksi jenis dari nama file (fallback: `invoice`):
`Invoice`/`INV_` → invoice · `FP_`/`Faktur` → tax_invoice · `Kwitansi`/`Kuitansi` → kwitansi · `PO_` → purchase_order · `DO_`/`Surat Jalan` → delivery_order

| Jenis | Field |
|-------|-------|
| invoice | invoice_number, invoice_date, due_date, **po_number, po_date**, supplier_name, customer_name, total_amount, tax_amount, currency, line_items[] |
| purchase_order | po_number, po_date, **cf_code**, supplier_name, buyer_name, total_amount, currency |
| delivery_order | do_number, do_date, **po_number, po_date**, supplier_name, recipient_name, delivery_address, total_items |
| kwitansi | kwitansi_number, kwitansi_date, company_name, invoice_reference, total_value, materai_present |
| tax_invoice | tax_invoice_number, invoice_date, supplier_name, supplier_tax_number, total_amount, tax_amount |

---

## 6. Cara Pakai

```bash
cd nu-paddle

# Batch SEMUA dokumen contoh (8 dokumen, ±12 menit)
./run.sh
# atau: venv/bin/python run_batch.py

# Dokumen tertentu
./run.sh "contoh invoice/Intisolusindo/Invoice_intisolusindo.pdf"

# UI Gradio (browser) — upload banyak dokumen sekaligus ATAU ekstrak semua contoh
./run.sh --ui                # http://127.0.0.1:7860
./run.sh --ui --share        # link publik sementara
```

Hasil: `results/<jenis>_<nama>.json` (JSON lengkap: preprocessing, fields,
normalised, validation, **seluruh teks OCR `ocr.all_text` + baris `ocr.lines`**,
waktu). UI menampilkan tabel ringkasan + JSON detail; upload otomatis
memproses (tanpa klik), tombol **"Ekstrak Semua Contoh"** memproses semua
dokumen contoh. Tidak ada pemilihan model — keduanya dipakai otomatis.

---

## 7. Hasil Fase-1 (8/8 PASSED)

Waktu rata-rata ± 87 detik/dokumen (CPU; PaddleOCR ~15-17s + NuExtract ~45-70s).

| Dokumen | Jenis | Status | Catatan kualitas |
|---------|-------|--------|------------------|
| Invoice_intisolusindo | invoice | PASSED | `0685/ISA/INV/VI/26`, tanggal 2026-06-02, supplier INTI SOLUSINDO, customer Surya Multi Cemerlang, total 116550 — **semua benar** |
| DO_intisolusindo | delivery_order | PASSED | 0692/ISA/DO/VI/26, semua benar; supplier minor typo logo "JNTI" |
| PO_intisolusindo | purchase_order | PASSED | POBSP-260529-000003, supplier/buyer benar |
| Kuitansi_intisolusindo | kwitansi | PASSED | KWT/0685-ISA/V1/2026 (V1 vs VV minor), total 116550 |
| FP_intisolusindo | tax_invoice | PASSED | No. FP benar; NPWP salah baca digit |
| INV_Ekadata | invoice | PASSED | invoice_number `12032` (INVOICE RETAIL-12032-AKSESCALL, benar); statement No. terpisah |
| FP_ekadata | tax_invoice | PASSED | No. FP benar; NPWP salah baca digit |
| KUITANSI_Ekadata | kwitansi | PASSED | Tanpa No. Kwitansi di dokumen (jujur `null`, validasi disesuaikan) |

**Perbaikan vs notebook RapidOCR:** field nama yang sebelumnya rusak
(`PT. INTISOLUSINDO GABADI`, `PT.JINTI SOLUSINDOABADI`) sekarang benar
(`PT. INTI SOLUSINDO ABADI`, `PT. Surya Multi Cemerlang`,
`PT. Platinum Ceramics Industry`).

**Masalah yang masih diketahui:**
1. Teks yang hanya ada di **logo/resolusi rendah** bisa kehilangan spasi/titik
   (DO supplier `PT.INTI SOLUSINDO ABADI` tanpa spasi setelah "PT.",
   FP supplier `PRIMEDIA ARMOEKADATA INTERNET`). Sudah jauh membaik lewat
   **OCR pas kedua area header** (`run_header`), nilai asli ada di
   `ocr.header_text`.
2. NPWP diambil dari baris OCR berlabel `NPWP:` (lebih andal), tapi belum
   dinormalisasi ke format 15-digit.
3. **RAM**: batch 8 dokumen dalam satu proses peak ±5.5GB; jika mesin sedang
   dipakai proses lain (RAM tersedia < 6GB), batch bisa kena OOM-kill — jalankan
   per-grup bila perlu.
4. `kwitansi_number` di Kuitansi bisa terbaca `V1` vs `VV` (huruf I/V kecil).

**Perbaikan yang sudah diterapkan (revisi):**
- `po_number`/`po_date` (No. & Tanggal PO/SPK) ditambah di schema invoice & DO.
- Prioritas instruksi: OCR = sumber nilai persis, gambar = cadangan.
- Koreksi field kode/nomor dari OCR ber-conf tinggi (`correct_codes_from_ocr`),
  termasuk NPWP dari baris `NPWP:`.
- OCR pas kedua area header (`run_header`): crop 22% atas + upscale 2x untuk
  membaca nama perusahaan di dalam logo (mis. `PT.INTI SOLUSINDO ABADI`).
- `due_date==po_date` yang duplikat dibuang; `kwitansi_number` hanya diisi jika
  ada nomor kwitansi tersendiri; `invoice_date` = tanggal terbit (bukan Due Date).

---

## 8. Integrasi VLM (GLM-4.6V-Flash) — Tulisan Tangan & Coretan

> **Model:** `zai-org/glm-4.6v-flash` (GLM-4.6V-Flash, Z.AI/Zhipu) via **API lokal**
> `http://10.0.1.250:1234/v1` (OpenAI-compatible, client `openai`). Model TIDAK
> dimuat lokal — cukup HTTP. Config: blok `VLM_API_*` di `config.py` (env
> `VLM_API_BASE_URL`/`VLM_API_KEY`/`VLM_API_MODEL`).

**Masalah yang dipecahkan:** PaddleOCR tidak bisa baca tulisan tangan dan salah
memahami kata/angka yang **dicoret (ciretan)** dengan **pembetulan** di
sebelahnya; NuExtract pada gambar ter-downscale bisa mengambil nilai yang dicoret
sebagai hasil akhir.

**Alur per dokumen (di `run_batch.process_one`):**

```text
PaddleOCR (teks cetak) + zone OCR (header/tabel)
   -> A) heuristik tulisan tangan/coretan (selection/selector, GRATIS, 0 API)
        |   bersih -> SKIP VLM -> langsung NuExtract (fast path, 0 call)
        |   mencurigakan ->
        v
   -> B) VLM detect (YA/TIDAK, max_tokens 1024, reasoning ikut dihitung)
        |   TIDAK -> SKIP
        |   YA ->
        v
   -> C) VLM read -> corrections[] + handwritten_notes + cleaned_text
        |   cleaned_text di-prepend "[KOREKSI]" ke teks OCR
        v
   NuExtract3-GGUF ekstrak JSON (instruksi: "pakai nilai pembetulan,
   jangan yang dicoret" dari extraction/schemas.py)
        |
   -> D) VLM review field: override field memakai nilai pembetulan
        |   nilai asli disimpan di result["vlm"]["review_changes"] (audit)
        v
   normalise -> validate -> results/
```

Hasil VLM dicatat di `result["vlm"]` (enabled/used/detected/skip_reason/error/
corrections/handwritten_notes/review_changes/elapsed_seconds). Kegagalan API
=> degrade halus: error tercatat, dokumen TETAP diproses (anti miss).

**Temuan teknis (uji sintetis + doc nyata):**
- Output GLM dibungkus `<|begin_of_box|>/<|end_of_box|>` dan model memakai
  `reasoning_content` dulu (gateway lokal MENGABAIKAN `thinking.disabled`) —
  `max_tokens` deteksi diperbesar (1024) + fallback parse dari reasoning.
- Foto halaman padat (banyak teks) membuat reasoning panjang sehingga budget
  token terpotong => `VLM_API_MAX_NEW_TOKENS=8192` (dokumentasi/pojok berisi
  coretan butuh budget besar).
- Heuristik dipakai 2 sinyal: confidence OCR **< 0.35** ATAU teks pendek dengan
  rasio simbol tinggi. Tanda baca umum cetak `(Rp)`, `PPH (%)` diabaikan.
  Semua 8 dokumen contoh = 0 baris mencurigakan => **0 panggilan API**.
- Uji sintetis (nilai `116.550` dicoret + pembetulan `125.000`): detect=YA,
  read menangkap `crossed_out=116.550, corrected=125.000`, review mengoreksi
  field `total_amount`; NuExtract mengambil nilai pembetulan.
- **Uji gambar asli (2 foto WhatsApp, test saja, bukan dokumen finance):**
  detect=YA pada keduanya. Read berhasil membaca coretan + pembetulan:
  gambar 1 -> `150 27001 : 2022`→`ISO/IEC 27001 ...`, `1EC`→`IEC`,
  `pelathan`→`security`; gambar 2 -> `asuransi`→`supplier`. cleaned_text
  memakai nilai pembetulan. (GLM terbukti jalan untuk tulisan tangan/coretan.)
- **Prompt read diperbaiki (2 aturan interpretasi):** tulisan tangan bisa
  PEMBETULAN (teks KETIK dicoret -> diganti tulisan tangan) ATAU TAMBAHAN kata
  (insertion, tanpa coretan). Output JSON kini punya `corrections[]` +
  `insertions[]` (struktur `{inserted, location}`). Test: insertion `security`
  di poin 2.11 terbaca benar; `(segregation of duties)`/`(pelatihan security
  awareness)` di gambar 1 terbaca benar.
- **Perbandingan model (gambar asli, data nyata):**
  - `qwen/qwen3-vl-30b` lebih akurat baca tulisan tangan: ejaan tepat
    (`segregation of duties`, `pelatihan security awareness`, `pihak ketiga`)
    dan lokasi poin benar.
  - `glm-4.6v-flash` (flash) membaca isi tapi ejaan sering garbled
    (`regregation`, `pelathan`, `pihak kelga`).
  - Keduanya masih keliru pada atribusi kata yang dicoret tanda X/garis
    (bukan nama kata yang salah dibaca, tapi arah coretan-vs-pembetulan).
  - Ganti model cukup ubah env `VLM_API_MODEL` (config model-agnostic).
- **Preprocessing membantu:** kirim gambar hasil `preprocess()` (bukan foto
  mentah) memperbaiki keterbacaan tulisan tangan. Di pipeline produksi VLM
  sudah menerima gambar ter-preprocess (`run_batch.prepare_image`).
- Keterbatasan: font yang di-render tidak bisa "menipu" PaddleOCR (conf tetap
  tinggi) — untuk pemicu penuh butuh tulisan tangan asli (sudah teruji pada
  foto asli). Threshold di `selection/selector.py` mudah di-tuning.

---

## 9. Roadmap

- **Fase-2 (selesai):** UI Gradio langsung di `run_batch.py --ui` + `run.sh`.
  - Upload banyak dokumen (ekstrak semua upload).
  - Tombol "Ekstrak Semua Contoh" (proses semua dokumen di `contoh invoice/`).
  - Model singleton: dimuat sekali, dipakai ulang antara klik (tanpa load ulang).
  - UI hanya memakai `models.registry` (pola AI-Document/app.py).
- **Fase-3 (selesai):**
  - `po_number`/`po_date` (No. & Tanggal PO/SPK) di schema invoice & DO.
  - OCR per-layout **PP-DocLayoutV3**: zona `table` dicrop + upscale; band
    header/logo `run_header` (nama di dalam logo terbaca, mis. `PT.INTI
    SOLUSINDO ABADI`). Model layout di-unload tiap dokumen (hemat RAM).
  - Prioritas instruksi OCR sebagai sumber nilai persis + `correct_codes_from_ocr`
    (kode/nomor + NPWP dari baris `NPWP:`).
  - `due_date` duplikat (`==po_date`) dibuang; `kwitansi_number` kosong bila
    tidak ada; `invoice_date` = tanggal terbit (bukan Due Date).
  - Temuan: zona `header_image` berisi logo grafis tanpa teks; nama perusahaan
    melewati batas zona `header_image`/`header` → pakai wide-band crop.
- **Fase-4 (selesai):** integrasi VLM GLM-4.6V-Flash via API untuk tulisan
  tangan/coretan (section 8). Model di `models/vlm_api_model.py`,
  dipanggil selektif (heuristik -> detect -> read -> review).
- **Fase-5 (opsional):** normalisasi NPWP 15 digit, penanganan spasi pada nama
  dari logo (`PT.INTI` → `PT. INTI`), indikator progress di UI untuk batch
  panjang, pengurangan RAM agar batch 8 dokumen bisa satu proses
  (saat ini jalankan per-grup bila RAM tersedia < 6GB), verifikasi VLM dengan
  dokumen asli ber-coretan (menunggu data dari user) + tuning threshold
  heuristik.
