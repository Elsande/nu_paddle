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
│   ├── paddleocr_model.py      #   PaddleOCR (role="support")
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
   + run_header(): OCR pas kedua area header/logo (crop atas + upscale 2x)
     khusus membaca nama perusahaan di dalam logo
   │
   ▼
3. selector.evaluate() — TANPA aturan pemblokiran:
   teks kosong  -> teks OCR dilewati
   selebihnya   -> teks OCR disertakan
   │
   ▼
4. NuExtract3-GGUF (main)
   gambar (di-fit <=900k px) + teks OCR (gabungan halaman + header) + schema
   -> JSON terstruktur (field)
   │
   ▼
5. koreksi dari OCR: `correct_codes_from_ocr()` perbaiki field kode/nomor &
   NPWP (dari baris OCR ber-conf tinggi) -> normalisasi (tanggal -> ISO,
   uang -> angka) -> validasi (field wajib, angka positif, tanggal valid)
   -> PASSED/FAILED
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

## 8. Roadmap

- **Fase-2 (selesai):** UI Gradio langsung di `run_batch.py --ui` + `run.sh`.
  - Upload banyak dokumen (ekstrak semua upload).
  - Tombol "Ekstrak Semua Contoh" (proses semua dokumen di `contoh invoice/`).
  - Model singleton: dimuat sekali, dipakai ulang antara klik (tanpa load ulang).
  - UI hanya memakai `models.registry` (pola AI-Document/app.py).
- **Fase-3 (opsional):** normalisasi NPWP 15 digit, perbaikan logo/name-field
  (crop + OCR khusus header), penanganan invoice_date/due_date yang ambigu,
  indikator progress di UI untuk batch panjang.
