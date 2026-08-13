# Arsitektur Pipeline `doc-validation` — Dokumentasi Lengkap

Dokumen ini menjelaskan alur final pipeline ekstraksi & validasi dokumen bisnis Indonesia (Invoice, Kwitansi, Faktur Pajak, Berita Acara, Delivery Order, PO), lengkap dengan alasan tiap keputusan arsitektur, model yang dipakai, dan prompt siap pakai.

---

## 1. Ringkasan Arsitektur

```
[Gambar Dokumen]
        │
        ├──► paddleocr-vl 1.6       (selalu jalan, ringan)
        ├──► qwen3-vl-30b        (selalu jalan, berat)
        ├──► glm-4.6v-flash      (selalu jalan, berat)
        └──► NuExtract3          (selalu jalan, ringan)
                    │
                    ▼
        ┌───────────────────────────┐
        │      FUSION LAYER          │  ← rule-based, field-aware
        └───────────────────────────┘
                    │
        (kondisional, hanya jika ada konflik qwen3-vl vs glm-4.6v)
                    │
                    ▼
        gemma4:31b (tie-breaker, on-demand)
                    │
                    ▼
        NuExtract3 (Structuring & Validasi Format)
                    │
                    ▼
        Arithmetic Validator
                    │
                    ▼
        ERP Fuzzy Matching
                    │
                    ▼
        [OUTPUT FINAL: JSON valid / flag review manual]
```

**Total model yang dipakai: 5** — paddleocr-vl 1.6, qwen3-vl-30b, glm-4.6v-flash, NuExtract3 (dipakai 2x di 2 slot berbeda), dan gemma4:31b (on-demand saja).

---

## 2. Kenapa Arsitekturnya Begini — Penjelasan Tiap Keputusan

### 2.1. Kenapa 4 sumber ekstraksi jalan PARALEL, bukan berurutan?

Keempat sumber (paddleocr-vl 1.6, qwen3-vl-30b, glm-4.6v-flash, NuExtract3) sama-sama membaca **gambar dokumen yang sama**, dari titik nol, secara independen — tidak ada satupun yang butuh hasil dari sumber lain untuk mulai bekerja.

- Kalau dijalankan berurutan (satu selesai, baru yang berikutnya mulai), total waktu proses = penjumlahan waktu semua model = lambat.
- Kalau dijalankan paralel, total waktu proses = waktu model yang paling lambat saja.

**Kesimpulan:** paralel adalah pilihan yang benar karena tidak ada dependency antar sumber di tahap ini.

### 2.2. Kenapa harus 4 sumber, bukan cukup 1?

Setiap sumber punya kekuatan dan kelemahan berbeda — kombinasinya saling menutupi:

| Sumber | Kekuatan | Kelemahan |
|---|---|---|
| paddleocr-vl 1.6 | Akurat di level karakter/posisi (bounding box) | Tidak paham konteks/makna, gagal di layout rumit |
| qwen3-vl-30b | Paham konteks semantik, bisa baca tulisan tangan | Berat, kadang halusinasi angka |
| glm-4.6v-flash | Kuat di dense document/table parsing, arsitektur beda dari Qwen (bagus untuk cross-check) | Berat |
| NuExtract3 | Sangat konsisten mengikuti schema JSON, ringan (4B) | **Tidak bisa membaca tulisan tangan sama sekali** |

Kalau cuma pakai 1 sumber, error dari sumber itu langsung lolos tanpa ada yang mengoreksi. Dengan 4 sumber, **fusion layer** bisa membandingkan dan mengambil keputusan per-field berdasarkan siapa yang paling bisa dipercaya untuk jenis field tersebut.

### 2.3. Kenapa paddleocr-vl 1.6 dan NuExtract3 "selalu jalan", tapi gemma4:31b cuma "on-demand"?

Alasannya murni **cost/beban komputasi**, bukan soal akurasi:

- **paddleocr-vl 1.6** bukan LLM generatif — dia OCR engine biasa, sangat ringan dan cepat. Tidak ada alasan untuk tidak menjalankannya tiap dokumen.
- **NuExtract3** cuma 4 miliar parameter — jauh lebih kecil dari qwen3-vl-30b (30B) atau glm-4.6v-flash. Biaya menjalankan dia tiap dokumen kecil.
- **gemma4:31b** setara besar dengan qwen3-vl-30b (puluhan miliar parameter) — mahal kalau dipanggil di setiap dokumen, padahal di sebagian besar kasus qwen3-vl-30b dan glm-4.6v-flash sudah **sepakat** satu sama lain. Maka gemma4:31b hanya dipanggil sebagai **tie-breaker**, khusus untuk field yang hasilnya berbeda antara qwen3-vl-30b dan glm-4.6v-flash — bukan untuk seluruh dokumen.

**Alur kondisionalnya:**
```
qwen3-vl-30b dan glm-4.6v-flash selesai
        │
        ▼
Ada field yang hasilnya BEDA di antara keduanya?
        │
   ┌────┴────┐
  Tidak      Ya
   │          │
langsung   panggil gemma4:31b, HANYA untuk field
ke fusion  yang berbeda tersebut (bukan re-run
           seluruh dokumen)
   │          │
   └────┬─────┘
        ▼
   Fusion layer
```

### 2.4. Kenapa NuExtract3 tidak boleh "vote" di field tulisan tangan?

NuExtract3 memang **tidak dilatih untuk membaca tulisan tangan/handwriting** — ini sudah dikonfirmasi lewat pengecekan langsung. Kalau dipaksa memberi nilai untuk field yang sebenarnya berasal dari coretan tangan (misalnya koreksi manual, paraf, catatan tambahan di margin dokumen), dia hanya akan:
- Mengembalikan `null` (karena memang tidak terbaca olehnya), atau
- Menebak salah, yang justru **menurunkan confidence score** hasil fusion.

**Solusi:** fusion layer harus bersifat **field-aware** — dia tahu field mana yang boleh menerima suara dari sumber mana:

| Jenis Field | Sumber yang Boleh Vote |
|---|---|
| Teks cetak (nomor dokumen, tanggal, nominal, nama pihak, item) | paddleocr-vl 1.6, qwen3-vl-30b, glm-4.6v-flash, NuExtract3 (4 suara) |
| Tulisan tangan / anotasi / koreksi manual | qwen3-vl-30b, glm-4.6v-flash saja (2 suara, paddleocr-vl 1.6 & NuExtract3 di-exclude) |

### 2.5. Kenapa fusion layer HARUS ada — tidak bisa langsung ke output?

Karena setiap sumber punya cara baca dan tingkat kepercayaan berbeda per jenis field (lihat tabel di 2.2), harus ada satu titik keputusan yang menentukan, untuk setiap field, nilai mana yang paling bisa dipercaya. Tanpa fusion layer, sistem akan punya 4 versi data berbeda tanpa cara menentukan mana yang benar.

**Rule prioritas per jenis field (dipakai fusion layer):**

| Tipe Field | Prioritas Utama | Alasan |
|---|---|---|
| Numerik (harga, qty, total, subtotal, pajak) | VLM (qwen3-vl / glm-4.6v) | Paham konteks tabel, bisa membedakan "Total" vs "Subtotal" walau posisinya berdekatan |
| ID/Kode (nomor dokumen, NPWP, nomor PO) | paddleocr-vl 1.6 | Akurat di level karakter literal |
| Teks bebas (nama pihak, catatan) | VLM, dengan fallback ke paddleocr-vl 1.6 jika confidence VLM rendah | Butuh pemahaman konteks, tapi tetap perlu validasi karakter |
| Schema-critical (format wajib konsisten) | NuExtract3 sebagai tie-breaker | Dilatih khusus schema-adherence |
| Tulisan tangan / anotasi | qwen3-vl-30b, glm-4.6v-flash (majority vote 2 sumber) | Hanya sumber ini yang punya kapabilitas baca handwriting |

### 2.6. Kenapa NuExtract3 dipakai DUA KALI — di awal (extraction) dan di akhir (structuring)?

Ini poin penting yang sering disalahpahami. NuExtract3 dipakai di **dua slot berbeda dengan input berbeda**, bukan dipanggil dua kali untuk kerjaan yang sama:

**Slot 1 — Extraction (sebelum fusion):**
- Input: **gambar dokumen** (mentah)
- Tugas: baca teks cetak langsung dari gambar, hasilkan JSON semi-final
- Nilai unik di sini: dia jadi salah satu "opini independen" tambahan yang membaca gambar yang sama dengan sudut pandang berbeda dari paddleocr-vl 1.6/qwen3-vl/glm-4.6v — menambah keragaman suara untuk voting di fusion layer, khususnya karena kekuatan schema-adherence-nya.

**Slot 2 — Structuring & Validasi Format (setelah fusion):**
- Input: **teks/JSON hasil fusion** (bukan gambar lagi)
- Tugas: normalisasi hasil fusion ke schema JSON final (format tanggal `YYYY-MM-DD`, angka jadi number murni, dsb)
- Nilai unik di sini: dia menggantikan qwen2.5-coder-32b/qwen3.6:35b yang jauh lebih besar, karena tugas reformat teks-ke-JSON tidak butuh model besar — kapabilitas vision NuExtract3 juga tidak dipakai sama sekali di slot ini (karena memang tidak ada gambar), murni soal efisiensi kecepatan/biaya.

**Kenapa TIDAK ditaruh HANYA di satu slot saja:**
- Kalau NuExtract3 hanya dipakai di slot extraction: peran structuring akhir tetap harus diisi model lain (qwen2.5-coder/qwen3.6:35b) yang jauh lebih besar — boros compute padahal NuExtract3 sendiri sudah cukup ringan dan mampu untuk kerjaan itu.
- Kalau NuExtract3 hanya dipakai di slot structuring: kapabilitas vision-nya (baca gambar langsung ke schema) jadi sama sekali tidak dimanfaatkan, padahal itu justru kekuatan utamanya.

**Kesimpulan:** memakainya di dua slot bukan duplikasi kerja — di slot 1 dia adalah salah satu "mata" (pembaca gambar), di slot 2 dia adalah "tangan" (perapi format). Dua peran berbeda, dua nilai berbeda, dan justru inilah yang membuat arsitektur ini jadi jauh lebih ringan dibanding memakai qwen2.5-coder-32b/qwen3.6:35b di slot structuring.

### 2.7. Kenapa Arithmetic Validator dan ERP Fuzzy Matching tetap terpisah di akhir, bukan digabung ke fusion/structuring?

- **Arithmetic Validator** mengecek logika matematis (qty × harga = subtotal baris, jumlah subtotal + pajak = total) — ini butuh SEMUA field sudah final dan dalam format angka murni. Kalau dijalankan sebelum structuring selesai, datanya belum tentu bersih (masih ada "Rp", titik ribuan, dsb), jadi hasil pengecekan bisa salah.
- **ERP Fuzzy Matching** mencocokkan data ke sistem ERP (PO number, nama vendor, kode barang) — ini butuh data yang sudah divalidasi secara matematis dulu, supaya tidak mencocokkan data yang sudah jelas salah secara logika.

Urutan ini memastikan setiap tahap bekerja dengan data yang sudah "bersih" dari tahap sebelumnya, sehingga error tidak menumpuk atau menyebar ke tahap berikutnya.

---

## 3. Alur Step-by-Step Lengkap

**Step 1 — Input**
Dokumen (Invoice/Kwitansi/Faktur Pajak/Berita Acara/Delivery Order/PO) masuk sebagai gambar (hasil scan/foto).

**Step 2 — Ekstraksi Paralel (4 sumber, selalu jalan bersamaan)**
- paddleocr-vl 1.6 membaca teks cetak berbasis deteksi karakter/posisi.
- qwen3-vl-30b membaca teks cetak dan tulisan tangan secara semantik.
- glm-4.6v-flash membaca teks cetak dan tulisan tangan sebagai pembanding arsitektur berbeda.
- NuExtract3 membaca teks cetak langsung ke format JSON schema-strict.

Keempatnya menghasilkan output JSON per-field dengan confidence score masing-masing.

**Step 3 — Pengecekan Konflik**
Sistem membandingkan hasil qwen3-vl-30b dan glm-4.6v-flash secara khusus, field per field.
- Jika semua field sepakat → lanjut ke Step 5.
- Jika ada field yang berbeda → lanjut ke Step 4.

**Step 4 — Tie-Breaking (kondisional)**
gemma4:31b dipanggil, tapi HANYA untuk field yang berbeda dari Step 3 — bukan seluruh dokumen. Hasilnya dipakai sebagai suara ketiga untuk majority vote di field tersebut.

**Step 5 — Fusion Layer**
Sistem menggabungkan semua hasil ekstraksi menjadi satu representasi konsolidasi, dengan aturan:
- Field teks cetak: voting dari paddleocr-vl 1.6, qwen3-vl-30b, glm-4.6v-flash, NuExtract3 (dan gemma4:31b jika dipanggil).
- Field tulisan tangan: voting hanya dari qwen3-vl-30b dan glm-4.6v-flash (dan gemma4:31b jika dipanggil) — paddleocr-vl 1.6 dan NuExtract3 di-exclude dari field ini.
- Setiap field diberi label `source_used` dan `had_conflict` untuk keperluan audit.

**Step 6 — Structuring & Validasi Format**
NuExtract3 (slot kedua, menerima teks/JSON hasil fusion — bukan gambar) merapikan hasil fusion menjadi JSON final sesuai schema baku: format tanggal `YYYY-MM-DD`, angka murni tanpa simbol mata uang/pemisah ribuan, field yang masih ada konflik ditandai di `review_flags`.

**Step 7 — Arithmetic Validator**
Sistem mengecek logika matematis dokumen: apakah qty × harga satuan = subtotal baris, apakah jumlah semua subtotal + pajak = total. Jika tidak cocok, field terkait ditandai untuk review manual.

**Step 8 — ERP Fuzzy Matching**
Data yang sudah tervalidasi dicocokkan ke data ERP (nomor PO, nama vendor, kode barang) menggunakan fuzzy matching untuk menangani variasi penulisan/typo.

**Step 9 — Output Final**
Hasil akhir berupa JSON terstruktur, siap dipakai sistem downstream, atau ditandai "perlu review manual" jika ada field dengan confidence rendah atau tidak cocok dengan data ERP.

---

## 4. Prompt Siap Pakai

### 4.1. Prompt Extraction — VLM (qwen3-vl-30b & glm-4.6v-flash)

Gunakan prompt yang SAMA PERSIS untuk kedua model ini, supaya hasilnya bisa dibandingkan secara adil di fusion layer.

```
SYSTEM:
Kamu adalah sistem ekstraksi dokumen bisnis Indonesia. Tugasmu HANYA membaca
dan mengekstrak informasi yang benar-benar terlihat di gambar. Jangan
menebak, menghitung, atau melengkapi data yang tidak tertulis.

Aturan:
1. Baca seluruh teks tercetak (typed/printed text) di dokumen.
2. Jika ada tulisan tangan (handwriting), coretan, atau catatan tambahan,
   ekstrak TERPISAH dari teks cetak dan tandai dengan field "is_handwritten": true.
3. Jika ada teks yang dicoret (strikethrough) dan diganti tulisan tangan,
   laporkan KEDUA versi: nilai asli (dicoret) dan nilai koreksi (tulisan tangan).
4. Jika suatu field tidak terbaca jelas atau tidak ada di dokumen, isi
   dengan null — JANGAN mengarang nilai.
5. Untuk setiap field, berikan confidence 0.0-1.0 berdasarkan kejelasan
   visual teks tersebut (bukan berdasarkan asumsi logis).
6. Jangan menerjemahkan istilah Indonesia ke Inggris. Pertahankan bahasa asli.

Output HARUS berupa JSON valid saja. Tidak ada teks pembuka, penutup,
atau penjelasan di luar JSON.

USER:
Berikut gambar dokumen bisnis. Ekstrak seluruh informasi yang terlihat
ke format berikut:

{
  "jenis_dokumen_terdeteksi": "<string atau null>",
  "nomor_dokumen": {"value": "<string atau null>", "confidence": 0.0},
  "tanggal": {"value": "<string atau null>", "confidence": 0.0},
  "pihak_pengirim": {"value": "<string atau null>", "confidence": 0.0},
  "pihak_penerima": {"value": "<string atau null>", "confidence": 0.0},
  "items": [
    {
      "nama_barang_jasa": {"value": "<string>", "confidence": 0.0},
      "qty": {"value": "<number atau null>", "confidence": 0.0},
      "satuan": {"value": "<string atau null>", "confidence": 0.0},
      "harga_satuan": {"value": "<number atau null>", "confidence": 0.0},
      "subtotal_baris": {"value": "<number atau null>", "confidence": 0.0}
    }
  ],
  "subtotal": {"value": "<number atau null>", "confidence": 0.0},
  "pajak": {"value": "<number atau null>", "confidence": 0.0},
  "total": {"value": "<number atau null>", "confidence": 0.0},
  "catatan_tercetak": {"value": "<string atau null>", "confidence": 0.0},
  "anotasi_tulisan_tangan": [
    {
      "teks": "<string>",
      "lokasi_referensi": "<field/bagian mana yang dianotasi>",
      "jenis": "<koreksi|catatan_tambahan|paraf|tidak_diketahui>",
      "confidence": 0.0
    }
  ]
}
```

### 4.2. Prompt Extraction — NuExtract3 (schema template, teks cetak saja)

NuExtract3 tidak perlu instruksi panjang seperti VLM di atas — dia dilatih untuk langsung mengikuti JSON template. Cukup berikan gambar + template berikut (tanpa field `anotasi_tulisan_tangan`, karena dia tidak akan bisa mengisinya):

```
Template:
{
  "jenis_dokumen_terdeteksi": "string",
  "nomor_dokumen": "string",
  "tanggal": "string",
  "pihak_pengirim": "string",
  "pihak_penerima": "string",
  "items": [
    {
      "nama_barang_jasa": "string",
      "qty": "number",
      "satuan": "string",
      "harga_satuan": "number",
      "subtotal_baris": "number"
    }
  ],
  "subtotal": "number",
  "pajak": "number",
  "total": "number",
  "catatan_tercetak": "string"
}

Instruksi: Ekstrak hanya teks tercetak/typed dari gambar dokumen ini
sesuai template di atas. Jangan menebak nilai yang tidak terlihat jelas —
isi dengan null.
```

### 4.3. Konfigurasi paddleocr-vl 1.6 (bukan prompt, tapi format output wajib)

```
Output paddleocr-vl 1.6 wajib di-post-process ke struktur:
{
  "raw_text_blocks": [
    {"text": "<string>", "bbox": [x1,y1,x2,y2], "confidence": 0.0}
  ]
}
```
Mapping `raw_text_blocks` ke field-field yang sama seperti schema VLM di atas dilakukan lewat heuristik posisi/label terdekat.

### 4.4. Prompt Fusion Layer (rule-based, direkomendasikan — tanpa LLM)

```python
FIELD_PRIORITY = {
    "numeric": ["vlm_qwen3", "vlm_glm", "gemma_tiebreak", "paddleocr-vl 1.6", "nuextract3"],
    "id_code": ["paddleocr-vl 1.6", "vlm_qwen3", "vlm_glm", "nuextract3"],
    "free_text": ["vlm_qwen3", "paddleocr-vl 1.6", "vlm_glm", "nuextract3"],
    "schema_critical": ["nuextract3", "vlm_qwen3", "vlm_glm"],
    "handwriting": ["vlm_qwen3", "vlm_glm", "gemma_tiebreak"],  # paddleocr-vl 1.6 & nuextract3 di-exclude
}
CONFIDENCE_OVERRIDE_THRESHOLD = 0.3

def fuse_field(field_name, field_type, sources: dict):
    # sources hanya berisi sumber yang diizinkan untuk field_type ini
    values = {k: v["value"] for k, v in sources.items() if v["value"] is not None}
    if len(set(values.values())) <= 1:
        return {"value": next(iter(values.values()), None),
                "confidence": avg([v["confidence"] for v in sources.values()]),
                "source_used": "consensus", "had_conflict": False}

    ordered = FIELD_PRIORITY[field_type]
    ordered = [s for s in ordered if s in sources]  # hanya sumber yang tersedia
    best_source = max(ordered, key=lambda s: sources[s]["confidence"])
    default_source = ordered[0]
    winner = best_source if (
        sources[best_source]["confidence"] - sources[default_source]["confidence"]
        > CONFIDENCE_OVERRIDE_THRESHOLD
    ) else default_source

    return {"value": sources[winner]["value"],
            "confidence": sources[winner]["confidence"],
            "source_used": winner, "had_conflict": True,
            "alternatives": {k: v["value"] for k, v in sources.items() if k != winner}}
```

### 4.5. Prompt Structuring & Validasi Format — NuExtract3 (slot kedua)

```
Template:
{
  "jenis_dokumen": "string",
  "nomor_dokumen": "string",
  "tanggal": "string (format YYYY-MM-DD)",
  "pihak_pengirim": "string",
  "pihak_penerima": "string",
  "items": [
    {"nama": "string", "qty": "number", "satuan": "string",
     "harga_satuan": "number", "subtotal_baris": "number"}
  ],
  "subtotal": "number",
  "pajak": "number",
  "total": "number",
  "catatan": "string",
  "anotasi_tulisan_tangan": [
    {"teks": "string", "lokasi_referensi": "string", "jenis": "string"}
  ],
  "review_flags": "array of string"
}

Instruksi: Input di bawah adalah hasil fusion dari beberapa sumber ekstraksi
dokumen (bukan gambar). Ubah menjadi JSON sesuai template:
- Normalisasi semua field numerik menjadi number murni (hilangkan "Rp",
  titik/koma pemisah ribuan).
- Normalisasi tanggal ke format YYYY-MM-DD.
- Field yang punya flag "had_conflict": true dari data input, masukkan
  nama field-nya ke "review_flags".
- Jangan menghitung ulang subtotal/total — hanya reformat nilai yang ada.

Input hasil fusion: {{fused_json}}
```

---

## 5. Rekomendasi Testing Sebelum Full Deploy

1. **A/B test NuExtract3 vs qwen2.5-coder-32b di slot structuring** — pakai sample hasil fusion yang sama, bandingkan output terutama di: normalisasi tanggal ambigu, normalisasi angka, dan kepatuhan mengisi `review_flags`.
2. **Uji akurasi handwriting** — pakai dokumen dengan anotasi tangan (seperti contoh Kebijakan Keamanan Informasi yang dites sebelumnya) untuk memastikan qwen3-vl-30b dan glm-4.6v-flash konsisten menangkap anotasi tersebut.
3. **Ukur frekuensi konflik qwen3-vl-30b vs glm-4.6v-flash** — dari sample dokumen riil, hitung berapa persen dokumen yang benar-benar butuh gemma4:31b sebagai tie-breaker. Kalau frekuensinya sangat rendah, ini mengonfirmasi keputusan on-demand sudah tepat secara cost.
4. **Ukur latency end-to-end** — bandingkan waktu proses per dokumen antara arsitektur ini dengan versi sebelumnya (qwen2.5-coder-32b di slot structuring) untuk memastikan penghematan compute benar-benar terasa.