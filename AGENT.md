# AGENT.md — Aturan Kerja di Project nu-paddle

Project gabungan **doc-validation + paddleocr** dengan preprocessing dari
**AI-Document**. Arsitektur: NuExtract3-GGUF (model utama, JSON terstruktur)
dibantu PaddleOCR (support, sumber teks + confidence).

## Struktur & Aturan Wajib

1. **Entry point**: `run_batch.py` (batch CLI + UI Gradio) via `run.sh`.
   - `./run.sh` -> batch SEMUA dokumen contoh.
   - `./run.sh --ui` -> UI Gradio di browser.
   - UI dibangun di `run_batch.build_ui()`. Belum ada `app.py` terpisah.
2. **Registry adalah satu-satunya daftar model** (`models/registry.py`).
   Jangan import class model spesifik dari `run_batch.py`/UI — tidak ada
   percabangan if/elif berdasarkan nama model.
3. **Setiap dokumen WAJIB lewat `preprocessing/pipeline.preprocess()`**
   sebelum dikirim ke model mana pun. Quality gate dihitung & dicatat;
   `STRICT_QUALITY_GATE=False` (default) => dokumen tetap diproses (anti miss).
4. **Model utama**: `models/nuextract_gguf_model.py` (NuExtract3-GGUF,
   llama.cpp, Q4_K_M + mmproj). **Support**: `models/paddleocr_model.py`
   (PaddleOCR detect-then-recognize + CLAHE + spatial sort).
5. **PENTING (jangan diubah tanpa alasan teknis)**:
   - Gambar WAJIB melewati `fit_image_for_vision` (<= 900k px) sebelum dikirim
     ke NuExtract — CLIP/mmproj llama.cpp segfault di atas ~1 MP.
   - Teks OCR adalah KONTEKS, bukan sumber kebenaran. TIDAK ada aturan yang
     membatasi: selama OCR menghasilkan teks, teks SELALU dikirim ke NuExtract
     (lihat `selection/selector.py`). Confidence tetap disimpan di hasil.
   - **Deskew DINONAKTIFKAN** (`USE_DESKEW=False` di `preprocessing/pipeline.py`).
     minAreaRect bisa memberi sudut ~±90° pada halaman yang lurus dan meng-rotasi
     -90°, menghilangkan baris teks kecil (header) dari deteksi PaddleOCR.
     Jangan aktifkan tanpa uji.
   - PaddleX dipatch offline (`_patch_paddlex_offline`) supaya memakai model
     lokal `~/.paddlex/official_models` — jangan hapus, hindari download.
   - Chat template NuExtract: `extraction/nuextract3_chat_template.jinja`.
     Jangan ganti dengan template lain (GGUF tidak meng-embed-nya).
   - Schema/instruksi per jenis: `extraction/schemas.py` (dari notebook yang
     sudah terbukti akurat).
6. **Tidak ada file lain yang boleh diubah saat menambah model** selain
   `models/` + `models/registry.py`.

## Perintah Verifikasi

- Batch semua dokumen: `./run.sh` atau `venv/bin/python run_batch.py`
- Batch dokumen tertentu: `./run.sh "contoh invoice/Intisolusindo/Invoice_intisolusindo.pdf"`
- UI Gradio: `./run.sh --ui` (host 127.0.0.1:7860; `--ui --share` untuk link publik)
- Model dimuat SEKALI via singleton di `run_batch.py` (`get_ocr_model()` / `get_extractor()`). Jangan buat load/unload berulang dalam satu proses.

## Catatan Lingkungan

- Model NuExtract3-GGUF: `data/models/nuextract3/` (Q4_K_M 2.6G + mmproj 645M).
- Model PaddleOCR: `~/.paddlex/official_models` (jangan download ulang).
- RAM saat proses ±5–6GB (PaddleOCR + NuExtract3 dalam satu proses).
- `results/` tidak di-commit (lihat `.gitignore`).
