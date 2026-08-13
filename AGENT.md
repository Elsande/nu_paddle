# AGENT.md — Aturan Kerja di Project nu-paddle

Arsitektur baru (rombak): **paddleocr-vl 1.6**, **qwen3-vl-30b**, dan
**glm-4.6v-flash** semuanya dipanggil lewat API OpenAI-compatible yang SAMA
(`config.VLM_API_BASE_URL`, default `http://10.0.1.250:1234/v1`) — beda nama
model. **Tidak ada lagi PaddleOCR yang dimuat lokal.** NuExtract3-GGUF
(lokal) SEMENTARA OFF (`config.NUEXTRACT_ENABLED=False`).

## Struktur & Aturan Wajib

1. **Entry point**: `run_batch.py` (batch CLI + UI Gradio) via `run.sh`.
   - `./run.sh` -> batch SEMUA dokumen contoh.
   - `./run.sh --ui` -> UI Gradio di browser.
   - UI dibangun di `run_batch.build_ui()`. Belum ada `app.py` terpisah.
2. **Registry adalah satu-satunya daftar model** (`models/registry.py`):
   `PaddleOCR-VL 1.6 (API)`, `Qwen3-VL-30B (API)`, `GLM-4.6V-Flash (API)`,
   `NuExtract3-GGUF` (off). Jangan import class model spesifik dari
   `run_batch.py`/UI — tidak ada percabangan if/elif berdasarkan nama model.
3. **Setiap dokumen WAJIB lewat `preprocessing/pipeline.preprocess()`**
   sebelum dikirim ke model mana pun. Quality gate dihitung & dicatat;
   `STRICT_QUALITY_GATE=False` (default) => dokumen tetap diproses (anti miss).
4. **Model API (semua via `models/vlm_api_model.OpenAIVLModel`)**:
   - `models/paddleocr_model.py` (`PaddleOCRVLApiModel`): OCR pengganti
     PaddleOCR lokal. Interface `run()`/`run_region()`/`run_header()` tetap
     (dipakai `run_batch._zone_ocr`). Output model di-parse jadi baris
     `text`+`bbox`+`confidence` (default netral bila tidak ada confidence).
   - `models/vlm_api_model.py`: `OpenAIVLModel` (base) + `VLMApiModel` (GLM)
     + `Qwen3VLModel` (qwen3-vl-30b). Punya `extract_document()` (JSON sesuai
     `extraction/schemas.py`) dan langkah tulisan tangan/coretan
     (detect/read/review).
   - **Layout**: `models/layout_model.py` (PP-DocLayoutV3) MASIH lokal/offline.
     Dipakai untuk OCR zona tabel (`config.LAYOUT_ZONE_TARGETS`). Model layout
     dimuat per dokumen lalu di-UNLOAD (jaga RAM) — jangan jadikan singleton
     persistent.
   - **NuExtract3-GGUF**: `models/nuextract_gguf_model.py` — JANGAN diubah,
     tetap terdaftar tapi hanya dijalankan bila `config.NUEXTRACT_ENABLED=True`.
5. **PENTING (jangan diubah tanpa alasan teknis)**:
   - Gambar untuk NuExtract WAJIB melewati `fit_image_for_vision`
     (CLIP/mmproj segfault > ~1 MP). Untuk model API cukup
     `_encode_image()` (JPEG, batas `VLM_API_MAX_SIDE`).
   - Teks OCR adalah KONTEKS, bukan sumber kebenaran. TIDAK ada aturan
     pemblokiran: selama OCR menghasilkan teks, teks SELALU dikirim sebagai
     konteks (lihat `selection/selector.py`). Confidence tetap disimpan.
   - **Deskew DINONAKTIFKAN** (`USE_DESKEW=False` di
     `preprocessing/pipeline.py`).
   - **Fusion**: `fusion.py` — rule-based, field-aware (numeric/id_code/
     free_text). paddleocr-vl TIDAK masuk fusion per-field; ia lewat
     `validation.correct_codes_from_ocr` untuk field kode/nomor.
   - **Tulisan tangan/coretan**: heuristik `selection/selector.py`
     (`detect_handwriting_heuristic`) -> GLM `detect_handwriting` YA/TIDAK ->
     `read_handwriting` (koreksi + cleaned_text) -> `review_fields` (nilai
     pembetulan menang, nilai asli dicatat di `result["vlm"]["review_changes"]`).
   - **3 skrip test latency** (tidak memuat model, murni benchmark API):
     `test_paddleocr_vl_api.py`, `test_qwen3_vl_api.py`, `test_glm_api.py`.
6. **Hasil per dokumen**: `result["extraction_sources"]` (tiap sumber qwen3/
   glm beserta fields+confidence+elapsed), `result["fusion"]` (per-field
   `{value, confidence, source_used, had_conflict}`), `result["ocr"]`,
   `result["vlm"]`.

## Perintah Verifikasi

- Batch semua dokumen: `./run.sh` atau `venv/bin/python run_batch.py`
- Batch dokumen tertentu: `./run.sh "contoh invoice/Intisolusindo/Invoice_intisolusindo.pdf"`
- UI Gradio: `./run.sh --ui` (host 127.0.0.1:7860; `--ui --share` untuk link publik)
- Test latency API:
  - `python test_paddleocr_vl_api.py`
  - `python test_qwen3_vl_api.py`
  - `python test_glm_api.py`
- Model dimuat SEKALI via singleton di `run_batch.py` (`get_ocr_model()` /
  `get_qwen3_model()` / `get_glm_model()`). PENGECUALIAN: `get_layout_model()`
  dimuat per dokumen lalu di-unload (hemat RAM — lihat aturan 4).

## Catatan Lingkungan

- Model NuExtract3-GGUF: `data/models/nuextract3/` (tidak dimuat selama off).
- Model PP-DocLayoutV3: `~/.paddlex/official_models` (lokal, jangan download ulang).
- Model VLM (paddleocr-vl 1.6 / qwen3-vl-30b / glm-4.6v-flash) tidak dimuat
  lokal — cukup HTTP ke `config.VLM_API_BASE_URL`. Bila nama model di gateway
  berbeda, override via env: `PADDLEOCR_VL_MODEL`, `QWEN3_VL_MODEL`, `GLM_VL_MODEL`.
- `results/` tidak di-commit (lihat `.gitignore`).
