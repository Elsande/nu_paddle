"""
config.py
==========
Konfigurasi terpusat untuk nu-paddle.

Arsitektur baru:
- paddleocr-vl 1.6, qwen3-vl-30b, dan glm-4.6v-flash semuanya dipanggil
  LEWAT API OpenAI-compatible yang SAMA (gateway lokal perusahaan), beda nama
  model. Tidak ada lagi PaddleOCR yang dimuat lokal.
- NuExtract3-GGUF (lokal) sementara OFF (flag NUEXTRACT_ENABLED).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# MODEL UTAMA — NuExtract3-GGUF (llama.cpp)
# ---------------------------------------------------------------------------
MODEL_DIR = PROJECT_DIR / "data" / "models" / "nuextract3"
GGUF_MODEL_PATH = MODEL_DIR / "NuExtract3-Q4_K_M.gguf"
GGUF_MMPROJ_PATH = MODEL_DIR / "mmproj-NuExtract3-BF16.gguf"

# llama.cpp sampling / konteks.
N_CTX = 8192
N_THREADS = 0            # 0 = auto (pakai semua core)
N_GPU_LAYERS = 0         # 0 = CPU-only
TEMPERATURE = 0.0        # greedy / deterministik
MAX_NEW_TOKENS = 1024
ENABLE_THINKING = False  # non-reasoning, cepat & deterministik

# ---------------------------------------------------------------------------
# NuExtract3-GGUF (lokal) — SEMENTARA OFF.
# Pipeline sekarang memakai 3 sumber VLM via API. NuExtract tetap ada di
# registry (`models/registry.py`) tapi tidak dimuat/dijalankan selama False.
# ---------------------------------------------------------------------------
NUEXTRACT_ENABLED = False

# ---------------------------------------------------------------------------
# OCR PENDUKUNG — paddleocr-vl 1.6 via API (gateway OpenAI-compatible).
# Model TIDAK dimuat lokal; cukup HTTP client ringan.
# ---------------------------------------------------------------------------
# Baris OCR dengan confidence di bawah ini dibuang (tidak dikirim ke extractor).
OCR_CONFIDENCE_THRESHOLD = 0.6

# Prompt OCR untuk paddleocr-vl 1.6. Output model di-parse menjadi baris OCR
# (text + bbox + confidence) — sama seperti dulu output PaddleOCR lokal.
PADDLEOCR_VL_SYSTEM_PROMPT = (
    "Kamu adalah mesin OCR dokumen. Baca SELURUH teks yang terlihat pada gambar, "
    "jangan menebak atau melengkapi. Jangan terjemahkan bahasa aslinya."
)
PADDLEOCR_VL_USER_PROMPT = (
    "Lakukan OCR pada gambar dokumen ini dan kembalikan HANYA JSON valid dengan "
    'format: {"texts": [{"text": "<teks>", "bbox": [x1,y1,x2,y2] atau null, '
    '"confidence": 0.0}]}. Tulis seluruh teks yang terlihat, urutkan sesuai '
    "posisi baca (atas ke bawah, kiri ke kanan). Tidak ada teks lain di luar JSON."
)
# Confidence default bila model tidak mengembalikannya (netral: tidak memicu
# heuristik tulisan tangan dan tidak dipakai untuk koreksi kode OCR).
PADDLEOCR_VL_DEFAULT_CONFIDENCE = 0.5

# ---------------------------------------------------------------------------
# VLM API — paddleocr-vl 1.6, qwen3-vl-30b, glm-4.6v-flash (SATU gateway).
# Ketiga model dipanggil lewat API OpenAI-compatible yang SAMA
# (default gateway lokal perusahaan); yang membedakan hanyalah nama model.
# ---------------------------------------------------------------------------
VLM_API_ENABLED = True
# Base URL gateway lokal (OpenAI-compatible). Override via env var.
VLM_API_BASE_URL = os.environ.get("VLM_API_BASE_URL", "http://10.0.1.250:1234/v1")
VLM_API_KEY = os.environ.get("VLM_API_KEY", "lm-studio")
VLM_API_TIMEOUT = int(os.environ.get("VLM_API_TIMEOUT", "180"))

# Nama model di gateway (override via env var).
PADDLEOCR_VL_MODEL = os.environ.get("PADDLEOCR_VL_MODEL", "paddleocr-vl-1.6")
QWEN3_VL_MODEL = os.environ.get("QWEN3_VL_MODEL", "qwen3-vl-30b")
GLM_VL_MODEL = os.environ.get("GLM_VL_MODEL", "zai-org/glm-4.6v-flash")

# Sisi terpanjang maksimal gambar yang dikirim (fit penuh halaman).
VLM_API_MAX_SIDE = 1600
# Read memakai resolusi lebih tinggi (tulisan tangan kecil lebih terbaca).
VLM_API_READ_MAX_SIDE = 2400
VLM_API_JPEG_QUALITY = 92
VLM_API_MAX_NEW_TOKENS = 8192
VLM_API_TEMPERATURE = 0.0
# Nonaktifkan chain-of-thought (lebih cepat & deterministik).
VLM_API_DISABLE_THINKING = True

# ---------------------------------------------------------------------------
# LAYOUT ANALYSIS — PP-DocLayoutV3 (deteksi zona, offline)
# ---------------------------------------------------------------------------
LAYOUT_MODEL_NAME = "PP-DocLayoutV3"

# Zona yang ditambah OCR per-layout (DI ATAS OCR halaman penuh).
# Catatan: header/logo TIDAK memakai bbox zona — teks nama di logo melewati
# batas zona header_image/header, lebih akurat pakai wide-band (run_header).
LAYOUT_ZONE_TARGETS = frozenset({"table"})

# Faktor upscale saat OCR per zona (teks logo kecil butuh perbesaran lebih besar).
LAYOUT_ZONE_SCALE = {
    "header_image": 3.0,
    "table": 1.8,
    "header": 1.0,
    "text": 1.0,
    "footer": 1.0,
}

# Padding (px) di sekitar bbox zona saat crop, agar teks tepi tidak terpotong.
LAYOUT_ZONE_MARGIN = 20

# Ukuran maksimal sisi terpanjang hasil upscale zona (batasi runtime OCR zona).
LAYOUT_ZONE_MAX_DIM = 2400

# Fallback bila deteksi layout gagal/kosong (perilaku lama run_header).
LAYOUT_HEADER_FALLBACK_TOP_RATIO = 0.22
LAYOUT_HEADER_FALLBACK_SCALE = 2.0

# ---------------------------------------------------------------------------
# PREPROCESSING (salinan dari AI-Document/preprocessing)
# ---------------------------------------------------------------------------
# True  -> dokumen yang gagal quality gate DITOLAK (tidak diproses).
# False -> quality metrics tetap dicatat, dokumen tetap diproses (anti miss).
STRICT_QUALITY_GATE = False

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ---------------------------------------------------------------------------
# DIREKTORI
# ---------------------------------------------------------------------------
SAMPLE_DIR = PROJECT_DIR / "contoh invoice"
RESULTS_DIR = PROJECT_DIR / "results"

# ---------------------------------------------------------------------------
# DETEKSI JENIS DOKUMEN DARI NAMA FILE
# ---------------------------------------------------------------------------
DOC_TYPE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(?i)invoice|inv_"), "invoice"),
    (re.compile(r"(?i)factur|faktur|fp_"), "tax_invoice"),
    (re.compile(r"(?i)kwitansi|kuitansi|kwt"), "kwitansi"),
    (re.compile(r"(?i)\bpo[_\- ]|purchase"), "purchase_order"),
    (re.compile(r"(?i)\bdo[_\- ]|delivery|surat\s*jalan"), "delivery_order"),
]


def detect_document_type(file_path: str | Path) -> str:
    """Infer jenis dokumen dari nama file. Default: ``invoice``."""
    name = Path(file_path).stem
    for pattern, doc_type in DOC_TYPE_PATTERNS:
        if pattern.search(name):
            return doc_type
    return "invoice"
