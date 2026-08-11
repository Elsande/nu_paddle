"""
config.py
=========
Konfigurasi terpusat untuk nu-paddle (NuExtract3-GGUF main + PaddleOCR support).
"""

from __future__ import annotations

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
# OCR PENDUKUNG — PaddleOCR
# ---------------------------------------------------------------------------
PADDLEOCR_PARAMS = {
    "lang": "en",
    "device": "cpu",
    "enable_mkldnn": False,          # hindari crash oneDNN/PIR di CPU
    "text_det_limit_side_len": 960,  # batasi ukuran saat deteksi -> hemat RAM
    "text_det_unclip_ratio": 2.0,    # cegah teks terpotong di tepi box
}

# Baris OCR dengan confidence di bawah ini dibuang (tidak dikirim ke NuExtract).
OCR_CONFIDENCE_THRESHOLD = 0.6

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
