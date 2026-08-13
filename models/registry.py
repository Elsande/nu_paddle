"""Registry — satu-satunya tempat daftar model yang aktif.

Untuk menambah/mengganti model: cukup import class baru dan tambahkan/ganti
satu baris di ``AVAILABLE_MODELS``. Tidak ada file lain yang perlu diubah.

Peran:
    PaddleOCR-VL 1.6 (API) -> "support" (OCR: teks + bbox + confidence)
    Qwen3-VL-30B (API)    -> "support" (ekstraksi VLM, dipakai pipeline)
    GLM-4.6V-Flash (API)  -> "support" (ekstraksi + tulisan tangan/coretan)
    NuExtract3-GGUF       -> "main"    (OFF sementara — lihat config.NUEXTRACT_ENABLED)

Semua model VLM dipanggil lewat API OpenAI-compatible yang SAMA
(``config.VLM_API_BASE_URL``); yang membedakan hanyalah nama model.
"""

from .nuextract_gguf_model import NuExtract3GGUFModel
from .paddleocr_model import PaddleOCRVLApiModel
from .vlm_api_model import Qwen3VLModel, VLMApiModel

AVAILABLE_MODELS = {
    "NuExtract3-GGUF": NuExtract3GGUFModel,
    "PaddleOCR-VL 1.6 (API)": PaddleOCRVLApiModel,
    "Qwen3-VL-30B (API)": Qwen3VLModel,
    "GLM-4.6V-Flash (API)": VLMApiModel,
}
