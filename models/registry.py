"""Registry — satu-satunya tempat daftar model yang aktif.

Untuk menambah/mengganti model: cukup import class baru dan tambahkan/ganti
satu baris di ``AVAILABLE_MODELS``. Tidak ada file lain yang perlu diubah.

Peran:
    NuExtract3-GGUF -> "main"   (sumber hasil akhir, JSON terstruktur)
    PaddleOCR       -> "support" (sumber teks pendukung + confidence)
"""

from .nuextract_gguf_model import NuExtract3GGUFModel
from .paddleocr_model import PaddleOCRModel

AVAILABLE_MODELS = {
    "NuExtract3-GGUF": NuExtract3GGUFModel,
    "PaddleOCR": PaddleOCRModel,
}
